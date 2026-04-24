"""
LangGraph quality loop: Generate → Evaluate → Refine → Regenerate.

This is the ONE place in the pipeline that is genuinely stateful, cyclical,
and benefits from agent-style orchestration. Everything else (resize, overlay,
compliance, file I/O) is deterministic and stays as plain Python functions.

Why LangGraph here:
  - State (prompt, score, feedback, attempt count) must persist across iterations
  - Routing is conditional on evaluation output, not a fixed retry schedule
  - Each node has a distinct responsibility and is independently testable
  - The cycle is bounded but variable length, not expressible as a for-loop cleanly
  - Future: add human-in-the-loop checkpoint between evaluate and refine
"""

import base64
import io
import logging
from typing import TypedDict

from PIL import Image
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.config.models import ProductConfig, RegionConfig
from src.generation.prompt_builder import build_image_prompt

# Module-level checkpointer: survives repeated calls within the same process.
# If the quality loop is interrupted mid-attempt (e.g. evaluator throws), calling
# run_quality_loop again with the same product.name resumes from the last saved
# node rather than restarting generation from scratch (saving Imagen 4 API calls).
# Future: replace with SqliteSaver / PostgresSaver for cross-process persistence.
_checkpointer = MemorySaver()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State schema: everything the graph needs, passed between nodes explicitly
# ---------------------------------------------------------------------------

class CreativeState(TypedDict):
    # Inputs (set once at graph invocation)
    product: ProductConfig
    region: RegionConfig
    brand_style: str
    target_audience: str
    quality_threshold: float
    max_attempts: int
    aspect_ratio: str               # "1:1" | "9:16" | "16:9", passed to generator

    # Evolving across iterations
    prompt: str
    image_bytes: bytes | None       # PNG bytes, avoids PIL coupling to state
    quality_score: float
    evaluation_feedback: str        # Natural-language critique from evaluator
    attempt_count: int

    # Outputs
    flagged_for_review: bool
    error: str | None


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def node_build_initial_prompt(state: CreativeState) -> dict:
    """Build the first generation prompt from product + brief context."""
    prompt = build_image_prompt(
        state["product"],
        state["region"],
        state["brand_style"],
        state["target_audience"],
    )
    logger.info(f"  [LangGraph] Initial prompt built for '{state['product'].name}'")
    return {"prompt": prompt, "attempt_count": 0}


def node_generate_image(state: CreativeState) -> dict:
    """Call GenAI provider and store raw PNG bytes in state."""
    from src.generation.factory import get_generator

    attempt = state["attempt_count"] + 1
    logger.info(f"  [LangGraph] Generation attempt {attempt}/{state['max_attempts']}")

    try:
        generator = get_generator()
        image: Image.Image = generator.generate(
            state["product"],
            state["region"],
            state["brand_style"],
            state["target_audience"],
            aspect_ratio=state.get("aspect_ratio", "1:1"),
        )
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        return {
            "image_bytes": buf.getvalue(),
            "attempt_count": attempt,
            "error": None,
        }
    except Exception as e:
        logger.error(f"  [LangGraph] Generation failed: {e}")
        return {"image_bytes": None, "attempt_count": attempt, "error": str(e)}


def node_evaluate_quality(state: CreativeState) -> dict:
    """
    Score the generated image.
    Uses Claude Vision if ANTHROPIC_API_KEY is set, falls back to heuristics.
    Returns quality_score (0.0–5.0) and evaluation_feedback.
    """
    if state.get("image_bytes") is None:
        return {"quality_score": 0.0, "evaluation_feedback": "No image generated"}

    from src.generation.evaluator import evaluate_image
    score, feedback = evaluate_image(
        state["image_bytes"],
        state["product"],
        state["brand_style"],
    )
    logger.info(f"  [LangGraph] Quality score: {score:.2f}/5.0, {feedback[:80]}")
    return {"quality_score": score, "evaluation_feedback": feedback}


def node_refine_prompt(state: CreativeState) -> dict:
    """
    Use an LLM to rewrite the prompt based on evaluation feedback.
    Distinct from generation, specialised for prompt engineering.
    """
    from src.generation.prompt_refiner import refine_prompt
    refined = refine_prompt(
        original_prompt=state["prompt"],
        feedback=state["evaluation_feedback"],
        product=state["product"],
    )
    logger.info(f"  [LangGraph] Prompt refined based on: {state['evaluation_feedback'][:60]}")
    return {"prompt": refined}


def node_flag_for_review(state: CreativeState) -> dict:
    """Max attempts reached without hitting threshold. Flag for human review."""
    logger.warning(
        f"  [LangGraph] '{state['product'].name}' flagged for human review "
        f"after {state['attempt_count']} attempts (best score: {state['quality_score']:.2f})"
    )
    return {"flagged_for_review": True}


# ---------------------------------------------------------------------------
# Routing (conditional edges)
# ---------------------------------------------------------------------------

def route_after_evaluation(state: CreativeState) -> str:
    """
    Decision point: the heart of the LangGraph loop.
    Three outcomes: accept, refine+retry, or escalate to human review.
    """
    score = state["quality_score"]
    threshold = state["quality_threshold"]
    attempts = state["attempt_count"]
    max_att = state["max_attempts"]

    if score >= threshold:
        logger.info(f"  [LangGraph] Score {score:.2f} ≥ threshold {threshold} → ACCEPT")
        return "accept"

    if attempts < max_att:
        logger.info(f"  [LangGraph] Score {score:.2f} < threshold → REFINE (attempt {attempts}/{max_att})")
        return "refine"

    logger.warning(f"  [LangGraph] Max attempts reached, score {score:.2f} → FLAG FOR REVIEW")
    return "flag"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_quality_loop(
    quality_threshold: float = 3.5,
    max_attempts: int = 3,
) -> StateGraph:
    """
    Construct and compile the creative quality loop graph.

    Graph topology:
        START
          → build_initial_prompt
          → generate_image
          → evaluate_quality
          → [accept → END]
          → [refine  → refine_prompt → generate_image → evaluate_quality → ...]
          → [flag    → flag_for_review → END]
    """
    graph = StateGraph(CreativeState)

    graph.add_node("build_initial_prompt", node_build_initial_prompt)
    graph.add_node("generate_image", node_generate_image)
    graph.add_node("evaluate_quality", node_evaluate_quality)
    graph.add_node("refine_prompt", node_refine_prompt)
    graph.add_node("flag_for_review", node_flag_for_review)

    graph.add_edge(START, "build_initial_prompt")
    graph.add_edge("build_initial_prompt", "generate_image")
    graph.add_edge("generate_image", "evaluate_quality")
    graph.add_conditional_edges(
        "evaluate_quality",
        route_after_evaluation,
        {
            "accept": END,
            "refine": "refine_prompt",
            "flag": "flag_for_review",
        },
    )
    graph.add_edge("refine_prompt", "generate_image")
    graph.add_edge("flag_for_review", END)

    return graph.compile(checkpointer=_checkpointer)


def run_quality_loop(
    product: ProductConfig,
    region: RegionConfig,
    brand_style: str,
    target_audience: str,
    quality_threshold: float = 3.5,
    max_attempts: int = 3,
    aspect_ratio: str = "1:1",
) -> tuple[Image.Image | None, bool, float]:
    """
    Run the full Generate → Evaluate → Refine loop for one product.

    aspect_ratio: "1:1" | "9:16" | "16:9"
      When --native-ratios is active the runner calls this once per ratio in
      parallel. Each call gets its own thread_id so their checkpoints are
      independent: a crash on the 9:16 loop does not affect the 1:1 loop.

    Returns:
        (final_image, flagged_for_review, quality_score)
        final_image is None if flagged_for_review is True.
    """
    app = build_quality_loop(quality_threshold, max_attempts)

    # Stable thread_id per product + ratio. LangGraph resumes from the last
    # checkpoint if this loop was previously interrupted in the same process.
    thread_id = f"{product.name}:{aspect_ratio}"
    config    = {"configurable": {"thread_id": thread_id}}

    initial_state: CreativeState = {
        "product": product,
        "region": region,
        "brand_style": brand_style,
        "target_audience": target_audience,
        "quality_threshold": quality_threshold,
        "max_attempts": max_attempts,
        "aspect_ratio": aspect_ratio,
        "prompt": "",
        "image_bytes": None,
        "quality_score": 0.0,
        "evaluation_feedback": "",
        "attempt_count": 0,
        "flagged_for_review": False,
        "error": None,
    }

    final_state = app.invoke(initial_state, config=config)

    if final_state["flagged_for_review"] or not final_state["image_bytes"]:
        return None, True, final_state["quality_score"]

    image = Image.open(io.BytesIO(final_state["image_bytes"])).convert("RGBA")
    return image, False, final_state["quality_score"]
