"""
Prompt refiner: LangGraph node refine_prompt.

Uses Google Gemini 2.0 Flash to rewrite generation prompts based on evaluator
feedback. This is a pure text task (rewrite a string), not a vision task.
Gemini 2.0 Flash-Lite would be cheaper for this role, but it has no free-tier
quota on standard Gemini API keys. Flash is used here for compatibility.

Model choice within Google's family:
  Gemini Flash (image modality) → image synthesis   (generator, auto mode)
  Gemini 2.0 Flash              → image evaluation  (evaluator, needs vision)
  Gemini 2.0 Flash              → prompt rewriting  (this file, text-only)

Fallback: keyword-based heuristic rewrite when GOOGLE_API_KEY absent.
"""

import logging
import os

from src.config.models import ProductConfig

logger = logging.getLogger(__name__)

REFINE_MODEL = "gemini-3.1-flash-lite-preview"  # Gemini 3.1 Flash Lite: 150K TPM, cheapest

REFINE_PROMPT = """\
You are an expert prompt engineer for AI image generation systems.

Rewrite the following image generation prompt to fix the specific feedback provided.
Keep the rewritten prompt under 400 words. Return ONLY the improved prompt, no explanation, no preamble.

Original prompt:
{original_prompt}

Creative director feedback:
{feedback}

Product context:
- Name: {product_name}
- Description: {product_description}
- Style keywords: {style_keywords}

Rewrite the prompt addressing the feedback while keeping product intent intact.
"""


def _refine_with_gemini(
    original_prompt: str, feedback: str, product: ProductConfig
) -> str:
    """Gemini 2.0 Flash-Lite: fast, cheap text rewriting task."""
    import google.genai as genai

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    prompt_text = REFINE_PROMPT.format(
        original_prompt=original_prompt,
        feedback=feedback,
        product_name=product.name,
        product_description=product.description,
        style_keywords=", ".join(product.style_keywords) if product.style_keywords else "professional",
    )

    response = client.models.generate_content(
        model=REFINE_MODEL,
        contents=[prompt_text],
    )
    return response.text.strip()


def _refine_heuristic(
    original_prompt: str, feedback: str, product: ProductConfig
) -> str:
    """Rule-based fallback: append style corrections derived from feedback keywords."""
    additions = []
    fb = feedback.lower()

    if any(w in fb for w in ["cluttered", "busy", "complex"]):
        additions.append("minimalist composition, clean empty background")
    if any(w in fb for w in ["dark", "underexposed", "dim"]):
        additions.append("bright studio lighting, well-lit, high key lighting")
    if any(w in fb for w in ["blurry", "soft", "unfocused"]):
        additions.append("tack sharp focus, ultra high resolution, crisp details")
    if any(w in fb for w in ["plain", "boring", "generic"]):
        additions.append("dynamic angle, aspirational lifestyle context, editorial quality")
    if any(w in fb for w in ["color", "dull", "muted"]):
        additions.append("vibrant saturated colors, punchy contrast, vivid")
    if any(w in fb for w in ["product", "recogniz", "visible", "accuracy"]):
        additions.append("product centered and prominently featured, hero product shot")

    if additions:
        return f"{original_prompt}, {', '.join(additions)}"

    return f"{original_prompt}, award-winning commercial photography, editorial quality, premium brand"


def refine_prompt(
    original_prompt: str, feedback: str, product: ProductConfig
) -> str:
    """Entry point called by LangGraph node_refine_prompt."""
    if os.environ.get("GOOGLE_API_KEY"):
        try:
            refined = _refine_with_gemini(original_prompt, feedback, product)
            logger.info(f"  [Gemini Flash-Lite refine] {len(refined)} chars")
            return refined
        except Exception as e:
            logger.warning(f"  Gemini prompt refiner failed ({e}), using heuristic")

    refined = _refine_heuristic(original_prompt, feedback, product)
    logger.info("  [Heuristic refine] keyword-based correction applied")
    return refined
