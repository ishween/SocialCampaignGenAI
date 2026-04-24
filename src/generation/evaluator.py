"""
Image quality evaluator: LangGraph node evaluate_quality.

Uses Google Gemini 2.0 Flash (vision-capable) to score generated images.
Gemini 2.0 Flash is the right model here: it understands image content,
brand aesthetics, and composition, tasks that require multimodal comprehension,
not just fast text generation.

Model choice within Google's family:
  Imagen 3          → image synthesis     (this pipeline's generator)
  Gemini 2.0 Flash  → image evaluation    (this file, understanding task)
  Gemini 2.0 Flash-Lite → text tasks      (prompt_refiner, input_validator)

Fallback: heuristic checks (resolution, variance) when GOOGLE_API_KEY absent.
"""

import base64
import logging
import os
import re

from src.config.models import ProductConfig

logger = logging.getLogger(__name__)

EVAL_MODEL = "gemini-3-flash-preview"  # Gemini 3 Flash: vision capable, 1K/day paid tier

EVAL_PROMPT = """\
You are a senior creative director evaluating an AI-generated product advertisement image.

Product: {product_name}
Product description: {product_description}
Brand style: {brand_style}

Score this image on each criterion from 1 (poor) to 5 (excellent):

1. brand_alignment: Does the image feel on-brand? (style, mood, professionalism)
2. product_accuracy: Is the correct product clearly depicted and recognizable?
3. composition: Is the framing clean, balanced, and visually appealing?
4. relevance: Does the image capture the campaign intent described above?

Respond ONLY in this exact format:
brand_alignment: <score>
product_accuracy: <score>
composition: <score>
relevance: <score>
feedback: <one sentence on the single most important improvement, or "looks great" if all ≥4>
"""

SCORE_WEIGHTS = {
    "brand_alignment": 0.30,
    "product_accuracy": 0.35,
    "composition": 0.20,
    "relevance": 0.15,
}


def _parse_scores(text: str) -> tuple[float, str]:
    scores: dict[str, float] = {}
    for key in SCORE_WEIGHTS:
        m = re.search(rf"{key}:\s*([1-5](?:\.\d+)?)", text, re.IGNORECASE)
        scores[key] = float(m.group(1)) if m else 3.0

    m = re.search(r"feedback:\s*(.+)", text, re.IGNORECASE)
    feedback = m.group(1).strip() if m else "No specific feedback"

    weighted = sum(scores[k] * SCORE_WEIGHTS[k] for k in SCORE_WEIGHTS)
    return round(weighted, 2), feedback


def _evaluate_with_gemini(
    image_bytes: bytes, product: ProductConfig, brand_style: str
) -> tuple[float, str]:
    """Gemini 2.0 Flash: multimodal image evaluation."""
    import google.genai as genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    prompt_text = EVAL_PROMPT.format(
        product_name=product.name,
        product_description=product.description,
        brand_style=brand_style,
    )

    response = client.models.generate_content(
        model=EVAL_MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            prompt_text,
        ],
    )
    return _parse_scores(response.text)


def _evaluate_heuristic(image_bytes: bytes) -> tuple[float, str]:
    """
    Fallback when no API key is available.
    Checks file size and pixel variance as proxies for image quality.
    Returns a conservative score. Vision-based eval is always preferred.
    """
    import io
    import statistics
    from PIL import Image

    size_kb = len(image_bytes) / 1024
    if size_kb < 50:
        return 1.5, "Image suspiciously small, likely blank or corrupt"

    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    pixels = list(img.getdata())
    variance = statistics.variance(pixels[:10_000])

    if variance < 100:
        return 2.0, "Low visual variance, image may be too plain or solid color"

    score = min(3.5, 2.5 + (variance / 5000))
    return round(score, 2), "Heuristic eval only. Set GOOGLE_API_KEY for vision-based scoring"


def evaluate_image(
    image_bytes: bytes,
    product: ProductConfig,
    brand_style: str,
) -> tuple[float, str]:
    """Entry point called by LangGraph node_evaluate_quality."""
    if os.environ.get("GOOGLE_API_KEY"):
        try:
            score, feedback = _evaluate_with_gemini(image_bytes, product, brand_style)
            logger.debug(f"  [Gemini 2.0 Flash eval] score={score}, feedback={feedback}")
            return score, feedback
        except Exception as e:
            logger.warning(f"  Gemini eval failed ({e}), falling back to heuristic")

    score, feedback = _evaluate_heuristic(image_bytes)
    logger.debug(f"  [Heuristic eval] score={score}, feedback={feedback}")
    return score, feedback
