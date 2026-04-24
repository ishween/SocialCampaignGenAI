"""
LLM-driven campaign text overlay via Gemini image generation.

Gemini image generation handles placement, typography, contrast treatment
and blending as one creative act, the way a human designer would.
Result: text that feels composed into the image, not stamped on top.

Two public functions:
  check_message_present(): vision check to determine if the campaign text is already in this image.
                            Called before overlaying existing assets to avoid duplicates.
  add_text_overlay_llm(): sends image + creative brief to Gemini image generation.
                           Returns edited PIL Image with message rendered.
                           Falls back to None on failure (caller handles gracefully).
"""

import io
import logging
import os

from PIL import Image

logger = logging.getLogger(__name__)

IMAGE_MODEL  = "gemini-3.1-flash-image-preview"   # multi-modal gen: image-in → image-out
VISION_MODEL = "gemini-3-flash-preview"            # vision only: image-in → text-out

# Ad format context sent to the LLM so it adapts composition conventions per ratio
FORMAT_CONTEXT: dict[str, str] = {
    "1:1":  "square Instagram / Facebook feed post, text typically bottom-center or bottom-left",
    "9:16": "vertical Instagram Stories / TikTok, text typically top or bottom third, never center",
    "16:9": "wide YouTube thumbnail / Twitter header, text typically left-aligned, product right",
    "4:5":  "portrait Instagram feed post, product center, text bottom",
    "2:3":  "tall portrait social post, text top or bottom, product fills mid-frame",
}


# ── Public: check if message is already baked into the image ──────────────────

def check_message_present(
    image: Image.Image,
    message: str,
    api_key: str = "",
) -> bool:
    """
    Ask Gemini vision whether the campaign message is already visible in the image.

    Useful for existing assets that may already carry a tagline.
    Returns False (add overlay) on any API error. Safe default.
    """
    api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        return False

    try:
        import google.genai as genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=80)

        response = client.models.generate_content(
            model=VISION_MODEL,
            contents=[
                types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"),
                (
                    f'Does this image already contain the campaign text "{message}" '
                    f"as a visible overlay or printed text on the product? "
                    f"Reply with only YES or NO."
                ),
            ],
        )

        answer = response.text.strip().upper()
        present = answer.startswith("YES")
        logger.info(
            f"  [LLMOverlay] Message-present check → {answer} "
            f"→ {'already there, skip' if present else 'absent, will add'}"
        )
        return present

    except Exception as e:
        logger.warning(f"  [LLMOverlay] check_message_present failed ({e}), assuming absent")
        return False


# ── Public: LLM adds the campaign message directly to the image ───────────────

def add_text_overlay_llm(
    image: Image.Image,
    message: str,
    aspect_ratio: str,
    brand_style: str = "modern, clean, professional",
    target_audience: str = "general consumers",
    brand_color: str = "",
    api_key: str = "",
) -> Image.Image | None:
    """
    Send image + creative direction to Gemini image generation.
    Gemini renders the campaign message directly onto the image, making
    holistic decisions about placement, font, contrast and blending.

    Returns edited PIL Image on success, None on failure.
    Caller should fall back to PIL overlay when None is returned.
    """
    api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        logger.warning("  [LLMOverlay] No API key, skipping LLM overlay")
        return None

    fmt         = FORMAT_CONTEXT.get(aspect_ratio, "social media ad")
    color_hint  = f"Brand accent color (use as subtle accent only): {brand_color}." if brand_color else ""

    prompt = f"""You are a senior creative director at an advertising agency that specialises
in health and wellness brands. Edit the product image I'm sending you to add
a campaign text overlay, following these exact requirements:

Campaign message: "{message}"
Ad format: {fmt}
Target audience: {target_audience}
Brand style: {brand_style}
{color_hint}

Creative requirements:
1. PLACEMENT: find the region with the most open background space.
   The message must NOT cover the product label, key ingredients, or brand logo.
2. TYPOGRAPHY: bold, large, immediately legible at a glance.
   Use a modern sans-serif with strong weight. No thin or script fonts.
3. CONTRAST: apply a subtle gradient darkening or a soft semi-transparent
   backdrop ONLY where the background is too light for white text.
   The treatment must feel like part of the photo, not a rectangle pasted on top.
4. TEXT COLOR: white or very light cream. Maximum contrast against the backdrop.
5. SHADOW: add a soft drop shadow behind every line of text for depth and polish.
6. AESTHETIC: the finished image must look like a premium editorial ad.
   Not a flyer. Not a coupon. Think Nike, Lululemon, or Liquid I.V.
7. PRESERVE: every original product detail, label text, logo, and background
   element must remain completely unchanged.

Return ONLY the edited image. No commentary. No border. No watermark."""

    try:
        import google.genai as genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=92)

        logger.info(f"  [LLMOverlay] {aspect_ratio} → {IMAGE_MODEL} ...")
        response = client.models.generate_content(
            model=IMAGE_MODEL,
            contents=[
                types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"),
                prompt,
            ],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )

        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                result = Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
                logger.info(f"  [LLMOverlay] ✓ {aspect_ratio} edited image received {result.size}")
                return result

        logger.warning(
            f"  [LLMOverlay] No image in response for {aspect_ratio}. "
            f"Text: {response.text[:200] if response.text else '(none)'}"
        )
        return None

    except Exception as e:
        logger.error(f"  [LLMOverlay] {aspect_ratio} failed: {e}")
        return None
