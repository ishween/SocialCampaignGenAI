"""
Smart image resizer: focal-point-aware crop for any aspect ratio.
"""

import io
import logging
import os
import re

from PIL import Image

logger = logging.getLogger(__name__)

ASPECT_RATIO_DIMENSIONS: dict[str, tuple[int, int]] = {
    "1:1":  (1080, 1080),
    "9:16": (1080, 1920),
    "16:9": (1920, 1080),
    "4:5":  (1080, 1350),
    "2:3":  (1080, 1620),
}

FOCAL_POINT_PROMPT = """\
Analyze this product image and identify the visual center of the main subject
(the product itself, not background).

Respond with ONLY two numbers on one line, no units, no explanation:
x_pct y_pct

Where x_pct is the horizontal position (0.0 = left edge, 1.0 = right edge)
and y_pct is the vertical position (0.0 = top edge, 1.0 = bottom edge)
of the product's center of mass.

Example response: 0.52 0.48
"""


def detect_focal_point(image: Image.Image) -> tuple[float, float]:
    """
    Use Gemini vision to find the product's focal point for smart crop.
    Returns (x_pct, y_pct) normalized 0–1. Falls back to center (0.5, 0.5).

    Called once per base image (before parallel ratio composition), so the
    API cost is one call per product, not per ratio.
    """
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        return 0.5, 0.5

    try:
        import google.genai as genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=85)
        image_bytes = buf.getvalue()

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                FOCAL_POINT_PROMPT,
            ],
        )

        text = response.text.strip()
        parts = re.findall(r"[0-9]*\.?[0-9]+", text)
        if len(parts) >= 2:
            x = max(0.0, min(1.0, float(parts[0])))
            y = max(0.0, min(1.0, float(parts[1])))
            logger.info(f"  [SmartCrop] focal point detected: ({x:.2f}, {y:.2f})")
            return x, y

        logger.warning(f"  [SmartCrop] unexpected Gemini response '{text}', using center")
    except Exception as e:
        logger.warning(f"  [SmartCrop] Gemini unavailable ({e}), using center crop")

    return 0.5, 0.5


def resize_to_aspect(
    image: Image.Image,
    aspect_ratio: str,
    focal_point: tuple[float, float] = (0.5, 0.5),
) -> Image.Image:
    """
    Resize and crop image to target dimensions using a focal point.

    The crop window is centered on focal_point, then clamped so it never
    exceeds the image boundary. This keeps the product in frame across all
    aspect ratios without manual adjustment per SKU.
    """
    if aspect_ratio not in ASPECT_RATIO_DIMENSIONS:
        raise ValueError(
            f"Unsupported aspect ratio: {aspect_ratio}. "
            f"Choose from {list(ASPECT_RATIO_DIMENSIONS)}"
        )

    target_w, target_h = ASPECT_RATIO_DIMENSIONS[aspect_ratio]
    target_ratio = target_w / target_h

    img = image.convert("RGBA")
    src_w, src_h = img.size
    src_ratio = src_w / src_h

    # Scale to cover the target canvas
    if src_ratio > target_ratio:
        # Source wider: match heights, crop width
        scale_h = target_h
        scale_w = int(src_w * (target_h / src_h))
    else:
        # Source taller: match widths, crop height
        scale_w = target_w
        scale_h = int(src_h * (target_w / src_w))

    resized = img.resize((scale_w, scale_h), Image.LANCZOS)

    # Focal-point crop: center the crop window on (fx, fy)
    fx, fy = focal_point
    ideal_left = int(fx * scale_w - target_w / 2)
    ideal_top  = int(fy * scale_h - target_h / 2)

    # Clamp so we never go out of bounds
    left = max(0, min(ideal_left, scale_w - target_w))
    top  = max(0, min(ideal_top,  scale_h - target_h))

    cropped = resized.crop((left, top, left + target_w, top + target_h))

    logger.debug(
        f"  SmartCrop {src_w}x{src_h} → {target_w}x{target_h} "
        f"({aspect_ratio}, focal={focal_point[0]:.2f},{focal_point[1]:.2f})"
    )
    return cropped


def get_dimensions(aspect_ratio: str) -> tuple[int, int]:
    return ASPECT_RATIO_DIMENSIONS[aspect_ratio]
