"""
PIL fallback overlay: campaign message rendered on image with a cinematic
bottom gradient vignette.

Used when Gemini image generation is unavailable (429, credits depleted, etc.).
The LLM overlay in llm_overlay.py is the primary path, this is the safety net.
"""

import logging
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from src.config.models import BrandConfig

logger = logging.getLogger(__name__)

SAFE_ZONE_MARGIN = 0.08   # 8% inset from edges
VIGNETTE_COVERAGE = 0.38  # gradient covers bottom 38% of image height


def _load_font(font_path: str | None, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if font_path and Path(font_path).exists():
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            logger.warning(f"  Failed to load font {font_path}, using default")
    for path in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _wrap_text(text: str, font, max_width: int, draw: ImageDraw.Draw) -> list[str]:
    words = text.split()
    lines, current = [], ""
    for word in words:
        test = f"{current} {word}".strip()
        if draw.textbbox((0, 0), test, font=font)[2] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _draw_vignette(
    canvas: Image.Image,
    w: int,
    h: int,
    brand_color: tuple[int, int, int],
) -> None:
    """
    Bottom gradient vignette: transparent at top of span → near-dark at bottom edge.
    Blend is 30% brand color + 70% black so it tints without clashing.
    """
    br, bg, bb = brand_color
    tr, tg, tb = int(br * 0.30), int(bg * 0.30), int(bb * 0.30)

    draw  = ImageDraw.Draw(canvas)
    span  = int(h * VIGNETTE_COVERAGE)
    start = h - span
    for y in range(start, h):
        t     = (y - start) / span          # 0.0 at top of span → 1.0 at bottom
        alpha = int(210 * (t ** 1.4))
        draw.line([(0, y), (w, y)], fill=(tr, tg, tb, alpha))


def add_text_overlay(
    image: Image.Image,
    message: str,
    brand_config: BrandConfig,
) -> Image.Image:
    """
    PIL fallback: render campaign message with a bottom cinematic gradient vignette.
    Text is centered horizontally, anchored above the bottom safe-zone margin.

    Called by runner.py when llm_overlay.add_text_overlay_llm() returns None.
    """
    img   = image.convert("RGBA")
    w, h  = img.size

    font_size = max(32, int(h * 0.052))
    font      = _load_font(brand_config.font_path, font_size)

    measure_draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    safe_w       = int(w * (1 - 2 * SAFE_ZONE_MARGIN))
    lines        = _wrap_text(message, font, safe_w, measure_draw)
    line_height  = int(font_size * 1.45)
    total_text_h = line_height * len(lines)
    pad_v        = int(font_size * 0.8)

    text_y_start = h - total_text_h - int(h * SAFE_ZONE_MARGIN) - pad_v

    brand_rgb = (
        _hex_to_rgb(brand_config.primary_colors[0])
        if brand_config.primary_colors else (0, 40, 80)
    )

    # Layer 1: gradient vignette
    vignette = Image.new("RGBA", img.size, (0, 0, 0, 0))
    _draw_vignette(vignette, w, h, brand_rgb)
    vignette = vignette.filter(ImageFilter.GaussianBlur(radius=2))
    result   = Image.alpha_composite(img, vignette)

    # Layer 2: text (centered, with multi-layer drop shadow)
    text_layer  = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw        = ImageDraw.Draw(text_layer)
    text_color  = _hex_to_rgb(brand_config.text_color)

    for i, line in enumerate(lines):
        y      = text_y_start + i * line_height
        bbox   = draw.textbbox((0, 0), line, font=font)
        text_w = bbox[2] - bbox[0]
        x      = (w - text_w) // 2

        if brand_config.text_shadow:
            for sx, sy, sa in [(3, 3, 140), (1, 1, 100), (0, 2, 80)]:
                draw.text((x + sx, y + sy), line, font=font, fill=(0, 0, 0, sa))

        draw.text((x, y), line, font=font, fill=(*text_color, 255))

    result = Image.alpha_composite(result, text_layer)
    logger.debug(f"  [PIL fallback] {len(lines)} lines rendered at bottom")
    return result
