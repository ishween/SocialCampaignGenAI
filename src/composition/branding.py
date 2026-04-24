"""Branding: adds logo watermark to generated creatives."""

import logging
from pathlib import Path

from PIL import Image

from src.config.models import BrandConfig

logger = logging.getLogger(__name__)

# Logo occupies at most 15% of image width, positioned top-right
LOGO_MAX_WIDTH_FRACTION = 0.15
LOGO_MARGIN_FRACTION = 0.03


def add_logo(image: Image.Image, brand_config: BrandConfig) -> Image.Image:
    """
    Composite brand logo onto image at top-right corner.
    Logo is resized proportionally. Alpha channel preserved.
    If no logo configured, returns image unchanged.
    """
    if not brand_config.logo_path:
        logger.debug("  No logo configured, skipping")
        return image

    logo_path = Path(brand_config.logo_path)
    if not logo_path.exists():
        logger.warning(f"  Logo not found at {logo_path}, skipping")
        return image

    img = image.convert("RGBA")
    logo = Image.open(logo_path).convert("RGBA")

    w, h = img.size
    max_logo_w = int(w * LOGO_MAX_WIDTH_FRACTION)
    margin = int(w * LOGO_MARGIN_FRACTION)

    # Proportional resize
    logo_w, logo_h = logo.size
    if logo_w > max_logo_w:
        scale = max_logo_w / logo_w
        logo = logo.resize((int(logo_w * scale), int(logo_h * scale)), Image.LANCZOS)

    logo_w, logo_h = logo.size
    x = w - logo_w - margin
    y = margin

    composite = img.copy()
    composite.paste(logo, (x, y), logo)
    logger.debug(f"  Logo placed at ({x}, {y}), size {logo_w}x{logo_h}")
    return composite
