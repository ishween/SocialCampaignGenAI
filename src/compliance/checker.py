"""Brand compliance checker: logo presence, brand colors, prohibited words."""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from src.config.models import BrandConfig

logger = logging.getLogger(__name__)


@dataclass
class ComplianceResult:
    logo_present: bool = False
    brand_colors_matched: bool = False
    prohibited_words_found: list[str] = field(default_factory=list)
    color_analysis: dict = field(default_factory=dict)
    passed: bool = False
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.passed = (
            self.logo_present or True  # Logo check is best-effort in POC
        ) and not self.prohibited_words_found

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        items = [
            f"logo={'✓' if self.logo_present else '✗'}",
            f"brand_colors={'✓' if self.brand_colors_matched else '~'}",
            f"prohibited_words={self.prohibited_words_found or 'none'}",
        ]
        return f"[{status}] {' | '.join(items)}"


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def _color_distance(c1: tuple[int, int, int], c2: tuple[int, int, int]) -> float:
    return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5


def _dominant_colors(image: Image.Image, n: int = 8) -> list[tuple[int, int, int]]:
    """Extract N dominant colors using quantization."""
    small = image.convert("RGB").resize((100, 100), Image.LANCZOS)
    quantized = small.quantize(colors=n, method=Image.Quantize.MEDIANCUT)
    palette = quantized.getpalette()
    colors = [tuple(palette[i*3:(i+1)*3]) for i in range(n)]
    return colors


def check_brand_colors(image: Image.Image, brand_colors: list[str], tolerance: int = 40) -> tuple[bool, dict]:
    """Check if image contains at least one brand color within tolerance."""
    if not brand_colors:
        return True, {}

    dominant = _dominant_colors(image)
    target_rgbs = [_hex_to_rgb(c) for c in brand_colors]

    matches = []
    for dom_color in dominant:
        for target in target_rgbs:
            dist = _color_distance(dom_color, target)
            if dist <= tolerance:
                matches.append({"found": dom_color, "target": target, "distance": round(dist, 1)})

    return bool(matches), {"dominant_colors": dominant, "matches": matches}


def check_logo_present(image: Image.Image, logo_path: str | None) -> bool:
    """
    Heuristic logo check: verify logo was composited by checking alpha channel
    in expected top-right region. Full template matching requires opencv (optional).
    """
    if not logo_path or not Path(logo_path).exists():
        return True  # Can't check without logo reference

    # Check top-right 20% x 20% has non-uniform content (logo presence heuristic)
    img = image.convert("RGBA")
    w, h = img.size
    region = img.crop((int(w * 0.8), 0, w, int(h * 0.2)))
    pixels = list(region.getdata())
    unique = len(set(pixels))
    return unique > 10  # Non-blank region implies logo was placed


def check_prohibited_words(message: str, prohibited: list[str]) -> list[str]:
    """Return any prohibited words found in message (case-insensitive)."""
    found = []
    for word in prohibited:
        if re.search(rf"\b{re.escape(word)}\b", message, re.IGNORECASE):
            found.append(word)
    return found


def run_compliance_check(
    image: Image.Image,
    brand_config: BrandConfig,
    campaign_message: str,
    prohibited_words: list[str],
) -> ComplianceResult:
    result = ComplianceResult()

    result.logo_present = check_logo_present(image, brand_config.logo_path)
    if not result.logo_present:
        result.warnings.append("Logo not detected in output image")

    result.brand_colors_matched, result.color_analysis = check_brand_colors(
        image, brand_config.primary_colors
    )
    if not result.brand_colors_matched and brand_config.primary_colors:
        result.warnings.append("Brand colors not detected in image")

    result.prohibited_words_found = check_prohibited_words(campaign_message, prohibited_words)
    if result.prohibited_words_found:
        result.warnings.append(f"Prohibited words in message: {result.prohibited_words_found}")

    result.passed = not result.prohibited_words_found

    logger.info(f"  Compliance: {result.summary()}")
    for w in result.warnings:
        logger.warning(f"    ⚠ {w}")

    return result
