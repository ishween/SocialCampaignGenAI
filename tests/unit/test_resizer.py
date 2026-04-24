"""Unit tests for image resizer (deterministic — pixel-perfect accuracy required)."""

import pytest
from PIL import Image

from src.composition.resizer import resize_to_aspect, ASPECT_RATIO_DIMENSIONS


def make_image(w: int, h: int) -> Image.Image:
    return Image.new("RGBA", (w, h), color=(100, 150, 200, 255))


@pytest.mark.parametrize("ratio,expected", [
    ("1:1",  (1080, 1080)),
    ("9:16", (1080, 1920)),
    ("16:9", (1920, 1080)),
])
def test_output_dimensions_exact(ratio, expected):
    img = make_image(2000, 1500)
    result = resize_to_aspect(img, ratio)
    assert result.size == expected, f"Expected {expected}, got {result.size}"


def test_landscape_to_square_no_distortion():
    img = make_image(2000, 1000)
    result = resize_to_aspect(img, "1:1")
    assert result.size == (1080, 1080)


def test_portrait_to_landscape():
    img = make_image(1000, 2000)
    result = resize_to_aspect(img, "16:9")
    assert result.size == (1920, 1080)


def test_unsupported_ratio_raises():
    img = make_image(1000, 1000)
    with pytest.raises(ValueError, match="Unsupported aspect ratio"):
        resize_to_aspect(img, "3:7")


def test_output_is_rgba():
    img = make_image(500, 500)
    result = resize_to_aspect(img, "1:1")
    assert result.mode == "RGBA"


def test_small_input_upscales_correctly():
    img = make_image(100, 100)
    result = resize_to_aspect(img, "1:1")
    assert result.size == (1080, 1080)


# ── focal point crop clamping ─────────────────────────────────────────────────

def test_focal_point_top_left_stays_in_bounds():
    """Focal point near (0, 0) must not produce negative crop coordinates."""
    img = make_image(2000, 2000)
    result = resize_to_aspect(img, "1:1", focal_point=(0.0, 0.0))
    assert result.size == (1080, 1080)


def test_focal_point_bottom_right_stays_in_bounds():
    """Focal point near (1, 1) must not exceed image boundary."""
    img = make_image(2000, 2000)
    result = resize_to_aspect(img, "1:1", focal_point=(1.0, 1.0))
    assert result.size == (1080, 1080)


def test_focal_point_center_default():
    img = make_image(2000, 1000)
    result_default = resize_to_aspect(img, "1:1")
    result_center = resize_to_aspect(img, "1:1", focal_point=(0.5, 0.5))
    assert result_default.size == result_center.size


def test_focal_point_affects_crop_position():
    """Different focal points on a wide gradient image should yield different crops."""
    from PIL import ImageDraw
    img = Image.new("RGBA", (2000, 500), (0, 0, 0, 255))
    draw = ImageDraw.Draw(img)
    # Left half red, right half green
    draw.rectangle([0, 0, 999, 499], fill=(255, 0, 0, 255))
    draw.rectangle([1000, 0, 1999, 499], fill=(0, 255, 0, 255))

    left = resize_to_aspect(img, "1:1", focal_point=(0.1, 0.5))
    right = resize_to_aspect(img, "1:1", focal_point=(0.9, 0.5))
    # The two crops should differ in pixel content
    assert left.tobytes() != right.tobytes()


# ── additional ratio support ──────────────────────────────────────────────────

def test_4_5_ratio_output_dimensions():
    img = make_image(2000, 2000)
    result = resize_to_aspect(img, "4:5")
    assert result.size == (1080, 1350)


def test_2_3_ratio_output_dimensions():
    img = make_image(2000, 2000)
    result = resize_to_aspect(img, "2:3")
    assert result.size == (1080, 1620)


# ── get_dimensions ────────────────────────────────────────────────────────────

def test_get_dimensions_returns_correct_tuple():
    from src.composition.resizer import get_dimensions
    assert get_dimensions("1:1") == (1080, 1080)
    assert get_dimensions("9:16") == (1080, 1920)
    assert get_dimensions("16:9") == (1920, 1080)


# ── detect_focal_point fallback ───────────────────────────────────────────────

def test_detect_focal_point_returns_center_when_no_api_key(monkeypatch):
    from src.composition.resizer import detect_focal_point
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    img = make_image(500, 500)
    fx, fy = detect_focal_point(img)
    assert fx == 0.5
    assert fy == 0.5
