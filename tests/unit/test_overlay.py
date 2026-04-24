"""Unit tests for text overlay composer."""

from PIL import Image

from src.composition.overlay import add_text_overlay
from src.config.models import BrandConfig


def make_brand() -> BrandConfig:
    return BrandConfig(name="TestBrand", text_color="#FFFFFF", text_shadow=True)


def make_image(w=1080, h=1080) -> Image.Image:
    return Image.new("RGBA", (w, h), (50, 100, 150, 255))


def test_overlay_returns_correct_size():
    img = make_image()
    brand = make_brand()
    result = add_text_overlay(img, "Stay Fresh.", brand)
    assert result.size == img.size


def test_overlay_preserves_mode():
    img = make_image()
    brand = make_brand()
    result = add_text_overlay(img, "Test message", brand)
    assert result.mode == "RGBA"


def test_overlay_long_message_wraps():
    img = make_image()
    brand = make_brand()
    long_msg = "This is a very long campaign message that should definitely wrap across multiple lines"
    result = add_text_overlay(img, long_msg, brand)
    assert result.size == img.size  # No crash, correct output


def test_overlay_story_format():
    img = make_image(1080, 1920)
    brand = make_brand()
    result = add_text_overlay(img, "Message", brand)
    assert result.size == (1080, 1920)


def test_overlay_landscape_format():
    img = make_image(1920, 1080)
    brand = make_brand()
    result = add_text_overlay(img, "Message", brand)
    assert result.size == (1920, 1080)
