"""Unit tests for branding — logo compositing on creatives."""

from pathlib import Path

import pytest
from PIL import Image

from src.composition.branding import add_logo, LOGO_MAX_WIDTH_FRACTION
from src.config.models import BrandConfig


def make_image(w=1080, h=1080) -> Image.Image:
    return Image.new("RGBA", (w, h), (50, 100, 150, 255))


def make_logo(tmp_path: Path, size=(200, 100)) -> Path:
    """Create a small PNG logo with transparency."""
    logo = Image.new("RGBA", size, (255, 255, 255, 200))
    path = tmp_path / "logo.png"
    logo.save(path, "PNG")
    return path


# ── No logo configured ────────────────────────────────────────────────────────

def test_no_logo_path_returns_image_unchanged():
    brand = BrandConfig(name="Brand", logo_path=None)
    img = make_image()
    result = add_logo(img, brand)
    # Same pixel data — image unchanged
    assert result.tobytes() == img.tobytes()


# ── Logo path does not exist ──────────────────────────────────────────────────

def test_logo_path_not_found_returns_image_unchanged():
    brand = BrandConfig(name="Brand", logo_path="/nonexistent/logo.png")
    img = make_image()
    result = add_logo(img, brand)
    assert result.size == img.size


# ── Logo composited ───────────────────────────────────────────────────────────

def test_logo_composited_preserves_canvas_size(tmp_path):
    logo_path = make_logo(tmp_path, size=(100, 50))
    brand = BrandConfig(name="Brand", logo_path=str(logo_path))
    img = make_image(1080, 1080)
    result = add_logo(img, brand)
    assert result.size == (1080, 1080)


def test_logo_composited_returns_rgba(tmp_path):
    logo_path = make_logo(tmp_path, size=(100, 50))
    brand = BrandConfig(name="Brand", logo_path=str(logo_path))
    img = make_image()
    result = add_logo(img, brand)
    assert result.mode == "RGBA"


def test_large_logo_is_scaled_down(tmp_path):
    """Logo wider than 15% of canvas width should be resized."""
    canvas_w = 1080
    # Make logo clearly wider than 15% of canvas
    oversized_logo = tmp_path / "big_logo.png"
    big = Image.new("RGBA", (800, 200), (255, 0, 0, 255))
    big.save(oversized_logo)

    brand = BrandConfig(name="Brand", logo_path=str(oversized_logo))
    img = make_image(canvas_w, canvas_w)
    result = add_logo(img, brand)

    # Image should still be composited (size preserved)
    assert result.size == (canvas_w, canvas_w)


def test_small_logo_not_upscaled(tmp_path):
    """Logo already within limits must not be enlarged."""
    canvas_w = 1080
    small_logo_path = make_logo(tmp_path, size=(50, 25))  # well under 15%
    brand = BrandConfig(name="Brand", logo_path=str(small_logo_path))
    img = make_image(canvas_w, canvas_w)
    result = add_logo(img, brand)
    assert result.size == (canvas_w, canvas_w)
