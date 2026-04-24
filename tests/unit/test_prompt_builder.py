"""Unit tests for prompt builder — all four label/background combinations."""

import pytest

from src.config.models import ProductConfig, RegionConfig
from src.generation.prompt_builder import build_image_prompt


def make_product(**kwargs) -> ProductConfig:
    defaults = {"name": "Test Bottle", "description": "A test product"}
    defaults.update(kwargs)
    return ProductConfig(**defaults)


def make_region(**kwargs) -> RegionConfig:
    defaults = {"code": "US", "language": "en"}
    defaults.update(kwargs)
    return RegionConfig(**defaults)


BRAND_STYLE = "vibrant, modern"
AUDIENCE = "Adults 25-40"


# ── label_text combinations ───────────────────────────────────────────────────

def test_label_text_included_in_prompt():
    product = make_product(label_text="HydraFresh\\nSport")
    prompt = build_image_prompt(product, make_region(), BRAND_STYLE, AUDIENCE)
    assert "HydraFresh" in prompt
    assert "product label" in prompt.lower()


def test_no_label_text_clean_shot():
    product = make_product(label_text=None)
    prompt = build_image_prompt(product, make_region(), BRAND_STYLE, AUDIENCE)
    assert "no text on product" in prompt.lower()


# ── background_scene combinations ────────────────────────────────────────────

def test_background_scene_included():
    product = make_product(background_scene="sports stadium at dusk")
    prompt = build_image_prompt(product, make_region(), BRAND_STYLE, AUDIENCE)
    assert "sports stadium at dusk" in prompt


def test_no_background_scene_uses_studio():
    product = make_product(background_scene=None)
    prompt = build_image_prompt(product, make_region(), BRAND_STYLE, AUDIENCE)
    assert "studio" in prompt.lower()


# ── style_keywords vs brand_style fallback ───────────────────────────────────

def test_style_keywords_override_brand_style():
    product = make_product(style_keywords=["cinematic", "premium"])
    prompt = build_image_prompt(product, make_region(), BRAND_STYLE, AUDIENCE)
    assert "cinematic" in prompt
    assert "premium" in prompt


def test_brand_style_used_when_no_keywords():
    product = make_product(style_keywords=[])
    prompt = build_image_prompt(product, make_region(), BRAND_STYLE, AUDIENCE)
    assert BRAND_STYLE in prompt


# ── locale hint ──────────────────────────────────────────────────────────────

def test_locale_hint_included_when_set():
    region = make_region(locale_name="United States")
    product = make_product()
    prompt = build_image_prompt(product, region, BRAND_STYLE, AUDIENCE)
    assert "United States" in prompt


def test_no_locale_hint_when_empty():
    region = make_region(locale_name="")
    product = make_product()
    prompt = build_image_prompt(product, region, BRAND_STYLE, AUDIENCE)
    # Should not contain "for " locale phrase
    assert " for " not in prompt or "for " not in prompt.split("Target audience")[1]


# ── product name and description always present ───────────────────────────────

def test_product_name_in_prompt():
    product = make_product(name="SuperDrink Pro")
    prompt = build_image_prompt(product, make_region(), BRAND_STYLE, AUDIENCE)
    assert "SuperDrink Pro" in prompt


def test_product_description_in_prompt():
    product = make_product(description="A refreshing electrolyte drink")
    prompt = build_image_prompt(product, make_region(), BRAND_STYLE, AUDIENCE)
    assert "A refreshing electrolyte drink" in prompt


def test_quality_keywords_always_present():
    product = make_product()
    prompt = build_image_prompt(product, make_region(), BRAND_STYLE, AUDIENCE)
    assert "photorealistic" in prompt.lower()
