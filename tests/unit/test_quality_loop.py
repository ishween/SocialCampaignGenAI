"""Unit tests for LangGraph quality loop — routing logic and end-to-end with mocks."""

import io
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from src.config.models import ProductConfig, RegionConfig
from src.generation.quality_loop import (
    route_after_evaluation,
    build_quality_loop,
    run_quality_loop,
    CreativeState,
)


def make_product(name="Test Bottle") -> ProductConfig:
    return ProductConfig(name=name, description="A test product")


def make_region() -> RegionConfig:
    return RegionConfig(code="US", language="en")


def make_png_bytes(color=(0, 200, 100)) -> bytes:
    img = Image.new("RGB", (64, 64), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def base_state(**overrides) -> CreativeState:
    state: CreativeState = {
        "product": make_product(),
        "region": make_region(),
        "brand_style": "modern",
        "target_audience": "Adults",
        "quality_threshold": 3.5,
        "max_attempts": 3,
        "aspect_ratio": "1:1",
        "prompt": "test prompt",
        "image_bytes": make_png_bytes(),
        "quality_score": 0.0,
        "evaluation_feedback": "",
        "attempt_count": 1,
        "flagged_for_review": False,
        "error": None,
    }
    state.update(overrides)
    return state


# ── route_after_evaluation (pure routing logic) ───────────────────────────────

def test_route_accept_when_score_meets_threshold():
    state = base_state(quality_score=3.5, quality_threshold=3.5, attempt_count=1, max_attempts=3)
    assert route_after_evaluation(state) == "accept"


def test_route_accept_when_score_exceeds_threshold():
    state = base_state(quality_score=4.8, quality_threshold=3.5, attempt_count=1, max_attempts=3)
    assert route_after_evaluation(state) == "accept"


def test_route_refine_when_score_low_and_attempts_remain():
    state = base_state(quality_score=2.0, quality_threshold=3.5, attempt_count=1, max_attempts=3)
    assert route_after_evaluation(state) == "refine"


def test_route_flag_when_max_attempts_reached():
    state = base_state(quality_score=1.0, quality_threshold=3.5, attempt_count=3, max_attempts=3)
    assert route_after_evaluation(state) == "flag"


def test_route_flag_when_attempts_exceed_max():
    state = base_state(quality_score=0.0, quality_threshold=3.5, attempt_count=5, max_attempts=3)
    assert route_after_evaluation(state) == "flag"


def test_route_refine_at_exactly_one_below_max():
    state = base_state(quality_score=2.0, quality_threshold=3.5, attempt_count=2, max_attempts=3)
    assert route_after_evaluation(state) == "refine"


# ── build_quality_loop ────────────────────────────────────────────────────────

def test_build_quality_loop_returns_compiled_graph():
    graph = build_quality_loop(quality_threshold=3.5, max_attempts=2)
    assert graph is not None
    # Compiled LangGraph exposes invoke
    assert callable(graph.invoke)


# ── run_quality_loop (mocked providers) ──────────────────────────────────────

def _make_mock_generator(image: Image.Image) -> MagicMock:
    mock_gen = MagicMock()
    mock_gen.generate.return_value = image
    return mock_gen


@patch("src.generation.prompt_builder.build_image_prompt", return_value="mocked prompt")
@patch("src.generation.prompt_refiner.refine_prompt", return_value="refined prompt")
@patch("src.generation.evaluator.evaluate_image", return_value=(4.0, "Looks great"))
@patch("src.generation.factory.get_generator")
def test_run_quality_loop_accept_path(mock_get_gen, mock_eval, mock_refine, mock_build):
    """High score on first attempt → accept without refine."""
    img = Image.new("RGBA", (64, 64), (0, 200, 100, 255))
    mock_get_gen.return_value = _make_mock_generator(img)

    result_img, flagged, score = run_quality_loop(
        product=make_product(name="AcceptProduct"),
        region=make_region(),
        brand_style="modern",
        target_audience="Adults",
        quality_threshold=3.5,
        max_attempts=3,
    )

    assert flagged is False
    assert result_img is not None
    assert score >= 3.5
    # Only one generation attempt needed
    mock_get_gen.return_value.generate.assert_called_once()


@patch("src.generation.prompt_builder.build_image_prompt", return_value="mocked prompt")
@patch("src.generation.prompt_refiner.refine_prompt", return_value="refined prompt")
@patch("src.generation.evaluator.evaluate_image", return_value=(1.0, "Terrible quality"))
@patch("src.generation.factory.get_generator")
def test_run_quality_loop_flag_path(mock_get_gen, mock_eval, mock_refine, mock_build):
    """Score always below threshold → flagged for human review after max attempts."""
    img = Image.new("RGBA", (64, 64), (200, 0, 0, 255))
    mock_get_gen.return_value = _make_mock_generator(img)

    result_img, flagged, score = run_quality_loop(
        product=make_product(name="FlagProduct"),
        region=make_region(),
        brand_style="modern",
        target_audience="Adults",
        quality_threshold=3.5,
        max_attempts=2,
    )

    assert flagged is True
    assert result_img is None


@patch("src.generation.prompt_builder.build_image_prompt", return_value="mocked prompt")
@patch("src.generation.prompt_refiner.refine_prompt", return_value="refined prompt")
@patch("src.generation.evaluator.evaluate_image", side_effect=[(2.0, "Bad"), (2.0, "Bad"), (4.5, "Good")])
@patch("src.generation.factory.get_generator")
def test_run_quality_loop_refines_then_accepts(mock_get_gen, mock_eval, mock_refine, mock_build):
    """Two low scores then a high score → accept on third attempt."""
    img = Image.new("RGBA", (64, 64), (0, 100, 200, 255))
    mock_get_gen.return_value = _make_mock_generator(img)

    result_img, flagged, score = run_quality_loop(
        product=make_product(name="RefineProduct"),
        region=make_region(),
        brand_style="modern",
        target_audience="Adults",
        quality_threshold=3.5,
        max_attempts=3,
    )

    assert flagged is False
    assert result_img is not None
    assert score >= 3.5


# ── aspect_ratio propagation ──────────────────────────────────────────────────

@patch("src.generation.prompt_builder.build_image_prompt", return_value="mocked prompt")
@patch("src.generation.prompt_refiner.refine_prompt", return_value="refined prompt")
@patch("src.generation.evaluator.evaluate_image", return_value=(4.5, "Looks great"))
@patch("src.generation.factory.get_generator")
def test_run_quality_loop_passes_aspect_ratio_to_generator(mock_get_gen, mock_eval, mock_refine, mock_build):
    """aspect_ratio is forwarded to generator.generate() on every call."""
    img = Image.new("RGBA", (64, 64), (0, 200, 100, 255))
    mock_generator = _make_mock_generator(img)
    mock_get_gen.return_value = mock_generator

    run_quality_loop(
        product=make_product(name="RatioProduct"),
        region=make_region(),
        brand_style="modern",
        target_audience="Adults",
        aspect_ratio="9:16",
    )

    _, call_kwargs = mock_generator.generate.call_args
    assert call_kwargs.get("aspect_ratio") == "9:16"


@patch("src.generation.prompt_builder.build_image_prompt", return_value="mocked prompt")
@patch("src.generation.prompt_refiner.refine_prompt", return_value="refined prompt")
@patch("src.generation.evaluator.evaluate_image", return_value=(4.5, "Great"))
@patch("src.generation.factory.get_generator")
def test_run_quality_loop_default_aspect_ratio_is_1x1(mock_get_gen, mock_eval, mock_refine, mock_build):
    """Omitting aspect_ratio defaults to 1:1 so existing callers are unaffected."""
    img = Image.new("RGBA", (64, 64), (0, 200, 100, 255))
    mock_generator = _make_mock_generator(img)
    mock_get_gen.return_value = mock_generator

    run_quality_loop(
        product=make_product(name="DefaultRatioProduct"),
        region=make_region(),
        brand_style="modern",
        target_audience="Adults",
    )

    _, call_kwargs = mock_generator.generate.call_args
    assert call_kwargs.get("aspect_ratio") == "1:1"
