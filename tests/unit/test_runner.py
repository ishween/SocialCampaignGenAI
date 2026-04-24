"""Unit tests for PipelineRunner helper methods and _process_ratio fault tolerance."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from src.compliance.checker import ComplianceResult
from src.config.models import BrandConfig, CampaignBrief, ProductConfig, RegionConfig
from src.pipeline.runner import AssetResult, PipelineResult, PipelineRunner


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_brief() -> CampaignBrief:
    return CampaignBrief(
        name="Test Campaign",
        message="Stay Fresh.",
        target_audience="Adults 25-40",
        products=[ProductConfig(name="Test Bottle", description="A water bottle")],
        regions=[RegionConfig(code="US", language="en")],
        aspect_ratios=["1:1", "9:16", "16:9"],
    )


def make_runner(tmp_path: Path) -> PipelineRunner:
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    return PipelineRunner(
        brief=make_brief(),
        output_dir=tmp_path / "output",
        assets_dir=assets_dir,
        brand_config_path=tmp_path / "nonexistent_brand.yaml",  # → uses defaults
    )


def make_image(w=200, h=200) -> Image.Image:
    return Image.new("RGBA", (w, h), (0, 150, 200, 255))


def make_compliance(passed=True) -> ComplianceResult:
    return ComplianceResult(
        logo_present=True,
        brand_colors_matched=True,
        prohibited_words_found=[],
        passed=passed,
    )


# ── _slug ─────────────────────────────────────────────────────────────────────

def test_slug_lowercases(tmp_path):
    runner = make_runner(tmp_path)
    assert runner._slug("HydraFresh") == "hydrafresh"


def test_slug_replaces_spaces(tmp_path):
    runner = make_runner(tmp_path)
    assert runner._slug("My Product") == "my_product"


def test_slug_replaces_dashes(tmp_path):
    runner = make_runner(tmp_path)
    assert runner._slug("sport-bottle") == "sport_bottle"


# ── _output_path ──────────────────────────────────────────────────────────────

def test_output_path_structure(tmp_path):
    runner = make_runner(tmp_path)
    product = ProductConfig(name="Hydrate Plus", description="desc")
    region = RegionConfig(code="US", language="en")
    path = runner._output_path(product, region, "1:1")

    assert path.parent.name == "us"
    assert path.parent.parent.name == "hydrate_plus"
    assert path.name == "1x1.png"


def test_output_path_ratio_colon_replaced(tmp_path):
    runner = make_runner(tmp_path)
    product = ProductConfig(name="Bottle", description="desc")
    region = RegionConfig(code="US", language="en")
    path = runner._output_path(product, region, "9:16")
    assert path.name == "9x16.png"


# ── _save ─────────────────────────────────────────────────────────────────────

def test_save_creates_parent_dirs(tmp_path):
    runner = make_runner(tmp_path)
    out_path = tmp_path / "output" / "sub" / "dir" / "file.png"
    img = make_image()
    runner._save(img, out_path)
    assert out_path.exists()


def test_save_writes_valid_png(tmp_path):
    runner = make_runner(tmp_path)
    out_path = tmp_path / "output" / "test.png"
    runner._save(make_image(), out_path)
    loaded = Image.open(out_path)
    assert loaded.size == (200, 200)


# ── _process_ratio: idempotency ───────────────────────────────────────────────

def test_process_ratio_skips_if_output_exists(tmp_path):
    runner = make_runner(tmp_path)
    product = ProductConfig(name="Bottle", description="desc")
    region = RegionConfig(code="US", language="en")

    # Pre-create the output file
    out_path = runner._output_path(product, region, "1:1")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    make_image().convert("RGB").save(out_path, "PNG")

    with patch("src.pipeline.runner.run_compliance_check", return_value=make_compliance()):
        result = runner._process_ratio(
            "1:1", make_image(), "Stay Fresh.", product, region,
            is_existing_asset=False, is_generated=False,
        )

    assert result is not None
    assert result.skipped is True


# ── _process_ratio: LLM overlay success ──────────────────────────────────────

def test_process_ratio_llm_overlay_used_and_saved(tmp_path):
    runner = make_runner(tmp_path)
    product = ProductConfig(name="FreshBottle", description="desc")
    region = RegionConfig(code="US", language="en")

    overlay_result = make_image()
    with patch("src.pipeline.runner.add_text_overlay_llm", return_value=overlay_result), \
         patch("src.pipeline.runner.add_logo", return_value=overlay_result), \
         patch("src.pipeline.runner.run_compliance_check", return_value=make_compliance()):

        result = runner._process_ratio(
            "1:1", make_image(), "Stay Fresh.", product, region,
            is_existing_asset=False, is_generated=True,
        )

    assert result is not None
    assert result.skipped is False
    out_path = runner._output_path(product, region, "1:1")
    assert out_path.exists()


# ── _process_ratio: PIL fallback ──────────────────────────────────────────────

def test_process_ratio_falls_back_to_pil_when_llm_returns_none(tmp_path):
    runner = make_runner(tmp_path)
    product = ProductConfig(name="PilBottle", description="desc")
    region = RegionConfig(code="US", language="en")

    pil_result = make_image()
    with patch("src.pipeline.runner.add_text_overlay_llm", return_value=None), \
         patch("src.pipeline.runner.add_text_overlay", return_value=pil_result) as mock_pil, \
         patch("src.pipeline.runner.add_logo", return_value=pil_result), \
         patch("src.pipeline.runner.run_compliance_check", return_value=make_compliance()):

        result = runner._process_ratio(
            "1:1", make_image(), "Stay Fresh.", product, region,
            is_existing_asset=False, is_generated=False,
        )

    mock_pil.assert_called_once()
    assert result is not None


# ── _process_ratio: no message ────────────────────────────────────────────────

def test_process_ratio_no_message_skips_overlay(tmp_path):
    runner = make_runner(tmp_path)
    product = ProductConfig(name="NoMsgBottle", description="desc")
    region = RegionConfig(code="US", language="en")

    logo_result = make_image()
    with patch("src.pipeline.runner.add_text_overlay_llm") as mock_llm, \
         patch("src.pipeline.runner.add_logo", return_value=logo_result), \
         patch("src.pipeline.runner.run_compliance_check", return_value=make_compliance()):

        result = runner._process_ratio(
            "1:1", make_image(), "", product, region,
            is_existing_asset=False, is_generated=False,
        )

    mock_llm.assert_not_called()
    assert result is not None


# ── _process_ratio: existing asset with message already present ───────────────

def test_process_ratio_skips_overlay_if_message_present_in_existing_asset(tmp_path):
    runner = make_runner(tmp_path)
    product = ProductConfig(name="ExistingAsset", description="desc")
    region = RegionConfig(code="US", language="en")

    logo_result = make_image()
    with patch("src.pipeline.runner.check_message_present", return_value=True), \
         patch("src.pipeline.runner.add_text_overlay_llm") as mock_llm, \
         patch("src.pipeline.runner.add_logo", return_value=logo_result), \
         patch("src.pipeline.runner.run_compliance_check", return_value=make_compliance()):

        result = runner._process_ratio(
            "1:1", make_image(), "Stay Fresh.", product, region,
            is_existing_asset=True, is_generated=False,
        )

    mock_llm.assert_not_called()
    assert result is not None


# ── _process_ratio: exception isolation ──────────────────────────────────────

def test_process_ratio_exception_returns_none(tmp_path):
    runner = make_runner(tmp_path)
    product = ProductConfig(name="BrokenProduct", description="desc")
    region = RegionConfig(code="US", language="en")

    with patch("src.pipeline.runner.add_text_overlay_llm", side_effect=RuntimeError("boom")):
        result = runner._process_ratio(
            "1:1", make_image(), "Stay Fresh.", product, region,
            is_existing_asset=False, is_generated=False,
        )

    assert result is None


# ── PipelineResult ────────────────────────────────────────────────────────────

def test_pipeline_result_total_assets():
    r = PipelineResult(campaign_name="Test")
    r.asset_results = [
        AssetResult("p", "US", "1:1", Path("a.png"), generated=True),
        AssetResult("p", "US", "9:16", Path("b.png"), generated=False),
    ]
    assert r.total_assets == 2


def test_pipeline_result_to_json_report(tmp_path):
    r = PipelineResult(campaign_name="Test", duration_sec=12.5)
    r.asset_results = [
        AssetResult(
            "Bottle", "US", "1:1", Path("output/bottle/us/1x1.png"),
            generated=True,
            compliance=make_compliance(),
            duration_sec=5.0,
        )
    ]
    report_path = tmp_path / "report.json"
    r.to_json_report(report_path)

    assert report_path.exists()
    data = json.loads(report_path.read_text())
    assert data["campaign"] == "Test"
    assert len(data["assets"]) == 1
    assert data["assets"][0]["product"] == "Bottle"


# ── _resolve_assets_parallel_ratios ──────────────────────────────────────────

def make_native_runner(tmp_path: Path) -> PipelineRunner:
    """Runner with native_ratios=True."""
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    return PipelineRunner(
        brief=make_brief(),
        output_dir=tmp_path / "output",
        assets_dir=assets_dir,
        brand_config_path=tmp_path / "nonexistent_brand.yaml",
        native_ratios=True,
    )


def test_resolve_assets_parallel_ratios_returns_all_ratios(tmp_path):
    """Parallel quality loops succeed → dict contains all three ratios."""
    runner = make_native_runner(tmp_path)
    product = make_brief().products[0]

    img = make_image()

    def _fake_loop(*args, **kwargs):
        return img, False, 4.5

    with patch("src.pipeline.runner.run_quality_loop", side_effect=_fake_loop), \
         patch.object(runner.asset_manager, "save_generated"):
        result = runner._resolve_assets_parallel_ratios(product)

    assert set(result.keys()) == {"1:1", "9:16", "16:9"}


def test_resolve_assets_parallel_ratios_skips_flagged_ratio(tmp_path):
    """A flagged ratio is excluded from the result; others are kept."""
    runner = make_native_runner(tmp_path)
    product = make_brief().products[0]

    img = make_image()

    def _fake_loop(*args, **kwargs):
        ratio = kwargs.get("aspect_ratio", "1:1")
        if ratio == "9:16":
            return None, True, 1.0   # flagged
        return img, False, 4.5

    with patch("src.pipeline.runner.run_quality_loop", side_effect=_fake_loop), \
         patch.object(runner.asset_manager, "save_generated"):
        result = runner._resolve_assets_parallel_ratios(product)

    assert "9:16" not in result
    assert "1:1" in result and "16:9" in result


def test_resolve_assets_parallel_ratios_each_call_gets_correct_ratio(tmp_path):
    """Each parallel call receives the expected aspect_ratio kwarg."""
    runner = make_native_runner(tmp_path)
    product = make_brief().products[0]

    img = make_image()
    seen_ratios: list[str] = []

    def _fake_loop(*args, **kwargs):
        seen_ratios.append(kwargs.get("aspect_ratio"))
        return img, False, 4.5

    with patch("src.pipeline.runner.run_quality_loop", side_effect=_fake_loop), \
         patch.object(runner.asset_manager, "save_generated"):
        runner._resolve_assets_parallel_ratios(product)

    assert sorted(seen_ratios) == sorted(["1:1", "9:16", "16:9"])


def test_pipeline_result_print_summary_does_not_crash(capsys):
    r = PipelineResult(campaign_name="Test", duration_sec=3.0)
    r.asset_results = [
        AssetResult("p", "US", "1:1", Path("x.png"), generated=True,
                    compliance=make_compliance(), duration_sec=1.0),
    ]
    r.print_summary()
    captured = capsys.readouterr()
    assert "Test" in captured.out
