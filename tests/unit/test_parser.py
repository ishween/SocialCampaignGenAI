"""Unit tests for config parser (deterministic — must be 100% pass rate)."""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.config.models import CampaignBrief
from src.config.parser import parse_campaign_brief

VALID_BRIEF = {
    "name": "Test Campaign",
    "message": "Test message",
    "target_audience": "Adults 25-40",
    "products": [
        {"name": "Product A", "description": "Description A"},
        {"name": "Product B", "description": "Description B"},
    ],
    "aspect_ratios": ["1:1", "9:16", "16:9"],
}


def write_yaml(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "brief.yaml"
    p.write_text(yaml.dump(data))
    return p


def test_valid_brief_parses(tmp_path):
    path = write_yaml(tmp_path, VALID_BRIEF)
    brief = parse_campaign_brief(path)
    assert brief.name == "Test Campaign"
    assert len(brief.products) == 2


def test_default_aspect_ratios(tmp_path):
    data = {**VALID_BRIEF}
    del data["aspect_ratios"]
    path = write_yaml(tmp_path, data)
    brief = parse_campaign_brief(path)
    assert "1:1" in brief.aspect_ratios


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        parse_campaign_brief("/nonexistent/brief.yaml")


def test_unsupported_format(tmp_path):
    p = tmp_path / "brief.txt"
    p.write_text("name: test")
    with pytest.raises(ValueError, match="Unsupported brief format"):
        parse_campaign_brief(p)


def test_missing_required_field_raises(tmp_path):
    data = {**VALID_BRIEF}
    del data["message"]
    path = write_yaml(tmp_path, data)
    with pytest.raises(ValueError, match="Invalid campaign brief"):
        parse_campaign_brief(path)


def test_single_product_accepted(tmp_path):
    """Single product is valid — validator now requires ≥ 1, not ≥ 2."""
    data = {**VALID_BRIEF, "products": [{"name": "Only One", "description": "desc"}]}
    path = write_yaml(tmp_path, data)
    brief = parse_campaign_brief(path)
    assert len(brief.products) == 1


def test_empty_products_rejected(tmp_path):
    """Empty product list should still be rejected."""
    data = {**VALID_BRIEF, "products": []}
    path = write_yaml(tmp_path, data)
    with pytest.raises(ValueError):
        parse_campaign_brief(path)


def test_invalid_aspect_ratio_rejected(tmp_path):
    data = {**VALID_BRIEF, "aspect_ratios": ["3:7"]}  # not in allowed set
    path = write_yaml(tmp_path, data)
    with pytest.raises(ValueError):
        parse_campaign_brief(path)
