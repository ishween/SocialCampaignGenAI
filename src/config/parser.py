"""YAML/JSON campaign brief parser with validation."""

import json
from pathlib import Path

import yaml
from pydantic import ValidationError

from src.config.models import BrandConfig, CampaignBrief


def parse_campaign_brief(path: str | Path) -> CampaignBrief:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Campaign brief not found: {path}")

    raw = path.read_text(encoding="utf-8")

    if path.suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(raw)
    elif path.suffix == ".json":
        data = json.loads(raw)
    else:
        raise ValueError(f"Unsupported brief format '{path.suffix}'. Use .yaml or .json")

    try:
        return CampaignBrief(**data)
    except ValidationError as e:
        raise ValueError(f"Invalid campaign brief:\n{e}") from e


def parse_brand_config(path: str | Path) -> BrandConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Brand config not found: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    try:
        return BrandConfig(**data)
    except ValidationError as e:
        raise ValueError(f"Invalid brand config:\n{e}") from e
