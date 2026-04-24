#!/usr/bin/env python3
"""
Creative Automation Pipeline: CLI entry point.

Usage:
    python src/main.py --brief config/example_campaign.yaml
"""

import argparse
import os
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file if present (so keys don't have to be exported every session)
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

from src.config.parser import parse_campaign_brief
from src.pipeline.runner import PipelineRunner
from src.utils.logger import setup_logger


def parse_args():
    p = argparse.ArgumentParser(
        description="Creative Automation Pipeline: generate localized social ad creatives",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--brief", required=True, help="Path to campaign brief (.yaml or .json)")
    p.add_argument("--output", default="output", help="Output directory")
    p.add_argument("--assets", default="assets/input", help="Input assets directory")
    p.add_argument("--brand-config", default="config/brand_config.yaml", help="Brand config file")
    p.add_argument(
        "--generator", default="auto", choices=["auto", "imagen"],
        help="Image generator: 'auto' = Gemini Flash (default, any API key), 'imagen' = Google Imagen 4 (requires billing)"
    )
    p.add_argument(
        "--quality-threshold", type=float, default=3.5,
        help="Minimum quality score (0-5) from LangGraph evaluator to accept an image"
    )
    p.add_argument(
        "--max-attempts", type=int, default=3,
        help="Max Generate→Evaluate→Refine iterations before flagging for human review"
    )
    p.add_argument(
        "--native-ratios", action="store_true", default=False,
        help=(
            "Generate one natively-composed image per aspect ratio in parallel "
            "instead of generating a single 1:1 base and resizing. "
            "Produces better-composed images at 3× the generation cost. "
            "Has no effect for products that supply an existing_asset."
        ),
    )
    p.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return p.parse_args()


def main():
    args = parse_args()
    log_file = Path(args.output) / "pipeline.log"
    setup_logger(level=args.log_level, log_file=log_file)

    import logging
    logger = logging.getLogger(__name__)

    logger.info("Creative Automation Pipeline starting...")

    try:
        brief = parse_campaign_brief(args.brief)
        logger.info(f"Campaign brief loaded: '{brief.name}' ({len(brief.products)} products)")
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to load campaign brief: {e}")
        sys.exit(1)

    runner = PipelineRunner(
        brief=brief,
        output_dir=Path(args.output),
        assets_dir=Path(args.assets),
        brand_config_path=Path(args.brand_config),
        generator_preference=args.generator,
        quality_threshold=args.quality_threshold,
        max_gen_attempts=args.max_attempts,
        native_ratios=args.native_ratios,
    )

    try:
        result = runner.run()
        if result.total_failed > 0:
            logger.warning(f"{result.total_failed} asset(s) failed, see logs for details")
            sys.exit(2)  # Partial failure: not a crash, but caller should notice
    except Exception as e:
        logger.exception(f"Pipeline crashed: {e}")
        sys.exit(1)

    logger.info("Done.")


if __name__ == "__main__":
    main()
