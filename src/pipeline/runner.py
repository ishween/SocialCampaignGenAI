"""
Pipeline Runner: four phases per product.

Phases:
  1 · Asset Resolution   existing_asset → load | null → LangGraph quality loop
  2 · Resize             Gemini focal-point crop to each aspect ratio (parallel)
  3+4 · Overlay + Save   LLM text overlay → compliance → save  (per ratio, parallel)
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from src.assets.manager import AssetManager
from src.compliance.checker import ComplianceResult, run_compliance_check
from src.composition.branding import add_logo
from src.composition.llm_overlay import add_text_overlay_llm, check_message_present
from src.generation.message_generator import generate_region_message
from src.composition.overlay import add_text_overlay          # PIL fallback
from src.composition.resizer import detect_focal_point, resize_to_aspect
from src.config.models import BrandConfig, CampaignBrief, ProductConfig, RegionConfig
from src.config.parser import parse_brand_config
from src.generation.base import ImageGenerator
from src.generation.factory import get_generator
from src.generation.quality_loop import run_quality_loop
from src.security.input_validator import validate_brief
from src.utils.logger import PhaseTimer, fmt_duration

logger = logging.getLogger(__name__)


# ── Result types ─────────────────────────────────────────────────────────────

@dataclass
class AssetResult:
    product: str
    region: str
    aspect_ratio: str
    output_path: Path
    generated: bool
    compliance: ComplianceResult | None = None
    duration_sec: float = 0.0
    skipped: bool = False   # True when idempotency hit (file already existed)


@dataclass
class PipelineResult:
    campaign_name: str
    asset_results: list[AssetResult] = field(default_factory=list)
    total_generated: int = 0
    total_reused: int = 0
    total_failed: int = 0
    duration_sec: float = 0.0

    @property
    def total_assets(self) -> int:
        return len(self.asset_results)

    def print_summary(self):
        print("\n" + "=" * 60)
        print(f"Campaign : {self.campaign_name}")
        print(f"Assets   : {self.total_assets}  (gen={self.total_generated}  reuse={self.total_reused}  fail={self.total_failed})")
        print(f"Duration : {fmt_duration(self.duration_sec)}")
        print("=" * 60)
        for r in self.asset_results:
            src  = "GEN"   if r.generated else ("SKIP" if r.skipped else "REUSE")
            ok   = "✓"     if (r.compliance and r.compliance.passed) else "⚠"
            dur  = f"  {fmt_duration(r.duration_sec)}" if not r.skipped else "  (cached)"
            print(f"  [{src}] {r.product}/{r.region}/{r.aspect_ratio} → {r.output_path.name} {ok}{dur}")
        print()

    def to_json_report(self, path: Path):
        data = {
            "campaign": self.campaign_name,
            "summary": {
                "total_assets": self.total_assets,
                "generated": self.total_generated,
                "reused": self.total_reused,
                "failed": self.total_failed,
                "duration": fmt_duration(self.duration_sec),
                "duration_sec": round(self.duration_sec, 2),
            },
            "assets": [
                {
                    "product": r.product,
                    "region": r.region,
                    "aspect_ratio": r.aspect_ratio,
                    "output": str(r.output_path),
                    "source": "generated" if r.generated else "reused",
                    "skipped_idempotent": r.skipped,
                    "compliance_passed": r.compliance.passed if r.compliance else None,
                    "compliance_warnings": r.compliance.warnings if r.compliance else [],
                    "duration": fmt_duration(r.duration_sec),
                    "duration_sec": round(r.duration_sec, 2),
                }
                for r in self.asset_results
            ],
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info(f"Report saved: {path}")


# ── Pipeline runner ───────────────────────────────────────────────────────────

class PipelineRunner:
    def __init__(
        self,
        brief: CampaignBrief,
        output_dir: Path,
        assets_dir: Path,
        brand_config_path: Path,
        generator_preference: str = "auto",
        quality_threshold: float = 3.5,
        max_gen_attempts: int = 3,
        native_ratios: bool = False,
    ):
        self.brief = brief
        self.output_dir = output_dir
        self.assets_dir = assets_dir
        self.brand_config = self._load_brand(brand_config_path)
        self.asset_manager = AssetManager(assets_dir)
        self.generator: ImageGenerator | None = None
        self.generator_preference = generator_preference
        self.quality_threshold = quality_threshold
        self.max_gen_attempts = max_gen_attempts
        self.native_ratios = native_ratios

        self._has_message = bool(brief.message.strip()) or any(
            r.message_override for r in brief.regions
        )

    def _load_brand(self, path: Path) -> BrandConfig:
        if path.exists():
            return parse_brand_config(path)
        logger.warning(f"Brand config not found at {path}, using defaults")
        return BrandConfig(name="Brand")

    def _get_generator(self) -> ImageGenerator:
        if self.generator is None:
            self.generator = get_generator(self.generator_preference)
        return self.generator

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _slug(name: str) -> str:
        return name.lower().replace(" ", "_").replace("-", "_")

    def _output_path(self, product: ProductConfig, region: RegionConfig, ratio: str) -> Path:
        """Compute the canonical output path for a single creative."""
        label = ratio.replace(":", "x")
        return self.output_dir / self._slug(product.name) / region.code.lower() / f"{label}.png"

    def _save(self, image: Image.Image, out_path: Path) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.convert("RGB").save(out_path, "PNG", optimize=True)
        logger.info(f"    Saved → {out_path}")
        return out_path

    # ── Phase 1: Asset resolution ─────────────────────────────────────────────

    def _resolve_asset(self, product: ProductConfig) -> tuple[Image.Image, bool]:
        """
        Returns (base_image, is_generated).
        Existing assets skip the quality loop. They are already production-ready.
        """
        asset_path = self.asset_manager.get_asset(product.name, product.existing_asset)
        if asset_path:
            logger.info(f"    Existing asset loaded: {asset_path}")
            return self.asset_manager.load_image(asset_path), False

        logger.info(f"    No existing asset, starting LangGraph quality loop...")
        region = self.brief.regions[0]
        image, flagged, score = run_quality_loop(
            product=product,
            region=region,
            brand_style=self.brief.brand_style,
            target_audience=self.brief.target_audience,
            quality_threshold=self.quality_threshold,
            max_attempts=self.max_gen_attempts,
        )
        if flagged or image is None:
            raise RuntimeError(
                f"'{product.name}' flagged for human review after {self.max_gen_attempts} "
                f"attempts (best score: {score:.2f})"
            )
        self.asset_manager.save_generated(image, product.name)
        logger.info(f"    Generated and cached  (quality score: {score:.2f})")
        return image, True

    # ── Phase 1b: Parallel native-ratio generation (--native-ratios) ─────────

    def _resolve_assets_parallel_ratios(
        self,
        product: ProductConfig,
    ) -> dict[str, Image.Image]:
        """
        Run one quality loop per aspect ratio concurrently.

        Used when --native-ratios is active and no existing_asset is present.
        Imagen 4 (and DALL-E 3) natively compose for the target format rather than
        generating a 1:1 base that gets cropped later.

        Cost: 3× Imagen 4 calls vs 1× in the default path (~$0.06 vs ~$0.02).
        Benefit: each ratio is composed for its frame, no focal-point crop needed.

        Each ratio gets its own LangGraph thread_id so checkpoints are independent:
        a mid-loop crash on the 9:16 job does not forfeit the completed 1:1 result.

        Returns a dict of {ratio: image} for every ratio that succeeded.
        Ratios that are flagged for human review or throw are omitted (logged as errors).
        """
        region = self.brief.regions[0]  # Generation uses the primary region

        def _run_one(ratio: str) -> tuple[str, Image.Image, float]:
            img, flagged, score = run_quality_loop(
                product=product,
                region=region,
                brand_style=self.brief.brand_style,
                target_audience=self.brief.target_audience,
                quality_threshold=self.quality_threshold,
                max_attempts=self.max_gen_attempts,
                aspect_ratio=ratio,
            )
            if flagged or img is None:
                raise RuntimeError(
                    f"'{product.name}' [{ratio}] flagged for human review "
                    f"after {self.max_gen_attempts} attempts (best score: {score:.2f})"
                )
            return ratio, img, score

        results: dict[str, Image.Image] = {}
        n = len(self.brief.aspect_ratios)
        logger.info(f"    Launching {n} parallel quality loop(s) for native ratios…")

        with ThreadPoolExecutor(max_workers=n) as ex:
            futures = {ex.submit(_run_one, r): r for r in self.brief.aspect_ratios}
            for future in as_completed(futures):
                ratio = futures[future]
                try:
                    _, img, score = future.result()
                    self.asset_manager.save_generated(img, f"{product.name}_{ratio.replace(':', 'x')}")
                    results[ratio] = img
                    logger.info(f"    [{ratio}] Generated ✓  (score: {score:.2f})")
                except Exception as e:
                    logger.error(f"    [{ratio}] ✗ Failed, skipping: {e}")

        return results

    # ── Phase 2: Resize ───────────────────────────────────────────────────────

    def _resize_all_ratios(
        self,
        base_image: Image.Image,
        focal_point: tuple[float, float],
    ) -> dict[str, Image.Image]:
        """
        Resize to all campaign aspect ratios in parallel.
        Per-future exception handling: one ratio failing does not discard the rest.
        """
        def resize_one(ratio: str) -> tuple[str, Image.Image]:
            return ratio, resize_to_aspect(base_image, ratio, focal_point)

        results: dict[str, Image.Image] = {}
        with ThreadPoolExecutor(max_workers=len(self.brief.aspect_ratios)) as ex:
            futures = {ex.submit(resize_one, r): r for r in self.brief.aspect_ratios}
            for future in as_completed(futures):
                ratio = futures[future]
                try:
                    _, img = future.result()
                    results[ratio] = img
                    logger.debug(f"    [{ratio}] resized ✓")
                except Exception as e:
                    logger.error(f"    [{ratio}] resize failed, skipping: {e}")
        return results

    # ── Phase 3+4: Overlay + Compliance + Save (per ratio) ───────────────────

    def _process_ratio(
        self,
        ratio: str,
        resized_img: Image.Image,
        message: str,
        product: ProductConfig,
        region: RegionConfig,
        is_existing_asset: bool,
        is_generated: bool,
    ) -> AssetResult | None:
        """
        Single-ratio pipeline: overlay → compliance → save.
        Self-contained so it runs safely in a ThreadPoolExecutor worker.

        Idempotency: if output already exists on disk, load it and return
        immediately, with no re-generation, no API calls.

        Returns AssetResult on success (including idempotent skips), None on failure.
        """
        t0 = time.time()
        out_path = self._output_path(product, region, ratio)

        # ── Idempotency check ────────────────────────────────────────────────
        if out_path.exists():
            logger.info(f"    [{ratio}] Already on disk, skipping (idempotent)")
            existing = Image.open(out_path)
            compliance = run_compliance_check(
                existing, self.brand_config, message, self.brief.prohibited_words
            )
            return AssetResult(
                product=product.name, region=region.code, aspect_ratio=ratio,
                output_path=out_path, generated=False,
                compliance=compliance, duration_sec=time.time() - t0, skipped=True,
            )

        try:
            brand_color = (
                self.brand_config.primary_colors[0]
                if self.brand_config.primary_colors else ""
            )

            # ── Overlay ──────────────────────────────────────────────────────
            if not message.strip():
                # No campaign message: just add logo
                creative = add_logo(resized_img, self.brand_config)

            elif is_existing_asset and check_message_present(resized_img, message):
                # Message already baked into the asset, don't double-stamp it
                logger.info(f"    [{ratio}] Campaign message already in image, overlay skipped")
                creative = add_logo(resized_img, self.brand_config)

            else:
                # LLM adds the text (placement + typography + contrast in one step)
                result = add_text_overlay_llm(
                    image          = resized_img,
                    message        = message,
                    aspect_ratio   = ratio,
                    brand_style    = self.brief.brand_style,
                    target_audience= self.brief.target_audience,
                    brand_color    = brand_color,
                )
                if result is None:
                    # Fallback: PIL gradient vignette
                    logger.warning(f"    [{ratio}] LLM overlay unavailable, PIL fallback")
                    result = add_text_overlay(resized_img, message, self.brand_config)

                creative = add_logo(result, self.brand_config)

            # ── Compliance ───────────────────────────────────────────────────
            compliance = run_compliance_check(
                creative, self.brand_config, message, self.brief.prohibited_words
            )

            # ── Save immediately ─────────────────────────────────────────────
            self._save(creative, out_path)
            duration = time.time() - t0
            logger.info(f"    [{ratio}] ✓ Complete  [{fmt_duration(duration)}]")

            return AssetResult(
                product=product.name, region=region.code, aspect_ratio=ratio,
                output_path=out_path, generated=is_generated,
                compliance=compliance, duration_sec=duration,
            )

        except Exception as e:
            logger.error(f"    [{ratio}] ✗ Failed: {e}")
            return None

    # ── Message resolution ────────────────────────────────────────────────────

    def _resolve_message(self, region: RegionConfig) -> str:
        """
        Resolve the campaign message for a region.

        Priority:
          1. region.message_override: explicit YAML value, used as-is
          2. LLM-adapted message: Gemini 3.1 Flash Lite adapts brief.message
                                   for the region's language and cultural context.
                                   Validated against prohibited_words and injection
                                   patterns before use.
          3. brief.message: original message, used when LLM unavailable
        """
        if region.message_override:
            logger.info(f"  [Message] {region.code}: using YAML override")
            return region.message_override

        adapted = generate_region_message(
            brief_message=self.brief.message,
            region=region,
            target_audience=self.brief.target_audience,
            prohibited_words=self.brief.prohibited_words,
        )
        if adapted:
            return adapted

        logger.info(f"  [Message] {region.code}: using brief message (no adaptation)")
        return self.brief.message

    # ── Orchestration ─────────────────────────────────────────────────────────

    def run(self) -> PipelineResult:
        t_campaign = time.time()
        result = PipelineResult(campaign_name=self.brief.name)

        validation = validate_brief(self.brief)
        if not validation.passed:
            raise ValueError(
                "Campaign brief failed validation:\n"
                + "\n".join(f"  - {v}" for v in validation.violations)
            )

        n_p = len(self.brief.products)
        n_r = len(self.brief.regions)
        n_a = len(self.brief.aspect_ratios)

        logger.info(f"\n{'='*60}")
        logger.info(f"Campaign : {self.brief.name}")
        logger.info(f"Products : {n_p}  Regions : {n_r}  Ratios : {n_a}  Total : {n_p*n_r*n_a}")
        logger.info(f"Overlay  : {'LLM (+ PIL fallback)' if self._has_message else 'OFF, no message'}")
        logger.info(f"Gen mode : {'native-ratios (parallel loops, no resize)' if self.native_ratios else 'base-image + resize'}")
        logger.info(f"{'='*60}\n")

        for product in self.brief.products:
            logger.info(f"── Product: {product.name}")

            # ── Phase 1: Asset resolution ─────────────────────────────────
            #
            # Two paths:
            #   a) native_ratios=True AND no existing_asset
            #      → parallel quality loops (one per ratio) → skip Phase 2
            #   b) existing_asset OR native_ratios=False
            #      → single quality loop (or load) + Phase 2 resize
            #
            has_existing = bool(product.existing_asset)
            use_parallel = self.native_ratios and not has_existing

            if use_parallel:
                with PhaseTimer(logger, "Phase 1 · Asset Resolution (native-ratios, parallel)"):
                    resized = self._resolve_assets_parallel_ratios(product)
                    is_generated = True

                if not resized:
                    logger.error(f"    All ratio loops failed, skipping product")
                    result.total_failed += n_r * n_a
                    continue
                # Phase 2 is skipped: resized already contains native-ratio images
            else:
                with PhaseTimer(logger, "Phase 1 · Asset Resolution"):
                    try:
                        base_image, is_generated = self._resolve_asset(product)
                    except Exception as e:
                        logger.error(f"    FAILED, skipping product: {e}")
                        result.total_failed += n_r * n_a
                        continue

                # ── Phase 2: Resize ───────────────────────────────────────
                with PhaseTimer(logger, "Phase 2 · Resize"):
                    try:
                        focal_point = detect_focal_point(base_image)
                        resized = self._resize_all_ratios(base_image, focal_point)
                    except Exception as e:
                        logger.error(f"    FAILED, skipping product: {e}")
                        result.total_failed += n_r * n_a
                        continue

                    if not resized:
                        logger.error(f"    No ratios resized, skipping product")
                        result.total_failed += n_r * n_a
                        continue

            # ── Phase 3+4: Overlay + Save (per region × ratio, parallel) ─
            for region in self.brief.regions:
                message = self._resolve_message(region)

                with PhaseTimer(logger, f"Phase 3+4 · Overlay + Save  [{region.code.upper()}]"):
                    with ThreadPoolExecutor(max_workers=len(resized)) as ex:
                        futures = {
                            ex.submit(
                                self._process_ratio,
                                ratio,
                                resized[ratio],
                                message,
                                product,
                                region,
                                not is_generated,   # is_existing_asset
                                is_generated,
                            ): ratio
                            for ratio in resized    # only successfully resized ratios
                        }
                        for future in as_completed(futures):
                            ratio = futures[future]
                            try:
                                asset = future.result()
                            except Exception as e:
                                logger.error(f"    [{ratio}] Unexpected error: {e}")
                                asset = None

                            if asset:
                                result.asset_results.append(asset)
                                if asset.skipped:
                                    result.total_reused += 1
                                elif is_generated:
                                    result.total_generated += 1
                                else:
                                    result.total_reused += 1
                            else:
                                result.total_failed += 1

        result.duration_sec = time.time() - t_campaign
        result.print_summary()
        result.to_json_report(self.output_dir / "pipeline_report.json")
        return result
