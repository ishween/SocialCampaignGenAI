# Evaluation Strategy: Creative Automation Pipeline POC

**Document Version:** 2.0
**Date:** 2026-04-23
**Status:** Active

---

## Overview

The pipeline contains two fundamentally different classes of components that require different evaluation approaches. Deterministic components (YAML parsing, image resizing, compliance regex checks, file I/O) must produce the same output every time for the same input. These are validated with exact assertions and must achieve 100% accuracy. Any deviation is a bug. Non-deterministic components (image generation, LLM quality scoring, LLM text overlay) produce different outputs on each run. These cannot be pixel-matched. Instead they require statistical quality gates, human evaluation rubrics, and behavioral invariant checks. Applying deterministic tests to generative components creates false failures. Applying non-deterministic quality sampling to config parsing creates false confidence. A mature evaluation strategy applies the right technique to each class.

---

## Evaluation Tier Structure

| Tier | Frequency | Owner | What is evaluated |
|---|---|---|---|
| Tier 1: Automated | Every run / every commit | CI / pytest | Structural correctness of all deterministic components. Structural checks on every generated image. |
| Tier 2: Human sample review | Weekly, 10% sample | Creative team reviewer (rotating) | Brand alignment, product accuracy, composition quality, regional relevance |
| Tier 3: Business metrics | Monthly | Creative ops lead | Throughput, cost per asset, efficiency gain vs manual baseline |

---

## Deterministic Component Evaluation

Deterministic components require 100% accuracy. These 29 unit tests run in CI on every commit and must pass before merge.

| Component | What is tested | Required accuracy | Method |
|---|---|---|---|
| YAML parser + Pydantic models | Valid inputs parsed correctly. Invalid inputs raise ConfigError with field-level message. Extra fields rejected. | 100% | Unit tests: known-good and known-bad YAML fixtures covering all field types and edge cases |
| Pydantic security validator | Prompt injection patterns blocked. Safe inputs pass. Gemini 3.1 Flash Lite classifier returns correct binary label. | 100% | Unit tests with fixture strings. Mock for classifier. |
| Image resizer | Output dimensions are exact (1080×1080, 1080×1920, 1920×1080). No pixel stretching. RGB mode preserved. LANCZOS resampling used. | 100% | Assert exact `.size` tuple post-resize for all three ratios and multiple source aspect ratios |
| Focal point detection | Gemini 3 Flash returns a coordinate within image bounds | 100% structural, sampled quality | Assert coordinate is within (0,0)–(W,H). Human spot-check of crop centering. |
| Compliance checker | Logo check, brand color check, prohibited word regex all return correct result for fixture inputs | 100% | Unit tests with synthetic fixture images and known-bad overlay text strings |
| File I/O | Written PNG is valid and non-empty. SHA-256 of file matches in-memory image at write time. Directory structure matches spec. | 100% | Hash comparison. PIL verify(). os.path.exists for all expected paths. |
| Idempotency check | Re-run skips existing files without making API calls. Zero-byte files are re-attempted. | 100% | Integration test: pre-seed output dir, assert API mock call count is zero |

---

## Non-Deterministic Quality Gates

### What is checked automatically (every run)

Every generated image passes through structural checks before composition proceeds. These are fast, deterministic checks on the image artifact itself rather than its content.

| Check | Threshold | On failure |
|---|---|---|
| Resolution | Long edge ≥ 1024px | Regenerate via quality loop (up to max attempts) |
| File size | 100 KB – 15 MB | Regenerate if too small, log if suspiciously large |
| Not blank | Pixel standard deviation > 10 | Regenerate |
| Provider safety | Not flagged by Imagen 4 built-in filter | Discard, do not regenerate with same prompt |
| Quality score (LangGraph loop) | Gemini 3 Flash score ≥ configured threshold (0-5 scale) | Refine prompt and regenerate, flag for human review after max attempts |

### What is checked by humans (weekly sample)

The LangGraph quality loop catches low-scoring images automatically, but it cannot replace a trained human eye for brand sensitivity, cultural appropriateness, and subtle composition issues. A rotating reviewer evaluates a random stratified 10% sample each week.

---

## Human Evaluation Rubric

Reviewers score each sampled asset on a 1-5 Likert scale across four criteria. The weighted average determines pass or fail.

| Criterion | Weight | Pass threshold | 1: Poor | 3: Acceptable | 5: Excellent |
|---|---|---|---|---|---|
| Brand alignment | 30% | ≥ 3.0 | Off-brand colors, wrong aesthetic | Mostly on-brand with minor deviations | Perfect brand expression |
| Product accuracy | 30% | ≥ 3.0 | Wrong product or unrecognizable | Correct category, minor inaccuracies | Accurate product depiction, recognizable SKU attributes |
| Composition quality | 20% | ≥ 3.0 | Cluttered, imbalanced, poor framing | Clean composition, product well-positioned | Professional, ratio-optimized, strong visual hierarchy |
| Regional relevance | 20% | ≥ 3.0 | Culturally inappropriate or generic | Neutral, not offensive | Authentically resonant for target region |

**Asset pass threshold:** weighted average ≥ 3.5
**Weekly batch pass rate target:** ≥ 85% of sampled assets pass

Escalation triggers:

- Weekly average score below 3.0: prompt engineering review required
- Pass rate below 70%: pipeline paused for investigation
- One criterion consistently low across a product category: targeted prompt adjustment for that category

---

## Business Metrics Baseline

| Metric | Manual process | POC target | Measurement method |
|---|---|---|---|
| Time per asset (design + review) | ~45 min | < 10 min total (generation + review) | Wall clock per asset in pipeline summary |
| Assets per campaign (avg, 7 products × 3 ratios) | ~15 hrs (21 assets × 45 min) | < 10 min pipeline + < 30 min review | PhaseTimer in pipeline report |
| Assets per designer per day | ~10 | ~120+ (review only, not design) | Throughput count |
| Efficiency gain (time) | Baseline | ≥ 5× | Manual hrs ÷ POC hrs |
| Throughput gain | Baseline | ≥ 12× | POC assets/day ÷ manual assets/day |
| Cost per asset (FTE loaded) | ~$120 | ~$0.04–0.08 API cost | Imagen 4 + Gemini pricing per run |
| Compliance pass rate | N/A | ≥ 90% | assets_compliant ÷ assets_generated |
| Human eval avg score | N/A | ≥ 3.5 / 5.0 | Weekly rubric aggregation |

**Decision point:** If the POC demonstrates ≥ 12× efficiency gain and ≥ 90% compliance pass rate sustained over four weeks, proceed to production architecture planning as documented in `FUTURE_SCOPE.md`.

---

## Production Readiness Checklist

**Deterministic quality gates**

- Config parser: 100% unit test pass rate, three consecutive CI runs
- Image resizer: exact dimension accuracy across all three ratios
- File I/O: byte-level integrity and idempotency verified
- Compliance checker: all rule checks verified with fixture inputs

**Non-deterministic quality gates**

- Tier 1 automated: ≥ 95% structural pass rate sustained over four weeks
- Tier 2 human eval: ≥ 3.5 weighted average score sustained over four weeks
- Tier 2 human eval: ≥ 85% pass rate sustained over four weeks
- Compliance: ≥ 90% pass rate sustained over four weeks
