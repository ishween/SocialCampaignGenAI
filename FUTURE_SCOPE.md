# Future Scope - Creative Automation Pipeline

**Document Version:** 2.0
**Date:** 2026-04-23
**Status:** Active

## 1. Persistent LangGraph Checkpointing

**What it is:** Replacing the in-process `MemorySaver` checkpointer in `src/generation/quality_loop.py` with `SqliteSaver` for single-machine deployments or `PostgresSaver` for distributed workers.

**Why it matters:** `MemorySaver` survives a crashed LangGraph node within the same Python process - if the evaluator throws mid-loop, a resume reloads the generated image from the checkpoint and does not repeat the ~$0.03 Imagen 4 call. However, `MemorySaver` does not survive a process kill or a server restart. `SqliteSaver` persists state to a local SQLite database and survives machine restarts. `PostgresSaver` enables multiple Kubernetes worker pods to share a single checkpoint store, so any worker can resume any product's quality loop from the last completed node.

**What is needed:** Swap the checkpointer in `quality_loop.py` to `SqliteSaver` or `PostgresSaver`, add the database connection string to environment variables, and ensure the `thread_id` (currently the product name) is stable and unique across runs.

---

## 2. Retry with Tenacity

**What it is:** Applying `tenacity` library decorators to every API-calling function in the pipeline to handle transient failures with exponential backoff and jitter.

**Why it matters:** `retry.py` was written early in the POC but was never wired to any API call. It was subsequently deleted. The 429 errors encountered during development were caused by billing credits being depleted rather than by genuine rate limiting - retry would not have helped, and the fix was topping up the credit balance. At production request volumes, 429 responses will be genuinely transient and retry with backoff is the correct response. The `tenacity` library handles all exception types (including `google.api_core.exceptions.ResourceExhausted`, which the old custom decorator missed) and expresses retry policy in a readable DSL.

**What is needed:** Add `tenacity` to `requirements.txt`. Apply `@retry` decorators to `llm_overlay.add_text_overlay_llm`, `llm_overlay.check_message_present`, `imagen.generate`, `evaluator.evaluate_image`, and `prompt_refiner.refine_prompt`. Configure `stop_after_attempt(4)`, `wait_exponential(multiplier=1, min=4, max=60)`, and `retry_if_exception_type((ResourceExhausted, ServiceUnavailable))`.

---

## 3. Provider-Side Prompt Caching

**What it is:** Using Google's `createCachedContent` API to cache repeated system prompts so that subsequent calls referencing the same cache token pay only for the incremental tokens.

**Why it matters:** The evaluator system prompt (brand criteria and scoring rubric) is sent up to three times per product through the quality loop. The LLM overlay creative direction system prompt is sent once per ratio - three times per product. At production scale with hundreds of products per day, these repeated prefixes represent meaningful token cost. Provider-side caching eliminates the cost of the prefix on every call after the first.

**What is needed:** Google's explicit caching requires a minimum of 32,768 tokens. Our current system prompts are 200–400 tokens, well below the threshold. This feature becomes applicable only if system prompts are expanded with full brand style guides, product catalogs, or extensive brand safety rules - which is the expected direction for enterprise campaigns. The implementation requires calling `client.caches.create` once per session and passing the returned cache name in all subsequent `GenerateContentConfig` objects for that prompt.

---

## 5. Human-in-the-Loop Checkpoint

**What it is:** Adding a LangGraph `interrupt_before=["refine_prompt"]` at graph compile time, pausing the quality loop after evaluation so a creative director can review the score, see the generated image, and approve or override the refinement direction before the loop continues.

**Why it matters:** The current quality loop auto-refines based on Gemini 3 Flash's score and feedback without any human judgment in the loop. For brand-sensitive campaigns - product launches, regulated industries, high-visibility placements - a creative director should have the opportunity to approve the refinement direction before another generation call is made. LangGraph's interrupt mechanism was designed precisely for this pattern.

**What is needed:** Add `interrupt_before=["refine_prompt"]` when compiling the graph in `quality_loop.py`. Build a minimal approval interface - either a CLI prompt that pauses and waits for keyboard input, or a lightweight web UI that surfaces the image and score. On approval or override, resume the graph with the reviewer's input baked into the state.

---

## 6. Brief Manifest for Idempotency Across Brief Changes

**What it is:** Writing a `pipeline_manifest.json` file alongside outputs that stores a SHA-256 hash of the campaign brief YAML. On re-run, the hash is compared against the stored value, and outputs are invalidated if the brief has changed.

**Why it matters:** Current file-based idempotency skips output files if they exist on disk, regardless of whether the brief that produced them has changed. If a creative director updates the campaign message, adds a new ratio, or changes the brand config, the old outputs will be silently reused on re-run. Stale outputs would be submitted for review without anyone realising the underlying brief changed. A brief hash makes this class of error impossible.

**What is needed:** Compute `sha256(brief_yaml_bytes)` at pipeline start. Read the stored hash from `pipeline_manifest.json` if it exists. If hashes differ, invalidate all existing outputs for the campaign and regenerate. Write the new hash alongside the completed outputs. The manifest structure maps each output path to its generation metadata (model, seed, timestamp) alongside the top-level brief hash.

---

## 7. Production Architecture

**What it is:** Replacing the sequential single-user CLI with a FastAPI service layer, a task queue backed by Redis as the broker, Kubernetes workers processing campaigns in parallel, S3 or GCS object storage for output assets, and webhook or email notifications on job completion.

**Why it matters:** The CLI processes one product at a time and writes to the local filesystem. At the expected production volume of 250 campaigns per month generating up to 55 GB of output, this is neither concurrent nor durable. Production requires multiple campaigns to run simultaneously, output to persist independently of the machine running the pipeline, and creative teams to receive completion notifications without polling a terminal. The Kubernetes worker model assigns disjoint product subsets to each pod, eliminating concurrent write conflicts without distributed locking.

**What is needed:** Wrap the pipeline runner in a FastAPI endpoint that accepts a campaign brief and returns a job ID. Submit jobs to queue via Redis. Deploy workers as Kubernetes pods with resource limits. Write outputs to S3 or GCS using the same `output/{product}/{region}/{ratio}.png` path convention. Send webhook callbacks or email notifications on job completion. This architecture is documented in `01_ARCHITECTURE_DECISIONS.md`.

---

## 8. Observability

**What it is:** OpenTelemetry spans per creative, and metrics pushed to Datadog or CloudWatch covering per-creative latency, compliance pass rate, cost per asset, and quality score trends.

**Why it matters:** Production needs alerting when quality scores degrade (indicating a model change or prompt drift), when compliance pass rates drop (indicating a brand config issue), or when per-asset costs spike (indicating retry storms or quality loop overruns). None of these alerts are possible without structured, searchable logs and a metrics backend.

**What is needed:** Wrap each phase (generation, resize, overlay, compliance) in an OpenTelemetry span. Emit cost and quality score as custom metrics. Configure Datadog or CloudWatch dashboards and alerts on the key signals. The `PhaseTimer` and `fmt_duration` utilities in `logger.py` already collect timing data - they need only to emit it as structured fields rather than formatted strings.

---

## 9. Atomic File Writes

**What it is:** Writing each output PNG to a temporary file first, then using `os.rename` (which is atomic on POSIX systems) to move it to the final path.

**Why it matters:** The current implementation writes directly to the final output path. If the process is killed mid-write - by an OOM event, a Kubernetes pod eviction, or a power loss - a partial PNG file is left on disk. On re-run, the idempotency check sees the file exists and skips it, silently leaving a corrupt asset. An atomic rename ensures that either the full file exists at the final path or nothing does.

**What is needed:** In `src/pipeline/runner.py` (and in any direct save calls), write to `path.with_suffix('.tmp')`, call `image.save(tmp_path)`, then `os.rename(tmp_path, final_path)`. Add cleanup of orphaned `.tmp` files to the pipeline startup pre-flight check.

---

## 10. Brief Validation Dry-Run Flag

**What it is:** A `--validate-only` CLI flag that parses and validates the campaign brief YAML and brand config against the Pydantic models, runs the Gemini 3.1 Flash Lite security classifier, and reports any errors without executing any generation or composition steps.

**Why it matters:** Users want to check YAML correctness before committing to a pipeline run that consumes API credits. Currently, a malformed YAML is caught at startup before any API calls, but there is no easy way to test a new brief without triggering the full pipeline. A validate-only mode costs nothing and gives the creative operations team confidence before scheduling a campaign run.

**What is needed:** Add `--validate-only` to the CLI argument parser. Route it to a function that runs only the config loading, Pydantic validation, and input security classification steps, then exits with code 0 on success or code 1 with field-level error messages on failure. No image generation or file I/O.

---
