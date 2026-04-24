# Design Decisions & Trade-offs

**Document Version:** 1.0
**Date:** 2026-04-23

Every significant decision made during this POC, with alternatives considered and the reason for the choice. Intended as a reference for reviewers.

---

## 1. Orchestration Framework: LangGraph

### Decision
Use LangGraph to orchestrate the Generate → Evaluate → Refine quality loop.

### Why LangGraph

| Requirement | How LangGraph satisfies it |
|---|---|
| State persists across iterations | `TypedDict` state schema passed between nodes. No global variables. |
| Conditional routing | `add_conditional_edges` maps evaluator output to accept / refine / flag without if-chains in application code |
| Crash recovery within a session | `MemorySaver` checkpointer saves state after each node. A crashed evaluator resumes from the last checkpoint rather than restarting generation. |
| Human-in-the-loop readiness | `interrupt_before=["refine_prompt"]` at compile time pauses the graph for a human review step. One config change, no refactor. |
| Independent node testability | Each node is a plain Python function accepting and returning a dict, unit tested without running the full graph. |

### Alternatives considered

**LangChain (chains / agents)**
LangChain is built for linear chains or opaque ReAct-style agents. It has no native graph topology, no built-in conditional routing between steps, and no checkpoint mechanism. Adding a cycle (generate → evaluate → regenerate) requires either a custom loop or a LangGraph dependency anyway. Rejected: wrong abstraction for a stateful cycle.

**CrewAI**
CrewAI is a role-based multi-agent framework where multiple AI agents collaborate on a shared goal. The quality loop is a single-product, single-concern task: generate one image, score it, improve the prompt. A crew of agents adds coordination overhead with no benefit. Rejected: designed for multi-agent collaboration, overkill for a bounded single-product loop.

**LlamaIndex**
LlamaIndex is a document retrieval and RAG (Retrieval-Augmented Generation) framework. Its primary abstraction is the index over a corpus of documents. It has no image generation, no quality scoring, and no graph orchestration relevant to this use case. Rejected: wrong domain entirely.

**Raw Python (for loop + retry)**
A `for attempt in range(max_attempts)` loop with a `break` on success is simpler to read. It works, but state is implicit (local variables), crash recovery is not possible without external serialization, adding a human-in-the-loop step requires restructuring the loop, and routing logic (accept / refine / flag) is embedded in if-chains rather than declared as graph edges. Rejected for production path. Acceptable only if the quality loop never needs to evolve beyond its initial form.

---

## 2. Two-Layer Input Validation: Regex + LLM Classifier

### Decision
Run deterministic regex blocklist first, then Gemini 3.1 Flash Lite binary classifier second, only when Layer 1 passes.

### Why both layers are needed

**Layer 1 - Regex (always runs, zero cost)**
Catches known injection signatures instantly: `ignore previous instructions`, `jailbreak`, `act as a`, `developer mode`, etc. These are fixed patterns that do not require reasoning to detect. Fast, deterministic, auditable. Does not require an API call.

**Layer 2 - LLM Classifier (runs only when Layer 1 passes)**
Novel and obfuscated attacks bypass regex: a subtle override embedded in a plausible-looking product description, a prompt that restructures itself mid-sentence, or a multi-step indirect injection. The LLM classifier reasons over intent and context, not just surface patterns. Catches what regex cannot.

### Why not LLM only
LLM classifiers are probabilistic: they can miss obvious regex-catchable patterns if they appear in an unexpected format. They also add ~$0.000075/call and ~0.5s latency. For simple known patterns, regex is more reliable and free. Layer 1 acts as a zero-cost pre-filter. Layer 2 adds depth for the cases that pass through.

### Why not regex only
Adversarial prompts evolve. A new injection technique not in the blocklist would pass Layer 1 silently. The LLM classifier generalises to patterns it was not explicitly trained on, providing a safety net for unknowns.

### Design invariant
Layer 2 is **never called if Layer 1 finds a violation**. This ensures a malicious brief cannot "confuse" the LLM classifier by embedding the injection inside the tested content. L1 failure → immediate reject, no further processing.

---

## 3. Evaluation Design - LLM-as-Judge + Deterministic Compliance

### Decision
Split evaluation into two distinct mechanisms with different purposes.

| Mechanism | What it evaluates | Method | When it runs |
|---|---|---|---|
| LangGraph quality loop | Response quality - does the generated image match the brand brief, product identity, and visual style? | Gemini 3 Flash scores 0–5 against brand criteria | During generation, before composition |
| Compliance checker | Brand rule adherence - logo present, brand colors in palette, no prohibited words | Deterministic: template match, histogram comparison, regex | After composition, before save |

### Why LLM-as-judge for quality
Brand alignment, composition quality, and product accuracy are semantic concepts that cannot be encoded as rules. "Does this image feel like a premium athletic brand?" requires visual reasoning. Gemini 3 Flash scores images against a structured rubric - the same rubric a human creative director would use - and provides natural-language feedback that drives prompt refinement.

### Why deterministic checks for compliance
Brand compliance must be auditable and reproducible. "Does this image contain a prohibited word?" must always give the same answer for the same input. An LLM answer to this question varies by temperature and phrasing. Regex is 100% deterministic, zero cost, and produces an exact match list. Logo detection via template match is explainable: here is the confidence score, here is the region where the match was found.

### Why not human-only evaluation
At 250 campaigns/month × 3 ratios × 5 products = 3,750 images/month, human review of every generated image at the generation stage is not viable. The quality loop automates the generation-time filter. Human review is reserved for the 10% weekly sample and for assets flagged after max attempts - where human judgment adds the most value.

---

## 4. LLM Text Overlay vs PIL-Only Approach

### Decision
Use Gemini 3.1 Flash Image as the primary text overlay method. PIL gradient vignette is the fallback.

### Why LLM overlay
A PIL-only approach requires two sequential steps:
1. Vision call to detect the best placement region (focal point, background brightness)
2. PIL rendering at the detected position

These are two independent decisions that can conflict: the vision model recommends "bottom-left" but the background there is too bright for white text. The PIL renderer does not know this.

Gemini 3.1 Flash Image handles placement, typography, contrast treatment, and blending as **one holistic creative act** - the same way a designer would. It sees the whole image before deciding where to put the text. The result looks composed into the photo rather than stamped on top.

### Why keep PIL as fallback
The LLM overlay path has failure modes: API quota exhaustion, provider outages, safety filter rejections. Discarding the asset when the overlay fails is unacceptable - the image generation step costs ~$0.02. The PIL fallback guarantees an asset is always written. Quality is lower (static bottom gradient, no adaptive placement) but the run completes and the asset reaches the reviewer.

---

## 5. Provider Selection - Single Google API Key

### Decision
Use only `GOOGLE_API_KEY`. All four models (Imagen 4, Gemini 3 Flash, Gemini 3.1 Flash Lite, Gemini 3.1 Flash Image) share one key and one SDK.

### Why single provider
- **Operational simplicity:** one key to manage, one SDK version to pin, one billing account to monitor
- **Consistent safety:** all models enforce the same Google AI safety policies
- **Cost tracking:** all spend aggregates in one Google Cloud project

### Why Imagen 4 over DALL-E 3 or Stable Diffusion for generation
Imagen 4 provides IP indemnification: Google's commercial usage policy covers the generated images against third-party IP claims. This matters for enterprise campaigns. DALL-E 3 and Stable Diffusion do not offer equivalent indemnification. A third-party provider would also break the single-key, single-SDK constraint and introduce split billing. If a non-Google provider is needed in the future, the `ImageGenerator` base class is the extension point.

### Why four different models rather than one
The pipeline uses the right model for each task rather than forcing every task through the most capable (and expensive) model:
- Image synthesis needs a dedicated image model (Imagen 4), not a text model pretending to generate images
- Quality scoring needs vision capability (Gemini 3 Flash), not text-only
- Prompt rewriting and safety classification are text tasks where speed matters more than power (Gemini 3.1 Flash Lite - ~5× cheaper than Flash)
- Text overlay needs image-in → image-out (Gemini 3.1 Flash Image), not just a chat response

---

## 6. Parallel Execution - Per-Ratio ThreadPoolExecutor

### Decision
Process products sequentially, resize and overlay each aspect ratio in parallel per product.

### Why not fully parallel across products
- Single-user CLI: no concurrent write contention between products at POC scale
- Sequential products make logs readable and debugging straightforward
- The expensive step (Imagen 4 generation) is one call per product regardless, parallelising it does not reduce API cost

### Why parallel at the ratio level
The three ratios (1:1, 9:16, 16:9) for a single product are completely independent after resize - no shared state, no write conflict (each writes to a different output path). Running them concurrently cuts composition wall-clock time by ~3×. `ThreadPoolExecutor` with `as_completed` and per-future exception handling means one ratio failing does not cancel the others.

### Production path
At 250 campaigns/month and multiple concurrent users, product-level parallelism is needed. The production design assigns disjoint product subsets to Kubernetes workers (one worker per product, or per campaign) with Redis preventing concurrent writes to the same output path. See FUTURE_SCOPE.md.

---

## 7. Idempotency: File Existence Check

### Decision
Before any API call for a given ratio, check whether the output file already exists on disk. If it does, load and compliance-check it, then skip all generation and overlay calls.

### Why file-based idempotency
- Zero infrastructure dependency: no database, no Redis, no external state
- Correct for the single-user sequential CLI model
- Makes re-runs after crashes or partial failures trivially safe - only missing outputs are rebuilt
- Consistent with how creative teams think: "does this output file exist?" is the natural question

### Why not hash-based idempotency
A hash-based system (store SHA-256 of brief + product + region → output path) would detect when the brief has changed and invalidate stale outputs. The current file check does not - if the brief message changes, existing output files are silently reused. This is documented in FUTURE_SCOPE.md (Brief Manifest for Idempotency) as the production fix.

### Production risk
If two workers attempt to write the same output path simultaneously, the file-existence check is not atomic. Worker A checks (file missing), Worker B checks (file missing), both generate, both write - last writer wins, one API call wasted. The production fix is Redis SETNX keyed on `{campaign_id}:{product}:{region}:{ratio}` before the check. Not needed at POC scale (sequential CLI, single user).

---

## 8. LangGraph Checkpointing - MemorySaver

### Decision
Use LangGraph `MemorySaver` for quality loop checkpointing.

### What it provides
After each node completes (build_prompt → generate → evaluate → refine), LangGraph saves the full state to an in-process dictionary. If the evaluator throws an exception after Imagen 4 has already returned an image, re-invoking the graph with the same `thread_id` resumes from the evaluate node - the image is loaded from the checkpoint, not regenerated.

### Limitation
`MemorySaver` is in-process only. A process kill (OOM, Kubernetes pod eviction, machine restart) loses all checkpoint state. On restart, the quality loop begins from scratch - one Imagen 4 credit wasted per interrupted product.

### Why not SqliteSaver or PostgresSaver now
`SqliteSaver` requires a database path, `PostgresSaver` requires a connection string and a running database. For a single-user CLI POC, adding a database dependency would not be evaluated and would complicate the setup. The in-process `MemorySaver` is correct for the POC scope. The migration path is one constructor swap - documented in FUTURE_SCOPE.md.

---

## 9. Compliance - Deterministic Rules, Not LLM

### Decision
Brand compliance is checked with three deterministic rules: logo template match, brand color histogram comparison, prohibited word regex. No LLM is involved in compliance gating.

### Why deterministic
Compliance is a hard gate - an asset either passes or it does not, and the decision must be reproducible. "Does this image contain the word 'guaranteed'?" must always be the same answer. An LLM answer to this question can vary between calls. For auditable brand governance, deterministic checks are the correct choice.

### What deterministic compliance cannot catch
Semantic brand alignment - "does this image feel like our brand?" - is not expressible as a rule. This is handled by the LangGraph quality loop (Gemini 3 Flash score), not by the compliance checker. The two mechanisms are complementary, not redundant.

---

## 10. Region Message Adaptation - LLM Generation + Validation

### Decision
When `message_override` is not set in the brief, Gemini 3.1 Flash Lite adapts `brief.message` for the region's language and cultural context. The adapted message is validated against `prohibited_words` and injection patterns before use. Failure falls back to `brief.message`.

### Why LLM adaptation rather than requiring manual overrides
At 10 regions × 50 campaigns, writing 500 manual override strings per quarter is not scalable. LLM adaptation generates culturally appropriate copy automatically. The human override field remains available for markets where exact copy control is required.

### Why validate the generated message
The LLM controls the output but does not know the brand's prohibited word list. Without validation, it could produce a message containing a prohibited term (e.g. "guaranteed" in German = "garantiert") which would cause downstream compliance failure and a wasted overlay call. Validation before the overlay step catches this class of error early. Security injection check is also applied - though the risk of the adaptation prompt itself being injected is low, defense-in-depth is correct practice.

### Why Gemini 3.1 Flash Lite, not Flash
Message adaptation is a short-text translation and cultural rewriting task - no vision, no complex reasoning. Flash Lite is ~5× cheaper, under 1s latency, and produces equivalent output for this task. Cost matters when this call runs once per region per product per campaign.
