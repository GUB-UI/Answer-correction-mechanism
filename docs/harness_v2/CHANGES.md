# Harness v2 changes

The existing app under `math-answer-diagnosis/` is unchanged. PR #9's branch is unchanged. vLLM guided JSON is not part of this harness.

FailureCase and ImprovementProposal use the field names in `docs/harness_v2/design_v0.1.md`. The summary is stored there: three grading states, the evaluation metrics, the development set (1-A 8/10, 2-B 4/10, 3-A 10/10), no special-casing of those formulas or scores, and one change type per cycle.

## Phase 0

| File | What | Why |
| --- | --- | --- |
| `docs/harness_v2/phase0_baseline.md` | Records `pytest` at `a383b29`: 132 passed in 8.72s. | Freeze the old suite before the harness. |
| `docs/harness_v2/old_pipeline.md` | Old inputs (config, problem, answer, OCR, diagnosis prompt) and outputs (JSONL OCR, diagnosis, review flags, teacher discrepancies). | The new harness has to be comparable without sharing the old store. |
| `scripts/legacy_runner.py` | Separate process. Parses `legacy_completion` with the old fence parser. | Old and new do not call each other. |
| `scripts/compare_old_vs_new.py` | Spawns the old runner and `python -m harness_v2` into different directories. | Same inputs, independent outputs. |
| `harness_v2/tests/test_compare.py` | Checks the two directories, and that a reference score never reaches the new prompt. | Locks the isolation. |

## Contracts and storage

| File | What | Why |
| --- | --- | --- |
| `harness_v2/core/contracts.py` | `RubricItem`, `ItemResult` (`item_id`, `max_points`, `score`, `evidence_spans`, `math_reason`, `uncertainty`, `review_reason`), `GradingResult` status `validated` / `review_required` / `failed`, `FailureCase`, `ImprovementProposal`, `RunRecord`. | One record shape for every provider. |
| `harness_v2/storage/sqlite_store.py`, `artifacts.py`, `secrets.py` | SQLite run rows plus write-once text artifacts. Secret-like values are redacted. Image suffixes are rejected. | Experiment rows need commit, model, versions, tokens, latency, checks, failure reason, and proposal link, without keys or student images. |
| `harness_v2/tests/test_contracts.py`, `test_run_store.py` | Missing score stays null. Token counts survive redaction. | Those two rules are easy to break by accident. |

## Providers

| File | What | Why |
| --- | --- | --- |
| `harness_v2/providers/gemma_ai_studio.py` | `google-genai`, model `gemma-4-31b-it`, key `GEMINI_API_KEY` only. Records request id, returned model id, settings, token counts, latency, and the raw body. 429 raises `QuotaExceeded` with no retry. Other transient errors retry with backoff, capped at 3 attempts. | A quota failure must stop the batch. A paid project or a second API is not a fallback. |
| `harness_v2/providers/codex_app_server.py` | JSON-RPC over stdio: `initialize`, `initialized`, `model/list`, `thread/start`, `turn/start`, `turn/completed` (`completed` / `failed` / `interrupted`), `turn/interrupt`. Default model `gpt-6.1-sol`. | Codex CLI login, no API key in this client. |
| `harness_v2/evaluation/independent.py` | Drops `reference_score` and `prior_ai_score` and scrubs those phrases in free text. | Independent evaluation must not see an earlier score. |
| `harness_v2/providers/grokbot_exchange.py`, `providers/schemas/` | Request and result JSON schemas sharing `request_id`. File exchange plus issue/branch/PR files. | Grokbot is a handoff, not an HTTP model API. |
| `harness_v2/providers/fixture.py` | Returns a saved completion. | Tests and the compare script stay offline. |
| `harness_v2/tests/test_gemma.py`, `test_codex.py`, `test_grokbot.py` | Mocked client, scripted stdio, file handoff. | No network. |

## Recognition and grading

| File | What | Why |
| --- | --- | --- |
| `harness_v2/recognition/ocr.py` | Import Xiaomi text. `run_cpu` refuses. | CPU page OCR is not a reasonable step here. See `cpu_ocr.md`. |
| `harness_v2/recognition/structure.py`, `candidates.py` | Lines, math spans, and alternate readings that do not replace the raw text. | Recognition stays evidence, not a silent correction. |
| `harness_v2/grading/rubric.py`, `scorer.py`, `validator.py`, `sympy_equiv.py` | Rubric load, one-fence JSON parse, checks for items, range, total, reason, evidence substring, and sympy equivalence when both sides are arithmetic or `sqrt`. Missing scores stay `None`. | A failed model response is `failed` or `review_required`, never a filled-in score. |
| `harness_v2/tests/test_recognition.py`, `test_grading.py` | Import, CPU refusal, missing item, range, sympy. | Locks those checks. |

## Pipeline skeleton

| File | What | Why |
| --- | --- | --- |
| `harness_v2/core/pipeline.py`, `orchestrator.py` | OCR import or provided text, prompt, provider, validate, store. `QuotaExceeded` saves the partial row and stops the rest of the batch. | Phase 1 run loop. |
| `harness_v2/improvement/proposals.py` | One `proposed` improvement from failure cases. Nothing is edited. | Phase 2 skeleton. |
| `harness_v2/__main__.py` | Offline one-file entry used by the compare script. | New side has its own process. |
| `harness_v2/tests/test_pipeline.py` | Quota stop, proposal stays `proposed`. | The batch must not continue after 429, and a proposal must not apply itself. |
| `harness_v2/requirements.txt` | `sympy` and `google-genai`. | App `requirements.txt` stays as it was. |

After these tests were added, `python3 -m pytest -q harness_v2/tests math-answer-diagnosis/tests` reported **154 passed**.

## Design v0.1 field names

| File | What | Why |
| --- | --- | --- |
| `harness_v2/core/contracts.py` | `FailureCase` and `ImprovementProposal` now use only the v0.1 names. | The earlier names (`failure_kind`, `status`, `source_case_ids`, and the rest) did not match the design. |
| `harness_v2/improvement/proposals.py` | A proposal has one entry in `changed_components`, an empty evaluation, and `decision=pending`. | One change type per cycle, and the proposal is not applied. |
| `harness_v2/core/pipeline.py` | A failed grade writes those fields. `reference_score` is copied onto the failure case and is still omitted from the prompt. | The metric needs the reference later. The model does not. |
| `docs/harness_v2/design_v0.1.md` | Three statuses, the metric list, and the development set 1-A 8/10, 2-B 4/10, 3-A 10/10. | The summary has to live in the repo. |
| `harness_v2/tests/test_contracts.py` | The serialized keys must equal the design lists, in that order. The design summary must name the three statuses, the metrics, and the development set. | An extra or renamed field fails the test. |

`python3 -m pytest -q harness_v2/tests math-answer-diagnosis/tests` after this alignment: **155 passed**.

## Review of 9290369

Status name stays `review_required`. The review's rename to `needs_review` was not applied.

| Finding | What changed |
| --- | --- |
| Sandbox | `thread/start` sends `sandbox: "read-only"` and checks it against `codex_sandbox_mode.schema.json` (`read-only`, `workspace-write`, `danger-full-access`). |
| Independent review | Grading input is an allowlist of problem, answer/OCR text, and rubric fields. Proposal review keeps failure-case fields except scores, plus a diff. Score keys inside JSON strings are removed. Each independent turn starts a new thread. |
| result.json | The CLI writes `redact(record)`. Secret redaction parses JSON embedded in strings. |
| Gemma | `genai.Client(vertexai=False, api_key=...)`. `GOOGLE_GENAI_USE_VERTEXAI` is rejected. No project or location is passed. |
| Scores | Item max is the rubric max. A different model max is `review_required`. NaN, inf, and non-numeric scores fail. |
| Rubric shape | Duplicate item ids and an empty rubric fail. |
| Review flags | Any model `review_reason`, or uncertainty at or above 0.5, is `review_required` when scores exist. |
| Pipeline review | Optional `codex_reviewer` is off by default. Disagreement is stored and sets `review_required`. |
| Codex transport | Responses are buffered by id. Server requests are answered and are not treated as our result. Reads can time out. Stderr is drained. The child is terminated on close. |
| Grokbot paths | `request_id` is a safe token. Resolved paths must stay in the exchange directory. |
| Grokbot files | Image data URIs are stripped. `read_result` validates the schema. |
| OCR bytes | Exact text is stored once under `ocr_private/` and hashed. A redacted sibling is the committable copy. Grading checks the hash. |
| CPU OCR | Float32 subprocess, revision `ff990be318c4c68f7e621dd690231200f47e48c4`, thinking off, greedy, 4096 tokens, no resize. bfloat16 and low memory are refused. Revision, dtype, and timing are recorded. |
| Prompt | The prompt lists the object the validator checks. |
| Gemma settings | `max_output_tokens` is kept. Settings, response id, and attempt count are stored on the run. Quota rows include grading status `failed`. |
| Requirements | `sympy` is in `math-answer-diagnosis/requirements.txt` so a clean venv can run the equivalence test. |

## PR #10 remaining re-review findings (2026-10-23)

All eight partial findings (2, 4, 5, 6, 9, 10, 12, 13), all three new MAJOR findings, and the new MINOR finding are addressed. Grading statuses remain `validated` / `review_required` / `failed`. Missing scores remain null. There are no formula-specific or score-specific branches, no paid API/fallback, no provider verification requests, and no changes under `math-answer-diagnosis/`.

| File | What | Why |
| --- | --- | --- |
| `harness_v2/evaluation/independent.py` | Scrubs generic/total/reference/AI score assignments, attributed prose, quoted ratios/units, and Japanese scoring annotations. Requires complete scalar values so symbolic expressions survive. Keeps unchanged embedded mathematical JSON verbatim and excludes non-scalar rubric text fields. | Finding 2: independent reviewers must not receive earlier scores, while intervals, fractions, geometric point counts and symbolic expressions stay intact. |
| `harness_v2/storage/secrets.py` | Keeps unchanged embedded JSON text verbatim during public redaction. | Finding 13: redaction must not gratuitously reformat mathematical text in prompt/handoff artifacts. |
| `harness_v2/grading/rubric.py` | Rejects malformed rows, blank/duplicate IDs, boolean/nonfinite/negative allocations and a nonfinite allocation sum. | Finding 6: invalid rubric data must fail before allowlisting or inference. |
| `harness_v2/grading/scorer.py` | Raises on malformed response rows or missing/invalid IDs. Checks every supplied `max_points` and `max_score`, including null, boolean, string and nonfinite values; rubric allocations stay authoritative. | Findings 5/6: bad model rows/maxima cannot disappear or validate. Missing item scores are never filled. |
| `harness_v2/grading/validator.py` | Independently validates direct rubric objects and avoids returning invalid maxima from failed rubric checks. | Finding 6: callers bypassing the JSON loader must fail closed too. |
| `harness_v2/providers/gemma_ai_studio.py` | Requires the literal config flag `free_tier_project=True` before SDK construction, pins `https://generativelanguage.googleapis.com` and `vertexai=False`, records `provider_policy` on successful/error/quota exchanges, and keeps that metadata out of generation config. Replaces opaque client injection with a factory receiving the pinned options. | Finding 4: the caller explicitly attests that the key is for a free-tier AI Studio project. This is a local config attestation, not a remotely verified billing claim. No tier-verification call, alternative endpoint, paid fallback or retry on 429. |
| `harness_v2/providers/codex_app_server.py` | Appends trusted grading instructions/schema after blinding; retains fresh threads. Filters fallback messages and completion notifications by thread and turn. Declines command/file approvals and returns JSON-RPC method-not-found for unsupported requests. Uses nonblocking byte framing/writes and a single absolute request/turn deadline, with bounded interruption and process cleanup on timeout. | Findings 9/10 and new MAJOR 1: prevent foreign output contamination, invalid server replies, buffered-line false timeouts, partial-line/write hangs and instruction-free reviews. |
| `harness_v2/evaluation/review.py` | Defines the trusted review contract and accepts only completed, error-free, schema-valid reviews that pass grading validation and match each primary item score and total. Rejects nonfinite JSON, flags and unsupported response fields. | New MAJOR 2: equal totals alone must not approve failed, flagged or item-disagreeing review. |
| `harness_v2/evaluation/schemas/grading_review.schema.json` | Closed JSON schema for independent items, evidence, reasons, uncertainty and nullable scores/total. | New MAJOR 1/2: the prompt and acceptance check share an explicit response contract. |
| `harness_v2/core/pipeline.py` | Validates raw rubrics before blinding; records exact raw/derived grading artifact paths and SHA256 values; validates evidence against the text actually graded. Persists OCR engine/revision/dtype/elapsed time. Catches reviewer exceptions and saves their separate failed outcome with primary grading routed to `review_required`. | Finding 13, new MAJOR 2/3 and new MINOR: preserve auditable inputs and completed grading even when review fails; retain CPU provenance. |
| `harness_v2/recognition/ocr.py` | Decodes imported and subprocess OCR from bytes without universal-newline translation. | Finding 13: CRLF and other UTF-8 source bytes survive the private copy and grading path exactly. |
| `harness_v2/providers/grokbot_exchange.py` | Validates publication before redaction, rejects image-bearing content and unsupported fields, enforces finite rubric numbers, and blinds independent payload strings. Both file and GitHub handoffs use the same publication boundary. | Finding 12: image keys, image data URIs (including non-base64), image markup/URLs and arbitrary nested payloads must not reach publishable files. |
| `harness_v2/providers/schemas/grokbot_request.schema.json` | Replaces unrestricted payload objects with declared text fields and a closed rubric-item schema. | Finding 12: publication has a text-field allowlist instead of accepting arbitrary image-bearing metadata. |
| `harness_v2/requirements.txt` | Declares `jsonschema>=4`. | Use one standard validator for the independent response contract; no legacy dependency changes. |
| `harness_v2/tests/test_rereview_independent.py` | Regressions for assignment/prose/Japanese/quoted score blinding, nested-field exclusion, untouched mathematical formatting, point counts and numeric-prefix symbolic math. | Finding 2/13 and final-review counterexamples. |
| `harness_v2/tests/test_rereview_validation.py` | Parameterized regressions for both maximum aliases, malformed rows and invalid loaded/direct rubrics. | Findings 5/6; no auto-filled scores. |
| `harness_v2/tests/test_rereview_gemma_policy.py` | Mock-only regressions for absent/non-boolean attestation, pinned SDK options despite environment overrides, 429 stopping and attestation persistence in SQLite. | Finding 4, without network calls. |
| `harness_v2/tests/test_rereview_codex.py` | Regressions for actual blinded/trusted prompt, scoped output, request refusals, buffered/fragmented UTF-8/partial frames, absolute response/turn deadlines, blocked writes and timeout cleanup. | Findings 9/10 and new MAJOR 1, including the final review's blocked-write counterexample. |
| `harness_v2/tests/test_rereview_pipeline.py` | Regressions for unsuccessful/malformed/flagged/item-disagreeing review, successful agreement, reviewer exceptions, exact CRLF grading, paired raw/derived hashes and persisted CPU metadata. | Finding 13, all new MAJOR findings affecting acceptance/storage, and new MINOR. |
| `harness_v2/tests/test_rereview_grokbot.py` | Regressions reject image/unknown/nested fields in both delivery modes and verify independent publication blinding. | Finding 12. |
| `harness_v2/tests/test_gemma.py` | Adapts SDK mocks to explicit attestation and factory construction, asserts pinned endpoint and preserves quota/retry tests. | Finding 4's intentional setup API change remains covered offline. |
| `harness_v2/tests/test_codex.py` | Adds thread IDs to notification fixtures and expects an explicit decline response. | Finding 9: exercise scoped protocol notifications and fail-closed approvals. |
| `harness_v2/tests/test_grokbot.py` | Uses embedded secret-bearing text within the publication allowlist and expects image-bearing requests to be rejected. | Finding 12: update earlier tests to the stricter publication contract while retaining credential redaction checks. |
| `docs/superpowers/plans/2026-10-23-pr10-rereview.md` | Records the findings, constraints, implementation sequence and final validation. | Keep the implementation/review rationale inspectable without git writes. |
| `commit_plan.json` | Groups every changed file exactly once into feature commits, with documentation last. Pre-existing untracked review/log files are excluded. | User requested a reviewable commit plan, without committing or writing git state. |
| `docs/harness_v2/CHANGES.md`, `docs/CHANGES_COLAB.md` | Record each changed file, behavior, reason and validation result. | Keep both existing change-log conventions complete. |

Final validation: `/tmp/pt2/bin/python -m pytest -q harness_v2/tests math-answer-diagnosis/tests` — **311 passed**. The six new re-review modules contain **136 parameterized test cases**. The initial sandbox run failed only because existing legacy tests bind localhost mock servers; the passing run used the same command with `PYTHONPATH=/tmp/pr10_offline`, a temporary guard blocking external connections/DNS, and sandbox permission for those localhost fixtures. No external provider calls or model downloads were made. A read-only reviewer found additional natural-language blinding, numeric-prefix math and blocked-stdin cases; all were reproduced as failing tests and fixed before final validation. `git diff --check` is clean. No git writes were performed.

## Free-tier attestation optional (2026-10-23)

| File | What | Why |
| --- | --- | --- |
| `harness_v2/providers/gemma_ai_studio.py` | `free_tier_project` is now optional (default `None`); the mandatory opt-in check is removed. `vertexai=False`, the pinned AI Studio endpoint, stop-on-429 and no fallback are unchanged. `provider_policy` records `free_tier_project` (or `null`) and `attestation_source` (`caller_config` or `none`). | The user confirmed the AI Studio key's project has no billing configured, so the extra opt-in only blocked runs. |
| `harness_v2/tests/test_rereview_gemma_policy.py` | Replaces the "attestation required" tests with: omitted attestation builds the pinned client and records `null`/`none`; quota still stops after one call. | Lock the relaxed rule without loosening routing or 429 behaviour. |
