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

## Gemma request timeout (2026-10-23, found in the first live run)

| File | What | Why |
| --- | --- | --- |
| `harness_v2/providers/gemma_ai_studio.py` | `REQUEST_TIMEOUT_MS = 180000` passed as `http_options.timeout` to the pinned AI Studio client. Endpoint, `vertexai=False`, stop-on-429 and no fallback unchanged. | In the Phase 2 live run, two AI Studio calls stalled with no response (one blocked the batch for >20 min). A bounded timeout turns a stall into a recorded error instead of a hang. |
| `harness_v2/tests/test_rereview_gemma_policy.py`, `harness_v2/tests/test_gemma.py` | Expect the timeout in the pinned SDK options; new test checks it is set and bounded. | Regression for the hang. |

## Post-hoc reference evaluation, reliable comparison, and private storage (2026-10-23)

Grading states remain `validated` / `review_required` / `failed`. Reference labels are post-hoc evaluation data only: a mismatch creates a failure/proposal without changing grading status or scores. Missing scores remain null; no formula or score is special-cased. No files under `math-answer-diagnosis/` were edited and no git write commands were run.

| Fix | Files | What | Why |
| --- | --- | --- | --- |
| 1 | `harness_v2/core/pipeline.py` | Compare the predicted total and referenced item scores after grading/review; persist mismatches through the existing FailureCase/proposal/RunRecord link path, including validated grades. Always blind model-facing input, including when `independent=False`. | A structurally validated grade can still disagree with its reference; labels must not influence inference or grading status. |
| 1 | `harness_v2/evaluation/independent.py` | Remove `reference_items` from nested JSON and labelled text assignments; non-JSON/malformed assignments redact the remaining text conservatively. | Item reference maps can contain no key named `score` and previously escaped blinding. Raw OCR remains available in its exact private artifact. |
| 1 | `harness_v2/tests/test_reference_evaluation.py` | Cover total/item mismatches, matching/missing references, zero references, null scores, provider prompts, actual Codex turns and JSON/Python/YAML label text. | Verify post-hoc detection and absence of reference keys/values at both model boundaries. |
| 2 | `harness_v2/evaluation/comparison.py` | Add reusable result loading, old/new validation and benchmark accounting. Require canonical fields, legacy required item fields, preserved item review flags, valid scores/evidence/totals and a validated new grade; incomplete cases stay in the denominator. | Exit zero, parse success, missing results or cross-schema aliases must not inflate success counts. |
| 2 | `scripts/compare_old_vs_new.py` | Inspect both saved results and exit codes; record `success`, comparison status and explicit reasons; return nonzero for incomplete/empty runs. Secure summary files/directories, including rewrites. | Report incomplete comparisons honestly; stderr/validation details can contain private text. Comparison statuses are separate from grading states. |
| 2 | `harness_v2/tests/test_comparison_reliability.py` | Cover missing/invalid completions and files, canonical-field alias attacks, legacy schema omissions/review flags, benchmark counts, empty inputs, CLI exit and summary permissions. | Regression coverage exercises real subprocess outputs as well as reusable evaluation. |
| 3 | `harness_v2/storage/permissions.py` | Add owner-only directory creation and descriptor-based opens/writes with explicit chmod before reading, truncating or writing existing files. | Set files to `0o600` and directories to `0o700` without relying on or changing the process umask. |
| 3 | `harness_v2/storage/artifacts.py` | Apply secure creation/opens to all artifacts and parent directories; chmod existing artifacts even when an immutable overwrite is refused. | Protect `ocr_private/`, `grading_private/`, raw responses and arbitrarily named private artifacts while preserving exact bytes and immutability. |
| 3 | `harness_v2/storage/sqlite_store.py` | Precreate/tighten the DB and existing journal/WAL/SHM before connecting; close connections explicitly and recheck permissions after transactions. | SQLite inherits secure DB permissions for newly created sidecars; existing databases and sidecars also need tightening. |
| 3 | `harness_v2/recognition/ocr.py`, `harness_v2/recognition/run_ocr_cpu.py` | Secure imported raw OCR on opening, plus generated OCR Markdown, stats and output directories on creation/rewrite. | Raw OCR must remain private before it enters the artifact store too. |
| 3 | `harness_v2/__main__.py`, `scripts/legacy_runner.py` | Secure CLI and legacy result/input output files and directories, including existing files. | Run JSON and legacy input copies contain student work and raw model text. |
| 3 | `harness_v2/providers/grokbot_exchange.py` | Secure request/result and handoff files, result reads and containing directories. | File-based provider artifacts also contain private student text. |
| 3 | `harness_v2/tests/test_private_permissions.py` | Use `os.umask(0o000)` and `os.stat` for new/existing artifacts, OCR import/output, CLI/legacy/exchange files, DB and live journal/WAL/SHM; restore umask afterward. | Prove privacy with a permissive caller umask and verify chmod on reopening/rewriting. CPU output tests use local model doubles without downloads. |
| All | `docs/harness_v2/CHANGES.md`, `docs/CHANGES_COLAB.md`, `commit_plan.json` | Record these changes and verification; group files into exactly three proposed commits in fix order, with both changelogs in the last group. | Keep implementation reviewable without modifying git state. |

Validation: `/workspace/venv_h2/bin/python -m pytest -q harness_v2/tests math-answer-diagnosis/tests` — **379 passed in 12.27s**. Existing app tests require localhost mock servers; the passing run used sandbox permission for loopback sockets. The initial restricted run had 345 passes and 14 socket-permission failures/errors. No external network requests or model downloads were needed. A read-only review found non-JSON reference text, empty comparisons and schema-alias counterexamples; regression tests now cover them. `git diff --check` passed.

## Phase 3 / 01: Evidence matching and explicit absence (2026-10-23)

| File | What | Why |
| --- | --- | --- |
| `harness_v2/core/contracts.py, harness_v2/grading/scorer.py, harness_v2/grading/validator.py` | Preserve evidence_absent/absent_reason; allow absent evidence only for numeric zero and literal true with a reason. Normalize paired math delimiters, leading math equals and whitespace only for matching. | Remove Phase 2 false flags without accepting invented evidence or rewriting raw text. |
| `harness_v2/evaluation/review.py, harness_v2/evaluation/schemas/grading_review.schema.json` | Document and validate the same absence contract for independent Codex review. | Keep both grading boundaries consistent. |
| `harness_v2/tests/test_p3_evidence.py` | Cover absence types, nonzero/null rejection, delimiter normalization, preserved spans and fabricated evidence. | Regression coverage for Phase 2 findings. |

Validation: `/workspace/venv_clean/bin/python -m pytest -q harness_v2/tests math-answer-diagnosis/tests` — **394 passed in 12.69s** (offline guard; loopback mocks allowed).

## Phase 3 / 02: Root runtime dependencies (2026-10-23)

| File | What | Why |
| --- | --- | --- |
| `requirements.txt` | Add bounded google-genai, jsonschema and sympy requirements in the existing range style. | Root installation must support every harness dependency, including the existing mechanical checker. |
| `harness_v2/tests/test_p3_requirements.py` | Compare parsed dependency names and lower bounds between harness and root requirements. | Prevent the Phase 2 missing-SDK setup failures from recurring. |

Validation: `/workspace/venv_clean/bin/python -m pytest -q harness_v2/tests math-answer-diagnosis/tests` — **395 passed in 12.30s** (offline guard; loopback mocks allowed).

## Phase 3 / 03: Persistent setup failures and retry-safe artifacts (2026-10-23)

| File | What | Why |
| --- | --- | --- |
| `harness_v2/core/pipeline.py` | Reserve per-request run attempts with exclusive markers; retain original first-attempt paths and suffix retries. Persist pre-return exceptions as failed records with error class/reason and artifact links. Persist quota before rethrow and reuse that record. | Setup failures must remain auditable and retryable even after crashes or orphaned write-once files. |
| `harness_v2/tests/test_p3_setup.py` | Exercise missing key, SDK/setup/runtime errors, same-id retries, orphaned artifacts and quota persistence. | Ensure retry does not overwrite history or double-record quota. |

Validation: `/workspace/venv_clean/bin/python -m pytest -q harness_v2/tests math-answer-diagnosis/tests` — **400 passed in 12.58s** (offline guard; loopback mocks allowed).

## Phase 3 / 04: Total and per-attempt latency (2026-10-23)

| File | What | Why |
| --- | --- | --- |
| `harness_v2/core/contracts.py, harness_v2/providers/gemma_ai_studio.py` | ProviderExchange retains per-attempt latency/status/error class/known usage. latency_ms now measures the entire generation loop including retry backoff, for success, errors and quota. | The final attempt alone understated Phase 2 request time and retry usage. |
| `harness_v2/core/pipeline.py, harness_v2/tests/test_p3_latency.py` | Persist attempt_count/history in completed and quota records; verify 20ms error + 500ms backoff + 30ms final attempt with a fake clock. | Keep total and attempt distributions auditable without network calls. |

Validation: `/workspace/venv_clean/bin/python -m pytest -q harness_v2/tests math-answer-diagnosis/tests` — **403 passed in 12.15s** (offline guard; loopback mocks allowed).

## Phase 3 / 05: Imported OCR provenance (2026-10-23)

| File | What | Why |
| --- | --- | --- |
| `harness_v2/recognition/ocr.py` | Import <file>.meta.json or an explicit metadata path; merge case metadata with precedence, preserve engine/model revision/dtype/source/timing and normalize elapsed_s. | CPU/imported OCR must carry the same auditable provenance as local inference without rewriting bytes. |
| `harness_v2/core/pipeline.py, harness_v2/tests/test_p3_ocr_metadata.py` | Pass case provenance to OCR import and persist the full metadata alongside canonical fields. Test all import modes, precedence, CRLF and SHA256. | Allow Phase 2 imported fp32 outputs to be evaluated with their actual provenance. |

Validation: `/workspace/venv_clean/bin/python -m pytest -q harness_v2/tests math-answer-diagnosis/tests` — **407 passed in 12.34s** (offline guard; loopback mocks allowed).

## Phase 3 / 06: Reusable post-inference reference FailureCases (2026-10-23)

| File | What | Why |
| --- | --- | --- |
| `harness_v2/evaluation/reference.py` | Share reference_errors and failure_from_run for RunRecord or SQLite-loaded dictionaries; report total/item mismatches without changing grades. | Stored validated runs need the same FailureCase path as fresh inference, without post-hoc scripts or new model calls. |
| `harness_v2/core/pipeline.py` | Use the shared evaluator for proposals and retain problem/input version metadata in run outputs. | Avoid divergent live/stored mismatch detection and preserve provenance. |
| `harness_v2/tests/test_p3_reference.py` | Test stored validated mismatches/matches and zero/null scores; existing reference tests retain actual Codex/provider blinding coverage. | References remain evaluation-only and scores are never filled. |

Validation: `/workspace/venv_clean/bin/python -m pytest -q harness_v2/tests math-answer-diagnosis/tests` — **409 passed in 12.23s** (offline guard; loopback mocks allowed).

## Phase 3 / 07: Versioned private dataset manifests (2026-10-23)

| File | What | Why |
| --- | --- | --- |
| `harness_v2/evaluation/dataset.py` | Read/write versioned manifests with case/input/rubric provenance; verify hashes, IDs, paths and reference ranges. Keep reference_scores/reference_items in their own hashed file and out of inference cases; secure files/directories. | Reproducible experiments need frozen, portable inputs and separately held labels. |
| `scripts/build_dev_dataset.py, scripts/dev_dataset_references.json` | Build manual C0/C1/manual and Xiaomi fp32 inputs from the supplied local sources and stats. Copy declared labels as data, with item labels only for 1-A/3-A. | Rebuild the Phase 2 seven-condition dev set offline without score-specific runtime logic or committing transcripts. |
| `harness_v2/tests/test_p3_dataset.py, docs/harness_v2/phase3_usage.md` | Cover temporary source fixtures, CRLF, separation, hash tampering, path escapes, duplicate cases and 0600/0700 modes; document schema and builder command. | Make input identity and privacy independently verifiable. |

Validation: `/workspace/venv_clean/bin/python -m pytest -q harness_v2/tests math-answer-diagnosis/tests` — **415 passed in 12.68s** (offline guard; loopback mocks allowed).

## Phase 3 / 08: Repeatable resumable run CLI (2026-10-23)

| File | What | Why |
| --- | --- | --- |
| `harness_v2/run.py` | Load explicit config/dataset; deterministically identify each repetition; resume from SQLite; checkpoint state/results atomically; save per-run failures; stop immediately on quota. Record constructor failures through a lazy provider boundary. | Replace one-off Phase 2 scripts with a repeatable offline-testable experiment entry point. |
| `harness_v2/providers/gemma_ai_studio.py, harness_v2/providers/fixture.py, harness_v2/core/pipeline.py, harness_v2/storage/sqlite_store.py` | Forward effective generation settings including seed, bind requested Gemma model without fallback, retain fixture settings and attach experiment provenance to stored runs. | Configuration must describe the actual call and immutable input/attempt history. |
| `harness_v2/tests/test_p3_runner.py, docs/harness_v2/phase3_usage.md` | Test deterministic repetitions, completed skips, retry after quota/setup errors, delay, offline CLI, settings and incompatible resume identities; document config and commands. | Ensure reproducibility and label isolation without live Gemma/Codex calls. |

Validation: `/workspace/venv_clean/bin/python -m pytest -q harness_v2/tests math-answer-diagnosis/tests` — **420 passed in 13.13s** (offline guard; loopback mocks allowed).

## Phase 3 / 09: Baseline/candidate comparability and CLI (2026-10-23)

| File | What | Why |
| --- | --- | --- |
| `harness_v2/evaluation/comparison.py` | Align saved runs by case/repetition; reject changed model/provider/problem/input/OCR/rubric/reference/settings identities or incomplete/duplicate results with reasons. Compute initial success/latency metrics and recursive deltas. | Baseline and candidate results are meaningful only under matching evaluation conditions. |
| `harness_v2/evaluation/compare.py` | Add a private-report CLI for run directories/results files; verify the dataset and return incomparable with reasons on invalid inputs. | Make comparisons repeatable without inference. |
| `harness_v2/tests/test_p3_comparison.py, docs/harness_v2/phase3_usage.md` | Exercise each refusal boundary, returned model, effective settings, incomplete cases, same-condition metrics and the real offline CLI. | Prevent accidental apples-to-oranges comparisons. |

Validation: `/workspace/venv_clean/bin/python -m pytest -q harness_v2/tests math-answer-diagnosis/tests` — **432 passed in 13.28s** (offline guard; loopback mocks allowed).

## Phase 3 / 10: Complete metrics and severe regression verdicts (2026-10-23)

| File | What | Why |
| --- | --- | --- |
| `harness_v2/evaluation/metrics.py` | Compute MAE/coverage, item agreement, traceable-span evidence accuracy, wrong grading with item/total addition-deduction split, review recall/false flags, success, total/per-attempt latency and retry/history usage. Keep missing scores unscored and undefined rates null. | The Phase 2 measurements must be reproducible inside evaluation, with honest denominators and unknown-token accounting. |
| `harness_v2/evaluation/comparison.py` | Compute both metric sets/deltas and quality verdicts. Any new wrong case/repetition/item or lower evidence accuracy forces regressed. Verify each expected/stored case and actual raw/derived hashes against the dataset, not just against the other report. | Aggregate MAE gains or identical corrupted provenance must not conceal severe regressions. |
| `harness_v2/core/pipeline.py, harness_v2/evaluation/independent.py` | Guard the manifest input hash before inference on every repetition; retain provenance/history on failure. Strip plural reference_scores maps from free text as well as JSON at both model boundaries. | Final regression checks reproduced between-repetition source changes and a plural-label leak; neither may contaminate metrics or prompts. |
| `harness_v2/tests/test_p3_metrics.py, harness_v2/tests/test_p3_comparison.py, harness_v2/tests/test_p3_runner.py, harness_v2/tests/test_p3_reference.py` | Use numerical fixtures for all metrics, unknown retries, empty/missing results, false flags, per-repetition severe errors despite lower MAE, mutated inputs and inconsistent provenance. Verify plural labels against a mocked provider and actual scripted Codex turn. | Regression tests cover quality, execution accounting and reference isolation offline. |
| `docs/harness_v2/phase3_usage.md` | Document metric denominators, nulls, latency/history semantics, unknown usage, verdict rules and the 07/08/09 commands. | Keep results interpretable and the workflow repeatable. |

Validation: `/workspace/venv_clean/bin/python -m pytest -q harness_v2/tests math-answer-diagnosis/tests` — **447 passed in 13.54s** (offline guard; loopback mocks allowed).

## Phase 3 review fix / 01: Immutable implementation identity (2026-10-23)

| File | What | Why |
| --- | --- | --- |
| `harness_v2/run.py` | Persist commit in experiment/results metadata and include it in the experiment identity; reject a different or missing stored commit before checkpoints or provider calls. Treat `unknown` as an exact identity value. | Resume must never mix implementation revisions or silently relabel existing runs. |
| `harness_v2/tests/test_p3_runner.py` | Cover same-commit resume, distinct commit identities, changes to/from `unknown`, and unchanged checkpoints on rejection. | Pin the implementation boundary and prevent provider calls under an incompatible revision. |

Validation: `python -m pytest -q harness_v2/tests math-answer-diagnosis/tests` — **452 passed in 13.48s** (using `/workspace/venv_clean/bin` on PATH; loopback mocks require sandbox escalation).

## Phase 3 review fix / 02: Recover interrupted experiment bookkeeping (2026-10-23)

| File | What | Why |
| --- | --- | --- |
| `harness_v2/core/pipeline.py, harness_v2/run.py` | Save experiment_case with the initial SQLite row. Before skipping completed requests, reconcile missing provenance and regenerate missing FailureCases for stored attempts; keep existing failure artifacts unchanged. | Interruptions after a durable grade must not leave incomplete reports or trigger another provider call for completed runs. |
| `harness_v2/tests/test_p3_runner.py` | Interrupt after row save and before failure-file write; exercise legacy provenance recovery, unchanged completed grades, historical failed attempts, idempotency and private permissions. | Prove recovery works from SQLite alone while preserving attempt history and reference isolation. |

Validation: `python -m pytest -q harness_v2/tests math-answer-diagnosis/tests` — **455 passed in 14.03s** (using `/workspace/venv_clean/bin` on PATH; loopback mocks allowed).

## Phase 3 review fix / 03: Validate recorded OCR provenance (2026-10-23)

| File | What | Why |
| --- | --- | --- |
| `harness_v2/recognition/ocr.py` | Hash exact imported text bytes as source_sha256 and exact sidecar bytes as sidecar_sha256, for automatic and explicit metadata files. Compute these after merging metadata so supplied claims cannot override them. | Unchanged transcript bytes do not prove that mutable sidecar provenance stayed unchanged. |
| `harness_v2/evaluation/comparison.py` | Check each run's recorded engine/revision/dtype/source hash against manifest expectations and its counterpart; honor declared source/sidecar hashes and revision aliases. Reject missing provenance on OCR-bearing runs and changed sidecar hashes; exclude elapsed time from the identity. | Matching manifests or identical corrupt run metadata must not conceal different actual OCR conditions. |
| `harness_v2/tests/test_p3_comparison.py, harness_v2/tests/test_p3_ocr_metadata.py` | Cover one/both-side mutations, missing fields, revision aliases, sidecar-only changes, exact CRLF hashes, forged claims and private permissions. | Keep comparisons fail-closed while preserving byte identity and offline evaluation. |

Validation: `python -m pytest -q harness_v2/tests math-answer-diagnosis/tests` — **473 passed in 14.33s** (using `/workspace/venv_clean/bin` on PATH; loopback mocks allowed).

## Phase 3 review fix / 04: Direction of total-only grading errors (2026-10-23)

| File | What | Why |
| --- | --- | --- |
| `harness_v2/evaluation/metrics.py` | Count unjust_addition/deduction whenever a total mismatch falls back to __total__ because no labeled item is wrong. Share the same branch with cases having no item labels. | Matching or partial item labels previously recorded a wrong total without counting its error direction. |
| `harness_v2/tests/test_p3_metrics.py` | Cover additions/deductions with matching full/partial item labels and verify no double counting when an item already explains the error. Assert input grades remain unchanged. | Report directional errors accurately without assigning unreferenced item errors or modifying scores. |

Validation: `python -m pytest -q harness_v2/tests math-answer-diagnosis/tests` — **479 passed in 14.09s** (using `/workspace/venv_clean/bin` on PATH; loopback mocks allowed).

## Phase 3 live-run fix / 05: Symmetric evidence normalization (2026-10-23)

| File | What | Why |
| --- | --- | --- |
| `harness_v2/grading/validator.py` | normalize_evidence no longer strips a leading `=` from math-delimited bodies; it only removes `$…$`, `\\(…\\)`, `\\[…\\]` delimiters and whitespace. | The strip was applied to the answer but not to an undelimited cited span, so a verbatim span such as `= \sqrt{9}-\sqrt{2}` against `$= \sqrt{9}-\sqrt{2}$` was judged untraceable. In the Phase 3 live baseline this caused all 3 review_required flags on 1-A xiaomi_cpu_fp32 (correctly graded). Comparison view only; stored text unchanged. |
| `harness_v2/tests/test_p3_evidence.py` | Regression: spans with/without leading `=` and with/without delimiters match a delimited answer. Existing negative cases (`=x+1` vs `x+1`, `$=$`) still fail. | Prevent the asymmetric false positive. |

Validation: `python -m pytest -q harness_v2/tests math-answer-diagnosis/tests` — **482 passed in 14.16s**.

## Phase 3 review fix / 06: Resolve the implementation identity by default (2026-10-23)

| File | What | Why |
| --- | --- | --- |
| `harness_v2/run.py` | Resolve omitted CLI/API commits from the harness source checkout's HEAD, adding `+dirty` for tracked modifications. Refuse unresolved, empty or `unknown` identities before writing checkpoints or calling providers. Explicit identities work without Git. | Default runs must preserve the commit boundary on resume, including legacy experiments recorded as `unknown`. |
| `harness_v2/tests/conftest.py, harness_v2/tests/test_p3_commit_identity.py, harness_v2/tests/test_p3_runner.py` | Mock resolution independently of repository state; cover clean/dirty HEAD, Git failures, CLI/API defaults, explicit identities, legacy rejection and unchanged files on refused resumes. | Make revision binding reproducible without depending on the real checkout or sending references to models. |

Validation: `python -m pytest -q harness_v2/tests math-answer-diagnosis/tests` — **506 passed in 14.31s** (using `/workspace/venv_clean/bin` on PATH; existing loopback mocks allowed with sandbox escalation).

## Phase 3 review fix / 07: Validate settings on failed provider calls (2026-10-23)

| File | What | Why |
| --- | --- | --- |
| `harness_v2/evaluation/comparison.py` | Validate recorded effective generation settings for current runs and historical attempts, including failed grades. Exempt only explicit setup errors with zero attempts and no response evidence; reject missing or invalid settings and uncertain call provenance. Continue excluding provider policy metadata from generation parameters. | Failed calls and retries contribute to comparison metrics, so their actual conditions must match the declared experiment settings. |
| `harness_v2/tests/test_p3_failed_settings.py` | Cover parse/HTTP/quota failures, one/both-side mismatches, missing/invalid settings, successful resumes with incompatible failed history, genuine setup failures, uncertain counters and contradictory response evidence. Assert comparison preserves records and leaves failed scores unscored. | Prevent status-based validation bypasses while retaining the pre-call setup exemption, reference isolation and honest metrics. |

Validation: `python -m pytest -q harness_v2/tests math-answer-diagnosis/tests` — **542 passed in 15.33s** (using `/workspace/venv_clean/bin` on PATH; existing loopback mocks allowed with sandbox escalation).

## Phase 3 review fix / 08: Distinguish tracked local modifications (2026-10-23)

| File | What | Why |
| --- | --- | --- |
| `harness_v2/run.py` | Append the SHA256 of exact tracked `git diff HEAD` bytes to dirty HEAD identities. Include staged, unstaged and binary changes; disable external diff/text conversion. Keep clean HEAD identities unchanged and fail closed if Git cannot resolve the diff. | Different local modifications must not share a resumable implementation identity. |
| `harness_v2/tests/test_p3_commit_identity.py` | Mock all Git resolution; cover clean and distinct dirty diffs, raw binary bytes, deterministic resolution, diff failures and refusal to resume another dirty tree before writes or provider calls. | Verify the revision boundary without depending on the real repository. |

Validation: `python -m pytest -q harness_v2/tests math-answer-diagnosis/tests` — **544 passed in 16.79s** (using `/workspace/venv_clean/bin` on PATH; existing loopback mocks allowed with sandbox escalation). No git writes.

## Phase 3 review fix / 09: Validate durable resume identity before metadata (2026-10-23)

| File | What | Why |
| --- | --- | --- |
| `harness_v2/run.py` | Read existing SQLite rows in read-only mode before writing experiment metadata. Validate every attempt's commit, experiment name and available case/config/dataset provenance; persist the full experiment identity in new row provenance. Reject mismatches, unreadable identities and legacy rows without sufficient metadata. Matching durable identities can recover missing experiment.json without inference; matching metadata still supports legacy provenance repair. | Missing or rewritten JSON checkpoints must not relabel durable results or mix implementation/experiment identities on resume. |
| `harness_v2/tests/test_p3_durable_identity.py` | Cover removed/rewritten metadata, unknown/different durable commits, corrupt databases/JSON/schema, experiment and case identity mismatches, historical failed rows, safe metadata recovery and private permissions. Assert rejected resumes preserve all files and never call a provider. | SQLite remains the durable source of truth and validation precedes checkpoint writes. |

Validation: `python -m pytest -q harness_v2/tests math-answer-diagnosis/tests` — **571 passed in 15.71s** (using `/workspace/venv_clean/bin` on PATH; existing loopback mocks allowed with sandbox escalation). No git writes.

## Phase 3 review fix / 10: Validate OCR provenance across contributing attempts (2026-10-23)

| File | What | Why |
| --- | --- | --- |
| `harness_v2/evaluation/comparison.py` | Validate case provenance, OCR metadata/source hashes and derived grading-input hashes on contributing historical attempt_runs as well as latest logical runs. Require consistent sidecar identity, including presence/absence, across a case's attempts and between experiments even when the latest failure precedes OCR. Exempt only failures without OCR or downstream response evidence. | Historical failures contribute usage and retry metrics; a later successful or pre-OCR attempt must not hide incompatible OCR conditions. |
| `harness_v2/storage/secrets.py` | Preserve numeric usage counters ending in _token_runs/_token_attempts while continuing to redact secret-valued strings and credential fields. | Unknown-token counts are metrics and must remain numeric in persisted comparison reports and deltas. |
| `harness_v2/tests/test_p3_historical_ocr.py` | Cover one/both-side historical metadata corruption, missing fields/hashes, changed/missing/declared sidecars, real sidecar changes on resume, all grading statuses, pre-OCR exemptions, cross-history comparison, derived input hashes and private CLI report counters. Assert comparison does not mutate records or fill failed scores. | Keep full attempt accounting auditable without inference, reference exposure or score substitution. |

Validation: `python -m pytest -q harness_v2/tests math-answer-diagnosis/tests` — **603 passed in 18.12s** (using `/workspace/venv_clean/bin` on PATH; existing loopback mocks allowed with sandbox escalation). No git writes; math-answer-diagnosis sources unchanged.

## Phase 3 final check fix / 11: Require full durable experiment identity (2026-10-23)

| File | What | Why |
| --- | --- | --- |
| `harness_v2/run.py` | Require every SQLite attempt to carry the matching full experiment_identity in experiment_case before any checkpoint writes or recovery. Remove metadata-based legacy provenance repair; absent provenance or a missing full identity now refuses resume even with matching experiment.json. Keep checkpoint and failure-artifact recovery for proven durable identities. | Mutable metadata and partial request/case hashes cannot prove the original problem text, OCR conditions or full experiment; repairing unproven rows would relabel old grades. |
| `harness_v2/tests/test_p3_durable_identity.py, harness_v2/tests/test_p3_runner.py` | Cover missing provenance/full identity with matching or rewritten metadata, changed problem text/OCR metadata and historical failed attempts. Preserve interruption recovery with durable identity, unchanged grades, idempotency and private permissions. | Refused resumes must leave all output files unchanged and never reach a provider; proven recovery must remain available without inference or score filling. |

Validation: `python -m pytest -q harness_v2/tests math-answer-diagnosis/tests` — **615 passed in 18.05s** (using `/workspace/venv_clean/bin` on PATH; existing loopback mocks allowed with sandbox escalation). No git writes; math-answer-diagnosis sources unchanged.

## Phase 3 final check fix / 12: Prove pre-OCR exemptions and reject malformed outputs (2026-10-23)

| File | What | Why |
| --- | --- | --- |
| `harness_v2/evaluation/comparison.py` | Exempt missing OCR provenance only for failed runs with an explicit integer attempt_count of zero, an empty attempts list and no OCR, prompt, grading-input or response artifacts. Validate output/grading objects before indexing or settings checks; return an incomparable report on compatibility errors before derived hashes or metrics consume malformed history. | Recorded calls/prompts and uncertain counters must not bypass OCR validation, and historical outputs such as null must produce a report instead of an AttributeError. |
| `harness_v2/tests/test_p3_ocr_exemption.py, harness_v2/tests/test_p3_historical_ocr.py` | Cover prompt/call evidence, missing/null/invalid counters, one/both sides, current/history runs, malformed output/grading objects with supplied or saved datasets, and the CLI's private incomparable report. Remove downstream artifacts from the synthetic pre-OCR fixture; retain the real pre-OCR failure regression. Assert records remain unchanged and failed scores remain unscored. | Require affirmative exemption evidence while preserving genuine early failures, reference isolation, status validation and 0600/0700 reports without inference or score substitution. |

Validation: `python -m pytest -q harness_v2/tests math-answer-diagnosis/tests` — **692 passed in 22.74s** (using `/workspace/venv_clean/bin` on PATH; existing loopback mocks allowed with sandbox escalation). No git writes; math-answer-diagnosis sources unchanged.
