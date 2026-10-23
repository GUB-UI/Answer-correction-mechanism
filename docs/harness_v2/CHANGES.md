# Harness v2 changes

The existing app under `math-answer-diagnosis/` is unchanged. PR #9's branch is unchanged. vLLM guided JSON is not part of this harness.

FailureCase and ImprovementProposal use the field set in `harness_v2/core/contracts.py`. A separate design file was not in the repository, so that module is the v0.1 contract.

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
