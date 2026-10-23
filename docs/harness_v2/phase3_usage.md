# Phase 3 offline dataset and experiment workflow

Build the seven development inputs without inference:

```sh
python scripts/build_dev_dataset.py --out /tmp/harness-dev-v1
```

`--kit`, `--c1`, `--ocr-dir`, `--references`, `--dataset-id` and `--version` override the local defaults. The checked-in `scripts/dev_dataset_references.json` contains only the declared evaluation labels, not student answers. The builder copies those labels into a separate private `references.json`. No labels are inferred from predictions. Item labels exist only for the two specified problems; the runner/evaluator has no problem-specific score logic. Generated transcripts, rubrics, manifests and reference copies belong outside git.

`manifest.json` has `schema_version: 1`, `dataset_id`, `version`, `cases`, and a `references: {path, sha256}` descriptor. Each case has `case_id`, `problem_id`, `input_version`, `problem_text`, `rubric_version`, `rubric: {path, sha256}`, `grading_input: {path, sha256}`, and `ocr_metadata`. Paths are relative to the manifest; path escapes, duplicate cases, invalid rubrics/references and changed file hashes are rejected. `references.json` has the same dataset identity/version, `reference_scores` keyed by problem ID and optional `reference_items` keyed by problem/item ID. Loading returns inference cases separately from references.

Generated directories use `0700` and files use `0600`, including rewrites under a permissive umask. Input bytes retain CRLF and original mathematical text. CPU metadata includes engine, pinned model revision, float32 dtype, source and available stats timings.

Run repeated experiments:

```sh
python -m harness_v2.run --config /tmp/baseline.json \
  --dataset /tmp/harness-dev-v1/manifest.json --repeats 3 \
  --delay 5 --out /tmp/harness-baseline
```

Example configuration (Gemma calls require a locally configured AI Studio key; this implementation was tested offline):

```json
{
  "name": "baseline",
  "provider": "gemma",
  "model": "gemma-4-31b-it",
  "generation_settings": {"temperature": 0, "seed": 7},
  "prompt_version": "harness_v2_default",
  "rubric_version": "v1.1"
}
```

Supported generation settings: `temperature`, `seed`, `max_output_tokens`, `top_p`, `top_k`. Unknown settings are rejected rather than silently ignored. `prompt_version` records the version of the current built-in prompt; change the prompt implementation/version together when testing a candidate. Configured rubric versions must match the dataset. For offline tests use `provider: "fixture"`, `model: "fixture"`, and `fixture_completion` (saved JSON string/object), or `fixture_completions` keyed by case ID; a list selects a response per repetition.

IDs hash dataset/case/input/rubric/config/repetition identities. Reusing an output directory with another identity is rejected. Completed `validated` and `review_required` runs are skipped; failed attempts can be retried with the same request ID and new immutable artifact paths. SQLite holds all attempt records; `results.json` includes latest logical results plus full attempt history. `experiment.json`, atomic `state.json`, and per-run `failure_cases/` are private. On 429 the failed row and checkpoint are saved and execution stops immediately (exit 3); other recorded failures produce exit 1. `--delay` applies between logical requests; the provider separately handles bounded transient retries. No references reach the provider factory, prompt or independent-review payload.

Compare a baseline with a candidate:

```sh
python -m harness_v2.evaluation.compare --baseline /tmp/harness-baseline \
  --candidate /tmp/harness-candidate --dataset /tmp/harness-dev-v1/manifest.json \
  --out /tmp/harness-comparison.json
```

Inputs can be run directories or their `results.json`. The optional `--dataset` overrides the saved manifest location; it must verify and match the stored manifest. The comparison refuses different requested/returned models, providers, problem/case/repetition sets, input hashes, OCR provenance, rubric hashes/versions, reference-file identities or effective generation settings. Prompt versions may differ to measure a prompt candidate. Missing results are reported as incomparable, with explicit reasons (exit 2). Reports include both sides' metrics and candidate-minus-baseline deltas. Comparison verdicts are separate from the three grading states.

The full comparison reports MAE (available finite totals), item agreement (all expected item labels, including unscored outcomes in the denominator), cited-span evidence accuracy, wrong-run counts, unjust additions/deductions, review recall and false-flag rate, success rate, total and per-attempt latency distributions, and known token usage/request counts including retries and earlier failed attempts on resume. Counts and coverage accompany ratios; no cited spans or no applicable labels yields `null`, never an invented perfect rate. Missing scores remain unscored and are never interpreted as zero. Review detection uses `review_required` or explicit item flags/uncertainty. Unscored failures are excluded from correct-grade/false-flag counts.

Wrong additions/deductions use referenced items when available; otherwise they compare totals. Thus compensating item errors count even when the total agrees. Unknown item labels are never inferred. Evidence matching uses the actual blinded grading text and the same comparison-only normalization as validation. Explicit absence at zero contributes no cited span. `latency.total_ms` describes latest logical runs, while `latency.per_attempt_ms` and usage include stored earlier attempts. Token totals are known reported tokens only; unknown-token run/attempt counts expose unavailable usage on errors.

Verdicts are `improved`, `not_improved`, `regressed`, `incomparable`. Any new wrong case/repetition/item unit or lower evidence accuracy is a severe regression and forces `regressed`, even if aggregate MAE improves. Otherwise at least one quality gain and no quality loss yields `improved`; no gain yields `not_improved`. Quality covers errors, reference coverage, evidence, success and review detection. Latency and token deltas are reported as tradeoffs and do not alone establish improved grading. Grading states and predictions are unchanged by evaluation.
