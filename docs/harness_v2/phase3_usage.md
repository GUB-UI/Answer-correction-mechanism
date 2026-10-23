# Phase 3 offline dataset and experiment workflow

Build the seven development inputs without inference:

```sh
python scripts/build_dev_dataset.py --out /tmp/harness-dev-v1
```

`--kit`, `--c1`, `--ocr-dir`, `--references`, `--dataset-id` and `--version` override the local defaults. The checked-in `scripts/dev_dataset_references.json` contains only the declared evaluation labels, not student answers. The builder copies those labels into a separate private `references.json`. No labels are inferred from predictions. Item labels exist only for the two specified problems; the runner/evaluator has no problem-specific score logic. Generated transcripts, rubrics, manifests and reference copies belong outside git.

`manifest.json` has `schema_version: 1`, `dataset_id`, `version`, `cases`, and a `references: {path, sha256}` descriptor. Each case has `case_id`, `problem_id`, `input_version`, `problem_text`, `rubric_version`, `rubric: {path, sha256}`, `grading_input: {path, sha256}`, and `ocr_metadata`. Paths are relative to the manifest; path escapes, duplicate cases, invalid rubrics/references and changed file hashes are rejected. `references.json` has the same dataset identity/version, `reference_scores` keyed by problem ID and optional `reference_items` keyed by problem/item ID. Loading returns inference cases separately from references.

Generated directories use `0700` and files use `0600`, including rewrites under a permissive umask. Input bytes retain CRLF and original mathematical text. CPU metadata includes engine, pinned model revision, float32 dtype, source and available stats timings.
