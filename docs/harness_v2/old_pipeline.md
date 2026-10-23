# Old pipeline inputs and outputs

The running app is `math-answer-diagnosis`. The v2 harness does not import its UI or storage. The compare script imports only `src.grading.parse` so a saved completion can be read with the old one-fence parser.

## Inputs

- Config: `config.yaml` / `config.colab.yaml`, overridden by environment variables. Providers, model ids, ports, and prompt selection live here.
- Problem: `ProblemRecord` (`problem_text`, `correct_answer`, `rubric`, `score_allocation`, optional problem and model-answer images).
- Answer: `AnswerRecord` (image paths or `typed_text`, anonymized student id).
- OCR: page images go to Xiaomi-OCR-0 or GLM-OCR. Optional UniMERNet formula replacement. Raw OCR text and the text actually graded are both kept.
- Diagnosis prompt: `prompt_type` plus the problem, rubric, allocation, and OCR text. The model is asked for JSON.
- Optional teacher CSV imported as `TeacherGradeRecord`.

## Stages

1. Intake writes problems and answers to JSONL.
2. OCR writes `OCRResult` (`raw_text`, `used_text`, `uncertain_parts`, `ocr_suspect`).
3. Diagnosis writes `DiagnosisResult` (`score`, `max_score`, `question_scores`, error categories, feedback, `possible_ocr_issue`, `review_status`, `raw_output`).
4. `raw_output` may include `fence_stripped`, `grading_validation`, and `review_flags`. Review flags are recorded. They do not change teacher routing unless `REVIEW_FLAGS_ROUTE` is set.
5. Teacher compare writes discrepancies when a teacher score and a system score differ.

Batch mode (`python -m src batch`) runs OCR, then formula refinement, then diagnosis, one GPU-backed service at a time.

## What the compare script uses

The old side of `scripts/compare_old_vs_new.py` does not start that batch. It parses `legacy_completion` with the existing strict JSON parser and writes `result.json`. A case with no saved completion is recorded as `no_legacy_completion` and no model is called.
