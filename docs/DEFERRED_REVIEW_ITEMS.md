# Deferred review items

Recorded from Sol re-reviews. These are not fixed.

## Sol r5 — PR #9 minor 2

- File: `math-answer-diagnosis/src/batch_pipeline.py` around line 287
- Incomplete saved grading metadata is rejected only by falling through the reuse filter. Regrading happens, but neither the batch item nor the new result records a reason (`reuse_unavailable_reason` stays `None`).
- Repro: save a diagnosis whose `grading_metadata` is `null` or `{}`, then rerun the batch with `skip_existing_diagnosis`. A new diagnosis is created and the model is called again, while `item.reuse_unavailable_reason` and the new result's `reuse_unavailable_reason` are both `None`. The missing reason should be `existing_grading_metadata_incomplete`, distinct from whether the new result itself can be reused later.
