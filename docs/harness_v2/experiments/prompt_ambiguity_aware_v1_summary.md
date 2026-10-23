# Experiment summary: `harness_v2_default` vs `harness_v2_ambiguity_aware_v1` (candidate)

Status: **candidate only. Not for default adoption until a human reviews these results.**
The default prompt is unchanged (hash-pinned). Privacy: aggregate numbers only — no images,
no answer text, no student identifiers, no teacher reference scores.

## Conditions (identical except the prompt)
| | old | new |
|---|---|---|
| prompt_version | harness_v2_default | harness_v2_ambiguity_aware_v1 |
| implementation commit | a608f6d | f09c6c5 (adds only prompt registration) |
| model / settings | gemma-4-31b-it, temperature 0 (same config apart from prompt_version/name) | same |
| dataset | harness-v2-dev v1, 7 inputs × 3 repetitions (byte-identical manifest) | same |
| OCR inputs | same stored transcriptions/OCR texts (manual, xiaomi CPU fp32) | same |

## Per-case results (model totals, 3 repetitions each)
| case | old status / total | new status / total | new items with uncertainty ≥ 0.5 |
|---|---|---|---|
| 1-A manual C0 | validated ×3 / 8, 8, 8 | validated ×3 / 8, 8, 8 | 0 |
| 1-A manual C1 | validated ×3 / 8, 8, 8 | validated ×3 / 8, 8, 8 | 0 |
| 1-A xiaomi OCR | validated ×3 / 8, 8, 8 | validated ×3 / 8, 8, 8 | 0 |
| 2-B manual | validated ×3 / 4, 4, 4 | validated ×3 / 4, 4, 4 | 0 |
| 2-B xiaomi OCR | validated ×3 / 4, 4, 4 | validated ×3 / 4, 4, 4 | 0 |
| 3-A manual | validated ×3 / 10, 10, 10 | validated ×3 / 10, 10, 10 | 0 |
| 3-A xiaomi OCR | review_required ×3 / 8, 8, 8 | review_required ×3 / 8, 8, 8 | 3 (R3–R5) |

Evidence: traceable evidence spans 99/105 (0.943) old → 108/111 (0.973) new.
Item scores and totals are identical in all 21 paired runs; agreement with the teacher
reference is unchanged (the same 3 item disagreements, all on 3-A xiaomi R5, caused by an OCR
misreading).

## Review flags
- Unnecessary review flags (review_required on runs whose grading agrees with the teacher):
  old 0/18, new 0/18 → no change.
- Label-free challenge set (Phase 5; formatting/spacing/notation variants of the dev answers,
  new prompt only): 0 flags in 27 scored variant runs, item agreement with the unmodified
  original 1.0. The old prompt was not run on this set, so it gives no old-vs-new comparison.

## Ambiguous-answer detection (defined honestly)
No pre-existing labelled “ambiguous answer” set exists. Operational definition used here:
an input is *should-be-flagged ambiguous* if its grading input is a degraded rendering of the
answer (OCR text differing from the manual transcription in a way that changes a gradable
step). In the dev set exactly one input meets it: 3-A xiaomi OCR.
- old: flagged 3/3 (review_required, reason: evidence mismatch only)
- new: flagged 3/3 (review_required, and the item-level review_reason names the garbled lines)
- detection rate 1.0 → 1.0; the change is in the explanation, not the rate.

## Limits
- One positive case, 7 inputs, 3 repetitions, development data used during design: this is a
  pilot, not evidence of generalization. A larger, independently labelled ambiguity set is
  required before any adoption decision.
- The comparator verdict “improved” rests only on evidence traceability; grading accuracy did
  not change.
- The OCR misreading itself is not fixed by any prompt; it must be addressed on the recognition side.
