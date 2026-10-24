# Design v0.1

Grading ends in one of three states: `validated`, `review_required`, or `failed`. A missing item score is left empty. It is not filled with 0 or with the item maximum.

## Metrics

Evaluation reports these measures. None of them is hardcoded for a problem id.

- MAE vs reference
- item agreement
- evidence accuracy
- wrong grading
- review detection
- success rate
- latency
- usage

`reference_score` is stored on a failure case for those measures. It is not sent to the model. Independent evaluation still removes prior AI scores and reference scores from the prompt.

## Development set

The development set is three problems, scored out of 10:

| Problem | Score |
| --- | --- |
| 1-A | 8/10 |
| 2-B | 4/10 |
| 3-A | 10/10 |

These totals are the reference for measurement. The harness does not special-case their formulas or their scores.

## One change type per cycle

An improvement cycle changes one component. `changed_components` holds that single name. `decision` stays `pending` until a person sets it. `evaluation_results`, `regressions`, and `teacher_reviews` stay empty until that evaluation exists. A proposal does not edit prompts, rubrics, or code by itself.

## FailureCase

`case_id`, `problem_id`, `input_version`, `model_version`, `harness_version`, `rubric_version`, `predicted_score`, `reference_score`, `rubric_results`, `evidence`, `errors`, `uncertainty`, `execution_log`.

## ImprovementProposal

`proposal_id`, `baseline_commit`, `candidate_commit`, `cause`, `changed_components`, `expected_effect`, `evaluation_results`, `regressions`, `teacher_reviews`, `decision`.
