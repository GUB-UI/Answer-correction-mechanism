"""Check a grading result. A failed check never fills in a missing score."""

from __future__ import annotations

import re

from harness_v2.core.contracts import Check, GradingResult, ItemResult, RubricItem
from harness_v2.grading.sympy_equiv import mechanical_equivalent

_SPACE = re.compile(r"\s+")


def _collapse(text: str) -> str:
    return _SPACE.sub("", text)


def validate(
    rubric: list[RubricItem],
    items: list[ItemResult],
    total_score: float | None,
    answer_text: str,
    *,
    request_id: str = "",
    provider: str = "",
    model_id: str = "",
    raw_response: str = "",
) -> GradingResult:
    by_id = {item.item_id: item for item in items}
    ordered: list[ItemResult] = []
    checks: list[Check] = []
    missing: list[str] = []
    for rubric_item in rubric:
        found = by_id.get(rubric_item.item_id)
        if found is None:
            missing.append(rubric_item.item_id)
            ordered.append(
                ItemResult(
                    item_id=rubric_item.item_id,
                    max_points=rubric_item.max_points,
                    score=None,
                    review_reason="missing_from_model",
                )
            )
            continue
        ordered.append(found)
    extra = sorted(set(by_id) - {item.item_id for item in rubric})
    checks.append(Check("items_present", not missing, ",".join(missing)))
    checks.append(Check("no_extra_items", not extra, ",".join(extra)))

    range_bad: list[str] = []
    reason_bad: list[str] = []
    evidence_bad: list[str] = []
    math_bad: list[str] = []
    score_missing = list(missing)
    haystack = _collapse(answer_text)
    rubric_by_id = {item.item_id: item for item in rubric}
    for item in ordered:
        if item.score is None:
            if item.item_id not in score_missing:
                score_missing.append(item.item_id)
            continue
        maximum = item.max_points
        if item.score < 0 or item.score > maximum:
            range_bad.append(f"{item.item_id}:{item.score}/{maximum}")
        if not item.math_reason.strip():
            reason_bad.append(item.item_id)
        if not item.evidence_spans or any(not span.text.strip() for span in item.evidence_spans):
            evidence_bad.append(item.item_id)
        else:
            for span in item.evidence_spans:
                if _collapse(span.text) not in haystack:
                    evidence_bad.append(item.item_id)
                    break
        rubric_item = rubric_by_id.get(item.item_id)
        expression = rubric_item.acceptable_expression if rubric_item else ""
        if expression and item.evidence_spans:
            verdict = mechanical_equivalent(item.evidence_spans[0].text, expression)
            if verdict is False:
                math_bad.append(item.item_id)
    checks.append(Check("scores_present", not score_missing, ",".join(score_missing)))
    checks.append(Check("score_range", not range_bad, ",".join(range_bad)))
    checks.append(Check("reason_present", not reason_bad, ",".join(reason_bad)))
    checks.append(Check("evidence_traceable", not evidence_bad, ",".join(evidence_bad)))
    checks.append(Check("sympy_equivalence", not math_bad, ",".join(math_bad)))

    numeric = [item.score for item in ordered if isinstance(item.score, (int, float))]
    all_numeric = len(numeric) == len(rubric) and not score_missing
    if total_score is None:
        checks.append(Check("total_matches", False, "total_missing"))
    elif not all_numeric:
        checks.append(Check("total_matches", False, "scores_incomplete"))
    else:
        summed = float(sum(numeric))
        matches = abs(summed - float(total_score)) <= 1e-6
        checks.append(
            Check("total_matches", matches, "" if matches else f"{summed}!={total_score}")
        )

    hard_fail = bool(missing or score_missing)
    review_names = [
        check.name
        for check in checks
        if not check.ok and check.name not in {"items_present", "scores_present"}
    ]
    if hard_fail:
        status = "failed"
        reason = "missing_score:" + ",".join(score_missing)
    elif review_names:
        status = "review_required"
        reason = ",".join(review_names)
    else:
        status = "validated"
        reason = ""
    return GradingResult(
        status=status,  # type: ignore[arg-type]
        items=ordered,
        total_score=total_score,
        max_points=sum(item.max_points for item in rubric),
        failure_reason=reason,
        request_id=request_id,
        provider=provider,
        model_id=model_id,
        raw_response=raw_response,
        checks=checks,
    )
