"""Check a grading result. A failed check never fills in a missing score."""

from __future__ import annotations

import math
import re

from harness_v2.core.contracts import Check, GradingResult, ItemResult, RubricItem
from harness_v2.grading.sympy_equiv import mechanical_equivalent
from harness_v2.grading.rubric import rubric_error

_SPACE = re.compile(r"\s+")
UNCERTAINTY_REVIEW_AT = 0.5


_MATH = re.compile(r"\$\$(.*?)\$\$|\$(.*?)\$|\\\((.*?)\\\)|\\\[(.*?)\\\]", re.DOTALL)


def normalize_evidence(text: str) -> str:
    """Comparison view only; never rewrite stored OCR or cited spans."""
    def body(match: re.Match) -> str:
        return next(group for group in match.groups() if group is not None)
    return _SPACE.sub("", _MATH.sub(body, text))


def evidence_traceable(text: str, answer_text: str) -> bool:
    needle = normalize_evidence(text)
    return bool(needle) and needle in normalize_evidence(answer_text)


def _finite(value: float | None) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _failed(
    rubric: list[RubricItem],
    items: list[ItemResult],
    total_score: float | None,
    reason: str,
    checks: list[Check],
    **meta: str,
) -> GradingResult:
    return GradingResult(
        status="failed",
        items=items,
        total_score=total_score,
        max_points=None if rubric_error(rubric) else sum(item.max_points for item in rubric),
        failure_reason=reason,
        checks=checks,
        request_id=meta.get("request_id", ""),
        provider=meta.get("provider", ""),
        model_id=meta.get("model_id", ""),
        raw_response=meta.get("raw_response", ""),
    )


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
    meta = {
        "request_id": request_id,
        "provider": provider,
        "model_id": model_id,
        "raw_response": raw_response,
    }
    invalid_rubric = rubric_error(rubric)
    if invalid_rubric:
        return _failed(
            rubric,
            [],
            total_score,
            invalid_rubric,
            [Check("rubric_valid", False, invalid_rubric)],
            **meta,
        )
    seen: dict[str, int] = {}
    for item in items:
        seen[item.item_id] = seen.get(item.item_id, 0) + 1
    duplicates = sorted(item_id for item_id, count in seen.items() if count > 1)
    if duplicates:
        return _failed(
            rubric,
            items,
            total_score,
            "duplicate_item_ids:" + ",".join(duplicates),
            [Check("unique_item_ids", False, ",".join(duplicates))],
            **meta,
        )
    by_id = {item.item_id: item for item in items}
    ordered: list[ItemResult] = []
    checks: list[Check] = [Check("rubric_nonempty", True, ""), Check("unique_item_ids", True, "")]
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
    review_bad: list[str] = []
    non_numeric: list[str] = []
    max_mismatch: list[str] = []
    score_missing = list(missing)
    rubric_by_id = {item.item_id: item for item in rubric}
    for item in ordered:
        rubric_item = rubric_by_id.get(item.item_id)
        maximum = rubric_item.max_points if rubric_item is not None else item.max_points
        if "item_max_mismatch" in item.review_reason:
            max_mismatch.append(item.item_id)
        if item.score is not None and not _finite(item.score):
            non_numeric.append(item.item_id)
            if item.item_id not in score_missing:
                score_missing.append(item.item_id)
            continue
        if item.score is None:
            if item.item_id not in score_missing:
                score_missing.append(item.item_id)
            continue
        if item.score < 0 or item.score > maximum:
            range_bad.append(f"{item.item_id}:{item.score}/{maximum}")
        if not item.math_reason.strip():
            reason_bad.append(item.item_id)
        absence_valid = (
            item.evidence_absent is True and item.score == 0
            and not item.evidence_spans and isinstance(item.absent_reason, str)
            and bool(item.absent_reason.strip())
        )
        if not isinstance(item.evidence_absent, bool) or (item.evidence_absent is True and not absence_valid):
            evidence_bad.append(item.item_id)
        elif absence_valid:
            pass
        elif not item.evidence_spans or any(not span.text.strip() for span in item.evidence_spans):
            evidence_bad.append(item.item_id)
        else:
            for span in item.evidence_spans:
                if not evidence_traceable(span.text, answer_text):
                    evidence_bad.append(item.item_id)
                    break
        expression = rubric_item.acceptable_expression if rubric_item else ""
        if expression and item.evidence_spans:
            verdict = mechanical_equivalent(item.evidence_spans[0].text, expression)
            if verdict is False:
                math_bad.append(item.item_id)
        reason = item.review_reason.strip()
        if reason and reason != "missing_from_model":
            review_bad.append(f"{item.item_id}:{reason}")
        if _finite(item.uncertainty) and float(item.uncertainty or 0) >= UNCERTAINTY_REVIEW_AT:
            review_bad.append(f"{item.item_id}:uncertainty")
    checks.append(Check("scores_present", not score_missing, ",".join(score_missing)))
    checks.append(Check("score_numeric", not non_numeric, ",".join(non_numeric)))
    checks.append(Check("score_range", not range_bad, ",".join(range_bad)))
    checks.append(Check("item_max_matches_rubric", not max_mismatch, ",".join(max_mismatch)))
    checks.append(Check("reason_present", not reason_bad, ",".join(reason_bad)))
    checks.append(Check("evidence_traceable", not evidence_bad, ",".join(evidence_bad)))
    checks.append(Check("sympy_equivalence", not math_bad, ",".join(math_bad)))
    checks.append(Check("review_flags", not review_bad, ",".join(review_bad)))

    numeric = [item.score for item in ordered if _finite(item.score)]
    all_numeric = len(numeric) == len(rubric) and not score_missing
    if total_score is None or not _finite(total_score):
        checks.append(Check("total_matches", False, "total_missing"))
    elif not all_numeric:
        checks.append(Check("total_matches", False, "scores_incomplete"))
    else:
        summed = float(sum(numeric))
        matches = abs(summed - float(total_score)) <= 1e-6
        checks.append(
            Check("total_matches", matches, "" if matches else f"{summed}!={total_score}")
        )

    hard_fail = bool(missing or score_missing or non_numeric)
    review_names = [
        check.name
        for check in checks
        if not check.ok and check.name not in {"items_present", "scores_present", "score_numeric", "rubric_nonempty"}
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
