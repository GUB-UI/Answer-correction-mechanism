from __future__ import annotations

from harness_v2.core.contracts import ItemResult, RubricItem
from harness_v2.grading.scorer import items_from_completion
from harness_v2.grading.sympy_equiv import mechanical_equivalent
from harness_v2.grading.validator import validate


def _rubric() -> list[RubricItem]:
    return [
        RubricItem("R1", 4, "process", "1+x"),
        RubricItem("R2", 2, "answer"),
    ]


def test_missing_item_is_not_given_a_score() -> None:
    raw = """```json
{"items":[{"item_id":"R1","score":4,"max_points":4,"evidence_spans":[{"text":"x+1"}],"math_reason":"shown"}],"total_score":4}
```"""
    items, total, _fence = items_from_completion(raw, _rubric())
    result = validate(_rubric(), items, total, "x+1")
    assert result.status == "failed"
    missing = next(item for item in result.items if item.item_id == "R2")
    assert missing.score is None
    assert all(item.score != 0 for item in result.items if item.item_id == "R2")


def test_range_total_reason_and_evidence() -> None:
    items = [
        ItemResult("R1", 4, 5, math_reason="", evidence_spans=[]),
        ItemResult("R2", 2, 2, math_reason="ok", evidence_spans=[]),
    ]
    from harness_v2.core.contracts import EvidenceSpan

    items[1].evidence_spans = [EvidenceSpan("not in the answer")]
    result = validate(_rubric(), items, 9, "x+1")
    assert result.status == "review_required"
    names = {check.name for check in result.checks if not check.ok}
    assert {"score_range", "total_matches", "reason_present", "evidence_traceable"} <= names
    assert items[0].score == 5


def test_sympy_equivalence_is_mechanical_only() -> None:
    assert mechanical_equivalent("x+1", "1+x") is True
    assert mechanical_equivalent("sqrt(x+1)", "sqrt(1+x)") is True
    assert mechanical_equivalent("x+1", "x+2") is False
    assert mechanical_equivalent("hello there", "1+x") is None
    items, total, _fence = items_from_completion(
        '{"items":[{"item_id":"R1","score":4,"max_points":4,"evidence_spans":[{"text":"x+1"}],"math_reason":"shown"},'
        '{"item_id":"R2","score":2,"max_points":2,"evidence_spans":[{"text":"x+1"}],"math_reason":"shown"}],'
        '"total_score":6}',
        _rubric(),
    )
    validated = validate(_rubric(), items, total, "x+1 is the work")
    assert validated.status == "validated"
    mismatch, mismatch_total, _fence = items_from_completion(
        '{"items":[{"item_id":"R1","score":4,"max_points":4,"evidence_spans":[{"text":"x+2"}],"math_reason":"shown"},'
        '{"item_id":"R2","score":2,"max_points":2,"evidence_spans":[{"text":"x+2"}],"math_reason":"shown"}],'
        '"total_score":6}',
        _rubric(),
    )
    reviewed = validate(_rubric(), mismatch, mismatch_total, "x+2")
    assert reviewed.status == "review_required"
    assert reviewed.items[0].score == 4
