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


def test_rubric_max_rejects_model_max_and_non_numeric() -> None:
    raw = (
        '{"items":[{"item_id":"R1","score":20,"max_points":20,"evidence_spans":[{"text":"x+1"}],"math_reason":"shown"},'
        '{"item_id":"R2","score":2,"max_points":2,"evidence_spans":[{"text":"x+1"}],"math_reason":"shown"}],'
        '"total_score":22}'
    )
    items, total, _fence = items_from_completion(raw, _rubric())
    assert items[0].max_points == 4
    leaked = validate(_rubric(), items, total, "x+1")
    assert leaked.status == "review_required"
    assert leaked.status != "validated"
    claimed = (
        '{"items":[{"item_id":"R1","score":4,"max_points":20,"evidence_spans":[{"text":"x+1"}],"math_reason":"shown"},'
        '{"item_id":"R2","score":2,"max_points":2,"evidence_spans":[{"text":"x+1"}],"math_reason":"shown"}],'
        '"total_score":6}'
    )
    mismatch_items, mismatch_total, _fence = items_from_completion(claimed, _rubric())
    mismatch = validate(_rubric(), mismatch_items, mismatch_total, "x+1 is the work")
    assert mismatch.status == "review_required"
    assert "item_max_mismatch" in mismatch_items[0].review_reason
    nan_items, nan_total, _fence = items_from_completion(
        '{"items":[{"item_id":"R1","score":"NaN","max_points":4,"evidence_spans":[{"text":"x+1"}],"math_reason":"shown"},'
        '{"item_id":"R2","score":2,"max_points":2,"evidence_spans":[{"text":"x+1"}],"math_reason":"shown"}],'
        '"total_score":2}',
        _rubric(),
    )
    assert nan_items[0].score is None
    assert validate(_rubric(), nan_items, nan_total, "x+1").status == "failed"
    infinite = [
        ItemResult("R1", 4, float("inf"), math_reason="shown"),
        ItemResult("R2", 2, 2, math_reason="shown"),
    ]
    assert validate(_rubric(), infinite, 2, "x+1").status == "failed"


def test_duplicate_ids_and_empty_rubric_fail() -> None:
    raw = (
        '{"items":[{"item_id":"R1","score":4,"max_points":4,"evidence_spans":[{"text":"x+1"}],"math_reason":"shown"},'
        '{"item_id":"R1","score":4,"max_points":4,"evidence_spans":[{"text":"x+1"}],"math_reason":"shown"}],'
        '"total_score":8}'
    )
    items, total, _fence = items_from_completion(raw, _rubric())
    duplicate = validate(_rubric(), items, total, "x+1")
    assert duplicate.status == "failed"
    assert "duplicate_item_ids" in duplicate.failure_reason
    empty = validate([], [], 0, "x+1")
    assert empty.status == "failed"
    assert empty.failure_reason == "empty_rubric"
    assert empty.total_score == 0


def test_review_reason_or_high_uncertainty_requires_review() -> None:
    def row(uncertainty: float, review_reason: str) -> str:
        return (
            '{"items":[{"item_id":"R1","score":4,"max_points":4,"evidence_spans":[{"text":"x+1"}],'
            f'"math_reason":"shown","uncertainty":{uncertainty},"review_reason":"{review_reason}"}},'
            '{"item_id":"R2","score":2,"max_points":2,"evidence_spans":[{"text":"x+1"}],'
            '"math_reason":"shown","uncertainty":0.1,"review_reason":""}],"total_score":6}'
        )

    unclear, unclear_total, _fence = items_from_completion(row(0.1, "cannot determine reading"), _rubric())
    reviewed = validate(_rubric(), unclear, unclear_total, "x+1 is the work")
    assert reviewed.status == "review_required"
    high, high_total, _fence = items_from_completion(row(0.5, ""), _rubric())
    assert validate(_rubric(), high, high_total, "x+1 is the work").status == "review_required"
    low, low_total, _fence = items_from_completion(row(0.49, ""), _rubric())
    assert validate(_rubric(), low, low_total, "x+1 is the work").status == "validated"


def test_prompt_lists_the_validator_schema() -> None:
    from harness_v2.grading.scorer import build_prompt

    text = build_prompt("problem", "x+1", _rubric())
    for token in (
        "item_id",
        "score",
        "evidence_spans",
        "math_reason",
        "uncertainty",
        "review_reason",
        "total_score",
        "null",
        "max_points",
    ):
        assert token in text
