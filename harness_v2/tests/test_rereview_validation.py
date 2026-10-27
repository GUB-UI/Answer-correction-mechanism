"""Remaining findings 5 and 6: reject malformed scoring contracts."""
import json

import pytest

from harness_v2.core.contracts import EvidenceSpan, ItemResult, RubricItem
from harness_v2.grading.rubric import rubric_from_data
from harness_v2.grading.scorer import ScoringParseError, items_from_completion
from harness_v2.grading.validator import validate


def completion(**changes):
    row = dict(item_id="a", score=3, evidence_spans=[{"text": "x"}], math_reason="shown")
    row.update(changes)
    return {"items": [row], "total_score": 3}


@pytest.mark.parametrize("field", ["max_points", "max_score"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), "NaN", True, None, "3"])
def test_invalid_returned_maximum_never_validates(field, value):
    rubric = [RubricItem("a", 3, "work")]
    items, total, *_ = items_from_completion(json.dumps(completion(**{field: value})), rubric)
    assert items[0].max_points == 3
    assert validate(rubric, items, total, "x").status != "validated"


def test_both_maximum_aliases_are_checked():
    rubric = [RubricItem("a", 3, "work")]
    items, total, *_ = items_from_completion(json.dumps(completion(max_points=3, max_score=float("nan"))), rubric)
    assert validate(rubric, items, total, "x").status != "validated"


@pytest.mark.parametrize("row", [None, 3, [], "garbage", {}, {"item_id": ""}, {"item_id": False}])
def test_malformed_response_rows_are_not_silently_skipped(row):
    data = completion()
    data["items"].append(row)
    with pytest.raises(ScoringParseError):
        items_from_completion(json.dumps(data), [RubricItem("a", 3, "work")])


@pytest.mark.parametrize("rows", [
    [{"item_id": "a", "max_points": 3}, {"item_id": "a", "max_points": 3}],
    *[[{"item_id": "a", "max_points": value}] for value in [float("nan"), float("inf"), -1, True, None]],
    [{"item_id": "", "max_points": 3}], [None],
])
def test_rubric_loader_rejects_invalid_allocations(rows):
    with pytest.raises(ValueError):
        rubric_from_data(rows)


@pytest.mark.parametrize("rubric", [
    [RubricItem("a", 3, ""), RubricItem("a", 3, "")],
    *[[RubricItem("a", value, "")] for value in [float("nan"), float("inf"), -1, True]],
    [RubricItem("", 3, "")],
])
def test_validator_defends_against_invalid_direct_rubrics(rubric):
    item = ItemResult("a", 3, 3, evidence_spans=[EvidenceSpan("x")], math_reason="shown")
    result = validate(rubric, [item], 3, "x")
    assert result.status == "failed"
    assert "rubric" in result.failure_reason
