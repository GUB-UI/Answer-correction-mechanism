from __future__ import annotations

from harness_v2.core.contracts import FailureCase, ImprovementProposal, ItemResult


def test_item_result_keeps_a_missing_score_as_none() -> None:
    item = ItemResult(item_id="R1", max_points=4, score=None, math_reason="")
    assert item.to_dict()["score"] is None


def test_failure_and_proposal_fields_round_trip() -> None:
    case = FailureCase(
        case_id="c",
        experiment_id="e",
        request_id="r",
        input_id="i",
        provider="p",
        model_id="m",
        rubric_version="rv",
        prompt_version="pv",
        failure_kind="missing_score",
        observed="R2 absent",
        expected_check="items_present",
    )
    proposal = ImprovementProposal(
        proposal_id="p1",
        source_case_ids=["c"],
        target="rubric",
        problem="missing item",
        recommended_action="review",
        severity="medium",
        pattern_count=1,
    )
    assert case.to_dict()["failure_kind"] == "missing_score"
    assert proposal.to_dict()["status"] == "proposed"
    assert proposal.to_dict()["pattern_count"] == 1
