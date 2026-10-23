from __future__ import annotations

from pathlib import Path

from harness_v2.core.contracts import (
    FAILURE_CASE_FIELDS,
    IMPROVEMENT_PROPOSAL_FIELDS,
    FailureCase,
    ImprovementProposal,
    ItemResult,
)

_DESIGN = Path(__file__).resolve().parents[2] / "docs" / "harness_v2" / "design_v0.1.md"


def test_item_result_keeps_a_missing_score_as_none() -> None:
    item = ItemResult(item_id="R1", max_points=4, score=None, math_reason="")
    assert item.to_dict()["score"] is None


def test_failure_and_proposal_fields_match_design_v0_1() -> None:
    case = FailureCase(
        case_id="c",
        problem_id="1-A",
        input_version="in-1",
        model_version="gemma-4-31b-it",
        harness_version="0.1.0",
        rubric_version="rv",
        predicted_score=4,
        reference_score=8,
        rubric_results=[{"item_id": "R1", "score": 4}],
        evidence=[{"text": "x+1"}],
        errors=["score_range"],
        uncertainty=0.2,
        execution_log=[{"step": "validate"}],
    )
    proposal = ImprovementProposal(
        proposal_id="p1",
        baseline_commit="abc",
        candidate_commit="",
        cause="score_range",
        changed_components=["rubric"],
        expected_effect="one rubric edit",
        evaluation_results={},
        regressions=[],
        teacher_reviews=[],
        decision="pending",
    )
    assert tuple(case.to_dict()) == FAILURE_CASE_FIELDS
    assert tuple(proposal.to_dict()) == IMPROVEMENT_PROPOSAL_FIELDS
    assert len(proposal.changed_components) == 1
    assert "status" not in proposal.to_dict()
    assert "failure_kind" not in case.to_dict()


def test_design_summary_records_metrics_and_dev_set() -> None:
    text = _DESIGN.read_text(encoding="utf-8")
    for status in ("validated", "review_required", "failed"):
        assert status in text
    for metric in (
        "MAE vs reference",
        "item agreement",
        "evidence accuracy",
        "wrong grading",
        "review detection",
        "success rate",
        "latency",
        "usage",
    ):
        assert metric in text
    assert "1-A" in text and "8/10" in text
    assert "2-B" in text and "4/10" in text
    assert "3-A" in text and "10/10" in text
    assert "one change type" in text.lower() or "One change type" in text
    for name in FAILURE_CASE_FIELDS:
        assert name in text
    for name in IMPROVEMENT_PROPOSAL_FIELDS:
        assert name in text
