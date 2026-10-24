from __future__ import annotations

import json

import pytest

from src.adapters.base import DiagnosisRunContext
from src.adapters.diagnosis_helpers import build_diagnosis_result, parse_diagnosis_json
from src.grading.parse import ExpectedItem, GradingOutputError, parse_grading_text, validate_grading
from src.grading.review_flags import evaluate_review_flags, normalize_for_match
from src.models import ProblemRecord, utc_now_iso


def _payload() -> dict:
    return {
        "question_id": "1-A",
        "rubric_scores": [
            {
                "item_id": "R1",
                "score": 0,
                "max_score": 4,
                "evidence": "2行目: $x=1$",
                "reason": "過程がないため0点。",
            },
            {
                "item_id": "R2",
                "score": 4,
                "max_score": 4,
                "evidence": "$y=2$",
                "reason": "形がある",
            },
        ],
        "score": 4,
        "max_score": 8,
        "needs_review": False,
        "grading_uncertain": False,
        "ocr_uncertain": False,
    }


def _expected() -> list[ExpectedItem]:
    return [ExpectedItem("R1", 4), ExpectedItem("R2", 4)]


def _context(used: str, allocation: list[dict] | None = None) -> DiagnosisRunContext:
    problem = ProblemRecord(
        problem_id="prob",
        title="t",
        created_at=utc_now_iso(),
        score_allocation=allocation
        or [
            {"question": "R1", "max_score": 4},
            {"question": "R2", "max_score": 4},
        ],
    )
    return DiagnosisRunContext(
        answer_id="ans",
        problem_id="prob",
        ocr_id="ocr",
        problem=problem,
        used_text=used,
        model_name="mock",
        prompt_type="basic_scoring",
    )


def test_fence_alone_does_not_need_review() -> None:
    text = "```json\n" + json.dumps(_payload(), ensure_ascii=False) + "\n```"
    parsed = parse_grading_text(text)
    flags = evaluate_review_flags(
        parsed.data,
        answer_text="$x=1$\n$y=2$",
        validation_errors=validate_grading(parsed.data, _expected()).errors,
        fence_stripped=True,
    )
    assert parsed.fence_stripped is True
    assert flags.needs_review is False
    assert flags.fence_stripped is True
    assert flags.is_grading_result is True
    assert "json_fence" not in "".join(flags.review_reasons)


def test_parse_failure_is_not_a_grading_result() -> None:
    with pytest.raises(GradingOutputError) as caught:
        parse_grading_text("not json")
    flags = evaluate_review_flags(
        None,
        answer_text="x=1",
        parse_error=str(caught.value),
        fence_stripped=caught.value.fence_stripped,
    )
    assert flags.is_grading_result is False
    assert flags.needs_review is True
    assert flags.grading_uncertain is True


def test_score_reason_mismatch_is_grading_uncertain() -> None:
    payload = _payload()
    payload["rubric_scores"][0]["score"] = 1
    payload["rubric_scores"][0]["reason"] = "基準を満たすので満点とする"
    payload["score"] = 5
    flags = evaluate_review_flags(
        payload,
        answer_text="$x=1$ $y=2$",
        validation_errors=validate_grading(payload, _expected()).errors,
    )
    assert flags.needs_review is True
    assert flags.grading_uncertain is True
    assert "score_reason_mismatch" in flags.review_reasons
    assert flags.ocr_uncertain is False


def test_missing_items_are_recorded() -> None:
    payload = _payload()
    payload["rubric_scores"] = payload["rubric_scores"][:1]
    payload["score"] = 0
    errors = validate_grading(
        payload,
        [ExpectedItem("R1", 4), ExpectedItem("R2", 4), ExpectedItem("R3", 2)],
    ).errors
    flags = evaluate_review_flags(payload, answer_text="x=1", validation_errors=errors)
    assert any(item.startswith("items_mismatch") for item in flags.review_reasons)
    assert flags.grading_uncertain is True


def test_evidence_tolerates_latex_space_and_ellipsis() -> None:
    payload = _payload()
    payload["rubric_scores"][0]["evidence"] = r"2行目: \(\sqrt{9}-\sqrt{2}\) ... その後"
    payload["rubric_scores"][1]["evidence"] = "記載なし"
    answer = "式は $\\sqrt{ 9 } - \\sqrt{ 2 }$ である"
    assert normalize_for_match(r"\sqrt{9}-\sqrt{2}") in normalize_for_match(answer)
    flags = evaluate_review_flags(payload, answer_text=answer, validation_errors=[])
    assert flags.evidence_missing == []
    assert "evidence_not_in_input" not in "".join(flags.review_reasons)


def test_paraphrase_without_a_formula_is_not_evidence_failure() -> None:
    payload = _payload()
    payload["rubric_scores"][1]["evidence"] = "二重根号を外す意図が読み取れる"
    flags = evaluate_review_flags(
        payload,
        answer_text="x=1\ny=2",
        validation_errors=[],
    )
    assert not any(item.startswith("R2:") for item in flags.evidence_missing)


def test_formula_corrected_away_from_the_input_is_missing() -> None:
    payload = _payload()
    payload["rubric_scores"][0]["evidence"] = "x=-1"
    payload["rubric_scores"][1]["evidence"] = "記載なし"
    flags = evaluate_review_flags(payload, answer_text="x=17''", validation_errors=[])
    assert flags.evidence_missing
    assert any(item.startswith("evidence_not_in_input") for item in flags.review_reasons)


def test_unreadable_formula_ignores_markdown_and_bare_pi() -> None:
    plain = evaluate_review_flags(
        _payload(),
        answer_text="```markdown\n\\pi\n```\n$x=1$\n$y=2$",
        validation_errors=[],
    )
    assert plain.ocr_reasons == []
    broken = evaluate_review_flags(
        _payload(),
        answer_text="$= √‾‾(11-2√2)$\n$x=1$\n$y=2$",
        validation_errors=[],
    )
    assert broken.ocr_uncertain is True
    assert any("√‾" in item or "解得" in item for item in broken.ocr_reasons)


def test_routing_is_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REVIEW_FLAGS_ROUTE", raising=False)
    payload = _payload()
    payload["score"] = 0
    payload["rubric_scores"][1]["score"] = 0
    payload["rubric_scores"][1]["reason"] = "基準を満たすので満点とする"
    raw = parse_diagnosis_json(json.dumps(payload, ensure_ascii=False))
    diagnosis = build_diagnosis_result(_context("$x=1$\n$y=2$"), raw, "diag")
    assert diagnosis.raw_output["review_flags"]["needs_review"] is True
    assert diagnosis.raw_output["review_flags"]["routing_applied"] is False
    assert diagnosis.review_status == "validated"
    assert diagnosis.score == 0

    monkeypatch.setenv("REVIEW_FLAGS_ROUTE", "true")
    routed = build_diagnosis_result(_context("$x=1$\n$y=2$"), raw, "diag2")
    assert routed.review_status == "review_required"
    assert routed.score == 0
    assert routed.possible_ocr_issue is False
