from __future__ import annotations

import json

import pytest

from src.adapters.base import DiagnosisRunContext
from src.adapters.diagnosis_helpers import build_diagnosis_result, parse_diagnosis_json
from src.grading.parse import (
    ExpectedItem,
    GradingOutputError,
    parse_grading_text,
    strip_one_outer_fence,
    validate_grading,
)
from src.models import ProblemRecord, utc_now_iso


def _rubric_text(**overrides: object) -> str:
    payload = {
        "question_id": "1-A",
        "rubric_scores": [
            {
                "item_id": "R1",
                "score": 0,
                "max_score": 4,
                "evidence": "x=1",
                "reason": "過程が無い",
            },
            {
                "item_id": "R2",
                "score": 4,
                "max_score": 4,
                "evidence": "y=2",
                "reason": "形がある",
            },
            {
                "item_id": "R3",
                "score": 0,
                "max_score": 2,
                "evidence": "z=3",
                "reason": "最終答が違う",
            },
        ],
        "score": 4,
        "max_score": 10,
        "needs_review": False,
        "grading_uncertain": False,
        "ocr_uncertain": False,
        "review_reason": "",
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


def test_plain_json_parses_without_fence_flag() -> None:
    parsed = parse_grading_text(_rubric_text())
    assert parsed.fence_stripped is False
    assert parsed.data["score"] == 4
    assert validate_grading(
        parsed.data,
        [ExpectedItem("R1", 4), ExpectedItem("R2", 4), ExpectedItem("R3", 2)],
    ).ok


def test_one_outer_fence_is_removed_and_content_is_unchanged() -> None:
    body = _rubric_text()
    wrapped = "```json\n" + body + "\n```\n"
    assert strip_one_outer_fence(wrapped) == body
    parsed = parse_grading_text(wrapped)
    assert parsed.fence_stripped is True
    assert parsed.data == json.loads(body)
    stored = parse_diagnosis_json(wrapped)
    assert stored["_fence_stripped"] is True
    assert stored["_grading_validation"] == []


def test_fence_inside_a_valid_json_string_is_not_stripped() -> None:
    text = json.dumps({"score": 1, "note": "```json\nnot a fence\n```"}, ensure_ascii=False)
    parsed = parse_grading_text(text)
    assert parsed.fence_stripped is False
    assert "```json" in parsed.data["note"]


def test_prose_around_json_is_an_error() -> None:
    with pytest.raises(GradingOutputError, match="json_parse_failed") as caught:
        parse_grading_text('結果は次のとおりです\n{"score": 1}\n')
    assert caught.value.fence_stripped is False


def test_broken_fence_is_an_error_and_not_a_result() -> None:
    with pytest.raises(GradingOutputError, match="json_parse_failed_after_fence") as caught:
        parse_grading_text("```json\n{not json}\n```")
    assert caught.value.fence_stripped is True


def test_only_one_fence_is_removed() -> None:
    inner = "```json\n" + _rubric_text() + "\n```"
    wrapped = "```\n" + inner + "\n```"
    with pytest.raises(GradingOutputError, match="json_parse_failed_after_fence"):
        parse_grading_text(wrapped)


def test_validation_checks_items_range_and_sum() -> None:
    payload = json.loads(_rubric_text())
    payload["rubric_scores"][0]["score"] = 9
    payload["score"] = 4
    errors = validate_grading(
        payload,
        [ExpectedItem("R1", 4), ExpectedItem("R2", 4), ExpectedItem("R3", 2)],
    ).errors
    assert any(item.startswith("score_out_of_range:R1") for item in errors)
    assert any(item.startswith("score_sum_mismatch:") for item in errors)

    payload = json.loads(_rubric_text())
    payload["rubric_scores"] = payload["rubric_scores"][:2]
    payload["score"] = 4
    errors = validate_grading(
        payload,
        [ExpectedItem("R1", 4), ExpectedItem("R2", 4), ExpectedItem("R3", 2)],
    ).errors
    assert any(item.startswith("items_mismatch:") for item in errors)


def test_fence_strip_does_not_set_teacher_review() -> None:
    problem = ProblemRecord(
        problem_id="prob",
        title="t",
        created_at=utc_now_iso(),
        score_allocation=[
            {"question": "R1", "max_score": 4},
            {"question": "R2", "max_score": 4},
            {"question": "R3", "max_score": 2},
        ],
    )
    context = DiagnosisRunContext(
        answer_id="ans",
        problem_id="prob",
        ocr_id="ocr",
        problem=problem,
        used_text="x=1",
        model_name="mock",
        prompt_type="basic_scoring",
    )
    raw = parse_diagnosis_json("```json\n" + _rubric_text() + "\n```")
    diagnosis = build_diagnosis_result(context, raw, "diag")
    assert diagnosis.raw_output["fence_stripped"] is True
    assert diagnosis.raw_output["grading_validation"] == []
    assert diagnosis.review_status == "validated"
    assert diagnosis.possible_ocr_issue is False
