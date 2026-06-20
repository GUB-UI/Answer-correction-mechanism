from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models import (
    DiagnosisResult,
    ERROR_CATEGORIES,
    OCRResult,
    ProblemRecord,
    utc_now_iso,
)


def test_problem_record_validation() -> None:
    problem = ProblemRecord(
        problem_id="prob_001",
        title="title",
        problem_text="text",
        correct_answer="answer",
        rubric="rubric",
        unit="unit",
        difficulty="basic",
        created_at=utc_now_iso(),
    )
    assert problem.problem_id == "prob_001"


def test_ocr_result_requires_required_fields() -> None:
    with pytest.raises(ValidationError):
        OCRResult.model_validate(
            {
                "ocr_id": "ocr_001",
                "answer_id": "ans_001",
            }
        )


def test_diagnosis_result_possible_ocr_issue() -> None:
    result = DiagnosisResult(
        diagnosis_id="diag_001",
        answer_id="ans_001",
        problem_id="prob_001",
        ocr_id="ocr_001",
        model_name="mock",
        prompt_type="basic_scoring",
        score=3.0,
        max_score=5.0,
        error_categories=["計算ミス"],
        error_locations=[],
        reasoning_summary="summary",
        student_feedback="feedback",
        teacher_review_notes="notes",
        possible_ocr_issue=True,
        confidence=0.5,
        created_at=utc_now_iso(),
    )
    assert result.possible_ocr_issue is True


def test_error_categories_contains_expected_items() -> None:
    assert "計算ミス" in ERROR_CATEGORIES
