from __future__ import annotations

from typing import Any

from src.adapters.base import DiagnosisRunContext
from src.analysis import normalize_error_categories
from src.models import DiagnosisResult, utc_now_iso
from src.utils import extract_json_object


def build_diagnosis_result(
    context: DiagnosisRunContext,
    raw_output: dict[str, Any],
    diagnosis_id: str,
    created_at: str | None = None,
) -> DiagnosisResult:
    return DiagnosisResult(
        diagnosis_id=diagnosis_id,
        answer_id=context.answer_id,
        problem_id=context.problem_id,
        ocr_id=context.ocr_id,
        model_name=context.model_name,
        prompt_type=context.prompt_type,
        score=float(raw_output.get("score", 0)),
        max_score=float(raw_output.get("max_score", 0)),
        error_categories=normalize_error_categories(
            list(raw_output.get("error_categories", [])),
        ),
        error_locations=list(raw_output.get("error_locations", [])),
        reasoning_summary=str(raw_output.get("reasoning_summary", "")),
        student_feedback=str(raw_output.get("student_feedback", "")),
        teacher_review_notes=str(raw_output.get("teacher_review_notes", "")),
        possible_ocr_issue=bool(raw_output.get("possible_ocr_issue", False)),
        confidence=float(raw_output.get("confidence", 0.0)),
        raw_output=raw_output,
        created_at=created_at or utc_now_iso(),
    )


def parse_diagnosis_json(text: str) -> dict[str, Any]:
    return extract_json_object(text)
