from __future__ import annotations

from typing import Any

from src.adapters.base import DiagnosisRunContext
from src.analysis import normalize_error_categories
from src.models import DiagnosisResult, utc_now_iso
from src.utils import extract_json_object


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_question_scores(
    value: Any,
    allocation: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """LLM 出力の question_scores を設問 ID の配列に整える。

    「全体」「合計」は答案合計であり設問 ID ではないので捨てる。
    配点があるときは、その設問 ID に正規化できるものだけ残す。
    """
    from src.teacher_compare import normalize_question_label

    if isinstance(value, dict):
        value = [
            {"question": key, **(item if isinstance(item, dict) else {"score": item})}
            for key, item in value.items()
        ]
    if not isinstance(value, list):
        return []
    canonical: dict[str, str] = {}
    for item in allocation or []:
        question = str(item.get("question") or "").strip()
        key = normalize_question_label(question)
        if key:
            canonical.setdefault(key, question)
    items: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        score = _to_float(item.get("score"))
        if score is None:
            continue
        question = str(item.get("question") or item.get("label") or "").strip()
        key = normalize_question_label(question)
        if not key:
            continue
        if canonical:
            if key not in canonical:
                continue
            question = canonical[key]
        items.append(
            {
                "question": question,
                "score": score,
                "max_score": _to_float(item.get("max_score")),
                "reason": str(item.get("reason") or ""),
            }
        )
    return items


def format_score_allocation(allocation: list[dict[str, Any]]) -> str:
    lines = []
    for item in allocation:
        question = str(item.get("question") or "").strip()
        max_score = _to_float(item.get("max_score"))
        if question and max_score is not None:
            lines.append(f"{question}: {max_score:g}点")
    return "\n".join(lines)


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
        question_scores=normalize_question_scores(
            raw_output.get("question_scores"),
            allocation=list(context.problem.score_allocation or []),
        ),
    )


def parse_diagnosis_json(text: str) -> dict[str, Any]:
    return extract_json_object(text)
