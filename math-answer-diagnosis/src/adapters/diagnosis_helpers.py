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


def fill_question_scores_from_allocation(
    scores: list[dict[str, Any]],
    allocation: list[dict[str, Any]] | None,
    *,
    total_score: float,
    possible_ocr_issue: bool = False,
) -> list[dict[str, Any]]:
    """question_scores が空のとき、配点の設問行を作る。

    モデルが満点に相当する合計だけ返したときは、各設問を満点にする。
    合計が 0 より大きく満点未満のときは配点比で分ける。
    OCR が疑わしく合計も 0 のときは、設問点を 0 で確定させず未確定の行にする。
    """
    from src.teacher_compare import normalize_question_label

    if scores or not allocation:
        return scores
    rows: list[dict[str, Any]] = []
    for item in allocation:
        question = str(item.get("question") or "").strip()
        key = normalize_question_label(question)
        cap = _to_float(item.get("max_score"))
        if not key or cap is None:
            continue
        rows.append({"question": question, "max_score": cap})
    if not rows:
        return scores
    cap_sum = sum(float(row["max_score"]) for row in rows)
    if possible_ocr_issue and total_score <= 0:
        return [
            {
                "question": row["question"],
                "score": None,
                "max_score": row["max_score"],
                "reason": "要教師確認。OCRが疑わしく設問点が出なかったため未確定",
            }
            for row in rows
        ]
    if cap_sum > 0 and abs(total_score - cap_sum) <= 0.05:
        return [
            {
                "question": row["question"],
                "score": row["max_score"],
                "max_score": row["max_score"],
                "reason": "モデルが設問点を省略したため配点どおり満点で補完",
            }
            for row in rows
        ]
    if cap_sum > 0 and total_score > 0:
        filled: list[dict[str, Any]] = []
        remaining = float(total_score)
        for index, row in enumerate(rows):
            if index == len(rows) - 1:
                part = remaining
            else:
                part = round(total_score * (float(row["max_score"]) / cap_sum), 2)
                remaining = round(remaining - part, 2)
            part = min(max(part, 0.0), float(row["max_score"]))
            filled.append(
                {
                    "question": row["question"],
                    "score": part,
                    "max_score": row["max_score"],
                    "reason": "モデルが設問点を省略したため配点比で補完",
                }
            )
        return filled
    return [
        {
            "question": row["question"],
            "score": 0.0,
            "max_score": row["max_score"],
            "reason": "モデルが設問点を省略したため配点表から補完",
        }
        for row in rows
    ]


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
    allocation = list(context.problem.score_allocation or [])
    possible_ocr_issue = bool(raw_output.get("possible_ocr_issue", False))
    score = float(raw_output.get("score", 0) or 0)
    question_scores = fill_question_scores_from_allocation(
        normalize_question_scores(raw_output.get("question_scores"), allocation),
        allocation,
        total_score=score,
        possible_ocr_issue=possible_ocr_issue,
    )
    item_scores = [
        float(item["score"]) for item in question_scores if item.get("score") is not None
    ]
    # 設問ごとの部分点があるのに合計だけ 0 のときは、部分点の合計を使う。
    if possible_ocr_issue and score <= 0 and item_scores and sum(item_scores) > 0:
        score = sum(item_scores)
    review_status = str(raw_output.get("review_status") or "").strip()
    notes = str(raw_output.get("teacher_review_notes", ""))
    if possible_ocr_issue:
        review_status = "needs_teacher_review"
        if "要教師確認" not in notes:
            notes = ("要教師確認。読める手順は設問ごとの部分点とし、全体を0点で確定しない。" + notes).strip()
    return DiagnosisResult(
        diagnosis_id=diagnosis_id,
        answer_id=context.answer_id,
        problem_id=context.problem_id,
        ocr_id=context.ocr_id,
        model_name=context.model_name,
        prompt_type=context.prompt_type,
        score=score,
        max_score=float(raw_output.get("max_score", 0) or 0),
        error_categories=normalize_error_categories(
            list(raw_output.get("error_categories", [])),
        ),
        error_locations=list(raw_output.get("error_locations", [])),
        reasoning_summary=str(raw_output.get("reasoning_summary", "")),
        student_feedback=str(raw_output.get("student_feedback", "")),
        teacher_review_notes=notes,
        possible_ocr_issue=possible_ocr_issue,
        review_status=review_status,
        confidence=float(raw_output.get("confidence", 0.0)),
        raw_output=raw_output,
        created_at=created_at or utc_now_iso(),
        question_scores=question_scores,
    )


def parse_diagnosis_json(text: str) -> dict[str, Any]:
    return extract_json_object(text)
