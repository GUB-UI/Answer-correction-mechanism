from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

ERROR_CATEGORIES = [
    "問題理解の誤り",
    "方針選択の誤り",
    "概念理解の誤り",
    "計算ミス",
    "論理接続の誤り",
    "表記・記述の不備",
    "見落とし・条件確認不足",
]


class ProblemRecord(BaseModel):
    problem_id: str
    title: str
    problem_text: str
    correct_answer: str
    rubric: str
    unit: str
    difficulty: str
    created_at: str


class AnswerRecord(BaseModel):
    answer_id: str
    problem_id: str
    image_path: str
    student_anonymized_id: str
    created_at: str


class OCRResult(BaseModel):
    ocr_id: str
    answer_id: str
    ocr_engine: str
    raw_text: str
    used_text: str
    uncertain_parts: list[str] = Field(default_factory=list)
    ocr_suspect: bool = False
    human_corrected: bool = False
    correction_note: str | None = None
    raw_output: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class DiagnosisResult(BaseModel):
    diagnosis_id: str
    answer_id: str
    problem_id: str
    ocr_id: str
    model_name: str
    prompt_type: str
    score: float
    max_score: float
    error_categories: list[str]
    error_locations: list[dict[str, Any]]
    reasoning_summary: str
    student_feedback: str
    teacher_review_notes: str
    possible_ocr_issue: bool
    confidence: float
    raw_output: dict[str, Any] = Field(default_factory=dict)
    created_at: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
