from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, model_validator

ERROR_CATEGORIES = [
    "問題理解の誤り",
    "方針選択の誤り",
    "概念理解の誤り",
    "計算ミス",
    "論理接続の誤り",
    "表記・記述の不備",
    "見落とし・条件確認不足",
]

INPUT_KINDS = ("text", "image", "mixed")
OCR_SOURCE_KINDS = (
    "student_answer",
    "problem",
    "model_answer",
    "combined",
)
DEFAULT_RUBRIC = (
    "入力された模範解答を基準に採点する。"
    "途中式が正しければ部分点可。"
    "最終答の境界値・不等号は厳密に見る。"
)


class ProblemRecord(BaseModel):
    problem_id: str
    title: str
    problem_text: str = ""
    correct_answer: str = ""
    rubric: str = ""
    unit: str = ""
    difficulty: str = "basic"
    created_at: str
    problem_input_kind: str = "text"
    answer_key_input_kind: str = "text"
    problem_image_paths: list[str] = Field(default_factory=list)
    model_answer_image_paths: list[str] = Field(default_factory=list)
    combined_image_paths: list[str] = Field(default_factory=list)
    problem_ocr_ids: list[str] = Field(default_factory=list)
    model_ocr_ids: list[str] = Field(default_factory=list)


class AnswerRecord(BaseModel):
    answer_id: str
    problem_id: str
    image_path: str = ""
    student_anonymized_id: str
    created_at: str
    image_paths: list[str] = Field(default_factory=list)
    typed_text: str | None = None
    input_kind: str = "image"

    @model_validator(mode="after")
    def sync_image_paths(self) -> AnswerRecord:
        if not self.image_paths and self.image_path:
            self.image_paths = [self.image_path]
        elif self.image_paths and not self.image_path:
            self.image_path = self.image_paths[0]
        return self


class OCRResult(BaseModel):
    ocr_id: str
    answer_id: str = ""
    ocr_engine: str
    raw_text: str
    used_text: str
    uncertain_parts: list[str] = Field(default_factory=list)
    ocr_suspect: bool = False
    human_corrected: bool = False
    correction_note: str | None = None
    raw_output: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    source_kind: str = "student_answer"
    source_id: str = ""
    image_paths: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def sync_source_fields(self) -> OCRResult:
        if not self.source_id:
            self.source_id = self.answer_id
        if (
            not self.answer_id
            and self.source_kind == "student_answer"
            and self.source_id
        ):
            self.answer_id = self.source_id
        return self


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
