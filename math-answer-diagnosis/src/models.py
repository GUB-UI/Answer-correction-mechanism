from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

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
    "intake_document",
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
    # 設問ごとの配点 [{"question": "(1)", "max_score": 5}]。教師採点の取り込みでも設定できる。
    score_allocation: list[dict[str, Any]] = Field(default_factory=list)
    tagging_metadata: dict[str, Any] = Field(default_factory=dict)


class AnswerRecord(BaseModel):
    answer_id: str
    problem_id: str
    image_path: str = ""
    student_anonymized_id: str
    created_at: str
    image_paths: list[str] = Field(default_factory=list)
    typed_text: str | None = None
    input_kind: str = "image"
    tagging_metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def sync_image_paths(self) -> AnswerRecord:
        if not self.image_paths and self.image_path:
            self.image_paths = [self.image_path]
        elif self.image_paths and not self.image_path:
            self.image_path = self.image_paths[0]
        return self


class OCRProvenance(BaseModel):
    engine: str
    model_id: str
    model_revision: str
    settings: dict[str, Any]
    image_sha256: list[str]


class OCRResult(BaseModel):
    ocr_id: str
    answer_id: str = ""
    ocr_engine: str
    input_type: Literal["ocr_output", "manual_transcription", "typed_text"] = "ocr_output"
    parent_ocr_id: str | None = None
    provenance: OCRProvenance | None = None
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
        # Legacy corrections and typed inputs must never enter the OCR cache.
        if self.human_corrected:
            self.input_type = "manual_transcription"
        elif self.ocr_engine == "typed_text":
            self.input_type = "typed_text"
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
    score: float | None
    max_score: float
    error_categories: list[str]
    error_locations: list[dict[str, Any]]
    reasoning_summary: str
    student_feedback: str
    teacher_review_notes: str
    possible_ocr_issue: bool
    review_status: Literal["validated", "review_required", "failed"] = "validated"
    confidence: float
    raw_output: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    # [{"question": "(1)", "score": 3, "max_score": 5, "reason": "..."}]
    question_scores: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("review_status", mode="before")
    @classmethod
    def migrate_review_status(cls, value: Any) -> Any:
        return {"": "validated", "needs_teacher_review": "review_required"}.get(value, value)

    @model_validator(mode="after")
    def require_review_for_undetermined_scores(self) -> DiagnosisResult:
        if self.review_status != "failed" and (
            self.score is None or any(item.get("score") is None for item in self.question_scores)
        ):
            self.review_status = "review_required"
        return self


class TeacherGradeRecord(BaseModel):
    """教師による採点結果（設問単位）。question が空なら答案全体の合計点。"""

    teacher_grade_id: str
    student_anonymized_id: str
    problem_id: str
    question: str = ""
    teacher_score: float
    max_score: float | None = None
    teacher_comment: str = ""
    grader: str = ""
    source_file: str = ""
    imported_at: str


class GradeDiscrepancy(BaseModel):
    """教師採点とシステム採点の差が閾値を超えた項目のログ。"""

    discrepancy_id: str
    comparison_key: str
    student_anonymized_id: str
    problem_id: str
    question: str = ""
    teacher_grade_id: str
    answer_id: str
    diagnosis_id: str
    ocr_id: str = ""
    teacher_score: float
    system_score: float
    max_score: float | None = None
    diff: float
    ratio_diff: float | None = None
    possible_ocr_issue: bool = False
    ocr_human_corrected: bool = False
    system_reasoning: str = ""
    teacher_comment: str = ""
    status: str = "open"
    recheck: dict[str, Any] | None = None
    detected_at: str
    updated_at: str = ""

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
