from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AppConfig, load_config
from src.models import AnswerRecord, DiagnosisResult, OCRResult, ProblemRecord
from src.storage import Storage


@st.cache_resource
def get_config() -> AppConfig:
    return load_config(PROJECT_ROOT)


@st.cache_resource
def get_storage() -> Storage:
    return Storage(get_config())


def resolve_image_path(image_path: str) -> Path:
    path = Path(image_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def show_answer_image(answer: AnswerRecord) -> None:
    image_path = resolve_image_path(answer.image_path)
    if image_path.is_file():
        st.image(str(image_path), caption=f"答案ID: {answer.answer_id}")
    else:
        st.warning(f"画像が見つかりません: {image_path}")


def problem_label(problem: ProblemRecord) -> str:
    return f"{problem.title} ({problem.problem_id})"


def answer_label(answer: AnswerRecord, problem: ProblemRecord | None = None) -> str:
    if problem is not None:
        return f"{answer.answer_id} / {problem.title} / {answer.student_anonymized_id}"
    return f"{answer.answer_id} / {answer.student_anonymized_id}"


def ocr_label(ocr: OCRResult) -> str:
    corrected = " [修正済]" if ocr.human_corrected else ""
    suspect = " [要確認]" if ocr.ocr_suspect else ""
    return f"{ocr.ocr_id} / {ocr.answer_id}{corrected}{suspect}"


def diagnosis_label(diagnosis: DiagnosisResult) -> str:
    ocr_flag = " [OCR疑い]" if diagnosis.possible_ocr_issue else ""
    return (
        f"{diagnosis.diagnosis_id} / {diagnosis.answer_id} "
        f"({diagnosis.score}/{diagnosis.max_score}){ocr_flag}"
    )
