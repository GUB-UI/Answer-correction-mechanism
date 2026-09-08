from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AppConfig, load_config
from src.models import AnswerRecord, DiagnosisResult, OCRResult, ProblemRecord
from src.sources import SOURCE_KIND_LABELS, answer_image_paths, resolve_stored_path
from src.storage import Storage


@st.cache_resource
def get_config() -> AppConfig:
    return load_config(PROJECT_ROOT)


@st.cache_resource
def get_storage() -> Storage:
    return Storage(get_config())


def resolve_image_path(image_path: str) -> Path:
    return resolve_stored_path(PROJECT_ROOT, image_path)


def show_images(image_paths: list[str], caption_prefix: str = "") -> None:
    if not image_paths:
        return
    for index, rel in enumerate(image_paths, start=1):
        image_path = resolve_image_path(rel)
        if image_path.is_file():
            caption = f"{caption_prefix}{index}" if caption_prefix else rel
            st.image(str(image_path), caption=caption)
        else:
            st.warning(f"画像が見つかりません: {image_path}")


def show_answer_image(answer: AnswerRecord) -> None:
    paths = answer_image_paths(answer)
    if not paths:
        if answer.typed_text:
            st.info("この答案はテキスト入力です（画像なし）。")
        else:
            st.warning("答案画像がありません。")
        return
    show_images(paths, caption_prefix=f"答案ID: {answer.answer_id} / ")


def show_problem_images(problem: ProblemRecord) -> None:
    if problem.combined_image_paths:
        st.caption("問題+模範（同一画像）")
        show_images(problem.combined_image_paths, caption_prefix="同一画像 ")
        return
    if problem.problem_image_paths:
        st.caption("問題画像")
        show_images(problem.problem_image_paths, caption_prefix="問題 ")
    if problem.model_answer_image_paths:
        st.caption("模範解答画像")
        show_images(problem.model_answer_image_paths, caption_prefix="模範 ")


def problem_label(problem: ProblemRecord) -> str:
    return f"{problem.title} ({problem.problem_id})"


def answer_label(answer: AnswerRecord, problem: ProblemRecord | None = None) -> str:
    if problem is not None:
        return f"{answer.answer_id} / {problem.title} / {answer.student_anonymized_id}"
    return f"{answer.answer_id} / {answer.student_anonymized_id}"


def ocr_label(ocr: OCRResult) -> str:
    corrected = " [修正済]" if ocr.human_corrected else ""
    suspect = " [要確認]" if ocr.ocr_suspect else ""
    kind = SOURCE_KIND_LABELS.get(ocr.source_kind, ocr.source_kind)
    return f"{ocr.ocr_id} / {kind} / {ocr.source_id or ocr.answer_id}{corrected}{suspect}"


def diagnosis_label(diagnosis: DiagnosisResult) -> str:
    ocr_flag = " [OCR疑い]" if diagnosis.possible_ocr_issue else ""
    return (
        f"{diagnosis.diagnosis_id} / {diagnosis.answer_id} "
        f"({diagnosis.score}/{diagnosis.max_score}){ocr_flag}"
    )


def provider_index(options: list[str], selected: str, default: int = 0) -> int:
    try:
        return options.index(selected)
    except ValueError:
        return default
