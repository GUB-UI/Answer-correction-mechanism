"""セット登録 — 問題・模範解答・答案をテキスト/画像/混在で登録する"""

from __future__ import annotations

import streamlit as st

from src.adapters import create_ocr_adapter
from src.models import AnswerRecord, DEFAULT_RUBRIC, ProblemRecord, utc_now_iso
from src.ocr_runner import (
    GLMOCR_SDK_ERROR_MESSAGE,
    run_ocr_for_images,
    typed_text_as_ocr,
)
from src.sources import (
    IMAGE_UPLOAD_TYPES,
    infer_input_kind,
    split_problem_and_solution,
)
from src.storage import validate_anonymized_id
from src.ui_common import get_config, get_storage, provider_index, show_images

OCR_PROVIDERS = ["mock", "glmocr_sdk"]
PROBLEM_SOURCE_OPTIONS = {
    "text": "テキスト入力",
    "image": "画像から読む",
    "combined": "問題と模範が同じ画像",
}
MODEL_SOURCE_OPTIONS = {
    "text": "テキスト入力",
    "image": "別画像から読む",
    "same": "問題と同じ画像",
}
ANSWER_SOURCE_OPTIONS = {
    "image": "答案画像",
    "text": "テキスト答案",
    "mixed": "画像とテキスト",
}


def _save_uploads(storage, files) -> list[str]:
    if not files:
        return []
    items = files if isinstance(files, list) else [files]
    return [storage.save_uploaded_image(item.getvalue(), item.name) for item in items]


st.set_page_config(page_title="セット登録", layout="wide")
st.title("セット登録")
st.caption(
    "問題文・模範解答・答案は、テキスト・画像・混在のどれでも登録できます。"
    "印刷物1枚に問題と模範が載っている場合は「同じ画像」を選んでください。"
)

storage = get_storage()
config = get_config()

st.subheader("入力形式")
col_a, col_b, col_c = st.columns(3)
with col_a:
    problem_source = st.radio(
        "問題文",
        options=list(PROBLEM_SOURCE_OPTIONS.keys()),
        format_func=lambda key: PROBLEM_SOURCE_OPTIONS[key],
        key="intake_problem_source",
    )
with col_b:
    model_disabled = problem_source == "combined"
    model_source = st.radio(
        "模範解答",
        options=list(MODEL_SOURCE_OPTIONS.keys()),
        format_func=lambda key: MODEL_SOURCE_OPTIONS[key],
        index=2 if model_disabled else 0,
        disabled=model_disabled,
        key="intake_model_source",
    )
    if model_disabled:
        model_source = "same"
        st.caption("問題と模範は同一画像として扱います。")
with col_c:
    answer_source = st.radio(
        "答案",
        options=list(ANSWER_SOURCE_OPTIONS.keys()),
        format_func=lambda key: ANSWER_SOURCE_OPTIONS[key],
        key="intake_answer_source",
    )

st.subheader("メタデータ")
meta1, meta2 = st.columns(2)
with meta1:
    title = st.text_input("問題タイトル", placeholder="例: 一次不等式 基本例題36")
    unit = st.text_input("単元", placeholder="例: 数学I 一次不等式")
    student_id = st.text_input("匿名化生徒ID", placeholder="例: student_001")
with meta2:
    difficulty = st.selectbox(
        "難易度",
        options=["basic", "standard", "advanced"],
        format_func=lambda x: {"basic": "基礎", "standard": "標準", "advanced": "発展"}[x],
    )
    rubric = st.text_area(
        "採点基準（空なら既定文）",
        height=90,
        placeholder=DEFAULT_RUBRIC,
    )

st.subheader("問題・模範解答")
combined_files = None
problem_files = None
model_files = None

if problem_source == "combined":
    combined_files = st.file_uploader(
        "問題と模範解答が写った画像（複数枚可）",
        type=IMAGE_UPLOAD_TYPES,
        accept_multiple_files=True,
        key="intake_combined_files",
    )
elif problem_source == "image":
    problem_files = st.file_uploader(
        "問題文の画像（複数枚可）",
        type=IMAGE_UPLOAD_TYPES,
        accept_multiple_files=True,
        key="intake_problem_files",
    )
if problem_source != "combined" and model_source == "image":
    model_files = st.file_uploader(
        "模範解答の画像（複数枚可）",
        type=IMAGE_UPLOAD_TYPES,
        accept_multiple_files=True,
        key="intake_model_files",
    )

problem_text = st.text_area(
    "問題文（画像OCR後に編集可）",
    height=140,
    key="intake_problem_text_area",
)
correct_answer = st.text_area(
    "模範解答（画像OCR後に編集可）",
    height=140,
    key="intake_correct_text_area",
)

st.subheader("答案")
answer_files = None
if answer_source in {"image", "mixed"}:
    answer_files = st.file_uploader(
        "答案画像（複数枚可）",
        type=IMAGE_UPLOAD_TYPES,
        accept_multiple_files=True,
        key="intake_answer_files",
    )
typed_text = ""
if answer_source in {"text", "mixed"}:
    typed_text = st.text_area(
        "テキスト答案",
        height=120,
        key="intake_typed_text_area",
    )

provider = st.selectbox(
    "OCR provider",
    options=OCR_PROVIDERS,
    index=provider_index(OCR_PROVIDERS, config.ocr.provider),
    help="画像がある場合に使います。本番は GLM-OCR + UniMERNet Small です。",
)

ocr_col, save_col = st.columns(2)
do_ocr = ocr_col.button("画像をOCRしてプレビュー", type="secondary")
do_save = save_col.button("セットを保存", type="primary")


def _engine() -> str:
    return config.ocr.model_name if provider == "glmocr_sdk" else "mock"


if do_ocr:
    try:
        adapter = create_ocr_adapter(provider, config)
        engine = _engine()
        with st.spinner("OCR実行中..."):
            if combined_files:
                paths = _save_uploads(storage, combined_files)
                st.session_state["intake_combined_paths"] = paths
                ocr = run_ocr_for_images(
                    adapter,
                    storage,
                    source_kind="combined",
                    source_id="intake_preview",
                    image_paths=paths,
                    engine=engine,
                )
                split_p, split_a = split_problem_and_solution(ocr.used_text)
                st.session_state["intake_problem_text_area"] = split_p
                st.session_state["intake_correct_text_area"] = split_a
                st.session_state["intake_combined_ocr_id"] = ocr.ocr_id
            if problem_files:
                paths = _save_uploads(storage, problem_files)
                st.session_state["intake_problem_paths"] = paths
                ocr = run_ocr_for_images(
                    adapter,
                    storage,
                    source_kind="problem",
                    source_id="intake_preview",
                    image_paths=paths,
                    engine=engine,
                )
                st.session_state["intake_problem_text_area"] = ocr.used_text
                st.session_state["intake_problem_ocr_id"] = ocr.ocr_id
            if model_files:
                paths = _save_uploads(storage, model_files)
                st.session_state["intake_model_paths"] = paths
                ocr = run_ocr_for_images(
                    adapter,
                    storage,
                    source_kind="model_answer",
                    source_id="intake_preview",
                    image_paths=paths,
                    engine=engine,
                )
                st.session_state["intake_correct_text_area"] = ocr.used_text
                st.session_state["intake_model_ocr_id"] = ocr.ocr_id
            if answer_files:
                paths = _save_uploads(storage, answer_files)
                st.session_state["intake_answer_paths"] = paths
                ocr = run_ocr_for_images(
                    adapter,
                    storage,
                    source_kind="student_answer",
                    source_id="intake_preview",
                    image_paths=paths,
                    engine=engine,
                )
                st.session_state["intake_typed_text_area"] = ocr.used_text
                st.session_state["intake_answer_preview_ocr_id"] = ocr.ocr_id
        if not any([combined_files, problem_files, model_files, answer_files]):
            st.warning("OCRする画像がありません。")
        else:
            st.success("OCR結果を下のテキスト欄に入れました。内容を確認してから保存してください。")
            st.rerun()
    except Exception as exc:
        if provider == "glmocr_sdk":
            st.error(GLMOCR_SDK_ERROR_MESSAGE)
            with st.expander("エラー詳細"):
                st.code(str(exc))
        else:
            st.error(f"OCRに失敗しました: {exc}")

if do_save:
    try:
        title_value = title.strip()
        if not title_value:
            raise ValueError("問題タイトルは必須です。")
        anonymized_id = validate_anonymized_id(student_id)

        combined_paths = list(st.session_state.get("intake_combined_paths", []))
        problem_paths = list(st.session_state.get("intake_problem_paths", []))
        model_paths = list(st.session_state.get("intake_model_paths", []))
        answer_paths = list(st.session_state.get("intake_answer_paths", []))
        if combined_files:
            combined_paths = _save_uploads(storage, combined_files)
        if problem_files:
            problem_paths = _save_uploads(storage, problem_files)
        if model_files:
            model_paths = _save_uploads(storage, model_files)
        if answer_files:
            answer_paths = _save_uploads(storage, answer_files)

        problem_text_value = (st.session_state.get("intake_problem_text_area") or "").strip()
        correct_value = (st.session_state.get("intake_correct_text_area") or "").strip()
        typed_value = (st.session_state.get("intake_typed_text_area") or "").strip()

        has_problem = bool(problem_text_value or problem_paths or combined_paths)
        has_model = bool(correct_value or model_paths or combined_paths)
        has_answer = bool(answer_paths or typed_value)
        if not has_problem:
            raise ValueError("問題文のテキストか画像が必要です。")
        if not has_model:
            raise ValueError("模範解答のテキストか画像が必要です。")
        if not has_answer:
            raise ValueError("答案の画像かテキストが必要です。")

        problem = ProblemRecord(
            problem_id=storage.generate_id("prob"),
            title=title_value,
            problem_text=problem_text_value,
            correct_answer=correct_value,
            rubric=(rubric.strip() or DEFAULT_RUBRIC),
            unit=unit.strip() or "未設定",
            difficulty=difficulty,
            created_at=utc_now_iso(),
            problem_input_kind=infer_input_kind(
                problem_text_value,
                problem_paths or combined_paths,
            ),
            answer_key_input_kind=infer_input_kind(
                correct_value,
                model_paths or combined_paths,
            ),
            problem_image_paths=problem_paths,
            model_answer_image_paths=model_paths,
            combined_image_paths=combined_paths,
            problem_ocr_ids=[
                oid
                for oid in [
                    st.session_state.get("intake_combined_ocr_id"),
                    st.session_state.get("intake_problem_ocr_id"),
                ]
                if oid
            ],
            model_ocr_ids=[
                oid
                for oid in [
                    st.session_state.get("intake_combined_ocr_id"),
                    st.session_state.get("intake_model_ocr_id"),
                ]
                if oid
            ],
        )
        storage.save_problem(problem)

        for ocr_id, kind in (
            (st.session_state.get("intake_combined_ocr_id"), "combined"),
            (st.session_state.get("intake_problem_ocr_id"), "problem"),
            (st.session_state.get("intake_model_ocr_id"), "model_answer"),
        ):
            if not ocr_id:
                continue
            preview = storage.get_ocr_result(ocr_id)
            if preview is None:
                continue
            storage.update_ocr_result(
                preview.model_copy(
                    update={"source_id": problem.problem_id, "source_kind": kind}
                )
            )

        answer_id = storage.generate_id("ans")
        answer = AnswerRecord(
            answer_id=answer_id,
            problem_id=problem.problem_id,
            image_path=answer_paths[0] if answer_paths else "",
            image_paths=answer_paths,
            typed_text=typed_value or None,
            input_kind=infer_input_kind(typed_value, answer_paths),
            student_anonymized_id=anonymized_id,
            created_at=utc_now_iso(),
        )
        storage.save_answer(answer)

        preview_ocr_id = st.session_state.get("intake_answer_preview_ocr_id")
        if preview_ocr_id:
            preview = storage.get_ocr_result(preview_ocr_id)
            if preview is not None:
                storage.update_ocr_result(
                    preview.model_copy(
                        update={
                            "answer_id": answer_id,
                            "source_id": answer_id,
                            "used_text": typed_value or preview.used_text,
                            "source_kind": "student_answer",
                        }
                    )
                )
        elif typed_value and not answer_paths:
            typed_text_as_ocr(storage, answer_id=answer_id, typed_text=typed_value)

        st.success(
            f"保存しました: {problem.problem_id} / {answer.answer_id}"
        )
        st.write("画像がある場合は「OCR実行」で本登録用に再実行できます。プレビューOCRは確認用です。")
        show_images(combined_paths or problem_paths, caption_prefix="問題側 ")
        show_images(model_paths, caption_prefix="模範 ")
        show_images(answer_paths, caption_prefix="答案 ")
        if problem_text_value:
            st.write("**問題文**")
            st.code(problem_text_value)
        if correct_value:
            st.write("**模範解答**")
            st.code(correct_value)
        if typed_value:
            st.write("**答案テキスト**")
            st.code(typed_value)
    except ValueError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"保存に失敗しました: {exc}")
