"""OCR確認・修正ページ"""

import streamlit as st

from src.adapters import ManualCorrectionOCRAdapter
from src.sources import SOURCE_KIND_LABELS
from src.ui_common import (
    get_storage,
    ocr_label,
    show_answer_image,
    show_images,
    show_problem_images,
)

st.set_page_config(page_title="OCR確認・修正", layout="wide")
st.title("OCR確認・修正")
st.caption(
    "診断結果が不自然な場合に、OCRの責任範囲を確認するためのページです。"
    "手書き答案の読み取り精度は、今後もここで観察してください。"
)

storage = get_storage()
ocr_results = storage.list_ocr_results()
answers = {a.answer_id: a for a in storage.list_answers()}
problems = {p.problem_id: p for p in storage.list_problems()}

if not ocr_results:
    st.warning("先に OCR を実行してください。")
    st.stop()

ocr_map = {o.ocr_id: o for o in ocr_results}
selected_ocr_id = st.selectbox(
    "OCR結果を選択",
    options=list(ocr_map.keys()),
    format_func=lambda oid: ocr_label(ocr_map[oid]),
)

ocr = ocr_map[selected_ocr_id]
st.write(
    f"**対象:** {SOURCE_KIND_LABELS.get(ocr.source_kind, ocr.source_kind)} / "
    f"`{ocr.source_id or ocr.answer_id}` / engine `{ocr.ocr_engine}`"
)

if ocr.image_paths:
    show_images(ocr.image_paths, caption_prefix="OCR画像 ")

answer = answers.get(ocr.answer_id)
if answer is not None:
    show_answer_image(answer)

problem = problems.get(ocr.source_id) or (
    problems.get(answer.problem_id) if answer else None
)
if problem is not None and ocr.source_kind != "student_answer":
    show_problem_images(problem)

def _formula_replacements(raw: dict) -> list:
    items: list = []
    if isinstance(raw.get("formula_replacements"), list):
        items.extend(raw["formula_replacements"])
    for page in raw.get("pages") or []:
        if isinstance(page, dict) and isinstance(page.get("formula_replacements"), list):
            items.extend(page["formula_replacements"])
    return items


replacements = _formula_replacements(ocr.raw_output or {})
if replacements:
    st.subheader("数式置換（GLM → UniMERNet Small）")
    st.caption("数字・数式領域だけ UniMERNet Small で読み直し、日本語は GLM-OCR のままです。")
    st.json(replacements)

st.subheader("raw_text（変更不可）")
st.code(ocr.raw_text)

with st.form("ocr_correction_form"):
    used_text = st.text_area(
        "used_text（診断LLMに渡すテキスト）",
        value=ocr.used_text,
        height=200,
    )
    ocr_suspect = st.checkbox("ocr_suspect（OCRが怪しい）", value=ocr.ocr_suspect)
    correction_note = st.text_input(
        "correction_note（修正理由）",
        value=ocr.correction_note or "",
        placeholder="例: 8x<40 が x<40 と誤読されていた",
    )
    submitted = st.form_submit_button("修正版を保存", type="primary")

if submitted:
    try:
        manual = ManualCorrectionOCRAdapter(storage)
        updated = manual.apply_correction(
            ocr.ocr_id,
            used_text.strip(),
            ocr_suspect=ocr_suspect,
            correction_note=correction_note.strip() or None,
        )
        st.success("OCR結果を更新しました。")
        st.write(f"**human_corrected:** {updated.human_corrected}")
        st.write(f"**ocr_suspect:** {updated.ocr_suspect}")
        if updated.correction_note:
            st.write(f"**correction_note:** {updated.correction_note}")
    except Exception as exc:
        st.error(f"保存に失敗しました: {exc}")
