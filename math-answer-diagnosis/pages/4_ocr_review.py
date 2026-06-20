"""OCR確認・修正ページ"""

import streamlit as st

from src.adapters import ManualCorrectionOCRAdapter
from src.ui_common import (
    get_storage,
    ocr_label,
    show_answer_image,
)

st.set_page_config(page_title="OCR確認・修正", layout="wide")
st.title("OCR確認・修正")
st.caption("診断結果が不自然な場合に、OCRの責任範囲を確認するためのページです。")

storage = get_storage()
ocr_results = storage.list_ocr_results()
answers = {a.answer_id: a for a in storage.list_answers()}

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
answer = answers.get(ocr.answer_id)
if answer is None:
    st.error("対応する答案が見つかりません。")
    st.stop()

show_answer_image(answer)

st.subheader("raw_text（変更不可）")
st.code(ocr.raw_text)

with st.form("ocr_correction_form"):
    used_text = st.text_area("used_text（診断LLMに渡すテキスト）", value=ocr.used_text, height=200)
    ocr_suspect = st.checkbox("ocr_suspect（OCRが怪しい）", value=ocr.ocr_suspect)
    correction_note = st.text_input(
        "correction_note（修正理由）",
        value=ocr.correction_note or "",
        placeholder="例: OCRが -5x を +5x と誤読していた",
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
