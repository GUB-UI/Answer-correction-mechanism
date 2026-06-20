"""診断結果確認ページ"""

import json
from io import StringIO

import pandas as pd
import streamlit as st

from src.ui_common import (
    diagnosis_label,
    get_storage,
    show_answer_image,
)

st.set_page_config(page_title="診断結果確認", layout="wide")
st.title("診断結果確認")

storage = get_storage()
diagnosis_results = storage.list_diagnosis_results()
ocr_results = {o.ocr_id: o for o in storage.list_ocr_results()}
answers = {a.answer_id: a for a in storage.list_answers()}
problems = {p.problem_id: p for p in storage.list_problems()}

if not diagnosis_results:
    st.warning("まだ診断結果がありません。")
    st.stop()

diag_map = {d.diagnosis_id: d for d in diagnosis_results}
selected_diag_id = st.selectbox(
    "診断結果を選択",
    options=list(diag_map.keys()),
    format_func=lambda did: diagnosis_label(diag_map[did]),
)

diagnosis = diag_map[selected_diag_id]
answer = answers.get(diagnosis.answer_id)
ocr = ocr_results.get(diagnosis.ocr_id)
problem = problems.get(diagnosis.problem_id)

if answer:
    show_answer_image(answer)

col1, col2 = st.columns(2)
with col1:
    st.metric("得点", f"{diagnosis.score} / {diagnosis.max_score}")
    st.metric("信頼度", f"{diagnosis.confidence:.2f}")
with col2:
    if diagnosis.possible_ocr_issue:
        st.error("possible_ocr_issue: **あり** — OCR誤りの可能性を確認してください。")
    else:
        st.success("possible_ocr_issue: なし")

st.subheader("teacher_review_notes")
if diagnosis.teacher_review_notes:
    st.info(diagnosis.teacher_review_notes)
else:
    st.write("（なし）")

if ocr:
    st.subheader("OCR結果")
    st.write(f"**OCR ID:** {ocr.ocr_id}")
    if ocr.human_corrected:
        st.warning("人間による修正あり")
    st.write("**raw_text**")
    st.code(ocr.raw_text)
    st.write("**used_text**")
    st.code(ocr.used_text)
    if ocr.raw_text != ocr.used_text:
        st.warning("raw_text と used_text が異なります。OCR修正の影響を確認してください。")
else:
    st.warning("対応する OCR 結果が見つかりません。")

st.subheader("診断結果")
st.write("**誤り分類:**", ", ".join(diagnosis.error_categories) or "なし")
st.write("**reasoning_summary**")
st.text(diagnosis.reasoning_summary)
st.write("**student_feedback**")
st.text(diagnosis.student_feedback)

if diagnosis.error_locations:
    st.write("**error_locations**")
    st.json(diagnosis.error_locations)

if problem:
    st.write(f"**問題:** {problem.title}")

st.subheader("エクスポート")
export_rows = [d.model_dump() for d in diagnosis_results]
df = pd.DataFrame(export_rows)

csv_buffer = StringIO()
df.to_csv(csv_buffer, index=False, encoding="utf-8")
st.download_button(
    label="diagnosis_results を CSV でダウンロード",
    data=csv_buffer.getvalue().encode("utf-8"),
    file_name="diagnosis_results.csv",
    mime="text/csv",
)

jsonl_lines = "\n".join(
    json.dumps(row, ensure_ascii=False) for row in export_rows
)
st.download_button(
    label="diagnosis_results を JSONL でダウンロード",
    data=jsonl_lines.encode("utf-8"),
    file_name="diagnosis_results.jsonl",
    mime="application/json",
)

st.subheader("診断結果一覧")
st.dataframe(
    df[
        [
            "diagnosis_id",
            "answer_id",
            "score",
            "max_score",
            "possible_ocr_issue",
            "prompt_type",
            "created_at",
        ]
    ],
    use_container_width=True,
)
