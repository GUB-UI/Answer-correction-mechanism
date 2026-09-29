"""診断実行ページ"""

import streamlit as st

from src.adapters import create_diagnosis_adapter
from src.adapters.base import DiagnosisRunContext
from src.models import utc_now_iso
from src.prompts import DIAGNOSIS_PROMPTS
from src.sources import student_ocr_results
from src.ui_common import (
    get_config,
    get_storage,
    ocr_label,
    problem_label,
    provider_index,
    show_answer_image,
    show_problem_images,
)

DIAGNOSIS_PROVIDERS = ["mock", "omlx"]

st.set_page_config(page_title="診断実行", layout="wide")
st.title("診断実行")

storage = get_storage()
config = get_config()
ocr_results = [
    item
    for item in student_ocr_results(storage.list_ocr_results())
    if item.answer_id in {a.answer_id for a in storage.list_answers()}
]
answers = {a.answer_id: a for a in storage.list_answers()}
problems = {p.problem_id: p for p in storage.list_problems()}

if not ocr_results:
    st.warning("先に答案の OCR（またはテキスト答案の登録）を実行してください。")
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

problem = problems.get(answer.problem_id)
if problem is None:
    st.error("対応する問題が見つかりません。")
    st.stop()

show_answer_image(answer)
show_problem_images(problem)

st.subheader("問題情報")
st.write(f"**問題:** {problem_label(problem)}")
st.write("**問題文**")
st.text(problem.problem_text)
st.write("**模範解答**")
st.text(problem.correct_answer)
st.write("**採点基準**")
st.text(problem.rubric)

st.subheader("OCR used_text（診断入力）")
st.code(ocr.used_text)
if ocr.human_corrected:
    st.info("この OCR 結果は人間により修正されています。")

provider = st.selectbox(
    "診断 provider",
    options=DIAGNOSIS_PROVIDERS,
    index=provider_index(DIAGNOSIS_PROVIDERS, config.diagnosis.provider),
    help="本番は omlx（Qwen3.8-27B-oQ4e-mtp + Lightning MTP）です。障害時のみ mock に切り替えてください。",
)
prompt_type = st.selectbox(
    "prompt_type",
    options=list(DIAGNOSIS_PROMPTS.keys()),
)

if st.button("診断を実行", type="primary"):
    try:
        adapter = create_diagnosis_adapter(provider, config)
        context = DiagnosisRunContext(
            answer_id=answer.answer_id,
            problem_id=problem.problem_id,
            ocr_id=ocr.ocr_id,
            problem=problem,
            used_text=ocr.used_text,
            model_name=config.diagnosis.model_name,
            prompt_type=prompt_type,
        )
        with st.spinner("診断実行中..."):
            raw = adapter.run(context)
        diagnosis = adapter.to_diagnosis_result(
            context,
            raw,
            diagnosis_id=storage.generate_id("diag"),
            created_at=utc_now_iso(),
        )
        storage.save_diagnosis_result(diagnosis)

        st.success(f"診断結果を保存しました: {diagnosis.diagnosis_id}")
        st.metric("得点", f"{diagnosis.score} / {diagnosis.max_score}")
        st.write("**誤り分類:**", ", ".join(diagnosis.error_categories) or "なし")
        st.write("**reasoning_summary**")
        st.text(diagnosis.reasoning_summary)
        st.write("**student_feedback**")
        st.text(diagnosis.student_feedback)
        st.write("**teacher_review_notes**")
        st.text(diagnosis.teacher_review_notes)
        if diagnosis.possible_ocr_issue:
            st.warning("possible_ocr_issue: true — OCR誤りの可能性があります。")
        st.json(diagnosis.model_dump())
    except Exception as exc:
        st.error(f"診断実行に失敗しました: {exc}")
