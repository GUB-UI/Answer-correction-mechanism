"""答案登録ページ"""

import streamlit as st

from src.models import AnswerRecord, utc_now_iso
from src.ocr_runner import typed_text_as_ocr
from src.sources import IMAGE_UPLOAD_TYPES, answer_image_paths, infer_input_kind
from src.storage import validate_anonymized_id
from src.ui_common import (
    answer_label,
    get_storage,
    problem_label,
    show_answer_image,
)

st.set_page_config(page_title="答案登録", layout="wide")
st.title("答案登録")
st.caption("答案は画像（複数枚）、テキスト、またはその両方で登録できます。")

storage = get_storage()
problems = storage.list_problems()

if not problems:
    st.warning("先に問題を登録してください。")
    st.stop()

problem_map = {p.problem_id: p for p in problems}
problem_options = list(problem_map.keys())

selected_problem_id = st.selectbox(
    "問題を選択",
    options=problem_options,
    format_func=lambda pid: problem_label(problem_map[pid]),
)
student_id = st.text_input("匿名化生徒ID", placeholder="例: student_001")
uploaded_files = st.file_uploader(
    "答案画像（任意・複数枚可）",
    type=IMAGE_UPLOAD_TYPES,
    accept_multiple_files=True,
)
typed_text = st.text_area("テキスト答案（任意。デジタル提出やOCR後の手入力）", height=120)

if st.button("保存", type="primary"):
    try:
        anonymized_id = validate_anonymized_id(student_id)
        files = uploaded_files or []
        typed_value = typed_text.strip()
        if not files and not typed_value:
            raise ValueError("答案画像かテキスト答案のどちらかが必要です。")

        answer_id = storage.generate_id("ans")
        image_paths = [
            storage.save_uploaded_image(item.getvalue(), item.name) for item in files
        ]
        answer = AnswerRecord(
            answer_id=answer_id,
            problem_id=selected_problem_id,
            image_path=image_paths[0] if image_paths else "",
            image_paths=image_paths,
            typed_text=typed_value or None,
            input_kind=infer_input_kind(typed_value, image_paths),
            student_anonymized_id=anonymized_id,
            created_at=utc_now_iso(),
        )
        storage.save_answer(answer)
        if typed_value and not image_paths:
            typed_text_as_ocr(storage, answer_id=answer_id, typed_text=typed_value)
        st.success(f"答案を保存しました: {answer.answer_id}")
        show_answer_image(answer)
        if typed_value:
            st.code(typed_value)
        if image_paths:
            st.info("画像答案は「OCR実行」でテキスト化してください。")
    except ValueError as exc:
        st.error(str(exc))

st.subheader("登録済み答案")
answers = storage.list_answers()
if not answers:
    st.info("まだ答案が登録されていません。")
else:
    for answer in reversed(answers):
        problem = problem_map.get(answer.problem_id)
        with st.expander(answer_label(answer, problem)):
            if problem:
                st.write(f"**問題:** {problem.title}")
            st.write(f"**匿名化生徒ID:** {answer.student_anonymized_id}")
            st.write(f"**入力形式:** {answer.input_kind}")
            show_answer_image(answer)
            if answer.typed_text:
                st.write("**テキスト答案**")
                st.code(answer.typed_text)
