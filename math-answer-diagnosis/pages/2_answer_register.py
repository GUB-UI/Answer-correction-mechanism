"""答案登録ページ"""

from pathlib import Path

import streamlit as st

from src.models import AnswerRecord, utc_now_iso
from src.storage import validate_anonymized_id
from src.ui_common import answer_label, get_storage, problem_label

st.set_page_config(page_title="答案登録", layout="wide")
st.title("答案登録")

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
uploaded = st.file_uploader(
    "答案画像をアップロード",
    type=["png", "jpg", "jpeg", "webp"],
)

if st.button("保存", type="primary"):
    try:
        if uploaded is None:
            raise ValueError("答案画像をアップロードしてください。")
        anonymized_id = validate_anonymized_id(student_id)

        answer_id = storage.generate_id("ans")
        suffix = Path(uploaded.name).suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
            suffix = ".png"

        image_path = storage.save_answer_image(
            answer_id,
            uploaded.getvalue(),
            extension=suffix,
        )
        answer = AnswerRecord(
            answer_id=answer_id,
            problem_id=selected_problem_id,
            image_path=image_path,
            student_anonymized_id=anonymized_id,
            created_at=utc_now_iso(),
        )
        storage.save_answer(answer)
        st.success(f"答案を保存しました: {answer.answer_id}")
        st.image(str(storage.config.project_root / image_path), caption=answer_id)
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
            st.write(f"**画像パス:** {answer.image_path}")
            image_path = storage.config.project_root / answer.image_path
            if image_path.is_file():
                st.image(str(image_path))
