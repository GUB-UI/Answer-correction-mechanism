"""問題登録ページ"""

import streamlit as st

from src.models import ProblemRecord, utc_now_iso
from src.ui_common import get_storage

st.set_page_config(page_title="問題登録", layout="wide")
st.title("問題登録")

storage = get_storage()

with st.form("problem_register_form"):
    title = st.text_input("問題タイトル")
    problem_text = st.text_area("問題文", height=120)
    correct_answer = st.text_area("模範解答", height=100)
    rubric = st.text_area("採点基準", height=100)
    unit = st.text_input("単元")
    difficulty = st.selectbox(
        "難易度",
        options=["basic", "standard", "advanced"],
        format_func=lambda x: {"basic": "基礎", "standard": "標準", "advanced": "発展"}[x],
    )
    submitted = st.form_submit_button("保存")

if submitted:
    if not all([title.strip(), problem_text.strip(), correct_answer.strip(), rubric.strip(), unit.strip()]):
        st.error("すべての項目を入力してください。")
    else:
        problem = ProblemRecord(
            problem_id=storage.generate_id("prob"),
            title=title.strip(),
            problem_text=problem_text.strip(),
            correct_answer=correct_answer.strip(),
            rubric=rubric.strip(),
            unit=unit.strip(),
            difficulty=difficulty,
            created_at=utc_now_iso(),
        )
        storage.save_problem(problem)
        st.success(f"問題を保存しました: {problem.problem_id}")
        st.json(problem.model_dump())

st.subheader("登録済み問題")
problems = storage.list_problems()
if not problems:
    st.info("まだ問題が登録されていません。")
else:
    for problem in reversed(problems):
        with st.expander(f"{problem.title} ({problem.problem_id})"):
            st.write(f"**単元:** {problem.unit} / **難易度:** {problem.difficulty}")
            st.write("**問題文**")
            st.text(problem.problem_text)
            st.write("**模範解答**")
            st.text(problem.correct_answer)
            st.write("**採点基準**")
            st.text(problem.rubric)
