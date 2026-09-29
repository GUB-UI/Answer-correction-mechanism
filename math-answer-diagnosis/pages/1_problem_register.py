"""問題登録ページ"""

import streamlit as st

from src.models import DEFAULT_RUBRIC, ProblemRecord, utc_now_iso
from src.sources import IMAGE_UPLOAD_TYPES, infer_input_kind
from src.ui_common import get_storage, show_problem_images

st.set_page_config(page_title="問題登録", layout="wide")
st.title("問題登録")
st.caption("テキストだけ、画像だけ、または混在で登録できます。画像の読み取りは「OCR実行」でも行えます。")

storage = get_storage()

title = st.text_input("問題タイトル")
unit = st.text_input("単元")
difficulty = st.selectbox(
    "難易度",
    options=["basic", "standard", "advanced"],
    format_func=lambda x: {"basic": "基礎", "standard": "標準", "advanced": "発展"}[x],
)
problem_text = st.text_area("問題文（任意。画像だけでも可）", height=120)
correct_answer = st.text_area("模範解答（任意。画像だけでも可）", height=100)
rubric = st.text_area("採点基準（空なら既定文）", height=80, placeholder=DEFAULT_RUBRIC)

same_image = st.checkbox("問題と模範解答が同じ画像に写っている")
combined_files = None
problem_files = None
model_files = None
if same_image:
    combined_files = st.file_uploader(
        "問題+模範の画像（複数枚可）",
        type=IMAGE_UPLOAD_TYPES,
        accept_multiple_files=True,
    )
else:
    problem_files = st.file_uploader(
        "問題文の画像（任意・複数枚可）",
        type=IMAGE_UPLOAD_TYPES,
        accept_multiple_files=True,
    )
    model_files = st.file_uploader(
        "模範解答の画像（任意・複数枚可）",
        type=IMAGE_UPLOAD_TYPES,
        accept_multiple_files=True,
    )

if st.button("保存", type="primary"):
    combined_paths = [
        storage.save_uploaded_image(item.getvalue(), item.name)
        for item in (combined_files or [])
    ]
    problem_paths = [
        storage.save_uploaded_image(item.getvalue(), item.name)
        for item in (problem_files or [])
    ]
    model_paths = [
        storage.save_uploaded_image(item.getvalue(), item.name)
        for item in (model_files or [])
    ]
    if not title.strip():
        st.error("問題タイトルは必須です。")
    elif not (
        problem_text.strip() or problem_paths or combined_paths
    ):
        st.error("問題文のテキストか画像が必要です。")
    elif not (correct_answer.strip() or model_paths or combined_paths):
        st.error("模範解答のテキストか画像が必要です。")
    else:
        problem = ProblemRecord(
            problem_id=storage.generate_id("prob"),
            title=title.strip(),
            problem_text=problem_text.strip(),
            correct_answer=correct_answer.strip(),
            rubric=(rubric.strip() or DEFAULT_RUBRIC),
            unit=unit.strip() or "未設定",
            difficulty=difficulty,
            created_at=utc_now_iso(),
            problem_input_kind=infer_input_kind(
                problem_text,
                problem_paths or combined_paths,
            ),
            answer_key_input_kind=infer_input_kind(
                correct_answer,
                model_paths or combined_paths,
            ),
            problem_image_paths=problem_paths,
            model_answer_image_paths=model_paths,
            combined_image_paths=combined_paths,
        )
        storage.save_problem(problem)
        st.success(f"問題を保存しました: {problem.problem_id}")
        if not problem.problem_text or not problem.correct_answer:
            st.info("テキストが空の項目は「OCR実行」の「問題・模範OCR」で読み取ってください。")

st.subheader("登録済み問題")
problems = storage.list_problems()
if not problems:
    st.info("まだ問題が登録されていません。")
else:
    for problem in reversed(problems):
        with st.expander(f"{problem.title} ({problem.problem_id})"):
            st.write(f"**単元:** {problem.unit} / **難易度:** {problem.difficulty}")
            st.write(
                f"**問題入力:** {problem.problem_input_kind} / "
                f"**模範入力:** {problem.answer_key_input_kind}"
            )
            show_problem_images(problem)
            st.write("**問題文**")
            st.text(problem.problem_text or "（未入力）")
            st.write("**模範解答**")
            st.text(problem.correct_answer or "（未入力）")
            st.write("**採点基準**")
            st.text(problem.rubric)
