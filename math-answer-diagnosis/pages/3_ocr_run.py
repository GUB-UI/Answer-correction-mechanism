"""OCR実行ページ"""

import streamlit as st

from src.adapters import create_ocr_adapter
from src.ocr_runner import GLMOCR_SDK_ERROR_MESSAGE, run_ocr_for_images
from src.sources import (
    apply_ocr_texts_to_problem,
    answer_image_paths,
    problem_ocr_targets,
    split_problem_and_solution,
)
from src.ui_common import (
    answer_label,
    get_config,
    get_storage,
    problem_label,
    provider_index,
    show_answer_image,
    show_problem_images,
)

OCR_PROVIDERS = ["mock", "glmocr_sdk"]

st.set_page_config(page_title="OCR実行", layout="wide")
st.title("OCR実行")
st.caption("答案画像だけでなく、問題文・模範解答の画像もここでテキスト化できます。数字・数式は UniMERNet Small を優先します。")

storage = get_storage()
config = get_config()
answers = storage.list_answers()
problems = {p.problem_id: p for p in storage.list_problems()}

provider = st.selectbox(
    "OCR provider",
    options=OCR_PROVIDERS,
    index=provider_index(OCR_PROVIDERS, config.ocr.provider),
    help="本番は glmocr_sdk（日本語は GLM-OCR、数字・数式は UniMERNet Small）。障害時のみ mock。",
)
ocr_engine = config.ocr.model_name if provider == "glmocr_sdk" else "mock"

tab_answer, tab_problem = st.tabs(["答案OCR", "問題・模範OCR"])


def _run_ocr(source_kind: str, source_id: str, image_paths: list[str]):
    adapter = create_ocr_adapter(provider, config)
    with st.spinner("OCR実行中..."):
        return run_ocr_for_images(
            adapter,
            storage,
            source_kind=source_kind,
            source_id=source_id,
            image_paths=image_paths,
            engine=ocr_engine,
        )


with tab_answer:
    image_answers = [a for a in answers if answer_image_paths(a)]
    if not image_answers:
        st.info("OCR対象の答案画像がありません。テキスト答案は登録時に診断入力へ入っています。")
    else:
        answer_map = {a.answer_id: a for a in image_answers}
        selected_answer_id = st.selectbox(
            "答案を選択",
            options=list(answer_map.keys()),
            format_func=lambda aid: answer_label(
                answer_map[aid], problems.get(answer_map[aid].problem_id)
            ),
        )
        answer = answer_map[selected_answer_id]
        problem = problems.get(answer.problem_id)
        if problem:
            st.write(f"**問題:** {problem_label(problem)}")
        show_answer_image(answer)

        existing_ocr = storage.get_ocr_results_for_answer(answer.answer_id)
        if existing_ocr:
            st.info(
                f"この答案には OCR 結果が {len(existing_ocr)} 件あります。"
                "再実行すると新しい結果が追加されます。"
            )

        if st.button("答案OCRを実行", type="primary"):
            try:
                result = _run_ocr(
                    "student_answer",
                    answer.answer_id,
                    answer_image_paths(answer),
                )
                st.success(f"OCR結果を保存しました: {result.ocr_id}")
                st.subheader("raw_text")
                st.code(result.raw_text)
                st.subheader("used_text")
                st.code(result.used_text)
            except Exception as exc:
                if provider == "glmocr_sdk":
                    st.error(GLMOCR_SDK_ERROR_MESSAGE)
                    with st.expander("エラー詳細"):
                        st.code(str(exc))
                else:
                    st.error(f"OCR実行に失敗しました: {exc}")

with tab_problem:
    pending = [p for p in problems.values() if problem_ocr_targets(p)]
    if not pending:
        st.info("問題・模範の画像が登録されていません。")
    else:
        problem_map = {p.problem_id: p for p in pending}
        selected_problem_id = st.selectbox(
            "問題を選択",
            options=list(problem_map.keys()),
            format_func=lambda pid: problem_label(problem_map[pid]),
        )
        problem = problem_map[selected_problem_id]
        show_problem_images(problem)
        st.write("**現在の問題文**")
        st.text(problem.problem_text or "（未入力）")
        st.write("**現在の模範解答**")
        st.text(problem.correct_answer or "（未入力）")

        fill_empty_only = st.checkbox(
            "空のテキストだけ埋める（既存テキストは上書きしない）",
            value=True,
        )
        if st.button("問題・模範OCRを実行", type="primary"):
            try:
                updated = problem
                for source_kind, paths in problem_ocr_targets(problem):
                    result = _run_ocr(source_kind, problem.problem_id, paths)
                    st.success(f"{source_kind}: {result.ocr_id}")
                    st.code(result.used_text)
                    if fill_empty_only:
                        updated = apply_ocr_texts_to_problem(updated, result)
                    elif source_kind == "combined":
                        problem_text, correct_answer = split_problem_and_solution(
                            result.used_text
                        )
                        updated = apply_ocr_texts_to_problem(updated, result)
                        updated = updated.model_copy(
                            update={
                                "problem_text": problem_text,
                                "correct_answer": correct_answer,
                            }
                        )
                    elif source_kind == "problem":
                        updated = apply_ocr_texts_to_problem(updated, result)
                        updated = updated.model_copy(
                            update={"problem_text": result.used_text}
                        )
                    elif source_kind == "model_answer":
                        updated = apply_ocr_texts_to_problem(updated, result)
                        updated = updated.model_copy(
                            update={"correct_answer": result.used_text}
                        )
                storage.update_problem(updated)
                st.info("問題レコードのテキストを更新しました。分割が粗い場合は問題登録のテキストを直してください。")
            except Exception as exc:
                if provider == "glmocr_sdk":
                    st.error(GLMOCR_SDK_ERROR_MESSAGE)
                    with st.expander("エラー詳細"):
                        st.code(str(exc))
                else:
                    st.error(f"OCR実行に失敗しました: {exc}")
