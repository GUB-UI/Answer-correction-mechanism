"""OCR実行ページ"""

import streamlit as st

from src.adapters import create_ocr_adapter
from src.adapters.base import OCRRunContext
from src.ui_common import (
    answer_label,
    get_config,
    get_storage,
    problem_label,
    resolve_image_path,
    show_answer_image,
)

GLMOCR_SDK_ERROR_MESSAGE = """GLM-OCR SDKの実行に失敗しました。
まずはMock OCRで動作確認してください。
GLM-OCRを使う場合は、公式SDKのインストールとGLMOCRSdkAdapterの設定を確認してください。"""

st.set_page_config(page_title="OCR実行", layout="wide")
st.title("OCR実行")

storage = get_storage()
config = get_config()
answers = storage.list_answers()
problems = {p.problem_id: p for p in storage.list_problems()}

if not answers:
    st.warning("先に答案を登録してください。")
    st.stop()

answer_map = {a.answer_id: a for a in answers}
selected_answer_id = st.selectbox(
    "答案を選択",
    options=list(answer_map.keys()),
    format_func=lambda aid: answer_label(answer_map[aid], problems.get(answer_map[aid].problem_id)),
)

answer = answer_map[selected_answer_id]
problem = problems.get(answer.problem_id)
if problem:
    st.write(f"**問題:** {problem_label(problem)}")

show_answer_image(answer)

existing_ocr = storage.get_ocr_results_for_answer(answer.answer_id)
if existing_ocr:
    st.info(f"この答案には OCR 結果が {len(existing_ocr)} 件あります。再実行すると新しい結果が追加されます。")

provider = st.selectbox(
    "OCR provider",
    options=["mock", "glmocr_sdk"],
    index=0,
    help="まずは mock で動作確認してください。glmocr_sdk は GLM-OCR 公式 SDK の導入が必要です。",
)

if st.button("OCRを実行", type="primary"):
    try:
        adapter = create_ocr_adapter(provider, config)
        image_path = resolve_image_path(answer.image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"画像が見つかりません: {image_path}")

        ocr_engine = config.ocr.model_name if provider == "glmocr_sdk" else "mock"
        with st.spinner("OCR実行中..."):
            result = adapter.run(
                OCRRunContext(
                    answer_id=answer.answer_id,
                    image_path=image_path,
                    ocr_engine=ocr_engine,
                )
            )
        result.ocr_id = storage.generate_id("ocr")
        storage.save_ocr_result(result)

        st.success(f"OCR結果を保存しました: {result.ocr_id}")
        st.subheader("raw_text")
        st.code(result.raw_text)
        st.subheader("used_text")
        st.code(result.used_text)
        if result.ocr_suspect:
            st.warning("OCR結果に不明箇所または疑わしい読み取りがあります。")
    except Exception as exc:
        if provider == "glmocr_sdk":
            st.error(GLMOCR_SDK_ERROR_MESSAGE)
            with st.expander("エラー詳細"):
                st.code(str(exc))
        else:
            st.error(f"OCR実行に失敗しました: {exc}")
