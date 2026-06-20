"""高校数学答案 OCR・診断LLMパイプライン — ホーム"""

import streamlit as st

from src.ui_common import get_config, get_storage

st.set_page_config(page_title="高校数学答案診断", layout="wide")
st.title("高校数学答案 OCR・診断LLMパイプライン")
st.caption("研究用MVP — 問題登録から診断結果確認まで")

storage = get_storage()
config = get_config()

problems = storage.list_problems()
answers = storage.list_answers()
ocr_results = storage.list_ocr_results()
diagnosis_results = storage.list_diagnosis_results()

col1, col2, col3, col4 = st.columns(4)
col1.metric("登録問題数", len(problems))
col2.metric("登録答案数", len(answers))
col3.metric("OCR結果数", len(ocr_results))
col4.metric("診断結果数", len(diagnosis_results))

st.subheader("現在の設定")
setting_col1, setting_col2 = st.columns(2)
setting_col1.info(
    f"**OCR provider:** `{config.ocr.provider}`  \n"
    f"**OCR model:** `{config.ocr.model_name}`"
)
setting_col2.info(
    f"**診断 provider:** `{config.diagnosis.provider}`  \n"
    f"**診断 model:** `{config.diagnosis.model_name}`"
)

st.subheader("使い方")
st.markdown(
    """
1. **問題登録** — 問題文・模範解答・採点基準を登録
2. **答案登録** — 答案画像と匿名化生徒IDを登録
3. **OCR実行** — 答案画像を読み取り、`raw_text` / `used_text` を保存
4. **OCR確認・修正** — 診断結果が不自然な場合に OCR を確認・修正
5. **診断実行** — `used_text` をもとに採点・フィードバックを生成
6. **診断結果確認** — 結果の確認と CSV エクスポート
"""
)
