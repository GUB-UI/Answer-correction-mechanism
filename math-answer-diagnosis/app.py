"""高校数学答案 OCR・診断LLMパイプライン — ホーム"""

import streamlit as st

from src.ui_common import require_password, get_config, get_storage

st.set_page_config(page_title="高校数学答案診断", layout="wide")
require_password()
st.title("高校数学答案 OCR・診断LLMパイプライン")
st.caption("研究用MVP — 問題登録から診断結果確認まで")

storage = get_storage()
config = get_config()

problems = storage.list_problems()
answers = storage.list_answers()
ocr_results = storage.list_ocr_results()
diagnosis_results = storage.list_diagnosis_results()

teacher_grades = storage.list_teacher_grades()
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("登録問題数", len(problems))
col2.metric("登録答案数", len(answers))
col3.metric("OCR結果数", len(ocr_results))
col4.metric("診断結果数", len(diagnosis_results))
col5.metric("教師採点", len(teacher_grades))

st.subheader("現在の設定")
setting_col1, setting_col2 = st.columns(2)
setting_col1.info(
    f"**OCR provider:** `{config.ocr.provider}`  \n"
    f"**OCR model:** `{config.ocr.model_name}`  \n"
    f"**数式:** `{'UniMERNet Small' if config.formula.enabled else 'off'}`  \n"
    f"**データ:** `{config.data_dir}`"
)
setting_col2.info(
    f"**診断 provider:** `{config.diagnosis.provider}`  \n"
    f"**診断 model:** `{config.diagnosis.model_name}`  \n"
    f"**実行モード:** `{config.services.mode}`"
)
if config.services.mode == "managed":
    from src.runtime.services import get_service_manager

    st.caption("GPU モデルは1つずつ起動します（OCR → UniMERNet → 診断LLM）。")
    st.dataframe(get_service_manager(config).status(), use_container_width=True)

st.subheader("使い方")
st.markdown(
    """
1. **セット登録** — 問題・模範・答案をテキスト/画像/混在でまとめて登録（推奨）
2. **問題登録 / 答案登録** — 個別に追加・補完する場合
3. **OCR実行** — 答案画像、および問題・模範の画像をテキスト化
4. **OCR確認・修正** — 手書きの誤読が疑われる場合に `used_text` を直す
5. **診断実行** — `used_text` をもとに採点・フィードバックを生成
6. **診断結果確認** — 結果の確認と CSV エクスポート
7. **教師採点との比較** — 教師の採点を取り込み、設問ごとの差・一致率を見る。差が大きい項目は再チェック
"""
)
