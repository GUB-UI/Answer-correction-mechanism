"""Local answer-correction workspace with task-oriented navigation."""
from pathlib import Path

import streamlit as st

from src.ui_common import get_config, require_password
from src.ui_theme import apply_theme

st.set_page_config(page_title="答案ワークスペース", page_icon="✦", layout="wide")
apply_theme()
require_password()

ROOT = Path(__file__).resolve().parent
sections = {
    "資料の管理": [
        st.Page(ROOT / "pages/0_set_register.py", title="資料を登録", icon=":material/add_circle:", default=True),
        st.Page(ROOT / "pages/1_problem_register.py", title="問題ライブラリ", icon=":material/library_books:"),
        st.Page(ROOT / "pages/2_answer_register.py", title="生徒の解答", icon=":material/edit_document:"),
    ],
    "確認・診断": [
        st.Page(ROOT / "pages/3_ocr_run.py", title="画像の読み取り", icon=":material/document_scanner:"),
        st.Page(ROOT / "pages/4_ocr_review.py", title="読み取りを確認", icon=":material/fact_check:"),
        st.Page(ROOT / "pages/5_diagnosis_run.py", title="答案を診断", icon=":material/psychology:"),
        st.Page(ROOT / "pages/6_diagnosis_viewer.py", title="診断結果", icon=":material/insert_chart:"),
        st.Page(ROOT / "pages/7_teacher_compare.py", title="教師採点との比較", icon=":material/compare_arrows:"),
    ],
}
navigation = st.navigation(sections, position="hidden")
config = get_config()
with st.sidebar:
    st.html('<div class="ac-brand"><span class="ac-brand-mark">✦</span><strong>答案ワークスペース</strong></div>'
            '<p class="ac-brand-sub">ANSWER CORRECTION</p>')
    for label, pages in sections.items():
        st.html(f'<div class="ac-nav-section">{label}</div>')
        for page in pages:
            st.page_link(page)
    st.html('<div class="ac-local-note"><span>◈ ローカルAIを使用</span><br>'
            '資料の整理から、学びの振り返りまで。</div>')
    st.caption(f"データ: {config.data_dir}")
    if config.services.mode == "managed":
        st.caption("GPU のモデルは、読み取りと診断で1つずつ起動します。")
navigation.run()
