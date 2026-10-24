"""資料を読み取り、ローカルLLMで同一問題に自動タグ付けする。"""
from __future__ import annotations

import hashlib
import json
from html import escape

import streamlit as st

from src.adapters import create_ocr_adapter
from src.intake import IntakeDocument, analyze_intake, assign_problem_tags, save_intake_batch
from src.models import DEFAULT_RUBRIC
from src.ocr_runner import run_ocr_for_images
from src.sources import IMAGE_UPLOAD_TYPES
from src.ui_common import get_config, get_storage, problem_label, provider_index, require_password, show_images
from src.ui_theme import apply_theme, intake_hero, intake_steps


MODES = {
    "auto": "おまかせで整理",
    "problem_student": "問題＋生徒の解答",
    "problem_model": "問題＋模範解答",
    "separate": "それぞれ別の資料",
}
SLOTS = {
    "auto": [("auto", "資料", "auto")],
    "problem_student": [("pair_student", "問題＋生徒の解答", "problem_student"),
                        ("model", "模範解答", "model_answer")],
    "problem_model": [("pair_model", "問題＋模範解答", "problem_model"),
                      ("student", "生徒の解答", "student_answer")],
    "separate": [("problem", "問題", "problem"), ("student", "生徒の解答", "student_answer"),
                 ("model", "模範解答", "model_answer")],
}


def _clear_intake():
    for key in list(st.session_state):
        if key.startswith("intake_"):
            del st.session_state[key]


st.set_page_config(page_title="資料登録・自動タグ付け", layout="wide")
apply_theme()
require_password()
intake_hero()
progress_slot = st.empty()

storage = get_storage()
config = get_config()
existing = storage.list_problems()

with st.expander("資料と登録情報", expanded="intake_batch" not in st.session_state):
    input_column, action_column = st.columns([2.8, 1], gap="large")
    with input_column:
        with st.container():
            with st.container(border=True, key="intake_picker"):
                st.html('<h2 class="ac-section-title">どんな資料を追加しますか？</h2>'
                        '<p class="ac-section-note">問題と解答が同じ紙にあっても、そのままで大丈夫です。</p>')
                mode = st.radio("資料の構成", options=list(MODES), format_func=MODES.get,
                                captions=["組み合わせが分からなくてもOK", "記入済みの答案用紙など", "問題集・解説プリントなど", "役割ごとに分けて追加"],
                            horizontal=True, label_visibility="collapsed", width="stretch", key="intake_mode")
            inputs = []
            slots = SLOTS[mode]
            columns = st.columns(len(slots))
            for index, (column, (key, label, hint)) in enumerate(zip(columns, slots), 1):
                with column, st.container(border=True, key=f"intake_source_{key}"):
                    st.html(f'<div class="ac-source-label">資料 {index:02d}</div>'
                            f'<div class="ac-source-title">{escape(label)}</div>')
                    files = st.file_uploader(f"{label}の画像", type=IMAGE_UPLOAD_TYPES,
                                             accept_multiple_files=True, key=f"intake_{key}_files",
                                             label_visibility="collapsed")
                    with st.expander("テキストを貼り付ける", expanded=bool(st.session_state.get(f"intake_{key}_text"))):
                        text = st.text_area(f"{label}のテキスト", height=145, key=f"intake_{key}_text",
                                            placeholder="問題や解答をここに貼り付けてください。", label_visibility="collapsed")
                    inputs.append((hint, text.strip(), files or []))
            st.caption("PNG・JPG・WebPに対応。複数枚を追加できます。不足している資料は、後から追加できます。")

    action_panel = action_column.container(border=True, key="intake_summary")
    with action_panel:
        st.html('<h2 class="ac-section-title">登録の準備</h2>')
        summary_slot = st.empty()
        student_id = st.text_input("匿名化生徒ID", key="intake_student_id", placeholder="例: student_001",
                                   help="生徒の解答を保存する場合に必要です。一回の登録では一人分を扱います。")
        st.caption("問題・模範解答だけなら、生徒IDは空欄で登録できます。")
        with st.expander("登録・読み取り設定"):
            unit = st.text_input("単元（任意）", key="intake_unit")
            rubric = st.text_area("採点基準（任意）", placeholder=DEFAULT_RUBRIC, key="intake_rubric")
            providers = ["glmocr_sdk", "mock"]
            provider = st.selectbox("画像の読み取り", options=providers,
                                    index=provider_index(providers, config.ocr.provider),
                                    format_func=lambda p: {"glmocr_sdk": "GLM-OCR（ローカル）", "mock": "動作確認用のダミーOCR"}[p],
                                    key="intake_provider")
            candidate_filter = st.text_input("既存問題の検索範囲（任意）", key="intake_candidate_filter",
                                             help="問題名・単元・タグで絞り込みます。空欄では全ての既存問題を照合します。")
            st.caption(f"自動タグ付けモデル: {config.tagging_model_name or config.diagnosis.model_name}")

    candidates = [p for p in existing if not candidate_filter.strip() or candidate_filter.strip().lower()
                  in f"{p.title} {p.unit} {p.problem_id}".lower()]
    document_count = sum(bool(text) + len(files) for _, text, files in inputs)
    summary_slot.html(f'<div class="ac-summary-row"><span>追加した資料</span><strong>{document_count} 点</strong></div>'
                      f'<div class="ac-summary-row"><span>照合する既存問題</span><strong>{len(candidates)} 問</strong></div>'
                      '<p class="ac-summary-help">問題と解答の対応を読み取り、同じ問題の資料に共通タグを自動で付けます。</p>')

# A preview belongs to these exact inputs. Changing files, mode, or scope invalidates it.
fingerprint = hashlib.sha256(json.dumps({
    "mode": mode, "provider": provider, "scope": candidate_filter,
    "endpoint": config.lmstudio.base_url,
    "model": config.tagging_model_name or config.diagnosis.model_name,
    "inputs": [(hint, text, [(f.name, hashlib.sha256(f.getvalue()).hexdigest()) for f in files])
               for hint, text, files in inputs],
}, ensure_ascii=False).encode()).hexdigest()
if st.session_state.get("intake_fingerprint") != fingerprint:
    if "intake_batch" in st.session_state:
        st.info("資料または設定が変わりました。もう一度解析して自動タグ付けしてください。")
    st.session_state.pop("intake_batch", None)
    st.session_state.pop("intake_saved_tags", None)

stage = 3 if st.session_state.get("intake_saved_tags") else 2 if "intake_batch" in st.session_state else 1
intake_steps(progress_slot, stage)
with action_panel:
    analyze_pressed = st.button("解析して自動タグ付け", icon=":material/auto_awesome:",
                                 type="primary" if stage == 1 else "secondary", width="stretch")
    st.caption("読み取り結果を確認してから保存できます。")
if analyze_pressed:
    st.session_state.pop("intake_batch", None)
    st.session_state.pop("intake_saved_tags", None)
    try:
        documents = []
        adapter = None
        with st.spinner("資料を読み取り、同じ問題を紐付けています…"):
            for hint, text, files in inputs:
                if text:
                    documents.append(IntakeDocument(source_id=f"doc_{len(documents) + 1}",
                                                     text=text, hint=hint))
                for file in files:
                    if adapter is None:
                        adapter = create_ocr_adapter(provider, config)
                    path = storage.save_uploaded_image(file.getvalue(), file.name)
                    source_id = f"doc_{len(documents) + 1}"
                    ocr = run_ocr_for_images(
                        adapter, storage, source_kind="intake_document", source_id=storage.generate_id("doc"),
                        image_paths=[path], engine=config.ocr.model_name if provider == "glmocr_sdk" else "mock",
                    )
                    documents.append(IntakeDocument(source_id=source_id, text=ocr.used_text,
                                                     hint=hint, image_paths=[path], ocr_ids=[ocr.ocr_id]))
            if not documents:
                raise ValueError("テキストか画像を追加してください。")
            if any(files for _, _, files in inputs) and provider == "mock":
                st.warning("画像はダミーOCRで読み取っています。実際の資料にはGLM-OCRを選択してください。")
            batch = analyze_intake(config, documents, candidates)
        st.session_state["intake_batch"] = batch
        st.session_state["intake_fingerprint"] = fingerprint
        st.rerun()
    except Exception as exc:
        st.error(f"自動タグ付けできませんでした: {exc}")
        st.caption("ローカルLLMと、画像を使用する場合はOCRサーバーの起動・設定を確認してください。")

if "intake_batch" not in st.session_state:
    st.stop()

batch = st.session_state["intake_batch"].model_copy(deep=True)
saved_tags = st.session_state.get("intake_saved_tags")
problem_map = {p.problem_id: p for p in existing}
tag_captions = []
st.html('<div class="ac-review-heading"><h2>読み取り結果を確認</h2>'
        f'<span class="ac-pill">{len(batch.groups)} 問に整理しました</span></div>')
st.caption("同じカードの内容を、ひとつの問題として保存します。気になる箇所は、その場で修正できます。")

for index, group in enumerate(batch.groups):
    extraction = group.extraction
    prefix = f"intake_{batch.batch_id}_{index}"
    with st.container(border=True, key=f"intake_result_{index}"):
        result_heading = st.empty()
        tag_captions.append((st.empty(), group))
        with st.expander("問題名・紐付け先を編集する"):
            title = st.text_input("問題名", value=extraction.title, key=f"{prefix}_title", disabled=bool(saved_tags))
            ids = [None, *problem_map]
            selected = st.selectbox("紐付け先", options=ids,
                                    index=ids.index(extraction.existing_problem_id),
                                    format_func=lambda pid: "新しい問題として登録" if pid is None else problem_label(problem_map[pid]),
                                    key=f"{prefix}_target", disabled=bool(saved_tags))
            st.caption(f"モデルが返した確信度（自己評価）: {extraction.confidence:.0%}")
        result_heading.html(f'<h3 class="ac-section-title">問題 {index + 1:02d} · {escape(title)}</h3>')
        target_changed = selected != extraction.existing_problem_id
        extraction.existing_problem_id = selected
        group.problem_id = selected or group.new_problem_id
        for reason in group.review_reasons:
            st.warning(reason)
        columns = st.columns(3)
        changed = target_changed or title != extraction.title
        extraction.title = title
        for column, role, label, color in zip(columns,
                                      [extraction.problem, extraction.student_answer, extraction.model_answer],
                                      ["問題", "生徒の解答", "模範解答"], ["problem", "student", "model"]):
            with column:
                st.html(f'<div class="ac-role-label {color}">{label}</div>')
                value = st.text_area(label, value=role.text, height=200,
                                     key=f"{prefix}_{label}", disabled=bool(saved_tags), label_visibility="collapsed")
                changed = changed or value != role.text
                role.text = value
                source_labels = {doc.source_id: f"資料 {number:02d}" for number, doc in enumerate(batch.documents, 1)}
                with st.expander("引用元を確認・修正"):
                    sources = st.multiselect("引用元", options=list(source_labels), format_func=source_labels.get,
                                              default=role.source_ids, key=f"{prefix}_{label}_sources",
                                              disabled=bool(saved_tags),
                                              help="役割を修正した場合は、元の資料も確認できます。")
                changed = changed or sources != role.source_ids
                role.source_ids = sources
        if not extraction.model_answer.text.strip() and not (selected and problem_map[selected].correct_answer.strip()):
            st.info("模範解答は未登録です。後から追加して同じタグに紐付けられます。")
        if changed:
            group.human_corrected = True
            group.review_reasons.append("読み取り内容または紐付け先を修正しました。")

assign_problem_tags(batch)
for placeholder, group in tag_captions:
    status = "既存問題に追加" if group.extraction.existing_problem_id else "新規問題"
    placeholder.caption(f"共通タグ: {group.problem_id} · {status}")

with st.expander("元資料を確認する"):
    for doc in batch.documents:
        st.markdown(f"**{doc.source_id}**")
        show_images(doc.image_paths)
        st.text(doc.text)

needs_review = any(group.review_reasons for group in batch.groups)
with st.container(border=True):
    save_note, save_action = st.columns([2, 1], vertical_alignment="center")
    with save_note:
        st.html('<div class="ac-section-title">この内容で保存しますか？</div>'
                '<div class="ac-save-note">共通タグで整理して保存し、生徒の解答を診断に引き継ぎます。</div>')
        if any(group.extraction.student_answer.text.strip() for group in batch.groups):
            st.caption(f"生徒ID: {student_id or '未入力（上の「資料と登録情報」から入力してください）'}")
        reviewed = st.checkbox("要確認の内容と紐付けを確認しました", key=f"intake_review_{batch.batch_id}",
                               disabled=bool(saved_tags)) if needs_review else False
    with save_action:
        save_pressed = st.button("確認した内容を保存", icon=":material/check:", type="primary", width="stretch",
                                  disabled=bool(saved_tags) or (needs_review and not reviewed))
if save_pressed:
    try:
        tags = save_intake_batch(storage, batch, student_id=student_id, reviewed=reviewed,
                                 rubric=rubric, unit=unit)
        st.session_state["intake_saved_tags"] = tags
        st.rerun()
    except Exception as exc:
        st.error(f"保存できませんでした: {exc}")
if saved_tags:
    st.success("保存しました。共通タグ: " + ", ".join(saved_tags))
    st.caption("生徒の解答は、抽出した内容でOCR確認・診断へ進めます。")
    st.button("次の資料を登録", on_click=_clear_intake)
