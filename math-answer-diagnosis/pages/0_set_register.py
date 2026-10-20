"""資料を読み取り、ローカルLLMで同一問題に自動タグ付けする。"""
from __future__ import annotations

import hashlib
import json

import streamlit as st

from src.adapters import create_ocr_adapter
from src.intake import IntakeDocument, analyze_intake, assign_problem_tags, save_intake_batch
from src.models import DEFAULT_RUBRIC
from src.ocr_runner import run_ocr_for_images
from src.sources import IMAGE_UPLOAD_TYPES
from src.ui_common import get_config, get_storage, provider_index, problem_label, show_images


MODES = {
    "auto": "自動判別",
    "problem_student": "問題＋生徒の解答がセット",
    "problem_model": "問題＋模範解答がセット",
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


st.set_page_config(page_title="セット登録・自動タグ付け", layout="wide")
st.title("資料をまとめて登録")
st.caption("問題・生徒の解答・模範解答を読み取り、同じ問題に共通タグを自動で付けます。")

storage = get_storage()
config = get_config()
existing = storage.list_problems()

st.subheader("1. 資料を入力")
mode = st.radio("資料の構成", options=list(MODES), format_func=MODES.get,
                horizontal=True, key="intake_mode")
st.info("同じ紙に問題と解答が載っていても、そのまま入力できます。模範解答や生徒の解答は後から追加できます。")

inputs = []
slots = SLOTS[mode]
columns = st.columns(len(slots))
for column, (key, label, hint) in zip(columns, slots):
    with column:
        st.markdown(f"**{label}**")
        files = st.file_uploader(f"{label}の画像", type=IMAGE_UPLOAD_TYPES,
                                 accept_multiple_files=True, key=f"intake_{key}_files")
        text = st.text_area(f"{label}のテキスト", height=180, key=f"intake_{key}_text",
                            placeholder="ここに貼り付けるか、画像を追加してください。")
        inputs.append((hint, text.strip(), files or []))

student_id = st.text_input("匿名化生徒ID", key="intake_student_id",
                           placeholder="例: student_001（生徒の解答を保存する場合に必要）",
                           help="一回の登録では一人の生徒の解答を扱います。問題・模範のみの場合は空欄で保存できます。")
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
st.caption(f"ローカルLLMで分類・紐付け · 既存問題 {len(candidates)} 件と照合")

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

if st.button("解析して自動タグ付け", type="primary"):
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
    except Exception as exc:
        st.error(f"自動タグ付けできませんでした: {exc}")
        st.caption("ローカルLLMと、画像を使用する場合はOCRサーバーの起動・設定を確認してください。")

if "intake_batch" not in st.session_state:
    st.stop()

st.subheader("2. 読み取り・紐付けを確認")
batch = st.session_state["intake_batch"].model_copy(deep=True)
saved_tags = st.session_state.get("intake_saved_tags")
problem_map = {p.problem_id: p for p in existing}
tag_captions = []

for index, group in enumerate(batch.groups):
    extraction = group.extraction
    prefix = f"intake_{batch.batch_id}_{index}"
    with st.container(border=True):
        title = st.text_input("問題名", value=extraction.title, key=f"{prefix}_title", disabled=bool(saved_tags))
        with st.expander("紐付け先を修正する（必要な場合のみ）"):
            ids = [None, *problem_map]
            selected = st.selectbox("紐付け先", options=ids,
                                    index=ids.index(extraction.existing_problem_id),
                                    format_func=lambda pid: "新しい問題として登録" if pid is None else problem_label(problem_map[pid]),
                                    key=f"{prefix}_target", disabled=bool(saved_tags))
        target_changed = selected != extraction.existing_problem_id
        extraction.existing_problem_id = selected
        group.problem_id = selected or group.new_problem_id
        tag_captions.append((st.empty(), group))
        st.caption(f"LLMの確信度: {extraction.confidence:.0%}")
        for reason in group.review_reasons:
            st.warning(reason)
        columns = st.columns(3)
        changed = target_changed or title != extraction.title
        extraction.title = title
        for column, role, label in zip(columns,
                                      [extraction.problem, extraction.student_answer, extraction.model_answer],
                                      ["問題", "生徒の解答", "模範解答"]):
            with column:
                value = st.text_area(label, value=role.text, height=200,
                                     key=f"{prefix}_{label}", disabled=bool(saved_tags))
                changed = changed or value != role.text
                role.text = value
                sources = st.multiselect("引用元", options=[doc.source_id for doc in batch.documents],
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

st.subheader("3. 保存")
needs_review = any(group.review_reasons for group in batch.groups)
reviewed = st.checkbox("要確認の内容と紐付けを確認しました", key=f"intake_review_{batch.batch_id}",
                       disabled=bool(saved_tags)) if needs_review else False
if st.button("確認した内容を保存", type="primary", disabled=bool(saved_tags) or (needs_review and not reviewed)):
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
