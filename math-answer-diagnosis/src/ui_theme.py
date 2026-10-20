"""Shared presentation for the local answer-correction workspace."""
from __future__ import annotations

from html import escape

import streamlit as st


def apply_theme() -> None:
    st.html("""
    <style>
    :root { --ac-ink:#172b35; --ac-muted:#62737e; --ac-line:#e2e8eb; --ac-teal:#0f766e; }
    .stApp { background:#f6f8fa; color:var(--ac-ink); }
    .stApp, .stApp input, .stApp textarea, .stApp button {
      font-family:Inter,"Noto Sans CJK JP","Hiragino Kaku Gothic ProN",sans-serif;
    }
    [data-testid="stMainBlockContainer"] { max-width:1440px; padding:4.4rem 3rem 4rem; }
    [data-testid="stHeader"] { background:rgba(246,248,250,.96); }
    [data-testid="stSidebar"] { background:#fff; border-right:1px solid var(--ac-line); }
    [data-testid="stSidebarUserContent"] { padding:1.8rem 1.25rem; }
    [data-testid="stSidebar"] [data-testid="stPageLink"] a {
      min-height:44px; padding:.65rem .8rem; border-radius:9px; color:#526571;
    }
    [data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"] {
      background:#eaf5f2; color:#0b685f; font-weight:650;
    }
    .ac-brand { display:flex; align-items:center; gap:10px; margin-bottom:5px; }
    .ac-brand-mark { display:grid; place-items:center; width:35px; height:35px;
      background:#0f766e; color:#fff; border-radius:11px; font-size:21px; font-weight:700; }
    .ac-brand strong { font-size:17px; letter-spacing:-.4px; }
    .ac-brand-sub { color:#7b8a94; font-size:11px; margin:0 0 35px 45px; }
    .ac-nav-section { margin:24px 10px 7px; color:#81909a; font-size:11px; font-weight:650; letter-spacing:.08em; }
    .ac-local-note { margin:36px 8px 0; border-top:1px solid var(--ac-line); padding-top:18px;
      color:#6c7d85; font-size:11px; line-height:1.9; }
    .ac-local-note span { color:#0f766e; }
    .ac-hero { display:flex; align-items:center; justify-content:space-between; gap:20px; margin:4px 0 25px; }
    .ac-eyebrow { color:#6b7d88; font-size:11px; font-weight:650; letter-spacing:.14em; margin-bottom:8px; }
    .ac-hero h1 { font-size:30px; font-weight:750; line-height:1.5; letter-spacing:-.8px; margin:0 0 7px; }
    .ac-hero p { color:var(--ac-muted); font-size:13px; line-height:1.9; margin:0; }
    .ac-pill { white-space:nowrap; border:1px solid #cce5df; border-radius:30px; padding:7px 12px;
      color:#12685f; background:#edf8f4; font-size:11px; font-weight:600; }
    .ac-steps { display:grid; grid-template-columns:repeat(3,1fr); background:#fff; border:1px solid var(--ac-line);
      border-radius:13px; padding:17px 22px; margin-bottom:28px; }
    .ac-step { display:flex; align-items:center; gap:11px; color:#778792; font-size:12px; position:relative; }
    .ac-step:not(:last-child)::after { content:""; height:1px; background:#e3e9ed; flex:1; margin:0 22px; }
    .ac-step b { display:grid; place-items:center; height:29px; width:29px; flex-shrink:0;
      border:1px solid #e0e7eb; border-radius:50%; background:#f8fafb; font-size:12px; }
    .ac-step.active { color:#0e6c63; font-weight:650; }
    .ac-step.active b, .ac-step.done b { color:#fff; background:#0f766e; border-color:#0f766e; }
    .ac-step.done { color:#3d665e; }
    [data-testid="stVerticalBlockBorderWrapper"] > div { border-color:var(--ac-line); border-radius:13px; }
    .st-key-intake_picker, .st-key-intake_summary, [class*="st-key-intake_source_"],
    [class*="st-key-intake_result_"] { background:#fff; border-radius:13px; }
    .st-key-intake_picker { padding:5px; }
    .st-key-intake_summary { padding:8px 6px; }
    .ac-section-title { margin:0 0 6px; font-size:17px; font-weight:700; letter-spacing:-.25px; }
    .ac-section-note { color:#6b7d88; margin:0 0 18px; font-size:12px; line-height:1.8; }
    .st-key-intake_picker [role="radiogroup"] { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; }
    .st-key-intake_picker [role="radiogroup"] > div[data-rac] {
      background:#fff; border:1px solid #dce5e9; border-radius:10px; padding:12px; margin:0; min-height:76px; position:relative;
    }
    .st-key-intake_picker [role="radiogroup"] > div[data-selected="true"] { background:#eff8f5; border-color:#168579; }
    .st-key-intake_picker [role="radiogroup"] > div[data-rac]:focus-within { outline:2px solid #0f766e; outline-offset:2px; }
    .st-key-intake_picker [data-testid="stRadioOption"] { width:100%; }
    .st-key-intake_picker [data-testid="stRadioOption"]::after { content:""; position:absolute; inset:0; border-radius:10px; }
    .st-key-intake_picker [data-testid="stRadio"] label[data-baseweb="radio"] {
      background:#fff; border:1px solid #dce5e9; border-radius:10px; padding:14px 12px; margin:0;
      min-height:70px; transition:border-color .15s,background .15s;
    }
    .st-key-intake_picker label[data-baseweb="radio"]:has(input:checked) { background:#eff8f5; border-color:#168579; }
    .st-key-intake_picker label[data-baseweb="radio"]:focus-within { outline:2px solid #0f766e; outline-offset:3px; }
    [data-testid="stRadio"] p { font-size:12px; }
    [data-testid="stWidgetLabel"] p { font-size:12px; font-weight:550; color:#364d59; }
    [data-testid="stCaptionContainer"] p { color:#6a7d89; font-size:11px; line-height:1.8; overflow-wrap:anywhere; }
    [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea { font-size:13px; }
    [data-testid="stTextArea"] textarea { line-height:1.9; }
    [data-testid="stFileUploaderDropzone"] { border:1px dashed #b9cbd0; background:#f8fbfb; border-radius:10px; }
    [data-testid="stFileUploaderDropzone"] button { font-size:12px; }
    [data-testid="stButton"] button { min-height:43px; font-size:12px; border-radius:9px; font-weight:600; }
    [data-testid="stBaseButton-primary"] { background:#0f766e; border-color:#0f766e; }
    [data-testid="stBaseButton-primary"]:hover { background:#0c655e; border-color:#0c655e; }
    [data-testid="stExpander"] { background:#fff; border-radius:10px; }
    [data-testid="stExpander"] summary { font-size:12px; }
    .ac-source-label { color:#69838d; font-size:10px; font-weight:650; letter-spacing:.1em; margin:0 0 7px; }
    .ac-source-title { color:#203b47; font-size:14px; font-weight:650; margin:0 0 14px; }
    .ac-summary-row { display:flex; justify-content:space-between; align-items:center; padding:12px 0;
      border-bottom:1px solid #edf1f3; font-size:12px; color:#617580; }
    .ac-summary-row strong { color:#233f4a; font-size:13px; }
    .ac-summary-help { font-size:11px; line-height:1.9; color:#6b7f89; margin:15px 0 5px; }
    .ac-review-heading { display:flex; align-items:center; justify-content:space-between; gap:12px; margin:16px 0; }
    .ac-review-heading h2 { margin:0; font-size:21px; letter-spacing:-.4px; }
    .ac-role-label { display:inline-flex; align-items:center; gap:7px; font-size:12px; font-weight:650; margin-bottom:5px; }
    .ac-role-label::before { content:""; width:7px; height:7px; border-radius:50%; background:#0f766e; }
    .ac-role-label.student::before { background:#4176bb; }
    .ac-role-label.model::before { background:#8971b0; }
    .ac-save-note { color:#647884; font-size:12px; line-height:1.9; }
    @media(max-width:1000px) {
      [data-testid="stMainBlockContainer"] { padding:4.2rem 1.5rem 3rem; }
      .ac-hero h1 { font-size:26px; }
      .ac-hero > .ac-pill { display:none; }
      .ac-step:not(:last-child)::after { margin:0 12px; }
    }
    @media(max-width:640px) {
      [data-testid="stMainBlockContainer"] { padding:4rem 1rem 3rem; }
      .ac-hero { margin:4px 0 20px; } .ac-hero h1 { font-size:23px; }
      .ac-hero p { font-size:12px; }
      .ac-steps { padding:13px 10px; margin-bottom:18px; }
      .ac-step { gap:6px; font-size:10px; align-items:center; }
      .ac-step b { height:23px; width:23px; font-size:10px; }
      .ac-step:not(:last-child)::after { display:none; }
      .st-key-intake_picker [role="radiogroup"] { gap:7px; }
      .st-key-intake_picker [data-testid="stRadio"] label[data-baseweb="radio"] { padding:11px 8px; }
    }
    </style>
    """)


def intake_hero() -> None:
    st.html("""<div class="ac-hero"><div>
      <div class="ac-eyebrow">WORKSPACE / 資料登録</div>
      <h1>答案の整理を、もっとかんたんに。</h1>
      <p>資料をそのまま追加。同じ問題の「問題・生徒の解答・模範解答」を、自動でひとまとめに。</p>
    </div><span class="ac-pill">✦ ローカルAIで自動整理</span></div>""")


def intake_steps(slot, stage: int) -> None:
    labels = ["資料を追加", "読み取り結果を確認", "保存して次へ"]
    items = []
    for index, label in enumerate(labels, 1):
        state = "done" if index < stage else "active" if index == stage else ""
        mark = "✓" if index < stage else str(index)
        items.append(f'<div class="ac-step {state}"><b>{mark}</b><span>{escape(label)}</span></div>')
    slot.html('<div class="ac-steps" aria-label="登録の進み具合">' + "".join(items) + '</div>')
