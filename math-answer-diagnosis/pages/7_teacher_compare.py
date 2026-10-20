"""教師採点の取り込みと、システム採点との比較。"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.adapters import create_diagnosis_adapter
from src.models import utc_now_iso
from src.teacher_compare import (
    STATUS_LABELS,
    TEACHER_CSV_TEMPLATE,
    build_comparisons,
    compute_metrics,
    derive_score_allocation,
    log_discrepancies,
    metrics_by_question,
    parse_teacher_csv,
    run_recheck,
    student_totals,
)
from src.ui_common import get_config, get_storage, problem_label, provider_index, require_password

DIAGNOSIS_PROVIDERS = ["mock", "omlx", "colab_cuda"]

st.set_page_config(page_title="教師採点との比較", layout="wide")
require_password()
st.title("教師採点との比較")
st.caption(
    "教師の採点を CSV で取り込み、AI の採点と設問ごとに比べます。"
    "差が大きい項目は再チェックし、それでも差が残るものは教師確認に回します。"
    "AI の採点結果そのものは書き換えません。"
)

storage = get_storage()
config = get_config()
problems = storage.list_problems()

st.subheader("1. 教師採点を取り込む")
st.download_button(
    "CSV テンプレート",
    data=TEACHER_CSV_TEMPLATE.encode("utf-8-sig"),
    file_name="teacher_grades_template.csv",
    mime="text/csv",
)
st.caption("1行が「生徒 × 設問」です。Excel の Shift_JIS でも、列名は日本語でも読めます。")

uploaded = st.file_uploader("教師採点 CSV", type=["csv"])
problem_map = {p.problem_id: p for p in problems}
default_problem = st.selectbox(
    "CSV に問題列が無いときの問題",
    options=[""] + list(problem_map),
    format_func=lambda pid: "（CSV の列を使う）" if not pid else problem_label(problem_map[pid]),
)
apply_allocation = st.checkbox(
    "配点列を、まだ配点が無い問題の配点として保存する",
    value=True,
)

if st.button("取り込む", type="primary", disabled=uploaded is None):
    grades, issues = parse_teacher_csv(
        uploaded.getvalue(),
        problems,
        source_file=uploaded.name,
        default_problem_id=default_problem or None,
        imported_at=utc_now_iso(),
        id_factory=lambda: storage.generate_id("tg"),
    )
    for issue in issues:
        st.warning(f"{issue.row_number}行目: {issue.message}")
    if grades:
        storage.save_teacher_grades(grades)
        st.success(f"{len(grades)} 件を保存しました。")
        if apply_allocation:
            for problem_id, allocation in derive_score_allocation(grades).items():
                problem = storage.get_problem(problem_id)
                if problem is not None and not problem.score_allocation:
                    storage.update_problem(
                        problem.model_copy(update={"score_allocation": allocation})
                    )
                    st.info(f"{problem.title} の配点を設定しました。")

grades = storage.list_teacher_grades()
if not grades:
    st.info("まだ教師採点がありません。上の CSV を取り込んでください。")
    st.stop()

st.subheader("2. 比較")
ratio = st.slider(
    "差が大きいとみなす割合（満点に対する差）",
    min_value=0.05,
    max_value=1.0,
    value=float(config.comparison.ratio_threshold),
    step=0.05,
)
abs_threshold = st.number_input(
    "絶対点差の閾値（0 なら使わない）",
    min_value=0.0,
    value=float(config.comparison.abs_threshold or 0.0),
    step=0.5,
)
model_names = sorted({d.model_name for d in storage.list_diagnosis_results()})
model_filter = st.selectbox("比較する診断モデル", options=["（すべて）"] + model_names)

rows = build_comparisons(
    grades,
    storage.list_answers(),
    storage.list_diagnosis_results(),
    storage.list_ocr_results(),
    model_name=None if model_filter.startswith("（") else model_filter,
)
abs_value = abs_threshold or None
metrics = compute_metrics(rows, ratio, abs_value)
logged = log_discrepancies(storage, rows, ratio_threshold=ratio, abs_threshold=abs_value)

m1, m2, m3, m4 = st.columns(4)
m1.metric("突合できた項目", f"{metrics.n_matched} / {metrics.n_teacher_rows}")
m2.metric("平均絶対誤差", "—" if metrics.mae is None else f"{metrics.mae:.2f}")
m3.metric("完全一致", "—" if metrics.exact_agreement_rate is None else f"{metrics.exact_agreement_rate:.0%}")
m4.metric("差が大きい項目", metrics.large_gap_count)
extra = []
if metrics.pearson_r is not None:
    extra.append(f"相関係数 r = {metrics.pearson_r:.3f}")
if metrics.quadratic_weighted_kappa is not None:
    extra.append(f"重み付きκ = {metrics.quadratic_weighted_kappa:.3f}")
if metrics.bias is not None:
    extra.append(f"平均の偏り（AI − 教師）= {metrics.bias:+.2f}")
if metrics.within_1pt_rate is not None:
    extra.append(f"±1点以内 = {metrics.within_1pt_rate:.0%}")
if extra:
    st.caption(" / ".join(extra))
if metrics.unmatched_reasons:
    st.warning(f"突合できなかった行: {metrics.unmatched_reasons}")

by_question = metrics_by_question(rows, ratio, abs_value)
if by_question:
    st.write("**設問ごと**")
    st.dataframe(
        pd.DataFrame(
            [{"question": key, **item.as_dict()} for key, item in by_question.items()]
        ),
        use_container_width=True,
    )

totals = student_totals(rows)
if totals:
    st.write("**生徒ごとの合計（設問を足したもの）**")
    st.dataframe(pd.DataFrame(totals), use_container_width=True)

table = pd.DataFrame([row.as_dict() for row in rows])
st.dataframe(table, use_container_width=True)
st.download_button(
    "比較表を CSV でダウンロード",
    data=table.to_csv(index=False).encode("utf-8-sig"),
    file_name="grade_comparison.csv",
    mime="text/csv",
)

st.subheader("3. 差が大きい項目の再チェック")
st.caption(
    "再チェックは新しい診断として保存します。blind は教師の点数を AI に見せません。"
    "再チェック後も差が残る項目は、画面上で教師確認待ちになります。"
)
open_items = [item for item in logged if item.status == "open"]
st.write(f"未再チェック: {len(open_items)} 件 / 記録済み: {len(storage.list_discrepancies())} 件")
provider = st.selectbox(
    "再チェックの診断 provider",
    options=DIAGNOSIS_PROVIDERS,
    index=provider_index(DIAGNOSIS_PROVIDERS, config.diagnosis.provider),
)
mode = st.radio(
    "再チェックの情報量",
    options=["blind", "informed"],
    index=0 if config.comparison.recheck_mode != "informed" else 1,
    format_func=lambda key: "教師の点数を見せない" if key == "blind" else "教師の点数とコメントも見せる",
    horizontal=True,
)
if st.button("差が大きい項目を再チェック", disabled=not open_items):
    adapter = create_diagnosis_adapter(provider, config)
    model_name = config.diagnosis.model_name if provider != "mock" else "mock"
    progress = st.progress(0.0)
    for index, discrepancy in enumerate(open_items, start=1):
        updated = run_recheck(
            storage,
            adapter,
            discrepancy,
            model_name=model_name,
            mode=mode,
            ratio_threshold=ratio,
            abs_threshold=abs_value,
        )
        st.write(
            f"{updated.student_anonymized_id} {updated.question or '合計'}: "
            f"{STATUS_LABELS.get(updated.status, updated.status)}"
        )
        progress.progress(index / len(open_items))
    st.success("再チェックが終わりました。上の比較は再読み込みすると最新になります。")

history = storage.list_discrepancies()
if history:
    st.write("**不一致ログ**")
    frame = pd.DataFrame([item.model_dump() for item in history])
    frame["status_label"] = frame["status"].map(lambda value: STATUS_LABELS.get(value, value))
    st.dataframe(frame, use_container_width=True)
