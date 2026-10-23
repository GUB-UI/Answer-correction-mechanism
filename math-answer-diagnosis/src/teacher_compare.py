"""教師採点の取り込みと、システム（AI）採点との比較・不一致ログ・自動再チェック。

設計方針（DESIGN_PHILOSOPHY.md 8章）:
- 教師採点を「正解」として AI の結果を黙って書き換えない。
- 差が大きい項目だけ再チェック（例外だけ高コスト処理）し、それでも差が残れば
  教師確認へ回す。判断過程はすべて JSONL に残す。
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import re
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

from src.models import (
    AnswerRecord,
    DiagnosisResult,
    GradeDiscrepancy,
    OCRResult,
    ProblemRecord,
    TeacherGradeRecord,
    utc_now_iso,
)
from src.prompts import RECHECK_PROMPT_TYPE

TOTAL_LABELS = {"", "合計", "total", "計", "全体", "総合", "sum"}

COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "student": ("student_anonymized_id", "student_id", "匿名化生徒id", "生徒id", "生徒"),
    "problem_id": ("problem_id", "問題id"),
    "problem_title": ("problem_title", "問題タイトル", "title", "問題"),
    "question": ("question", "設問", "小問", "question_label"),
    "score": ("teacher_score", "教師得点", "得点", "score", "点数"),
    "max_score": ("max_score", "配点", "満点"),
    "comment": ("teacher_comment", "comment", "コメント", "備考"),
    "grader": ("grader", "採点者"),
}

TEACHER_CSV_TEMPLATE = (
    "student_anonymized_id,problem_id,question,teacher_score,max_score,teacher_comment\n"
    "student_001,prob_xxxxxxxx,(1),2,2,\n"
    "student_001,prob_xxxxxxxx,(2),1,3,等号の扱いが不十分\n"
)

STATUS_LABELS = {
    "open": "未再チェック",
    "recheck_agrees_with_teacher": "再チェックで教師採点に近づいた（差が閾値未満）",
    "recheck_keeps_gap": "再チェック後も差が残る（教師確認が必要）",
    "recheck_failed": "再チェック失敗",
}


def normalize_question_label(label: Any) -> str:
    """設問ラベルを比較用に正規化する。"(1)" "（１）" "問1" "1" はすべて "1"。合計は ""。"""
    text = unicodedata.normalize("NFKC", str(label or "")).strip().lower()
    if text in TOTAL_LABELS:
        return ""
    text = re.sub(r"^(設問|小問|問|第)", "", text)
    text = re.sub(r"[()\[\]{}【】「」\s.．、,:：問]", "", text)
    return "" if text in TOTAL_LABELS else text


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = unicodedata.normalize("NFKC", str(value)).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _decode_csv(data: bytes | str) -> str:
    if isinstance(data, str):
        return data
    for encoding in ("utf-8-sig", "cp932"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _column_map(fieldnames: Iterable[str]) -> dict[str, str]:
    normalized = {
        unicodedata.normalize("NFKC", name).strip().lower(): name for name in fieldnames if name
    }
    mapping: dict[str, str] = {}
    for key, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in normalized:
                mapping[key] = normalized[alias]
                break
    return mapping


@dataclass
class ImportIssue:
    row_number: int
    message: str


def parse_teacher_csv(
    data: bytes | str,
    problems: list[ProblemRecord],
    *,
    source_file: str = "",
    default_problem_id: str | None = None,
    imported_at: str | None = None,
    id_factory: Callable[[], str] | None = None,
) -> tuple[list[TeacherGradeRecord], list[ImportIssue]]:
    """縦持ち CSV（1行 = 生徒 × 設問）を TeacherGradeRecord に変換する。

    UTF-8（BOM 付き可）と Excel の Shift_JIS(cp932) に対応。列名は日本語でもよい。
    """
    text = _decode_csv(data)
    reader = csv.DictReader(io.StringIO(text))
    columns = _column_map(reader.fieldnames or [])
    issues: list[ImportIssue] = []
    for required in ("student", "score"):
        if required not in columns:
            label = "/".join(COLUMN_ALIASES[required][:3])
            issues.append(ImportIssue(1, f"必須列がありません: {label}"))
    if issues:
        return [], issues
    if "problem_id" not in columns and "problem_title" not in columns and not default_problem_id:
        return [], [ImportIssue(1, "problem_id 列（または問題タイトル列）か、取り込み先の問題指定が必要です")]

    by_id = {p.problem_id: p for p in problems}
    by_title: dict[str, list[ProblemRecord]] = defaultdict(list)
    for problem in problems:
        by_title[problem.title.strip()].append(problem)

    stamp = imported_at or utc_now_iso()
    counter = iter(range(1, 1_000_000))
    make_id = id_factory or (lambda: f"tg_{hashlib.sha1(f'{stamp}{next(counter)}'.encode()).hexdigest()[:8]}")
    grades: list[TeacherGradeRecord] = []
    for row_number, row in enumerate(reader, start=2):
        def cell(key: str) -> str:
            column = columns.get(key)
            return str(row.get(column) or "").strip() if column else ""

        student = unicodedata.normalize("NFKC", cell("student"))
        if not student:
            if any(str(v or "").strip() for v in row.values()):
                issues.append(ImportIssue(row_number, "生徒IDが空です"))
            continue
        if re.search(r"[\s/\\]", student):
            issues.append(ImportIssue(row_number, f"生徒IDに空白や区切り文字があります: {student}"))
            continue

        problem_id = cell("problem_id")
        if problem_id and problem_id not in by_id:
            issues.append(ImportIssue(row_number, f"未登録の problem_id です: {problem_id}"))
            continue
        if not problem_id and cell("problem_title"):
            matches = by_title.get(cell("problem_title"), [])
            if len(matches) == 1:
                problem_id = matches[0].problem_id
            else:
                reason = "見つかりません" if not matches else "同名の問題が複数あります"
                issues.append(
                    ImportIssue(row_number, f"問題タイトル「{cell('problem_title')}」が{reason}")
                )
                continue
        if not problem_id:
            problem_id = default_problem_id or ""
        if not problem_id:
            issues.append(ImportIssue(row_number, "問題を特定できません"))
            continue

        score = _to_float(cell("score"))
        if score is None:
            issues.append(ImportIssue(row_number, f"得点が数値ではありません: {cell('score')!r}"))
            continue
        max_score = _to_float(cell("max_score"))
        if max_score is not None and (score < 0 or score > max_score + 1e-9):
            issues.append(
                ImportIssue(row_number, f"得点 {score:g} が配点 {max_score:g} の範囲外です")
            )
            continue

        grades.append(
            TeacherGradeRecord(
                teacher_grade_id=make_id(),
                student_anonymized_id=student,
                problem_id=problem_id,
                question=unicodedata.normalize("NFKC", cell("question")),
                teacher_score=score,
                max_score=max_score,
                teacher_comment=cell("comment"),
                grader=cell("grader"),
                source_file=source_file,
                imported_at=stamp,
            )
        )
    return grades, issues


def derive_score_allocation(
    grades: list[TeacherGradeRecord],
) -> dict[str, list[dict[str, Any]]]:
    """教師採点の配点列から、問題ごとの設問配点を作る（AI に同じ配点で採点させるため）。"""
    allocation: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for grade in grades:
        label = normalize_question_label(grade.question)
        if not label or grade.max_score is None:
            continue
        allocation[grade.problem_id].setdefault(
            label, {"question": grade.question, "max_score": grade.max_score}
        )
    return {pid: list(items.values()) for pid, items in allocation.items()}


@dataclass
class ComparisonRow:
    comparison_key: str
    student_anonymized_id: str
    problem_id: str
    question: str
    teacher_grade_id: str
    teacher_score: float
    teacher_max: float | None
    teacher_comment: str = ""
    answer_id: str = ""
    diagnosis_id: str = ""
    ocr_id: str = ""
    model_name: str = ""
    system_score: float | None = None
    system_max: float | None = None
    system_reasoning: str = ""
    possible_ocr_issue: bool = False
    ocr_human_corrected: bool = False
    status: str = "matched"

    @property
    def matched(self) -> bool:
        return self.status == "matched" and self.system_score is not None

    @property
    def max_score(self) -> float | None:
        return self.teacher_max if self.teacher_max is not None else self.system_max

    @property
    def system_score_on_teacher_scale(self) -> float | None:
        if self.system_score is None:
            return None
        if (
            self.teacher_max is not None
            and self.system_max
            and abs(self.teacher_max - self.system_max) > 1e-9
        ):
            return self.system_score * self.teacher_max / self.system_max
        return self.system_score

    @property
    def diff(self) -> float | None:
        scaled = self.system_score_on_teacher_scale
        return None if scaled is None else scaled - self.teacher_score

    @property
    def ratio_diff(self) -> float | None:
        if self.system_score is None:
            return None
        teacher_max = self.teacher_max or self.system_max
        system_max = self.system_max or self.teacher_max
        if not teacher_max or not system_max:
            return None
        return self.system_score / system_max - self.teacher_score / teacher_max

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(
            system_score_on_teacher_scale=self.system_score_on_teacher_scale,
            diff=self.diff,
            ratio_diff=self.ratio_diff,
        )
        return data


def comparison_key(student: str, problem_id: str, question: str) -> str:
    return f"{student}|{problem_id}|{normalize_question_label(question) or 'total'}"


def system_score_for_question(
    diagnosis: DiagnosisResult, question: str
) -> tuple[float | None, float | None]:
    label = normalize_question_label(question)
    if not label:
        return diagnosis.score, diagnosis.max_score
    for item in diagnosis.question_scores:
        if normalize_question_label(item.get("question")) == label:
            return item.get("score"), item.get("max_score")
    return None, None


def select_system_diagnoses(
    answers: list[AnswerRecord],
    diagnoses: list[DiagnosisResult],
    *,
    model_name: str | None = None,
    prompt_types: set[str] | None = None,
) -> dict[tuple[str, str], DiagnosisResult]:
    """(生徒, 問題) ごとに比較対象の診断（再チェックを除く最新）を1件選ぶ。"""
    answer_map = {a.answer_id: a for a in answers}
    chosen: dict[tuple[str, str], DiagnosisResult] = {}
    for diagnosis in diagnoses:
        if diagnosis.prompt_type == RECHECK_PROMPT_TYPE:
            continue
        if prompt_types and diagnosis.prompt_type not in prompt_types:
            continue
        if model_name and diagnosis.model_name != model_name:
            continue
        answer = answer_map.get(diagnosis.answer_id)
        if answer is None:
            continue
        key = (answer.student_anonymized_id, answer.problem_id)
        current = chosen.get(key)
        if current is None or diagnosis.created_at >= current.created_at:
            chosen[key] = diagnosis
    return chosen


def build_comparisons(
    teacher_grades: list[TeacherGradeRecord],
    answers: list[AnswerRecord],
    diagnoses: list[DiagnosisResult],
    ocr_results: list[OCRResult] | None = None,
    *,
    model_name: str | None = None,
    prompt_types: set[str] | None = None,
) -> list[ComparisonRow]:
    selected = select_system_diagnoses(
        answers, diagnoses, model_name=model_name, prompt_types=prompt_types
    )
    ocr_map = {o.ocr_id: o for o in ocr_results or []}
    rows: list[ComparisonRow] = []
    for grade in teacher_grades:
        row = ComparisonRow(
            comparison_key=comparison_key(
                grade.student_anonymized_id, grade.problem_id, grade.question
            ),
            student_anonymized_id=grade.student_anonymized_id,
            problem_id=grade.problem_id,
            question=grade.question,
            teacher_grade_id=grade.teacher_grade_id,
            teacher_score=grade.teacher_score,
            teacher_max=grade.max_score,
            teacher_comment=grade.teacher_comment,
        )
        diagnosis = selected.get((grade.student_anonymized_id, grade.problem_id))
        if diagnosis is None:
            row.status = "no_system_grade"
            rows.append(row)
            continue
        score, max_score = system_score_for_question(diagnosis, grade.question)
        ocr = ocr_map.get(diagnosis.ocr_id)
        row.answer_id = diagnosis.answer_id
        row.diagnosis_id = diagnosis.diagnosis_id
        row.ocr_id = diagnosis.ocr_id
        row.model_name = diagnosis.model_name
        row.system_reasoning = diagnosis.reasoning_summary
        row.possible_ocr_issue = diagnosis.possible_ocr_issue
        row.ocr_human_corrected = bool(ocr and ocr.human_corrected)
        if score is None:
            row.status = "system_missing_question"
        else:
            row.system_score = float(score)
            row.system_max = float(max_score) if max_score is not None else None
        rows.append(row)
    return rows


def is_large_gap(
    row: ComparisonRow,
    ratio_threshold: float = 0.2,
    abs_threshold: float | None = None,
) -> bool:
    if not row.matched:
        return False
    ratio = row.ratio_diff
    if ratio is not None and abs(ratio) >= ratio_threshold - 1e-9:
        return True
    diff = row.diff
    if abs_threshold is not None and diff is not None and abs(diff) >= abs_threshold - 1e-9:
        return True
    return False


def pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0 or var_y <= 0:
        return None
    return cov / math.sqrt(var_x * var_y)


def quadratic_weighted_kappa(a: list[int], b: list[int]) -> float | None:
    """2評価者間の重み付きκ（二次重み）。全員同じ値で分散が無いときは None。"""
    if not a or len(a) != len(b):
        return None
    low = min(min(a), min(b))
    high = max(max(a), max(b))
    k = high - low + 1
    if k <= 1:
        return None
    observed = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        observed[x - low][y - low] += 1
    n = len(a)
    hist_a = [sum(row) for row in observed]
    hist_b = [sum(observed[i][j] for i in range(k)) for j in range(k)]
    numerator = 0.0
    denominator = 0.0
    for i in range(k):
        for j in range(k):
            weight = ((i - j) ** 2) / ((k - 1) ** 2)
            expected = hist_a[i] * hist_b[j] / n
            numerator += weight * observed[i][j]
            denominator += weight * expected
    if denominator == 0:
        return None
    return 1.0 - numerator / denominator


@dataclass
class AgreementMetrics:
    n_teacher_rows: int
    n_matched: int
    mae: float | None = None
    rmse: float | None = None
    bias: float | None = None
    exact_agreement_rate: float | None = None
    within_1pt_rate: float | None = None
    mean_abs_ratio_diff: float | None = None
    pearson_r: float | None = None
    quadratic_weighted_kappa: float | None = None
    large_gap_count: int = 0
    unmatched_reasons: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_metrics(
    rows: list[ComparisonRow],
    ratio_threshold: float = 0.2,
    abs_threshold: float | None = None,
) -> AgreementMetrics:
    matched = [row for row in rows if row.matched]
    unmatched: dict[str, int] = defaultdict(int)
    for row in rows:
        if not row.matched:
            unmatched[row.status] += 1
    metrics = AgreementMetrics(
        n_teacher_rows=len(rows),
        n_matched=len(matched),
        unmatched_reasons=dict(unmatched),
    )
    if not matched:
        return metrics
    teacher = [row.teacher_score for row in matched]
    system = [float(row.system_score_on_teacher_scale) for row in matched]
    diffs = [s - t for s, t in zip(system, teacher)]
    n = len(diffs)
    metrics.mae = sum(abs(d) for d in diffs) / n
    metrics.rmse = math.sqrt(sum(d * d for d in diffs) / n)
    metrics.bias = sum(diffs) / n
    metrics.exact_agreement_rate = sum(1 for d in diffs if abs(d) < 1e-6) / n
    metrics.within_1pt_rate = sum(1 for d in diffs if abs(d) <= 1 + 1e-9) / n
    ratios = [abs(row.ratio_diff) for row in matched if row.ratio_diff is not None]
    metrics.mean_abs_ratio_diff = sum(ratios) / len(ratios) if ratios else None
    metrics.pearson_r = pearson(system, teacher)
    metrics.quadratic_weighted_kappa = quadratic_weighted_kappa(
        [int(round(t)) for t in teacher], [int(round(s)) for s in system]
    )
    metrics.large_gap_count = sum(
        1 for row in matched if is_large_gap(row, ratio_threshold, abs_threshold)
    )
    return metrics


def metrics_by_question(
    rows: list[ComparisonRow],
    ratio_threshold: float = 0.2,
    abs_threshold: float | None = None,
) -> dict[str, AgreementMetrics]:
    groups: dict[str, list[ComparisonRow]] = defaultdict(list)
    for row in rows:
        label = normalize_question_label(row.question)
        groups[f"{row.problem_id}|{label or 'total'}"].append(row)
    return {
        key: compute_metrics(group, ratio_threshold, abs_threshold)
        for key, group in sorted(groups.items())
    }


def student_totals(rows: list[ComparisonRow]) -> list[dict[str, Any]]:
    """設問単位の行を生徒×問題で合計する（合計点だけの行は二重計上しない）。"""
    totals: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if not row.matched or not normalize_question_label(row.question):
            continue
        key = (row.student_anonymized_id, row.problem_id)
        item = totals.setdefault(
            key,
            {
                "student_anonymized_id": row.student_anonymized_id,
                "problem_id": row.problem_id,
                "teacher_total": 0.0,
                "system_total": 0.0,
                "max_total": 0.0,
                "n_questions": 0,
            },
        )
        item["teacher_total"] += row.teacher_score
        item["system_total"] += float(row.system_score_on_teacher_scale or 0.0)
        item["max_total"] += float(row.max_score or 0.0)
        item["n_questions"] += 1
    result = []
    for item in totals.values():
        item["diff"] = item["system_total"] - item["teacher_total"]
        result.append(item)
    return sorted(result, key=lambda x: (x["problem_id"], x["student_anonymized_id"]))


def discrepancy_id_for(row: ComparisonRow) -> str:
    digest = hashlib.sha1(f"{row.comparison_key}|{row.diagnosis_id}".encode()).hexdigest()
    return f"disc_{digest[:10]}"


def log_discrepancies(
    storage: Any,
    rows: list[ComparisonRow],
    *,
    ratio_threshold: float = 0.2,
    abs_threshold: float | None = None,
) -> list[GradeDiscrepancy]:
    """閾値を超えた項目を grade_discrepancies.jsonl に記録する（同じ項目は更新）。"""
    existing = {d.discrepancy_id: d for d in storage.list_discrepancies()}
    now = utc_now_iso()
    logged: list[GradeDiscrepancy] = []
    for row in rows:
        if not is_large_gap(row, ratio_threshold, abs_threshold):
            continue
        discrepancy_id = discrepancy_id_for(row)
        previous = existing.get(discrepancy_id)
        record = GradeDiscrepancy(
            discrepancy_id=discrepancy_id,
            comparison_key=row.comparison_key,
            student_anonymized_id=row.student_anonymized_id,
            problem_id=row.problem_id,
            question=row.question,
            teacher_grade_id=row.teacher_grade_id,
            answer_id=row.answer_id,
            diagnosis_id=row.diagnosis_id,
            ocr_id=row.ocr_id,
            teacher_score=row.teacher_score,
            system_score=float(row.system_score_on_teacher_scale or 0.0),
            max_score=row.max_score,
            diff=float(row.diff or 0.0),
            ratio_diff=row.ratio_diff,
            possible_ocr_issue=row.possible_ocr_issue,
            ocr_human_corrected=row.ocr_human_corrected,
            system_reasoning=row.system_reasoning,
            teacher_comment=row.teacher_comment,
            status=previous.status if previous else "open",
            recheck=previous.recheck if previous else None,
            detected_at=previous.detected_at if previous else now,
            updated_at=now,
        )
        storage.save_discrepancy(record)
        logged.append(record)
    return logged


def build_recheck_instructions(
    discrepancy: GradeDiscrepancy,
    original: DiagnosisResult,
    *,
    mode: str = "blind",
) -> str:
    """再チェック用の追加指示。

    blind: 教師の点数は見せない（AI が教師に迎合して点を合わせるのを防ぐ）。
    informed: 教師の点数・コメントも示し、答案の証拠からどちらが妥当かを判断させる。
    """
    total = f"{original.score:g}" if original.score is not None else "未確定"
    lines = [
        "【前回のAI採点（参考。誤っている可能性があります）】",
        f"合計: {total} / {original.max_score:g}",
    ]
    for item in original.question_scores:
        max_part = f" / {item['max_score']:g}" if item.get("max_score") is not None else ""
        score_text = f"{item['score']:g}" if item.get("score") is not None else "未確定"
        lines.append(f"{item['question']}: {score_text}{max_part}  {item.get('reason', '')}")
    if original.reasoning_summary:
        lines.append(f"前回の根拠: {original.reasoning_summary}")
    target = discrepancy.question or "答案全体"
    lines.append("")
    lines.append(f"【再チェック対象】{target}")
    if mode == "informed":
        max_part = (
            f" / {discrepancy.max_score:g}" if discrepancy.max_score is not None else ""
        )
        lines.append(f"教師の採点: {discrepancy.teacher_score:g}{max_part}")
        if discrepancy.teacher_comment:
            lines.append(f"教師コメント: {discrepancy.teacher_comment}")
        lines.append(
            "教師の採点が常に正しいとは限りません。答案（OCR結果）の証拠だけに基づいて採点し、"
            "教師とAIのどちらの判断を支持するか、その根拠を teacher_review_notes に書いてください。"
        )
    return "\n".join(lines)


def run_recheck(
    storage: Any,
    adapter: Any,
    discrepancy: GradeDiscrepancy,
    *,
    model_name: str,
    mode: str = "blind",
    ratio_threshold: float = 0.2,
    abs_threshold: float | None = None,
) -> GradeDiscrepancy:
    """差の大きい項目を再採点し、結果を新しい診断（prompt_type=recheck）として保存する。

    元の診断は上書きしない。再チェック後の差が閾値未満なら
    recheck_agrees_with_teacher、まだ大きければ recheck_keeps_gap（教師確認へ）。
    """
    from src.adapters.base import DiagnosisRunContext

    original = storage.get_diagnosis_result(discrepancy.diagnosis_id)
    problem = storage.get_problem(discrepancy.problem_id)
    ocr = storage.get_ocr_result(discrepancy.ocr_id) if discrepancy.ocr_id else None
    now = utc_now_iso()
    if original is None or problem is None or ocr is None:
        updated = discrepancy.model_copy(
            update={
                "status": "recheck_failed",
                "recheck": {"error": "元の診断・問題・OCR結果のいずれかが見つかりません", "at": now},
                "updated_at": now,
            }
        )
        storage.save_discrepancy(updated)
        return updated

    context = DiagnosisRunContext(
        answer_id=original.answer_id,
        problem_id=original.problem_id,
        ocr_id=ocr.ocr_id,
        problem=problem,
        used_text=ocr.used_text,
        model_name=model_name,
        prompt_type=RECHECK_PROMPT_TYPE,
        extra_instructions=build_recheck_instructions(discrepancy, original, mode=mode),
    )
    try:
        raw = adapter.run(context)
    except Exception as exc:
        updated = discrepancy.model_copy(
            update={
                "status": "recheck_failed",
                "recheck": {"error": str(exc), "mode": mode, "at": now},
                "updated_at": now,
            }
        )
        storage.save_discrepancy(updated)
        return updated
    raw["_recheck_of"] = original.diagnosis_id
    raw["_recheck_mode"] = mode
    raw["_discrepancy_id"] = discrepancy.discrepancy_id
    recheck_diag = adapter.to_diagnosis_result(
        context,
        raw,
        diagnosis_id=storage.generate_id("diag"),
        created_at=now,
    )
    storage.save_diagnosis_result(recheck_diag)

    score, max_score = system_score_for_question(recheck_diag, discrepancy.question)
    row = ComparisonRow(
        comparison_key=discrepancy.comparison_key,
        student_anonymized_id=discrepancy.student_anonymized_id,
        problem_id=discrepancy.problem_id,
        question=discrepancy.question,
        teacher_grade_id=discrepancy.teacher_grade_id,
        teacher_score=discrepancy.teacher_score,
        teacher_max=discrepancy.max_score,
        diagnosis_id=recheck_diag.diagnosis_id,
        system_score=float(score) if score is not None else None,
        system_max=float(max_score) if max_score is not None else None,
        status="matched" if score is not None else "system_missing_question",
    )
    still_large = row.matched and is_large_gap(row, ratio_threshold, abs_threshold)
    if not row.matched:
        status = "recheck_failed"
    elif still_large:
        status = "recheck_keeps_gap"
    else:
        status = "recheck_agrees_with_teacher"
    updated = discrepancy.model_copy(
        update={
            "status": status,
            "recheck": {
                "diagnosis_id": recheck_diag.diagnosis_id,
                "mode": mode,
                "system_score": row.system_score_on_teacher_scale,
                "diff": row.diff,
                "ratio_diff": row.ratio_diff,
                "possible_ocr_issue": recheck_diag.possible_ocr_issue,
                "teacher_review_notes": recheck_diag.teacher_review_notes,
                "at": now,
            },
            "possible_ocr_issue": discrepancy.possible_ocr_issue
            or recheck_diag.possible_ocr_issue,
            "updated_at": now,
        }
    )
    storage.save_discrepancy(updated)
    return updated
