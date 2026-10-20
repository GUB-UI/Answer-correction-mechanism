from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.adapters.base import DiagnosisRunContext
from src.config import load_config
from src.models import (
    AnswerRecord,
    DiagnosisResult,
    OCRResult,
    ProblemRecord,
    utc_now_iso,
)
from src.storage import Storage
from src.teacher_compare import (
    build_comparisons,
    build_recheck_instructions,
    compute_metrics,
    derive_score_allocation,
    is_large_gap,
    log_discrepancies,
    normalize_question_label,
    parse_teacher_csv,
    pearson,
    quadratic_weighted_kappa,
    run_recheck,
    student_totals,
)


@pytest.fixture
def storage(tmp_path: Path) -> Storage:
    root = tmp_path / "project"
    root.mkdir()
    (root / "config.yaml").write_text(
        yaml.safe_dump({"runtime": {"device": "test"}, "comparison": {"ratio_threshold": 0.2}}),
        encoding="utf-8",
    )
    return Storage(load_config(root))


def _problem(problem_id: str = "prob_1", title: str = "不等式") -> ProblemRecord:
    return ProblemRecord(
        problem_id=problem_id,
        title=title,
        problem_text="問題",
        correct_answer="解答",
        rubric="基準",
        unit="数学I",
        created_at=utc_now_iso(),
        score_allocation=[{"question": "(1)", "max_score": 4}, {"question": "(2)", "max_score": 6}],
    )


def _diagnosis(
    *,
    diagnosis_id: str,
    answer_id: str,
    scores: list[tuple[str, float, float]],
    created_at: str = "2026-10-01T00:00:00+00:00",
    prompt_type: str = "step_by_step_check",
    model_name: str = "Qwen3.8-27B-bnb4",
    possible_ocr_issue: bool = False,
) -> DiagnosisResult:
    return DiagnosisResult(
        diagnosis_id=diagnosis_id,
        answer_id=answer_id,
        problem_id="prob_1",
        ocr_id=f"ocr_{answer_id}",
        model_name=model_name,
        prompt_type=prompt_type,
        score=sum(item[1] for item in scores),
        max_score=sum(item[2] for item in scores),
        error_categories=[],
        error_locations=[],
        reasoning_summary="方針は妥当",
        student_feedback="確認を",
        teacher_review_notes="",
        possible_ocr_issue=possible_ocr_issue,
        confidence=0.8,
        created_at=created_at,
        question_scores=[
            {"question": q, "score": s, "max_score": m, "reason": ""} for q, s, m in scores
        ],
    )


def test_normalize_question_label_folds_variants() -> None:
    assert normalize_question_label("(1)") == "1"
    assert normalize_question_label("（１）") == "1"
    assert normalize_question_label("問1") == "1"
    assert normalize_question_label("合計") == ""
    assert normalize_question_label("  ") == ""


def test_parse_teacher_csv_accepts_japanese_headers_and_cp932(storage: Storage) -> None:
    problem = _problem()
    storage.save_problem(problem)
    text = "生徒ID,問題タイトル,設問,得点,配点,コメント\nstudent_001,不等式,（1）,4,4,\nstudent_001,不等式,(2),2,6,等号\n"
    grades, issues = parse_teacher_csv(text.encode("cp932"), storage.list_problems(), source_file="t.csv")
    assert issues == []
    assert len(grades) == 2
    assert grades[0].question == "(1)"
    assert grades[1].teacher_comment == "等号"
    allocation = derive_score_allocation(grades)
    assert allocation["prob_1"][0]["max_score"] == 4


def test_parse_teacher_csv_reports_row_problems(storage: Storage) -> None:
    storage.save_problem(_problem())
    text = (
        "student_anonymized_id,problem_id,question,teacher_score,max_score\n"
        ",prob_1,(1),1,4\n"
        "bad id,prob_1,(1),1,4\n"
        "student_001,missing,(1),1,4\n"
        "student_001,prob_1,(1),9,4\n"
        "student_001,prob_1,(1),x,4\n"
    )
    grades, issues = parse_teacher_csv(text, storage.list_problems())
    assert grades == []
    assert len(issues) == 5


def test_parse_teacher_csv_requires_score_column() -> None:
    grades, issues = parse_teacher_csv("student_anonymized_id\nstudent_001\n", [])
    assert grades == []
    assert issues


def test_compare_matches_per_question_and_scales_totals(storage: Storage) -> None:
    answer = AnswerRecord(
        answer_id="ans_1",
        problem_id="prob_1",
        student_anonymized_id="student_001",
        created_at=utc_now_iso(),
    )
    older = _diagnosis(
        diagnosis_id="diag_old",
        answer_id="ans_1",
        scores=[("(1)", 0, 4), ("(2)", 0, 6)],
        created_at="2026-09-01T00:00:00+00:00",
    )
    latest = _diagnosis(
        diagnosis_id="diag_new",
        answer_id="ans_1",
        scores=[("(1)", 4, 4), ("(2)", 2, 6)],
        possible_ocr_issue=True,
    )
    recheck = _diagnosis(
        diagnosis_id="diag_re",
        answer_id="ans_1",
        scores=[("(1)", 4, 4), ("(2)", 6, 6)],
        prompt_type="recheck",
        created_at="2026-10-02T00:00:00+00:00",
    )
    other_model = _diagnosis(
        diagnosis_id="diag_9b",
        answer_id="ans_1",
        scores=[("(1)", 1, 4), ("(2)", 1, 6)],
        model_name="Qwen3.5-9B-bnb4",
        created_at="2026-10-03T00:00:00+00:00",
    )
    grades, _ = parse_teacher_csv(
        "student_anonymized_id,problem_id,question,teacher_score,max_score\n"
        "student_001,prob_1,(1),4,4\n"
        "student_001,prob_1,問2,4,6\n"
        "student_002,prob_1,(1),4,4\n",
        [_problem()],
    )
    rows = build_comparisons(
        grades,
        [answer],
        [older, latest, recheck, other_model],
        model_name="Qwen3.8-27B-bnb4",
    )
    by_key = {
        (row.student_anonymized_id, normalize_question_label(row.question)): row for row in rows
    }
    first = by_key[("student_001", "1")]
    second = by_key[("student_001", "2")]
    missing = by_key[("student_002", "1")]
    assert first.system_score == 4
    assert first.diagnosis_id == "diag_new"
    assert first.possible_ocr_issue is True
    assert second.diff == -2
    assert missing.status == "no_system_grade"
    assert is_large_gap(second, ratio_threshold=0.2) is True
    assert is_large_gap(first, ratio_threshold=0.2) is False

    metrics = compute_metrics(rows, ratio_threshold=0.2)
    assert metrics.n_matched == 2
    assert metrics.large_gap_count == 1
    assert metrics.unmatched_reasons == {"no_system_grade": 1}
    assert metrics.mae == pytest.approx(1.0)
    totals = student_totals(rows)
    assert totals[0]["teacher_total"] == 8
    assert totals[0]["system_total"] == 6


def test_metrics_correlation_and_kappa() -> None:
    assert pearson([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert pearson([1, 1], [2, 3]) is None
    assert quadratic_weighted_kappa([0, 1, 2], [0, 1, 2]) == pytest.approx(1.0)
    assert quadratic_weighted_kappa([1, 1], [1, 1]) is None


def test_log_discrepancies_updates_same_item(storage: Storage) -> None:
    answer = AnswerRecord(
        answer_id="ans_1", problem_id="prob_1", student_anonymized_id="student_001", created_at=utc_now_iso()
    )
    diagnosis = _diagnosis(diagnosis_id="diag_1", answer_id="ans_1", scores=[("(1)", 1, 4)])
    grades, _ = parse_teacher_csv(
        "student_anonymized_id,problem_id,question,teacher_score,max_score\nstudent_001,prob_1,(1),4,4\n",
        [_problem()],
    )
    rows = build_comparisons(grades, [answer], [diagnosis])
    first = log_discrepancies(storage, rows, ratio_threshold=0.2)
    second = log_discrepancies(storage, rows, ratio_threshold=0.2)
    assert len(storage.list_discrepancies()) == 1
    assert first[0].discrepancy_id == second[0].discrepancy_id
    assert first[0].status == "open"


class _FakeAdapter:
    def __init__(self, payload: dict, fail: bool = False) -> None:
        self.payload = payload
        self.fail = fail
        self.contexts: list[DiagnosisRunContext] = []

    def run(self, context: DiagnosisRunContext) -> dict:
        self.contexts.append(context)
        if self.fail:
            raise RuntimeError("model down")
        return dict(self.payload)

    def to_diagnosis_result(self, context, raw_output, diagnosis_id, created_at):
        from src.adapters.diagnosis_helpers import build_diagnosis_result

        return build_diagnosis_result(context, raw_output, diagnosis_id, created_at)


def _seed(storage: Storage, system_score: float) -> tuple:
    storage.save_problem(_problem())
    storage.save_answer(
        AnswerRecord(
            answer_id="ans_1",
            problem_id="prob_1",
            student_anonymized_id="student_001",
            created_at=utc_now_iso(),
        )
    )
    storage.save_ocr_result(
        OCRResult(
            ocr_id="ocr_ans_1",
            answer_id="ans_1",
            ocr_engine="glm",
            raw_text="x<5",
            used_text="x<5",
            created_at=utc_now_iso(),
        )
    )
    diagnosis = _diagnosis(diagnosis_id="diag_1", answer_id="ans_1", scores=[("(1)", system_score, 4)])
    storage.save_diagnosis_result(diagnosis)
    grades, _ = parse_teacher_csv(
        "student_anonymized_id,problem_id,question,teacher_score,max_score,teacher_comment\n"
        "student_001,prob_1,(1),4,4,境界\n",
        storage.list_problems(),
    )
    storage.save_teacher_grades(grades)
    rows = build_comparisons(
        storage.list_teacher_grades(),
        storage.list_answers(),
        storage.list_diagnosis_results(),
        storage.list_ocr_results(),
    )
    return log_discrepancies(storage, rows, ratio_threshold=0.2)[0]


def test_recheck_blind_hides_teacher_score_and_can_close_gap(storage: Storage) -> None:
    discrepancy = _seed(storage, system_score=1)
    adapter = _FakeAdapter(
        {"score": 4, "max_score": 4, "question_scores": [{"question": "(1)", "score": 4, "max_score": 4}]}
    )
    updated = run_recheck(storage, adapter, discrepancy, model_name="mock", mode="blind")
    assert updated.status == "recheck_agrees_with_teacher"
    assert "4" not in adapter.contexts[0].extra_instructions or "前回" in adapter.contexts[0].extra_instructions
    assert "教師の採点" not in adapter.contexts[0].extra_instructions
    assert adapter.contexts[0].prompt_type == "recheck"
    saved = storage.list_diagnosis_results()
    assert any(item.prompt_type == "recheck" for item in saved)
    assert storage.get_diagnosis_result("diag_1").score == 1


def test_recheck_informed_includes_teacher_comment(storage: Storage) -> None:
    discrepancy = _seed(storage, system_score=1)
    text = build_recheck_instructions(
        discrepancy, storage.get_diagnosis_result("diag_1"), mode="informed"
    )
    assert "境界" in text
    assert "教師の採点: 4" in text


def test_recheck_keeps_gap_and_records_failure(storage: Storage) -> None:
    discrepancy = _seed(storage, system_score=1)
    kept = run_recheck(
        storage,
        _FakeAdapter(
            {"score": 1, "max_score": 4, "question_scores": [{"question": "(1)", "score": 1, "max_score": 4}]}
        ),
        discrepancy,
        model_name="mock",
    )
    assert kept.status == "recheck_keeps_gap"
    failed = run_recheck(storage, _FakeAdapter({}, fail=True), discrepancy, model_name="mock")
    assert failed.status == "recheck_failed"
    assert "model down" in failed.recheck["error"]
