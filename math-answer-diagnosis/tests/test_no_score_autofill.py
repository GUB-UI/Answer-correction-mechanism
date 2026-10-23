import json

import pytest
from pydantic import ValidationError

from src.adapters.base import DiagnosisRunContext
from src.adapters.diagnosis_helpers import build_diagnosis_result
from src.adapters.mock_adapters import MockDiagnosisAdapter
from src.config import load_config
from src.models import DiagnosisResult, ProblemRecord, utc_now_iso
from src.storage import Storage
from src.teacher_compare import system_score_for_question


@pytest.fixture
def context():
    problem = ProblemRecord(
        problem_id="p", title="test", created_at=utc_now_iso(),
        score_allocation=[{"question": "(1)", "max_score": 2},
                          {"question": "(2)", "max_score": 7},
                          {"question": "(3)", "max_score": 11}],
    )
    return DiagnosisRunContext(answer_id="a", problem_id="p", ocr_id="o", problem=problem,
                               used_text="student work", model_name="test", prompt_type="basic_scoring")


def build(context, **payload):
    return build_diagnosis_result(context, {"score": 10, "max_score": 20, **payload}, "d")


@pytest.mark.parametrize("total", [0, 8.75, 20, -1, 22])
@pytest.mark.parametrize("ocr_issue", [False, True])
def test_missing_scores_are_never_inferred_from_any_total(context, monkeypatch, total, ocr_issue):
    monkeypatch.setenv("REVIEW_FLAGS_ROUTE", "false")
    result = build(context, score=total, question_scores=[], possible_ocr_issue=ocr_issue)
    assert result.score == total
    assert [row["score"] for row in result.question_scores] == [None, None, None]
    assert [row["max_score"] for row in result.question_scores] == [2, 7, 11]
    assert all(row["status"] == "review_required" for row in result.question_scores)
    assert result.review_status == "review_required"
    assert result.raw_output["review_flags"]["needs_review"] is True
    assert "question_scores_missing" in result.raw_output["grading_validation"]


@pytest.mark.parametrize("scores", [None, {}, [{"question": "total", "score": 10}]])
def test_omitted_or_total_only_question_scores_are_undetermined(context, scores):
    result = build(context, question_scores=scores)
    assert [row["score"] for row in result.question_scores] == [None, None, None]
    assert result.review_status == "review_required"


@pytest.mark.parametrize("score", [None, "", "unknown", "NaN", "Infinity", True])
def test_invalid_or_null_question_score_is_not_replaced(context, score):
    result = build(context, question_scores=[{"question": "(1)", "score": score, "reason": "unresolved"}])
    assert [row["score"] for row in result.question_scores] == [None, None, None]
    assert result.review_status == "review_required"


def test_partial_output_preserves_explicit_zero_and_does_not_distribute_remainder(context):
    result = build(context, question_scores=[{"question": "問1", "score": 0},
                                           {"question": "(3)", "score": 4}])
    scores = {row["question"]: row["score"] for row in result.question_scores}
    assert scores == {"(1)": 0, "(2)": None, "(3)": 4}
    assert result.score == 10
    assert result.review_status == "review_required"


def test_missing_score_key_is_preserved_without_an_allocation(context):
    context.problem.score_allocation = []
    result = build(context, question_scores=[{"question": "(1)", "max_score": 5}])
    assert result.question_scores[0]["score"] is None
    assert result.review_status == "review_required"


def test_absent_scores_without_allocation_require_review(context):
    context.problem.score_allocation = []
    result = build(context)
    assert result.question_scores == []
    assert result.review_status == "review_required"


def test_missing_allocation_cap_does_not_drop_undetermined_question(context):
    context.problem.score_allocation[1].pop("max_score")
    result = build(context)
    row = next(row for row in result.question_scores if row["question"] == "(2)")
    assert row["score"] is None
    assert row["max_score"] is None
    assert result.review_status == "review_required"


def test_explicit_rubric_item_scores_are_preserved_and_missing_items_not_inferred(context):
    raw_items = [
        {"item_id": "(1)", "score": 1, "max_score": 2, "reason": "work", "evidence": "記載なし"},
        {"item_id": "(2)", "score": None, "max_score": 7, "reason": "unresolved", "evidence": "記載なし"},
    ]
    result = build(context, rubric_scores=raw_items)
    assert [row["score"] for row in result.question_scores] == [1, None, None]
    assert result.raw_output["rubric_scores"] == raw_items
    assert result.review_status == "review_required"


def test_explicit_complete_scores_can_be_validated_without_changes(context):
    result = build(context, score=6, question_scores=[
        {"question": "(1)", "score": 0}, {"question": "(2)", "score": 2},
        {"question": "(3)", "score": 4},
    ])
    assert [row["score"] for row in result.question_scores] == [0, 2, 4]
    assert result.review_status == "validated"


def test_contradictory_total_is_not_recomputed(context):
    result = build(context, score=0, possible_ocr_issue=True, question_scores=[
        {"question": "(1)", "score": 0}, {"question": "(2)", "score": 2},
        {"question": "(3)", "score": 4},
    ])
    assert result.score == 0
    assert [row["score"] for row in result.question_scores] == [0, 2, 4]
    assert result.review_status == "review_required"
    assert "question_score_sum_mismatch" in result.raw_output["grading_validation"]


def test_missing_total_is_not_filled_from_question_scores(context):
    result = build(context, score=None, question_scores=[{"question": "(1)", "score": 1}])
    assert result.score is None
    assert result.review_status == "review_required"


def test_mock_does_not_calculate_question_scores_from_allocations(context):
    adapter = MockDiagnosisAdapter()
    raw = adapter.run(context)
    assert raw["question_scores"] == []
    result = build_diagnosis_result(context, raw, "d")
    assert all(row["score"] is None for row in result.question_scores)
    assert result.review_status == "review_required"


def test_null_scores_survive_storage_and_teacher_comparison(context, tmp_path):
    storage = Storage(load_config(tmp_path))
    result = build(context)
    storage.save_diagnosis_result(result)
    stored = storage.get_diagnosis_result(result.diagnosis_id)
    assert stored.question_scores == result.question_scores
    assert stored.review_status == "review_required"
    assert system_score_for_question(stored, "(1)") == (None, 2)
    payload = json.loads(stored.model_dump_json())
    assert payload["question_scores"][0]["score"] is None


def test_unknown_review_status_is_not_accepted_as_validated(context):
    result = build(context, review_status="unexpected")
    assert result.review_status == "review_required"
    with pytest.raises(ValidationError):
        DiagnosisResult.model_validate({**result.model_dump(), "review_status": "unexpected"})


def test_loading_record_with_null_question_score_requires_review(context):
    result = build(context)
    payload = {**result.model_dump(), "review_status": "validated"}
    assert DiagnosisResult.model_validate(payload).review_status == "review_required"


def test_batch_keeps_missing_scores_and_review_status_on_cache_hit(context, tmp_path):
    from src.batch_pipeline import BatchOptions, run_batch
    from src.models import AnswerRecord

    class MissingQuestionScores(MockDiagnosisAdapter):
        calls = 0

        def run(self, context):
            self.calls += 1
            return {"score": 10, "max_score": 20, "question_scores": []}

    config = load_config(tmp_path)
    storage = Storage(config)
    storage.save_problem(context.problem)
    answer = AnswerRecord(answer_id="a", problem_id="p", created_at=utc_now_iso(),
                          student_anonymized_id="student", image_path=storage.save_image(b"test input"))
    storage.save_answer(answer)
    adapter = MissingQuestionScores()
    options = BatchOptions(ocr_provider="mock", diagnosis_provider="mock")
    first = run_batch(storage, config, [answer], options, diagnosis_adapter=adapter).items[0]
    again = run_batch(storage, config, [answer], options, diagnosis_adapter=adapter).items[0]
    assert first.status == again.status == "review_required"
    assert first.diagnosis_id == again.diagnosis_id
    assert adapter.calls == 1
    result = storage.get_diagnosis_result(first.diagnosis_id)
    assert [row["score"] for row in result.question_scores] == [None, None, None]


def test_recheck_instructions_preserve_undetermined_scores(context):
    from types import SimpleNamespace
    from src.teacher_compare import build_recheck_instructions

    result = build(context, score=None)
    instructions = build_recheck_instructions(SimpleNamespace(question="(2)"), result)
    assert "合計: 未確定 / 20" in instructions
    assert "(2): 未確定 / 7" in instructions


@pytest.mark.parametrize("scores_present", [False, True])
@pytest.mark.parametrize("review_status", [None, "", "validated", "review_required", "failed"])
def test_legacy_empty_question_scores_require_review(context, scores_present, review_status):
    payload = build(context).model_dump()
    if scores_present:
        payload["question_scores"] = []
    else:
        payload.pop("question_scores")
    if review_status is None:
        payload.pop("review_status")
    else:
        payload["review_status"] = review_status
    result = DiagnosisResult.model_validate_json(json.dumps(payload))
    assert result.review_status == ("failed" if review_status == "failed" else "review_required")
    assert result.question_scores == []
    assert result.score == payload["score"]
    assert result.max_score == payload["max_score"]


@pytest.mark.parametrize("scores_present", [False, True])
@pytest.mark.parametrize("review_status", [None, "", "validated", "failed"])
def test_batch_cache_reuses_legacy_empty_scores_with_review_status(
    context, tmp_path, scores_present, review_status,
):
    from src.batch_pipeline import BatchOptions, run_batch
    from src.models import AnswerRecord

    class CountingDiagnosis(MockDiagnosisAdapter):
        calls = 0

        def run(self, context):
            self.calls += 1
            return {"score": 10, "max_score": 20, "question_scores": []}

    config = load_config(tmp_path)
    storage = Storage(config)
    storage.save_problem(context.problem)
    answer = AnswerRecord(answer_id="a", problem_id="p", created_at=utc_now_iso(),
                          student_anonymized_id="student", image_path=storage.save_image(b"test input"))
    storage.save_answer(answer)
    adapter = CountingDiagnosis()
    options = BatchOptions(ocr_provider="mock", diagnosis_provider="mock")
    first = run_batch(storage, config, [answer], options, diagnosis_adapter=adapter).items[0]
    payload = storage.get_diagnosis_result(first.diagnosis_id).model_dump()
    if scores_present:
        payload["question_scores"] = []
    else:
        payload.pop("question_scores")
    if review_status is None:
        payload.pop("review_status")
    else:
        payload["review_status"] = review_status
    # Write a legacy record directly so model validation happens on cache load.
    (config.records_dir / "diagnosis_results.jsonl").write_text(json.dumps(payload) + "\n")
    reused = run_batch(storage, config, [answer], options, diagnosis_adapter=adapter).items[0]
    assert reused.ocr_id == first.ocr_id
    assert reused.diagnosis_id == first.diagnosis_id
    assert adapter.calls == 1
    assert reused.status == ("failed" if review_status == "failed" else "review_required")
    assert reused.score == first.score == payload["score"]
    assert reused.max_score == first.max_score == payload["max_score"]
    assert storage.get_diagnosis_result(reused.diagnosis_id).question_scores == []


def test_legacy_blank_status_with_complete_explicit_scores_can_validate(context):
    payload = build(context, score=6, question_scores=[
        {"question": "(1)", "score": 0}, {"question": "(2)", "score": 2},
        {"question": "(3)", "score": 4},
    ]).model_dump()
    payload["review_status"] = ""
    result = DiagnosisResult.model_validate(payload)
    assert result.review_status == "validated"
    assert result.question_scores == payload["question_scores"]
    assert result.score == payload["score"]
