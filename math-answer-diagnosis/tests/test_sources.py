from __future__ import annotations

from src.models import AnswerRecord, OCRResult, ProblemRecord, utc_now_iso
from src.sources import (
    apply_ocr_texts_to_problem,
    infer_input_kind,
    split_problem_and_solution,
    student_ocr_results,
    suffix_for_upload,
)


def test_split_problem_and_solution_on_repeated_items() -> None:
    text = (
        "4-[C]\n(1) 不等式 5x-29 < -3x+11 を満たす自然数 x の値をすべて求めよ。\n"
        "青チャート → 数学I 基本例題36\n"
        "(1) 不等式から 8x < 40 よって x < 5\n"
        "x は自然数であるから x=1,2,3,4"
    )
    problem, solution = split_problem_and_solution(text)
    assert "満たす自然数" in problem
    assert "8x < 40" in solution
    assert "満たす自然数" not in solution


def test_infer_input_kind() -> None:
    assert infer_input_kind("text", []) == "text"
    assert infer_input_kind("", ["a.png"]) == "image"
    assert infer_input_kind("text", ["a.png"]) == "mixed"


def test_suffix_for_upload() -> None:
    assert suffix_for_upload("note.JPEG") == ".jpeg"
    assert suffix_for_upload("note") == ".png"


def test_student_ocr_results_filters_problem_ocr() -> None:
    now = utc_now_iso()
    items = [
        OCRResult(
            ocr_id="ocr_a",
            answer_id="ans_1",
            ocr_engine="mock",
            raw_text="a",
            used_text="a",
            created_at=now,
            source_kind="student_answer",
            source_id="ans_1",
        ),
        OCRResult(
            ocr_id="ocr_p",
            answer_id="",
            ocr_engine="mock",
            raw_text="p",
            used_text="p",
            created_at=now,
            source_kind="combined",
            source_id="prob_1",
        ),
    ]
    filtered = student_ocr_results(items)
    assert [item.ocr_id for item in filtered] == ["ocr_a"]


def test_apply_ocr_fills_empty_problem_fields_only() -> None:
    problem = ProblemRecord(
        problem_id="prob_1",
        title="t",
        problem_text="",
        correct_answer="already",
        rubric="r",
        unit="u",
        difficulty="basic",
        created_at=utc_now_iso(),
        combined_image_paths=["x.png"],
    )
    ocr = OCRResult(
        ocr_id="ocr_c",
        ocr_engine="mock",
        raw_text="(1) 問題文です。\n" + "x" * 40 + "\n(1) 模範です。",
        used_text="(1) 問題文です。\n" + "x" * 40 + "\n(1) 模範です。",
        created_at=utc_now_iso(),
        source_kind="combined",
        source_id="prob_1",
    )
    updated = apply_ocr_texts_to_problem(problem, ocr)
    assert "問題文です" in updated.problem_text
    assert updated.correct_answer == "already"
    assert "ocr_c" in updated.problem_ocr_ids
    assert "ocr_c" in updated.model_ocr_ids


def test_answer_record_syncs_legacy_image_path() -> None:
    answer = AnswerRecord(
        answer_id="ans_1",
        problem_id="prob_1",
        image_path="data/images/ans_1.png",
        student_anonymized_id="student_001",
        created_at=utc_now_iso(),
    )
    assert answer.image_paths == ["data/images/ans_1.png"]
