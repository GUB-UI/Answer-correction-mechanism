from __future__ import annotations

from pathlib import Path

from src.models import AnswerRecord, OCRResult, ProblemRecord

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
IMAGE_UPLOAD_TYPES = ["png", "jpg", "jpeg", "webp"]

SOURCE_KIND_LABELS = {
    "student_answer": "答案",
    "problem": "問題文",
    "model_answer": "模範解答",
    "combined": "問題+模範",
}


def resolve_stored_path(project_root: Path, image_path: str) -> Path:
    path = Path(image_path)
    if not path.is_absolute():
        path = project_root / path
    return path


def infer_input_kind(text: str | None, image_paths: list[str]) -> str:
    has_text = bool((text or "").strip())
    has_images = bool(image_paths)
    if has_text and has_images:
        return "mixed"
    if has_images:
        return "image"
    return "text"


def suffix_for_upload(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    return suffix if suffix in IMAGE_SUFFIXES else ".png"


def split_problem_and_solution(text: str) -> tuple[str, str]:
    """同一画像に問題と模範が載っている場合の分割ヒューリスティック。"""
    stripped = text.strip()
    if not stripped:
        return "", ""

    markers = ["【解答】", "解答\n", "\n解答", "【解】", "解\n(1)", "解\n１"]
    for marker in markers:
        idx = stripped.find(marker)
        if idx >= 30:
            return stripped[:idx].strip(), stripped[idx:].strip()

    first = stripped.find("(1)")
    second = stripped.find("(1)", first + 3) if first >= 0 else -1
    if first >= 0 and second > first + 40:
        return stripped[:second].strip(), stripped[second:].strip()
    return stripped, stripped


def answer_image_paths(answer: AnswerRecord) -> list[str]:
    if answer.image_paths:
        return list(answer.image_paths)
    if answer.image_path:
        return [answer.image_path]
    return []


def problem_ocr_targets(problem: ProblemRecord) -> list[tuple[str, list[str]]]:
    """OCRすべき問題側画像を (source_kind, paths) で返す。"""
    targets: list[tuple[str, list[str]]] = []
    if problem.combined_image_paths:
        targets.append(("combined", list(problem.combined_image_paths)))
        return targets
    if problem.problem_image_paths:
        targets.append(("problem", list(problem.problem_image_paths)))
    if problem.model_answer_image_paths:
        targets.append(("model_answer", list(problem.model_answer_image_paths)))
    return targets


def student_ocr_results(ocr_results: list[OCRResult]) -> list[OCRResult]:
    return [
        item
        for item in ocr_results
        if item.source_kind == "student_answer" or not item.source_kind
    ]


def apply_ocr_texts_to_problem(
    problem: ProblemRecord,
    ocr: OCRResult,
) -> ProblemRecord:
    used = ocr.used_text.strip()
    ocr_ids_problem = list(problem.problem_ocr_ids)
    ocr_ids_model = list(problem.model_ocr_ids)
    if ocr.ocr_id not in ocr_ids_problem and ocr.source_kind in {
        "problem",
        "combined",
    }:
        ocr_ids_problem.append(ocr.ocr_id)
    if ocr.ocr_id not in ocr_ids_model and ocr.source_kind in {
        "model_answer",
        "combined",
    }:
        ocr_ids_model.append(ocr.ocr_id)

    updates: dict[str, object] = {
        "problem_ocr_ids": ocr_ids_problem,
        "model_ocr_ids": ocr_ids_model,
    }
    if ocr.source_kind == "combined":
        problem_text, correct_answer = split_problem_and_solution(used)
        if not problem.problem_text.strip():
            updates["problem_text"] = problem_text
        if not problem.correct_answer.strip():
            updates["correct_answer"] = correct_answer
    elif ocr.source_kind == "problem" and not problem.problem_text.strip():
        updates["problem_text"] = used
    elif ocr.source_kind == "model_answer" and not problem.correct_answer.strip():
        updates["correct_answer"] = used
    return problem.model_copy(update=updates)
