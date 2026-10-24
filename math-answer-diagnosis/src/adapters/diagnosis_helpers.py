from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from src.adapters.base import DiagnosisRunContext
from src.analysis import normalize_error_categories
from src.models import DiagnosisResult, utc_now_iso
from src.grading.parse import (
    ExpectedItem,
    parse_grading_text,
    validate_grading,
)
from src.grading.review_flags import evaluate_review_flags, review_routing_enabled


def valid_backend_fingerprint(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    model = value.get("model_id")
    settings = value.get("settings")
    if not (isinstance(model, str) and model.strip() and isinstance(settings, dict) and settings):
        return False
    revision = settings.get("model_revision")
    weights = settings.get("weights_sha256")
    if not ((isinstance(revision, str) and revision.strip())
            or (isinstance(weights, str) and re.fullmatch(r"[0-9a-f]{64}", weights))):
        return False
    source = value.get("server_source_sha256")
    versions = value.get("library_versions")
    return (
        isinstance(source, str) and re.fullmatch(r"[0-9a-f]{64}", source) is not None
        and isinstance(versions, dict)
        and all(isinstance(versions.get(name), str) and versions[name].strip()
                for name in ("torch", "transformers", "bitsandbytes", "tokenizers", "accelerate"))
    )


def complete_grading_metadata(value: Any) -> bool:
    """Equality is meaningful only for complete, well-formed provenance."""
    if not isinstance(value, dict):
        return False
    if any(not isinstance(value.get(key), str) or not value[key].strip()
           for key in ("provider", "model_id", "model_alias")):
        return False
    if any(not isinstance(value.get(key), str) or re.fullmatch(r"[0-9a-f]{64}", value[key]) is None
           for key in ("prompt_version", "rubric_identity")):
        return False
    post_processing = value.get("post_processing")
    if (not isinstance(post_processing, dict)
            or type(post_processing.get("review_flags_route")) is not bool):
        return False
    settings = value.get("generation_settings")
    if not isinstance(settings, dict):
        return False
    if value["provider"] == "mock":
        version = settings.get("implementation_version")
        return type(version) is int and version > 0
    request = settings.get("request")
    if not isinstance(request, dict) or not valid_backend_fingerprint(settings.get("backend")):
        return False
    if value["model_id"] != settings["backend"]["model_id"]:
        return False
    temperature = request.get("temperature")
    max_tokens = request.get("max_tokens")
    return (
        type(temperature) in (int, float) and math.isfinite(temperature) and temperature >= 0
        and type(max_tokens) is int and max_tokens > 0
        and type(request.get("response_format_json")) is bool
        and type(request.get("disable_thinking")) is bool
    )


def build_grading_metadata(
    context: DiagnosisRunContext,
    *,
    provider: str,
    messages: list[dict[str, Any]],
    generation_settings: dict[str, Any],
) -> dict[str, Any]:
    """Use content hashes as versions of the rendered prompt and grading rubric."""
    def digest(value: Any) -> str:
        return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    problem = context.problem
    return {
        "post_processing": {"review_flags_route": review_routing_enabled()},
        "provider": provider,
        "model_id": (context.model_name if provider == "mock"
                     else generation_settings.get("backend", {}).get("model_id")),
        "model_alias": context.model_name,
        "prompt_version": digest(messages),
        "rubric_identity": digest({
            "problem_id": problem.problem_id,
            "problem_text": problem.problem_text,
            "correct_answer": problem.correct_answer,
            "rubric": problem.rubric,
            "score_allocation": problem.score_allocation,
        }),
        # Snapshot nested settings so later changes cannot alter stored metadata.
        "generation_settings": json.loads(json.dumps(generation_settings)),
    }


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def normalize_question_scores(
    value: Any,
    allocation: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """LLM 出力の question_scores を設問 ID の配列に整える。

    「全体」「合計」は答案合計であり設問 ID ではないので捨てる。
    配点があるときは、その設問 ID に正規化できるものだけ残す。
    """
    from src.teacher_compare import normalize_question_label

    if isinstance(value, dict):
        value = [
            {"question": key, **(item if isinstance(item, dict) else {"score": item})}
            for key, item in value.items()
        ]
    if not isinstance(value, list):
        return []
    canonical: dict[str, str] = {}
    for item in allocation or []:
        question = str(item.get("question") or item.get("item_id") or "").strip()
        key = normalize_question_label(question)
        if key:
            canonical.setdefault(key, question)
    items: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        score = _to_float(item.get("score"))
        question = str(item.get("question") or item.get("label") or item.get("item_id") or "").strip()
        key = normalize_question_label(question)
        if not key:
            continue
        if canonical:
            if key not in canonical:
                continue
            question = canonical[key]
        items.append(
            {
                "question": question,
                "score": score,
                "status": "validated" if score is not None else "review_required",
                "max_score": _to_float(item.get("max_score")),
                "reason": str(item.get("reason") or ""),
            }
        )
    return items


def mark_missing_question_scores(
    scores: list[dict[str, Any]],
    allocation: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Keep model scores, adding only undetermined rows for omitted questions."""
    from src.teacher_compare import normalize_question_label

    rows = [dict(item) for item in scores]
    present = {normalize_question_label(item["question"]): item for item in rows}
    for item in allocation or []:
        question = str(item.get("question") or item.get("item_id") or "").strip()
        key = normalize_question_label(question)
        if not key:
            continue
        cap = _to_float(item.get("max_score"))
        if key in present:
            if present[key].get("max_score") is None:
                present[key]["max_score"] = cap
            continue
        row = {
            "question": question,
            "score": None,
            "max_score": cap,
            "status": "review_required",
            "reason": "要教師確認。設問点が出力されていないため未確定",
        }
        rows.append(row)
        present[key] = row
    return rows


def format_score_allocation(allocation: list[dict[str, Any]]) -> str:
    lines = []
    for item in allocation:
        question = str(item.get("question") or item.get("item_id") or "").strip()
        max_score = _to_float(item.get("max_score"))
        if question and max_score is not None:
            lines.append(f"{question}: {max_score:g}点")
    return "\n".join(lines)


def build_diagnosis_result(
    context: DiagnosisRunContext,
    raw_output: dict[str, Any],
    diagnosis_id: str,
    created_at: str | None = None,
) -> DiagnosisResult:
    allocation = list(context.problem.score_allocation or [])
    raw_output = dict(raw_output)
    fence_stripped = bool(raw_output.pop("_fence_stripped", False))
    raw_output.pop("_grading_validation", None)
    raw_output["fence_stripped"] = fence_stripped
    if isinstance(raw_output.get("rubric_scores"), list):
        expected = expected_items_from_allocation(allocation)
        output_ids = {
            str(item.get("item_id"))
            for item in raw_output["rubric_scores"]
            if isinstance(item, dict) and item.get("item_id")
        }
        alloc_ids = {item.item_id for item in expected}
        if alloc_ids and output_ids and alloc_ids.isdisjoint(output_ids):
            expected = []
        raw_output["grading_validation"] = validate_grading(
            raw_output, expected or None
        ).errors
    possible_ocr_issue = bool(raw_output.get("possible_ocr_issue", False))
    score = _to_float(raw_output.get("score"))
    # Rubric items may supply explicit scores for matching IDs. Never derive them
    # from the total or from allocation weights.
    provided = raw_output.get("question_scores") or raw_output.get("rubric_scores")
    question_scores = mark_missing_question_scores(
        normalize_question_scores(provided, allocation), allocation,
    )
    missing_scores = not question_scores or any(item["score"] is None for item in question_scores)
    review_status = str(raw_output.get("review_status") or "validated").strip()
    if review_status == "needs_teacher_review":
        review_status = "review_required"
    if review_status not in {"validated", "review_required", "failed"}:
        review_status = "review_required"
    notes = str(raw_output.get("teacher_review_notes", ""))
    validation_errors = list(raw_output.get("grading_validation") or [])
    if missing_scores:
        validation_errors.append("question_scores_missing")
        notes = ("要教師確認。設問点が未確定。" + notes).strip()
    if score is None:
        validation_errors.append("total_score_missing")
    if not missing_scores and score is not None:
        if abs(sum(item["score"] for item in question_scores) - score) > 1e-6:
            validation_errors.append("question_score_sum_mismatch")
    if (possible_ocr_issue or validation_errors) and review_status != "failed":
        review_status = "review_required"
    if possible_ocr_issue and "要教師確認" not in notes:
        notes = ("要教師確認。OCRの疑いがあるため原文を確認する。" + notes).strip()
    raw_output["grading_validation"] = validation_errors
    routing = review_routing_enabled()
    flags = evaluate_review_flags(
        raw_output,
        answer_text=context.used_text,
        validation_errors=list(raw_output.get("grading_validation") or []),
        fence_stripped=fence_stripped,
        routing_applied=routing,
    )
    raw_output["review_flags"] = flags.as_dict()
    # 既定では記録だけ。possible_ocr_issue による確認状態は変えない。
    if routing and flags.needs_review and review_status != "failed":
        review_status = "review_required"
    return DiagnosisResult(
        diagnosis_id=diagnosis_id,
        answer_id=context.answer_id,
        problem_id=context.problem_id,
        ocr_id=context.ocr_id,
        model_name=context.model_name,
        prompt_type=context.prompt_type,
        score=score,
        max_score=float(raw_output.get("max_score", 0) or 0),
        error_categories=normalize_error_categories(
            list(raw_output.get("error_categories", [])),
        ),
        error_locations=list(raw_output.get("error_locations", [])),
        reasoning_summary=str(raw_output.get("reasoning_summary", "")),
        student_feedback=str(raw_output.get("student_feedback", "")),
        teacher_review_notes=notes,
        possible_ocr_issue=possible_ocr_issue,
        review_status=review_status,
        confidence=float(raw_output.get("confidence", 0.0)),
        raw_output=raw_output,
        created_at=created_at or utc_now_iso(),
        question_scores=question_scores,
    )


def parse_diagnosis_json(text: str) -> dict[str, Any]:
    """採点出力を読む。フェンスを1組外しただけでは要確認にしない。

    読めない出力は GradingOutputError になり、採点結果にはしない。
    rubric_scores があるときはスキーマ、点数の範囲、合計を検証して
    ``_grading_validation`` に残す。項目の過不足は配点を渡した
    ``validate_grading`` で見る。
    """
    parsed = parse_grading_text(text)
    data = dict(parsed.data)
    data["_fence_stripped"] = parsed.fence_stripped
    if isinstance(data.get("rubric_scores"), list):
        data["_grading_validation"] = validate_grading(data).errors
    return data


def expected_items_from_allocation(
    allocation: list[dict[str, Any]] | None,
) -> list[ExpectedItem]:
    items: list[ExpectedItem] = []
    for item in allocation or []:
        question = str(item.get("question") or item.get("item_id") or "").strip()
        max_score = _to_float(item.get("max_score"))
        if question and max_score is not None:
            items.append(ExpectedItem(question, max_score))
    return items
