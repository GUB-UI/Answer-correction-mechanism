"""Fail-closed old/new result checks and benchmark accounting.

A completed comparison means both sides produced valid grades; it does not
mean their scores agree, or that either score agrees with a reference label.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from harness_v2.evaluation.independent import prepare_independent_input
from harness_v2.grading.rubric import rubric_from_data
from harness_v2.grading.scorer import ScoringParseError, items_from_completion
from harness_v2.grading.validator import validate


def load_result(path: Path) -> tuple[Any, str]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), ""
    except FileNotFoundError:
        return None, "missing result.json"
    except (OSError, ValueError, UnicodeError) as exc:
        return None, f"invalid result.json: {type(exc).__name__}"


def _invalid_result(payload: Any, side: str, case: dict[str, Any]) -> str:
    if payload is None:
        return "missing grading result"
    if not isinstance(payload, dict):
        return "invalid result object"
    if payload.get("failure_reason"):
        return f"result failure: {payload['failure_reason']}"
    if side == "old":
        if payload.get("status") != "parsed":
            return f"no valid grade: status={payload.get('status')}"
        grade = payload.get("data")
    else:
        if payload.get("status") != "ran":
            return f"no valid grade: status={payload.get('status')}"
        run = payload.get("run")
        outputs = run.get("outputs") if isinstance(run, dict) else None
        grade = outputs.get("grading") if isinstance(outputs, dict) else None
        if not isinstance(grade, dict) or grade.get("status") != "validated":
            return "missing or non-validated grading result"
    if not isinstance(grade, dict):
        return "missing grading object"
    if grade.get("failure_reason") or any(grade.get(flag) for flag in (
        "needs_review", "review_required", "grading_uncertain", "ocr_uncertain", "review_reason"
    )):
        return "grading requires review or reports a failure"
    try:
        view = prepare_independent_input(case)
        rubric = rubric_from_data(view.get("rubric") or [])
        if side == "old":
            maximum = grade.get("max_score")
            if (isinstance(maximum, bool) or not isinstance(maximum, (int, float))
                    or not math.isfinite(maximum)
                    or abs(maximum - sum(item.max_points for item in rubric)) > 1e-6):
                return "invalid legacy max_score"
            rows = grade.get("rubric_scores")
            required = {"item_id", "score", "max_score", "reason", "evidence"}
            if not isinstance(rows, list) or any(
                not isinstance(row, dict) or not required.issubset(row)
                or not isinstance(row["reason"], str) or not isinstance(row["evidence"], str)
                for row in rows
            ):
                return "invalid legacy rubric_scores schema"
            normalized = {"items": [
                {"item_id": row["item_id"], "score": row["score"], "max_points": row["max_score"],
                 "math_reason": row["reason"], "evidence_spans": [{"text": row["evidence"]}],
                 "review_reason": row.get("review_reason"), "uncertainty": row.get("uncertainty")}
                for row in rows
            ], "total_score": grade.get("score")}
        else:
            normalized = {"items": grade.get("items"), "total_score": grade.get("total_score")}
        # Each runner's canonical fields must be present. Cross-schema aliases
        # may never hide a missing/invalid old or new score or item array.
        items, total, _ = items_from_completion(json.dumps(normalized), rubric)
        answer = view.get("answer_text") or ""
        if case.get("ocr_import_path"):
            answer = prepare_independent_input({
                "answer_text": Path(case["ocr_import_path"]).read_bytes().decode("utf-8")
            })["answer_text"]
        checked = validate(rubric, items, total, answer)
    except (ScoringParseError, ValueError, TypeError, OSError, OverflowError) as exc:
        return f"invalid grading result: {exc}"
    if checked.status != "validated":
        return f"invalid grading result: {checked.failure_reason}"
    return ""


def evaluate_comparison(old: Any, new: Any, case: dict[str, Any]) -> dict[str, Any]:
    reasons = []
    for side, payload in (("old", old), ("new", new)):
        reason = _invalid_result(payload, side, case)
        if reason:
            reasons.append(f"{side}: {reason}")
    return {"success": not reasons, "status": "incomplete" if reasons else "complete",
            "reason": "; ".join(reasons)}


def summarize_comparisons(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # Incomplete cases remain in the denominator; empty benchmarks have no rate.
    success_count = sum(row.get("success") is True and row.get("status") == "complete"
                        and not row.get("reason") for row in rows)
    return {"rows": rows, "case_count": len(rows), "success_count": success_count,
            "incomplete_count": len(rows) - success_count,
            "success_rate": success_count / len(rows) if rows else None}
