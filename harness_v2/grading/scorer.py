"""Map a model completion onto ItemResult rows.

A missing score stays None. This module does not copy a 0 or a max into it.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from harness_v2.core.contracts import EvidenceSpan, ItemResult, RubricItem

_OPENER = re.compile(r"```(?:json)?[ \t]*\r?\n")


class ScoringParseError(ValueError):
    def __init__(self, message: str, *, raw: str, fence_stripped: bool = False) -> None:
        super().__init__(message)
        self.raw = raw
        self.fence_stripped = fence_stripped


def strip_one_outer_fence(text: str) -> str | None:
    """Same one-fence rule as the existing grader. The body is not rewritten."""
    if "```" not in text:
        return None
    lead = len(text) - len(text.lstrip(" \t\r\n"))
    trail_body = text[lead:]
    trail = len(trail_body) - len(trail_body.rstrip(" \t\r\n"))
    core = text[lead : len(text) - trail] if trail else text[lead:]
    opener = _OPENER.match(core)
    if opener is None or opener.start() != 0 or not core.endswith("```"):
        return None
    body = core[opener.end() : len(core) - 3]
    if body.endswith("\r\n"):
        return body[:-2]
    if body.endswith("\n") or body.endswith("\r"):
        return body[:-1]
    return None


def parse_completion(raw: str) -> tuple[dict[str, Any], bool]:
    try:
        loaded = json.loads(raw)
        fence = False
    except json.JSONDecodeError as first_error:
        body = strip_one_outer_fence(raw)
        if body is None:
            raise ScoringParseError(f"json_parse_failed: {first_error}", raw=raw) from first_error
        try:
            loaded = json.loads(body)
        except json.JSONDecodeError as second_error:
            raise ScoringParseError(
                f"json_parse_failed_after_fence: {second_error}",
                raw=raw,
                fence_stripped=True,
            ) from second_error
        fence = True
    if not isinstance(loaded, dict):
        raise ScoringParseError("json object expected", raw=raw, fence_stripped=fence)
    return loaded, fence


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _spans(value: Any) -> list[EvidenceSpan]:
    if isinstance(value, str):
        return [EvidenceSpan(text=value)]
    if not isinstance(value, list):
        return []
    spans: list[EvidenceSpan] = []
    for item in value:
        if isinstance(item, str):
            spans.append(EvidenceSpan(text=item))
        elif isinstance(item, dict) and isinstance(item.get("text"), str):
            start = item.get("start")
            end = item.get("end")
            spans.append(
                EvidenceSpan(
                    text=item["text"],
                    start=start if isinstance(start, int) else None,
                    end=end if isinstance(end, int) else None,
                    source=str(item.get("source") or "answer"),
                )
            )
    return spans


def items_from_completion(raw: str, rubric: list[RubricItem]) -> tuple[list[ItemResult], float | None, bool]:
    payload, fence = parse_completion(raw)
    maxima = {item.item_id: item.max_points for item in rubric}
    rows = payload.get("items")
    if rows is None:
        rows = payload.get("rubric_scores")
    if not isinstance(rows, list):
        raise ScoringParseError("items array missing", raw=raw, fence_stripped=fence)
    items: list[ItemResult] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ScoringParseError("malformed_item_row", raw=raw, fence_stripped=fence)
        item_id = row.get("item_id")
        if not isinstance(item_id, str) or not item_id.strip():
            raise ScoringParseError("invalid_item_id", raw=raw, fence_stripped=fence)
        claimed = _number(row.get("max_points", row.get("max_score")))
        rubric_max = maxima.get(item_id)
        max_points = float(rubric_max if rubric_max is not None else (claimed if claimed is not None else 0))
        reasons: list[str] = []
        supplied_reason = str(row.get("review_reason") or "")
        if supplied_reason:
            reasons.append(supplied_reason)
        for field in ("max_points", "max_score"):
            if field in row:
                supplied = _number(row[field])
                if supplied is None:
                    reasons.append("item_max_invalid")
                elif rubric_max is not None and abs(supplied - float(rubric_max)) > 1e-6:
                    reasons.append("item_max_mismatch")
        if "score" in row and row["score"] is not None and _number(row["score"]) is None:
            reasons.append("score_not_number")
            score = None
        else:
            raw_score = row["score"] if "score" in row else None
            score = _number(raw_score) if raw_score is not None else None
        uncertainty = row.get("uncertainty")
        items.append(
            ItemResult(
                item_id=item_id,
                max_points=max_points,
                score=score,
                evidence_spans=_spans(row.get("evidence_spans", row.get("evidence"))),
                math_reason=str(row.get("math_reason") or row.get("reason") or ""),
                uncertainty=_number(uncertainty) if uncertainty is not None else None,
                review_reason="; ".join(reasons),
                evidence_absent=row.get("evidence_absent", False),
                absent_reason=row.get("absent_reason", ""),
            )
        )
    total = payload.get("total_score", payload.get("score"))
    return items, _number(total) if total is not None else None, fence


AMBIGUITY_AWARE_V1 = (
    " The answer text may be a transcription and can contain transcription errors. "
    "Give credit only for mathematics supported by the supplied text; do not reconstruct content that is "
    "not written. When a passage that affects an item's score looks garbled or ambiguous, cite it verbatim, "
    "explain in math_reason, set uncertainty to at least 0.5, and name the passage in review_reason. "
    "Do not raise uncertainty for clear work, including clear mistakes."
)

# Registered instruction variants. "v0" and "harness_v2_default" render the historical prompt byte for
# byte. Every other name is rejected, so a typo cannot silently run the default prompt.
PROMPT_VERSIONS = {
    "v0": "",
    "harness_v2_default": "",
    "harness_v2_ambiguity_aware_v1": AMBIGUITY_AWARE_V1,
}


def prompt_suffix(prompt_version: str) -> str:
    if prompt_version not in PROMPT_VERSIONS:
        raise ValueError(f"unregistered prompt_version: {prompt_version!r}")
    return PROMPT_VERSIONS[prompt_version]


def build_prompt(problem_text: str, answer_text: str, rubric: list[RubricItem],
                 prompt_version: str = "harness_v2_default") -> str:
    """Prompt body. Reference scores are not a field on this object."""
    suffix = prompt_suffix(prompt_version)
    return json.dumps(
        {
            "problem_text": problem_text,
            "answer_text": answer_text,
            "rubric": [
                {
                    "item_id": item.item_id,
                    "max_points": item.max_points,
                    "description": item.description,
                }
                for item in rubric
            ],
            "instructions": (
                "Return one JSON object and no other text. "
                "The object has items and total_score. "
                "items is an array of objects with item_id, score, evidence_spans, "
                "math_reason, uncertainty, and review_reason. "
                "score is a finite number, or null when the item cannot be graded. Do not invent a score. "
                "evidence_spans is an array of objects with text and optional start, end, and source. "
                "Only for score exactly 0 and no written evidence, set evidence_absent to the JSON boolean true, "
                "give a nonempty absent_reason string, and use empty evidence_spans. "
                "Otherwise evidence_absent is false and evidence_spans must cite answer text. "
                "math_reason is a string. "
                "uncertainty is a number from 0 to 1, or null. "
                "review_reason is a string and is empty when the item is clear. "
                "total_score is the sum of the item scores, or null if any score is null. "
                "Rubric max_points is the point allocation. Do not return a different maximum."
            ) + suffix,
        },
        ensure_ascii=False,
    )
