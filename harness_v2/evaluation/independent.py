"""Allowlist what an independent reviewer is allowed to see.

Scores are removed by keeping known fields, not by trying to name every
score key. Strings are scanned for score assignments and for JSON objects
embedded inside the text.
"""

from __future__ import annotations

import json
import re
from typing import Any

_PROBLEM_KEYS = ("problem_text", "problem")
_ANSWER_KEYS = ("answer_text", "ocr_text", "answer")
_RUBRIC_FIELDS = ("item_id", "max_points", "description", "acceptable_expression")
_FAILURE_FIELDS = (
    "case_id",
    "problem_id",
    "input_version",
    "model_version",
    "harness_version",
    "rubric_version",
    "rubric_results",
    "evidence",
    "errors",
    "uncertainty",
    "execution_log",
)
# Scalar score values must end before punctuation/prose, rather than consume a
# numeric prefix of a student's symbolic expression such as score = 2*x + 1.
_SCALAR = r"[-+−]?(?:\d+(?:[.．]\d+)?(?:[eE][-+]?\d+)?|NaN|Inf(?:inity)?)"
_SCORE_VALUE = rf"{_SCALAR}(?:[ \t]*(?:[/／]|out[ \t]+of)[ \t]*{_SCALAR})?"
_VALUE = (
    rf"(?:[\"']{_SCORE_VALUE}(?:[ \t]*(?:points?|marks?|点))?[\"']|{_SCORE_VALUE}|null)"
    r"(?:[ \t]*点|(?![\w.．/／]))(?!(?:[ \t]*[-+*/^=]))"
)
_SCORE_TEXT = re.compile(
    r"(?i)(?<!\w)[\"']?(?:(?:predicted|reference|teacher|ai|total|correct[_\s-]*answer|"
    r"prior[_\s-]*ai)[_\s-]*)?score(?:_strict)?[\"']?"
    rf"\s*(?:[:=：]|was\b|is\b)\s*{_VALUE}"
)
_ATTRIBUTED_SCORE = re.compile(
    r"(?i)(?<!\w)(?:(?:prior|previous|earlier|reference|teacher|ai|model|predicted|total)[_\s-]+)+"
    r"(?:scores?|scored|grades?|graded|marks?|awarded|gave|assigned)"
    r"(?:[ \t]+(?:is|was|of|a|the|score|grade|total))*[ \t]*[:=：]?[ \t]*"
    + _VALUE
)
_POINT_TEXT = re.compile(
    r"(?:\d+(?:[.．]\d+)?\s*[/／]\s*\d+(?:[.．]\d+)?\s*点|"
    r"(?:採点(?:結果)?|参照得点|教師採点|得点|点数|評価点|合計点|スコア)\s*[:：=は]?\s*\d+(?:[.．]\d+)?\s*点?)"
)


def _drop_key(key: str) -> bool:
    lowered = str(key).lower().replace("-", "_")
    if lowered in {"max_points", "max_score"}:
        return False
    return "score" in lowered


def _strip_loaded(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _strip_loaded(item)
            for key, item in value.items()
            if not _drop_key(str(key))
        }
    if isinstance(value, list):
        return [_strip_loaded(item) for item in value]
    if isinstance(value, str):
        return scrub_score_text(value)
    return value


def _scrub_embedded(text: str) -> str:
    decoder = json.JSONDecoder()
    pieces: list[str] = []
    index = 0
    while index < len(text):
        if text[index] in "{[":
            try:
                loaded, end = decoder.raw_decode(text, index)
            except json.JSONDecodeError:
                pieces.append(text[index])
                index += 1
                continue
            if isinstance(loaded, (dict, list)):
                cleaned = _strip_loaded(loaded)
                # Mathematical lists/objects are text too: do not reserialize them
                # unless blinding actually changed their contents.
                pieces.append(text[index:end] if cleaned == loaded else json.dumps(cleaned, ensure_ascii=False))
                index = end
                continue
        pieces.append(text[index])
        index += 1
    return "".join(pieces)


def scrub_score_text(text: str) -> str:
    text = _scrub_embedded(text)
    for pattern in (_SCORE_TEXT, _ATTRIBUTED_SCORE, _POINT_TEXT):
        text = pattern.sub("[redacted-score]", text)
    return text


def _text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return scrub_score_text(value)


def grading_allowlist(payload: dict[str, Any]) -> dict[str, Any]:
    """Problem, answer or OCR text, and rubric fields. No scores."""
    kept: dict[str, Any] = {}
    for key in _PROBLEM_KEYS:
        if isinstance(payload.get(key), str):
            kept["problem_text"] = _text(payload[key])
            break
    for key in _ANSWER_KEYS:
        if isinstance(payload.get(key), str):
            kept["answer_text"] = _text(payload[key])
            break
    items: list[dict[str, Any]] = []
    rubric = payload.get("rubric")
    if isinstance(rubric, list):
        for item in rubric:
            if not isinstance(item, dict):
                continue
            row: dict[str, Any] = {}
            for field in _RUBRIC_FIELDS:
                if field not in item:
                    continue
                value = item[field]
                if field == "max_points":
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        row[field] = value
                elif isinstance(value, str):
                    row[field] = _text(value)
            items.append(row)
    kept["rubric"] = items
    return kept


def proposal_review_allowlist(payload: dict[str, Any]) -> dict[str, Any]:
    """Failure-case fields except scores, plus a diff."""
    source = payload.get("failure_case")
    if not isinstance(source, dict):
        source = payload
    kept: dict[str, Any] = {}
    for key in _FAILURE_FIELDS:
        if key in source:
            kept[key] = _strip_loaded(source[key])
    diff = payload.get("diff")
    if isinstance(diff, str):
        kept["diff"] = scrub_score_text(diff)
    return kept


def prepare_independent_input(payload: dict[str, Any]) -> dict[str, Any]:
    """Model-facing copy of a grading case."""
    return grading_allowlist(payload)


def independent_text(text: str) -> str:
    """Scrub a turn string. JSON objects are reduced to the allowlist."""
    stripped = text.strip()
    if stripped[:1] in "{[":
        try:
            loaded = json.loads(stripped)
        except json.JSONDecodeError:
            loaded = None
        if isinstance(loaded, dict):
            if any(key in loaded for key in ("diff", "failure_case", "rubric_results", "case_id")):
                return json.dumps(proposal_review_allowlist(loaded), ensure_ascii=False)
            if any(key in loaded for key in ("problem_text", "problem", "answer_text", "answer", "rubric")):
                return json.dumps(grading_allowlist(loaded), ensure_ascii=False)
            return json.dumps(_strip_loaded(loaded), ensure_ascii=False)
        if isinstance(loaded, list):
            return json.dumps(_strip_loaded(loaded), ensure_ascii=False)
    return scrub_score_text(text)
