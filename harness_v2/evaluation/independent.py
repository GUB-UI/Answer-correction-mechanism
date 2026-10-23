"""Drop prior AI scores and reference scores before an independent evaluation."""

from __future__ import annotations

import re
from typing import Any

_DROPPED_KEYS = {
    "prior_ai_score",
    "reference_score",
    "teacher_score",
    "ai_score",
    "score_strict",
    "correct_answer_score",
}
_SCORE_TEXT = re.compile(
    r"(?i)(prior[_\s-]*ai[_\s-]*score|reference[_\s-]*score|teacher[_\s-]*score)\s*[:=]\s*\S+"
)


def drop_score_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): drop_score_fields(item)
            for key, item in value.items()
            if str(key) not in _DROPPED_KEYS
        }
    if isinstance(value, list):
        return [drop_score_fields(item) for item in value]
    return value


def scrub_score_text(text: str) -> str:
    return _SCORE_TEXT.sub("[redacted-score]", text)


def prepare_independent_input(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a copy safe to show a model that must not see earlier scores."""
    return drop_score_fields(payload)
