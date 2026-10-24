"""Load a rubric. Point values here are the allocation, not a teacher score."""

from __future__ import annotations

import json
import math
from pathlib import Path

from harness_v2.core.contracts import RubricItem


def rubric_error(rubric: list[RubricItem]) -> str:
    if not rubric:
        return "empty_rubric"
    seen: set[str] = set()
    for item in rubric:
        if not isinstance(item.item_id, str) or not item.item_id.strip():
            return "invalid_rubric_id"
        if item.item_id in seen:
            return "duplicate_rubric_ids"
        seen.add(item.item_id)
        maximum = item.max_points
        if (isinstance(maximum, bool) or not isinstance(maximum, (int, float))
                or not math.isfinite(maximum) or maximum < 0):
            return "invalid_rubric_maximum"
    if not math.isfinite(sum(item.max_points for item in rubric)):
        return "invalid_rubric_total"
    return ""


def rubric_from_data(items: list[dict]) -> list[RubricItem]:
    if not isinstance(items, list):
        raise ValueError("rubric must be an array")
    rubric: list[RubricItem] = []
    for item in items:
        if not isinstance(item, dict) or "item_id" not in item or "max_points" not in item:
            raise ValueError("invalid_rubric_row")
        rubric.append(
            RubricItem(
                item_id=item["item_id"],
                max_points=item["max_points"],
                description=str(item.get("description") or ""),
                acceptable_expression=str(item.get("acceptable_expression") or ""),
            )
        )
    error = rubric_error(rubric)
    if error:
        raise ValueError(error)
    return rubric


def load_rubric(path: Path) -> tuple[str, list[RubricItem]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    version = str(payload.get("version") or "unversioned")
    return version, rubric_from_data(payload["items"])
