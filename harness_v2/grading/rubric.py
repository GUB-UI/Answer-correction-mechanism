"""Load a rubric. Point values here are the allocation, not a teacher score."""

from __future__ import annotations

import json
from pathlib import Path

from harness_v2.core.contracts import RubricItem


def rubric_from_data(items: list[dict]) -> list[RubricItem]:
    rubric: list[RubricItem] = []
    for item in items:
        rubric.append(
            RubricItem(
                item_id=str(item["item_id"]),
                max_points=float(item["max_points"]),
                description=str(item.get("description") or ""),
                acceptable_expression=str(item.get("acceptable_expression") or ""),
            )
        )
    return rubric


def load_rubric(path: Path) -> tuple[str, list[RubricItem]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    version = str(payload.get("version") or "unversioned")
    return version, rubric_from_data(payload["items"])
