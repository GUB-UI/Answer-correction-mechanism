"""採点出力の JSON を厳密に読む。

まずそのまま ``json.loads`` する。失敗したときだけ、外側のコードフェンスを
1組だけ外して同じ読み方をやり直す。中の文字は削らないし、``{...}`` の切り出しもしない。
読めなかった出力は採点結果にしない。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_SCHEMA_PATH = Path(__file__).with_name("schema.json")
_OPENER = re.compile(r"```(?:json)?[ \t]*\r?\n")


class GradingOutputError(ValueError):
    """修復しても JSON にならなかった採点出力。通常の採点結果としては扱わない。"""

    def __init__(self, message: str, *, fence_stripped: bool = False) -> None:
        super().__init__(message)
        self.fence_stripped = fence_stripped


@dataclass(frozen=True)
class ParsedGradingText:
    data: dict[str, Any]
    fence_stripped: bool


@dataclass(frozen=True)
class ExpectedItem:
    item_id: str
    max_score: float


def strip_one_outer_fence(text: str) -> str | None:
    """文字列全体が ``` または ```json の1組なら、中身をそのまま返す。

    先頭と末尾の空白だけフェンスの外側として捨てる。中の空白や改行は残す。
    """
    if "```" not in text:
        return None
    lead = len(text) - len(text.lstrip(" \t\r\n"))
    trail_body = text[lead:]
    trail = len(trail_body) - len(trail_body.rstrip(" \t\r\n"))
    core = text[lead : len(text) - trail] if trail else text[lead:]
    opener = _OPENER.match(core)
    if opener is None or opener.start() != 0:
        return None
    if not core.endswith("```"):
        return None
    body = core[opener.end() : len(core) - 3]
    if body.endswith("\r\n"):
        body = body[:-2]
    elif body.endswith("\n"):
        body = body[:-1]
    elif body.endswith("\r"):
        body = body[:-1]
    else:
        return None
    return body


def parse_grading_text(text: str) -> ParsedGradingText:
    """採点出力を読む。フェンスを外した事実は ``fence_stripped`` にだけ残す。"""
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as first_error:
        body = strip_one_outer_fence(text)
        if body is None:
            raise GradingOutputError(f"json_parse_failed: {first_error}") from first_error
        try:
            loaded = json.loads(body)
        except json.JSONDecodeError as second_error:
            raise GradingOutputError(
                f"json_parse_failed_after_fence: {second_error}",
                fence_stripped=True,
            ) from second_error
        if not isinstance(loaded, dict):
            raise GradingOutputError(
                "json_parse_failed_after_fence: JSON object expected",
                fence_stripped=True,
            )
        return ParsedGradingText(loaded, True)
    if not isinstance(loaded, dict):
        raise GradingOutputError("json_parse_failed: JSON object expected")
    return ParsedGradingText(loaded, False)


def load_grading_schema() -> dict[str, Any]:
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _schema_errors(value: Any, schema: dict[str, Any], path: str) -> list[str]:
    expected = schema.get("type")
    if expected == "object":
        if not isinstance(value, dict):
            return [f"schema:{path or 'object'} must be object"]
        errors: list[str] = []
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"schema:{path + '.' if path else ''}{key} required")
        properties = schema.get("properties") or {}
        for key, sub in properties.items():
            if key not in value:
                continue
            child = f"{path}.{key}" if path else key
            errors.extend(_schema_errors(value[key], sub, child))
        return errors
    if expected == "array":
        if not isinstance(value, list):
            return [f"schema:{path} must be array"]
        item_schema = schema.get("items") or {}
        errors = []
        for index, item in enumerate(value):
            errors.extend(_schema_errors(item, item_schema, f"{path}[{index}]"))
        return errors
    if expected == "string":
        if not isinstance(value, str):
            return [f"schema:{path} must be string"]
        return []
    if expected == "number":
        if not _is_number(value):
            return [f"schema:{path} must be number"]
        return []
    if expected == "boolean":
        if not isinstance(value, bool):
            return [f"schema:{path} must be boolean"]
        return []
    return []


def _numbers_equal(left: float, right: float) -> bool:
    if isinstance(left, int) and isinstance(right, int):
        return left == right
    return abs(float(left) - float(right)) <= 1e-6


@dataclass
class GradingValidation:
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_grading(
    obj: dict[str, Any],
    expected_items: list[ExpectedItem] | None = None,
) -> GradingValidation:
    """スキーマ、項目数、点数の範囲、項目合計と score の一致を見る。

    ``expected_items`` は配点（項目 ID と満点）で、参照得点ではない。
    渡さないときは、出力に書かれた max_score の範囲と合計だけを見る。
    """
    errors = _schema_errors(obj, load_grading_schema(), "")
    items = obj.get("rubric_scores")
    if not isinstance(items, list):
        return GradingValidation(errors)
    parsed_items: list[dict[str, Any]] = [item for item in items if isinstance(item, dict)]
    ids = [str(item.get("item_id")) for item in parsed_items if item.get("item_id") is not None]
    if expected_items is not None:
        expected_ids = [item.item_id for item in expected_items]
        if sorted(ids) != sorted(expected_ids) or len(ids) != len(expected_ids):
            errors.append(f"items_mismatch:{ids}")
        expected_max = {item.item_id: item.max_score for item in expected_items}
    else:
        expected_max = {}
    total = 0.0
    total_ok = True
    for item in parsed_items:
        item_id = str(item.get("item_id") or "")
        score = item.get("score")
        own_max = item.get("max_score")
        if not _is_number(score):
            total_ok = False
            continue
        ceiling = expected_max.get(item_id, own_max if _is_number(own_max) else None)
        if ceiling is None or not _is_number(ceiling):
            total_ok = False
            continue
        if float(score) < 0 or float(score) > float(ceiling):
            errors.append(f"score_out_of_range:{item_id}:{score}>{ceiling}")
        if item_id in expected_max and _is_number(own_max) and not _numbers_equal(own_max, expected_max[item_id]):
            errors.append(f"item_max_mismatch:{item_id}")
        total += float(score)
    score = obj.get("score")
    if total_ok and _is_number(score) and not _numbers_equal(total, score):
        errors.append(f"score_sum_mismatch:{total}!={score}")
    elif not total_ok and isinstance(items, list) and items:
        errors.append("score_not_number")
    return GradingValidation(errors)
