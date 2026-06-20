from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

from src.models import ERROR_CATEGORIES


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", stripped)
    if fence:
        stripped = fence.group(1).strip()
    return json.loads(stripped)


def image_to_base64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


def normalize_error_categories(categories: list[str]) -> list[str]:
    normalized: list[str] = []
    for category in categories:
        if category in ERROR_CATEGORIES and category not in normalized:
            normalized.append(category)
    return normalized


def detect_uncertain_parts(text: str) -> list[str]:
    return re.findall(r"\[不明\]", text)


def is_ocr_suspect(text: str, uncertain_parts: list[str] | None = None) -> bool:
    parts = uncertain_parts if uncertain_parts is not None else detect_uncertain_parts(text)
    return bool(parts) or "[不明]" in text
