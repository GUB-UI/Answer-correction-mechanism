"""OCR 本文が採点に渡してよいかの粗い判定。モデルの読み取り品質は保証しない。"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from src.adapters.formula_merge import latex_diverges_from_base
from src.utils import detect_uncertain_parts

# 手書き日本語の答案に GLM 系が混ぜやすい中国語の定型。漢字だけの日本語とは分ける。
_CHINESE_MARKERS = (
    "最小值",
    "最大值",
    "当且仅当",
    "解得",
    "所以",
    "因为",
    "可得",
    "即得",
    "不等式组",
    "联立",
    "解集",
)
_SIMPLIFIED_RE = re.compile(r"[仅这为与东对时们个贝经过值]")
_KANA_RE = re.compile(r"[\u3040-\u30ff]")


def _chinese_in_japanese(text: str) -> bool:
    if any(marker in text for marker in _CHINESE_MARKERS):
        return True
    # ひらがな・カタカナがある文に簡体字が混ざるときだけ追加で見る。
    return bool(_KANA_RE.search(text) and _SIMPLIFIED_RE.search(text))


def _repeated_lines(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    counts = Counter(lines)
    if any(count >= 3 and len(line) >= 8 for line, count in counts.items()):
        return True
    run = 1
    for previous, current in zip(lines, lines[1:]):
        if current == previous and len(current) >= 8:
            run += 1
            if run >= 3:
                return True
        else:
            run = 1
    # 同じ短い断片が1行の中で繰り返されるループ。
    for line in lines:
        compact = re.sub(r"\s+", "", line)
        if len(compact) >= 12 and compact[: len(compact) // 3] * 3 == compact[: (len(compact) // 3) * 3]:
            unit = compact[: len(compact) // 3]
            if unit and compact.count(unit) >= 3 and len(unit) >= 4:
                return True
    return False


def _unimernet_disagreement(replacements: list[dict[str, Any]]) -> bool:
    for item in replacements:
        skipped = str(item.get("skipped") or "")
        if skipped in {"unimernet_diverged", "unimernet_implausible"}:
            return True
        latex = item.get("unimernet")
        base = item.get("glm")
        if isinstance(latex, str) and isinstance(base, str) and latex_diverges_from_base(latex, base):
            return True
    return False


def assess_ocr_text(
    text: str,
    replacements: list[dict[str, Any]] | None = None,
) -> list[str]:
    """怪しい理由のリスト。空なら ocr_suspect にしない。"""
    reasons: list[str] = []
    if detect_uncertain_parts(text):
        reasons.append("uncertain_marker")
    if _chinese_in_japanese(text):
        reasons.append("chinese_in_japanese")
    if _repeated_lines(text):
        reasons.append("repeated_lines")
    if replacements and _unimernet_disagreement(replacements):
        reasons.append("unimernet_disagreement")
    return reasons
