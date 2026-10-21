"""採点結果への規則フラグ。記録用で、既定では教師確認への振り分けを変えない。

実行時に参照得点は使わない。配点（項目 ID と満点）だけを使う。
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any

_FULL_CREDIT = re.compile(r"満点とする|基準を満たす")
_NOT_FULL = re.compile(r"満たさない|満たしていない")
_ZERO_CREDIT = re.compile(r"0点とする|0点。|認められない")
_SKIP_EVIDENCE = re.compile(
    r"記載なし|未記載|記載がない|記述がない|示されていない|記載されていない"
)
_HAN_FORMULA = re.compile(r"最小值|最大值|解得|当且仅当")
_BROKEN_TOKEN = re.compile(r"\d+\s*(?:°|∘|''|′′|″)|√‾|\[不明\]")
_MATH_SPAN = re.compile(r"\$\$[\s\S]+?\$\$|\$[^$\n]+\$|\\\([\s\S]+?\\\)|\\\[[\s\S]+?\\\]")
_FORMULA_LINE = re.compile(r"√|\\sqrt|=|≤|≥|≦|≧")


def review_routing_enabled() -> bool:
    """True のときだけ、規則の needs_review を教師確認の状態に写す。既定はオフ。"""
    return os.environ.get("REVIEW_FLAGS_ROUTE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@dataclass
class ReviewFlags:
    needs_review: bool
    grading_uncertain: bool
    ocr_uncertain: bool
    review_reasons: list[str] = field(default_factory=list)
    grading_reasons: list[str] = field(default_factory=list)
    ocr_reasons: list[str] = field(default_factory=list)
    evidence_missing: list[str] = field(default_factory=list)
    fence_stripped: bool = False
    routing_applied: bool = False
    is_grading_result: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_for_match(text: str) -> str:
    """LaTeX、空白、記号のゆれを揃える。意味の言い換えは揃えない。"""
    value = text
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("\\cdot", "·").replace("\\times", "×")
    value = value.replace("\\geq", "≥").replace("\\leq", "≤")
    value = value.replace("\\ge", "≥").replace("\\le", "≤")
    value = value.replace("≧", "≥").replace("≦", "≤")
    value = value.replace("−", "-").replace("–", "-").replace("—", "-")
    value = value.replace("\\sqrt", "√").replace("^2", "²")
    value = value.replace("\\frac", "")
    value = re.sub(r"\\[a-zA-Z]+", "", value)
    value = re.sub(r"[\s$「」『』{}()（）\[\]\\_^]", "", value)
    return value


def _formula_fragments(evidence: str) -> list[str]:
    quoted = re.findall(r"[「$]([^」$]{1,})[」$]", evidence)
    source = " ".join(quoted) if quoted else evidence
    source = re.sub(r"^[^:：]{0,16}[:：]", "", source.strip())
    parts = re.split(r"\.\.\.|…|、|，|,|および", source)
    fragments: list[str] = []
    for part in parts:
        normalized = normalize_for_match(part)
        if len(normalized) >= 3 and re.search(r"[0-9√=<>≤≥+\-*/²]", normalized):
            fragments.append(normalized)
    return fragments


def _items(obj: dict[str, Any]) -> list[dict[str, Any]]:
    raw = obj.get("rubric_scores")
    if not isinstance(raw, list):
        raw = obj.get("question_scores")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _item_id(item: dict[str, Any]) -> str:
    return str(item.get("item_id") or item.get("question") or "")


def unreadable_formula_hits(answer_text: str) -> list[str]:
    """数式の中に、式として読めない印があるときだけ拾う。

    ページ全体の Markdown や、式の外の \\pi では立てない。
    """
    spans = [match.group(0) for match in _MATH_SPAN.finditer(answer_text)]
    regions = spans or [line for line in answer_text.splitlines() if _FORMULA_LINE.search(line)]
    if "√‾" in answer_text and not any("√‾" in region for region in regions):
        regions.append(answer_text)
    hits: list[str] = []
    for region in regions:
        for pattern in (_HAN_FORMULA, _BROKEN_TOKEN):
            found = pattern.search(region)
            if found and found.group(0) not in hits:
                hits.append(found.group(0))
    return hits


def evidence_missing_from_input(obj: dict[str, Any], answer_text: str) -> list[str]:
    haystack = normalize_for_match(answer_text)
    missing: list[str] = []
    for item in _items(obj):
        evidence = str(item.get("evidence") or "")
        if not evidence.strip() or _SKIP_EVIDENCE.search(evidence):
            continue
        fragments = _formula_fragments(evidence)
        if not fragments:
            continue
        absent = [fragment for fragment in fragments if fragment not in haystack]
        if len(absent) == len(fragments):
            missing.append(f"{_item_id(item)}:{evidence[:60]}")
    return missing


def score_reason_mismatches(obj: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for item in _items(obj):
        score = item.get("score")
        maximum = item.get("max_score")
        if isinstance(score, bool) or isinstance(maximum, bool):
            continue
        if not isinstance(score, (int, float)) or not isinstance(maximum, (int, float)):
            continue
        reason = str(item.get("reason") or "")
        item_id = _item_id(item)
        if score < maximum and _FULL_CREDIT.search(reason) and not _NOT_FULL.search(reason):
            reasons.append(f"{item_id}:理由は満点だが{score}/{maximum}")
        if score == maximum and maximum > 0 and _ZERO_CREDIT.search(reason):
            reasons.append(f"{item_id}:理由は0点だが満点")
    return reasons


def evaluate_review_flags(
    obj: dict[str, Any] | None,
    *,
    answer_text: str | None,
    validation_errors: list[str] | None = None,
    fence_stripped: bool = False,
    parse_error: str | None = None,
    routing_applied: bool = False,
) -> ReviewFlags:
    """規則で needs_review / grading_uncertain / ocr_uncertain を付ける。

    ``fence_stripped`` だけが真でも needs_review にはしない。
    ``answer_text`` が None のときは、OCR と根拠の照合はしない。
    """
    if parse_error:
        return ReviewFlags(
            needs_review=True,
            grading_uncertain=True,
            ocr_uncertain=False,
            review_reasons=[parse_error],
            grading_reasons=["parse_failed"],
            fence_stripped=fence_stripped,
            routing_applied=routing_applied,
            is_grading_result=False,
        )
    assert obj is not None
    review_reasons: list[str] = []
    grading_reasons = score_reason_mismatches(obj)
    structural = list(validation_errors or [])
    if structural:
        review_reasons.extend(structural)
        grading_reasons = structural + grading_reasons
    if grading_reasons and "score_reason_mismatch" not in review_reasons:
        if any("理由は" in item for item in grading_reasons):
            review_reasons.append("score_reason_mismatch")
    ocr_reasons: list[str] = []
    evidence_missing: list[str] = []
    if answer_text is not None:
        for hit in unreadable_formula_hits(answer_text):
            ocr_reasons.append(f"key_formula_unreadable:{hit}")
        if ocr_reasons:
            review_reasons.append("key_formula_unreadable")
        evidence_missing = evidence_missing_from_input(obj, answer_text)
        if evidence_missing:
            review_reasons.append(f"evidence_not_in_input:{len(evidence_missing)}")
    model_grading = bool(obj.get("grading_uncertain"))
    model_ocr = bool(obj.get("ocr_uncertain"))
    return ReviewFlags(
        needs_review=bool(review_reasons),
        grading_uncertain=bool(grading_reasons) or model_grading,
        ocr_uncertain=bool(ocr_reasons) or model_ocr,
        review_reasons=review_reasons,
        grading_reasons=grading_reasons,
        ocr_reasons=ocr_reasons,
        evidence_missing=evidence_missing,
        fence_stripped=fence_stripped,
        routing_applied=routing_applied,
        is_grading_result=True,
    )
