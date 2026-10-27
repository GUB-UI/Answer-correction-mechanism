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
# Bumped when repaired text can change a grade. Reuse identity includes this
# so a grade parsed by an older pass is not treated as this parser's output.
COMPLETION_PARSER_VERSION = "json_latex_escapes_v12"
_JSON_SIMPLE_ESCAPES = set('"\\/')
_JSON_CONTROL_ESCAPES = set("bfnrt")
_JSON_HEX = set("0123456789abcdefABCDEF")
# Backspace and form feed are not used in grading prose. \b and \f followed
# by an ASCII letter are always a literal backslash, whitelist or not.
# Newline, tab, and carriage return stay control characters unless the
# letter run is exactly one name below and a non-letter ends it.
# A name that starts with n is the same bytes as a newline plus the rest
# (\neq is a newline plus eq). Those are not rewritten.
_LATEX_COMMANDS = frozenset({
    "nabla", "ne", "neg", "neq", "ngeq", "ngtr", "nleq", "nless",
    "not", "notin", "nu",
    "rangle", "rbrace", "rceil", "rfloor", "rho", "right", "rightarrow", "rvert",
    "tan", "tau", "text", "textbf", "textit", "textrm",
    "tfrac", "therefore", "theta", "tilde", "times", "to",
    "triangle", "triangleleft", "triangleright",
})


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


def _latex_command_name(text: str, start: int) -> str:
    """ASCII letter run beginning at the control letter, else empty."""
    end = start + 1
    length = len(text)
    while end < length and text[end].isascii() and text[end].isalpha():
        end += 1
    return text[start:end]


def repair_json_string_escapes(text: str) -> tuple[str, bool]:
    """Keep unambiguous LaTeX backslashes inside JSON strings.

    A backslash that is not a JSON escape is doubled, so ``\\le`` and
    ``\\left`` survive. ``\\b`` and ``\\f`` followed by an ASCII letter are
    always doubled, so ``\\boxed`` and ``\\begin`` are not a backspace or a
    form feed. ``\\n \\t \\r`` are doubled only when the following letters are
    exactly one known Math I/A command and a non-letter ends that name. A
    command that starts with ``n`` is left as a newline and reported as
    ambiguous, because ``\\neq`` is also a newline plus ``eq``. ``\\\\`` and
    ``\\uXXXX`` are left alone. Returns the repaired text and whether any
    newline collision was left undecided.
    """
    out: list[str] = []
    ambiguous = False
    in_string = False
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if not in_string:
            out.append(char)
            if char == '"':
                in_string = True
            index += 1
            continue
        if char == '"':
            out.append(char)
            in_string = False
            index += 1
            continue
        if char != "\\":
            out.append(char)
            index += 1
            continue
        if index + 1 >= length:
            out.append("\\\\")
            index += 1
            continue
        nxt = text[index + 1]
        if nxt in _JSON_SIMPLE_ESCAPES:
            out.append("\\")
            out.append(nxt)
            index += 2
            continue
        if nxt == "u":
            hexpart = text[index + 2:index + 6]
            if len(hexpart) == 4 and all(item in _JSON_HEX for item in hexpart):
                out.append("\\u")
                out.append(hexpart)
                index += 6
                continue
            out.append("\\\\")
            index += 1
            continue
        if nxt in _JSON_CONTROL_ESCAPES:
            follow = text[index + 2] if index + 2 < length else ""
            if nxt in "bf" and follow.isascii() and follow.isalpha():
                out.append("\\\\")
                out.append(nxt)
                index += 2
                continue
            command = _latex_command_name(text, index + 1)
            if command in _LATEX_COMMANDS and command.startswith("n"):
                ambiguous = True
                out.append("\\")
                out.append(nxt)
                index += 2
                continue
            if command in _LATEX_COMMANDS:
                out.append("\\\\")
                out.append(nxt)
                index += 2
                continue
            out.append("\\")
            out.append(nxt)
            index += 2
            continue
        out.append("\\\\")
        index += 1
    return "".join(out), ambiguous


# A stray brace or bracket is leftover JSON. A comma between words is prose.
# A numeric list or a JSON literal list is leftover JSON even when the
# characters beside the comma are letters or digits.
_JSON_STRUCTURAL = re.compile(r"[{}\[\]]")
_NUMERIC_LIST = re.compile(
    r"(?<![A-Za-z0-9_.])[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\s*,\s*"
    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
)
_LITERAL_LIST = re.compile(
    r"(?<![A-Za-z0-9_])(?:true|false|null)\s*,\s*(?:true|false|null)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
# A quoted key followed by a colon is leftover JSON. A closed quote in a
# sentence is prose.
_JSON_KEY_COLON = re.compile(r'"(?:[^"\\\n]|\\.)*"\s*:')
_PROSE_LETTER = re.compile(r"[A-Za-z\u3040-\u30ff\u4e00-\u9fff]")
# A quote, or a colon introducing a number that is not a score statement.
_COLON_VALUE = re.compile(
    r":\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
)
# A full numeric token, so 4e-1 is 0.4 and is not also read as 4.
_NUMBER_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_.])[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?![A-Za-z0-9_])"
)
# 点数 and 採点 are score words. A bare 点 is not, so 3点を通ります stays prose.
_SCORE_WORD = re.compile(
    r"(?<![A-Za-z_])(?:total_score|score|revised|total)(?![A-Za-z_])|点数|採点|合計|得点|訂正",
    re.IGNORECASE,
)
# N点 is a score only as a score phrase. 以下の3点を確認 is not one.
_SCORE_TEN = re.compile(
    r"(?<![A-Za-z0-9_.])(?P<number>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*点"
    r"(?!を確認|について|がある|を挙げ|を述べ|を見る|を説明|を通)"
)
_LIST_INTRO = ("以下の", "次の", "この", "その", "あの")


def _number_tokens(text: str) -> list[float]:
    found: list[float] = []
    for match in _NUMBER_TOKEN.finditer(text):
        try:
            value = float(match.group(0))
        except ValueError:
            continue
        if math.isfinite(value):
            found.append(value)
    return found


def _parsed_total(loaded: dict) -> float | None:
    """The object's total only. An item score is not a total."""
    if "total_score" in loaded:
        return _number(loaded.get("total_score"))
    if "score" in loaded:
        return _number(loaded.get("score"))
    return None


def _number_ending_at(text: str, index: int) -> float | None:
    """The number token that ends at index, ignoring trailing spaces."""
    prefix = text[:index].rstrip()
    for match in _NUMBER_TOKEN.finditer(prefix):
        if match.end() != len(prefix):
            continue
        try:
            value = float(match.group(0))
        except ValueError:
            return None
        return value if math.isfinite(value) else None
    return None


def _clause_end(text: str, start: int) -> int:
    """End of the sentence or clause that contains start. A decimal point stays."""
    index = start
    while index < len(text):
        char = text[index]
        if char in "。！？?!；;\n":
            return index
        if char == "." and not (
            index > 0 and text[index - 1].isdigit()
            and index + 1 < len(text) and text[index + 1].isdigit()
        ):
            return index
        index += 1
    return len(text)


def _score_mention_differs(rest: str, loaded: dict) -> bool:
    """Compare numbers in the same sentence or clause as each score word."""
    total = _parsed_total(loaded)
    matches = list(_SCORE_WORD.finditer(rest))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(rest)
        end = min(end, _clause_end(rest, match.start()))
        numbers = _number_tokens(rest[match.start():end])
        before = _number_ending_at(rest, match.start())
        if before is not None:
            numbers.append(before)
        if not numbers:
            continue
        if total is None or any(abs(value - total) > 1e-6 for value in numbers):
            return True
    for match in _SCORE_TEN.finditer(rest):
        if not _ten_is_score_phrase(rest, match):
            continue
        try:
            value = float(match.group("number"))
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        if total is None or abs(value - total) > 1e-6:
            return True
    return False


def _ten_is_score_phrase(text: str, match: re.Match[str]) -> bool:
    window = text[max(0, match.start() - 6):match.start()]
    return not any(intro in window for intro in _LIST_INTRO)


def _prose_comma(text: str, index: int) -> bool:
    before = text[:index].rstrip()
    after = text[index + 1:].lstrip()
    if not before or not after:
        return False
    return _wordish(before[-1]) and _wordish(after[0])


def _wordish(char: str) -> bool:
    """A letter or kana. A digit is not a word, so 0,0 is not prose."""
    if char.isdecimal():
        return False
    return char.isalpha() or "\u3040" <= char <= "\u30ff" or "\u4e00" <= char <= "\u9fff"


def _structural_comma(rest: str) -> bool:
    if _NUMERIC_LIST.search(rest) or _LITERAL_LIST.search(rest):
        return True
    if "," not in rest:
        return False
    # A sentence with letters or kana may contain a comma before punctuation.
    if _PROSE_LETTER.search(rest):
        return False
    for index, char in enumerate(rest):
        if char == "," and not _prose_comma(rest, index):
            return True
    return False


def _colon_is_score_statement(rest: str, colon_at: int) -> bool:
    """True when a score word sits immediately before this colon."""
    prefix = rest[:colon_at].rstrip()
    match = None
    for found in _SCORE_WORD.finditer(prefix):
        match = found
    return match is not None and match.end() == len(prefix)


def _unbalanced_quotes(rest: str) -> bool:
    count = 0
    escaped = False
    for char in rest:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            count += 1
    return count % 2 == 1


def _kv_fragment(rest: str) -> bool:
    """An unmatched quote, a quoted key, or a colon-value that is not a score."""
    if _unbalanced_quotes(rest) or _JSON_KEY_COLON.search(rest):
        return True
    for match in _COLON_VALUE.finditer(rest):
        if not _colon_is_score_statement(rest, match.start()):
            return True
    return False


def _trailing_is_ambiguous(rest: str, loaded: dict) -> bool:
    if _JSON_STRUCTURAL.search(rest):
        return True
    if _structural_comma(rest):
        return True
    if _kv_fragment(rest):
        return True
    return _score_mention_differs(rest, loaded)


def _prefix_object(text: str) -> tuple[dict[str, Any] | None, bool]:
    """One complete object with trailing text, or None when that is not the shape."""
    try:
        loaded, end = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError:
        return None, False
    if not isinstance(loaded, dict):
        return None, False
    rest = text[end:]
    if not rest.strip():
        return None, False
    return loaded, _trailing_is_ambiguous(rest, loaded)


def _load_repaired_json(text: str) -> tuple[Any, bool, bool, str]:
    repaired_text, escape_ambiguous = repair_json_string_escapes(text)
    escape_repaired = repaired_text != text
    try:
        return json.loads(repaired_text), escape_repaired, escape_ambiguous, ""
    except json.JSONDecodeError as exc:
        loaded, trailing_ambiguous = _prefix_object(repaired_text)
        if loaded is None:
            raise exc
        labels = ["newline_or_latex"] if escape_ambiguous else []
        labels.append("trailing_text")
        return loaded, True, escape_ambiguous or trailing_ambiguous, ",".join(labels)


def parse_completion(raw: str) -> tuple[dict[str, Any], bool, bool, bool, str]:
    """Return the object, fence removal, repair, newline ambiguity, and marker.

    Repair runs even when ``json.loads`` would succeed, so ``\\frac`` and
    ``\\boxed`` are not stored as a form feed or a backspace. A newline that
    also spells a known command is not rewritten. One complete object followed
    only by prose is that object, with marker ``trailing_text``. A brace,
    bracket, JSON comma, quote, colon-value fragment, or score statement
    after it is ``parse_ambiguous``. A comma in a sentence is prose. A numeric
    list or a JSON literal list is not. A score word is compared only with
    numbers in its sentence. A quoted phrase is prose; a quoted key is not.
    ``点`` counts only inside a score phrase.
    """
    try:
        loaded, repaired, ambiguous, marker = _load_repaired_json(raw)
        fence = False
    except json.JSONDecodeError as first_error:
        body = strip_one_outer_fence(raw)
        if body is None:
            raise ScoringParseError(f"json_parse_failed: {first_error}", raw=raw) from first_error
        try:
            loaded, repaired, ambiguous, marker = _load_repaired_json(body)
        except json.JSONDecodeError as second_error:
            raise ScoringParseError(
                f"json_parse_failed_after_fence: {second_error}",
                raw=raw,
                fence_stripped=True,
            ) from second_error
        fence = True
    if not isinstance(loaded, dict):
        raise ScoringParseError("json object expected", raw=raw, fence_stripped=fence)
    return loaded, fence, repaired, ambiguous, marker


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


def items_from_completion(
    raw: str, rubric: list[RubricItem]
) -> tuple[list[ItemResult], float | None, bool, bool, bool, str]:
    payload, fence, repaired, ambiguous, marker = parse_completion(raw)
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
    return items, _number(total) if total is not None else None, fence, repaired, ambiguous, marker


AMBIGUITY_AWARE_V1 = (
    " The answer text may be a transcription and can contain transcription errors. "
    "Give credit only for mathematics supported by the supplied text; do not reconstruct content that is "
    "not written. When a passage that affects an item's score looks garbled or ambiguous, cite it verbatim, "
    "explain in math_reason, set uncertainty to at least 0.5, and name the passage in review_reason. "
    "Do not raise uncertainty for clear work, including clear mistakes."
)

_OCR_UNCERTAINTY = (
    " The answer text is an OCR transcript that may contain misreads. "
    "When a line contradicts neighbouring lines or the student's own later correct work, "
    "or contains implausible symbols, do not treat it as a student error by default. "
    "State the plausible reading and the uncertainty, and lower confidence or mark it for review "
    "instead of deducting. Never invent or fill in a correct value the transcript does not support. "
    "Deduct only where the error is clearly the student's, for example when it is consistent with how "
    "the work continues and is not explainable as a character misread."
)

OCR_UNCERTAINTY_V2 = AMBIGUITY_AWARE_V1 + _OCR_UNCERTAINTY

_PARTIAL_CREDIT = (
    " For each rubric criterion, quote the transcript evidence and award its listed points if met, "
    "even if the final answer is wrong. Never award points outside the rubric. "
    "The total is the sum of the awarded criterion points."
)

PARTIAL_CREDIT_V3 = OCR_UNCERTAINTY_V2 + _PARTIAL_CREDIT

# Registered instruction variants. "v0" and "harness_v2_default" render the historical prompt byte for
# byte. Every other name is rejected, so a typo cannot silently run the default prompt.
PROMPT_VERSIONS = {
    "v0": "",
    "harness_v2_default": "",
    "harness_v2_ambiguity_aware_v1": AMBIGUITY_AWARE_V1,
    "harness_v2_ocr_uncertainty_v2": OCR_UNCERTAINTY_V2,
    "harness_v2_partial_credit_v3": PARTIAL_CREDIT_V3,
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
