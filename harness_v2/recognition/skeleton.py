"""Notation-neutral content skeleton and CER for OCR-only checks.

Both the OCR text and the manual transcription are passed through the same
function. Mathematical notation is canonicalized *before* formatting is
removed, so LaTeX and plain/Unicode spellings of the same expression
collapse to the same skeleton while different mathematics stays different.
Nothing here names a problem, formula, or score.
"""

from __future__ import annotations

import re
import unicodedata

_FENCE = re.compile(r"```[A-Za-z]*")
_FRAC_HEAD = re.compile(r"\\[dt]?frac(?![A-Za-z])\s*")
_ROOT_HEAD = re.compile(r"\\sqrt(?![A-Za-z])|√|\bsqrt(?=\s*\()")
_SUPERSCRIPTS = re.compile(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+")
_SUPERSCRIPT_MAP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_BARRED = re.compile(r"(?:[^\s\u0304\u0305][\u0304\u0305]+)+")
_COMMANDS = [
    (r"\\leqq|\\leq|\\le(?![A-Za-z])", "<="),
    (r"\\geqq|\\geq|\\ge(?![A-Za-z])", ">="),
    (r"\\neq|\\ne(?![A-Za-z])", "!="),
    (r"\\lt(?![A-Za-z])", "<"),
    (r"\\gt(?![A-Za-z])", ">"),
    (r"\\minus(?![A-Za-z])", "-"),
    (r"\\times|\\cdot(?![A-Za-z])", "*"),
    (r"\\pm(?![A-Za-z])", "+-"),
]
_UNICODE = {"≤": "<=", "≦": "<=", "≥": ">=", "≧": ">=", "≠": "!=", "−": "-",
            "×": "*", "·": "*", "±": "+-"}
_COMMAND_WORD = re.compile(r"\\[A-Za-z]+")
_DROP = set("$\\{}()[]_") | {"（", "）", "\u0304", "\u0305"}


def _group(text: str, start: int, *, delimiters: str = "{") -> tuple[str, int] | None:
    """Balanced group (or a single token) starting at ``start``."""
    while start < len(text) and text[start].isspace():
        start += 1
    if start >= len(text):
        return None
    if text[start] not in delimiters:
        return text[start], start + 1
    opening = text[start]
    closing = {"{": "}", "(": ")", "[": "]"}[opening]
    depth = 0
    for index in range(start, len(text)):
        depth += {opening: 1, closing: -1}.get(text[index], 0)
        if depth == 0:
            return text[start + 1:index], index + 1
    return None


def _fractions(text: str) -> str:
    out, pos = [], 0
    while True:
        match = _FRAC_HEAD.search(text, pos)
        if match is None:
            out.append(text[pos:])
            return "".join(out)
        numerator = _group(text, match.end())
        denominator = numerator and _group(text, numerator[1])
        if not denominator:
            out.append(text[pos:match.end()])
            pos = match.end()
            continue
        out.append(text[pos:match.start()])
        out.append(f"({_fractions(numerator[0])})/({_fractions(denominator[0])})")
        pos = denominator[1]


def _radicand(text: str, start: int) -> tuple[str, int] | None:
    """Read a grouped expression, nested root, number, or single variable."""
    while start < len(text) and text[start].isspace():
        start += 1
    root = _ROOT_HEAD.match(text, start)
    if root:
        inner = _radicand(text, root.end())
        end = inner[1] if inner else root.end()
        return text[start:end], end
    barred = _BARRED.match(text, start)
    if barred:
        return barred.group(), barred.end()
    number = _NUMBER.match(text, start)
    if number:
        return number.group(), number.end()
    return _group(text, start, delimiters="{([")


def _content(text: str) -> str:
    """Strip formatting while retaining the delimiters of canonical roots."""
    out, pos = [], 0
    while pos < len(text):
        root = _ROOT_HEAD.match(text, pos)
        if root:
            argument = _radicand(text, root.end())
            inner, pos = argument if argument else ("", root.end())
            out.append(f"sqrt({_content(inner)})")
        else:
            ch = text[pos]
            if not ch.isspace() and ch not in _DROP:
                out.append(ch)
            pos += 1
    return "".join(out)


def skeleton(text: str) -> str:
    # NFKC alone would flatten superscripts into ordinary baseline digits.
    text = _SUPERSCRIPTS.sub(lambda m: "^" + m.group().translate(_SUPERSCRIPT_MAP),
                             _FENCE.sub("", text))
    text = unicodedata.normalize("NFKC", text)
    text = _fractions(text)
    for pattern, repl in _COMMANDS:
        text = re.sub(pattern, repl, text)
    text = _COMMAND_WORD.sub(lambda m: m.group() if m.group() == r"\sqrt" else "", text)
    for key, value in _UNICODE.items():
        text = text.replace(key, value)
    return _content(text)


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (ca != cb))
    return row[-1]


def cer(hypothesis: str, reference: str, *, normalize=skeleton) -> float:
    ref = normalize(reference)
    if not ref:
        raise ValueError("empty reference transcription")
    return levenshtein(normalize(hypothesis), ref) / len(ref)


def raw_normalize(text: str) -> str:
    return "".join(ch for ch in _FENCE.sub("", text) if not ch.isspace())


TOLERANCE = 1e-9


def gate_a1(baseline: dict[str, float], candidate: dict[str, float], *, target: str,
            min_gain: float = 0.02, max_loss: float = 0.01) -> dict:
    """Preregistered Stage A1 gate on skeleton CER per image.

    Boundaries are inclusive (gain exactly 0.02 passes, loss exactly 0.01 is
    allowed) up to a 1e-9 floating-point tolerance.
    """
    if set(baseline) != set(candidate) or target not in baseline:
        raise ValueError("baseline/candidate images differ or target missing")
    gain = baseline[target] - candidate[target]
    worse = sorted(k for k in baseline
                   if k != target and candidate[k] - baseline[k] > max_loss + TOLERANCE)
    return {"target_gain": gain, "worse_images": worse,
            "passed": gain >= min_gain - TOLERANCE and not worse}
