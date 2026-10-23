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
_FRAC = re.compile(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
_COMMANDS = [
    (r"\\leqq|\\leq|\\le(?![A-Za-z])", "<="),
    (r"\\geqq|\\geq|\\ge(?![A-Za-z])", ">="),
    (r"\\neq|\\ne(?![A-Za-z])", "!="),
    (r"\\lt(?![A-Za-z])", "<"),
    (r"\\gt(?![A-Za-z])", ">"),
    (r"\\minus(?![A-Za-z])", "-"),
    (r"\\times|\\cdot(?![A-Za-z])", "*"),
    (r"\\pm(?![A-Za-z])", "+-"),
    (r"\\sqrt(?![A-Za-z])", ""),
]
_UNICODE = {"≤": "<=", "≦": "<=", "≥": ">=", "≧": ">=", "≠": "!=", "−": "-",
            "×": "*", "·": "*", "±": "+-", "√": ""}
_COMMAND_WORD = re.compile(r"\\[A-Za-z]+")
_DROP = set("$\\{}()[]^_") | {"（", "）"}


def skeleton(text: str) -> str:
    text = _FENCE.sub("", text)
    previous = None
    while previous != text:  # innermost-first, so nested fractions work
        previous = text
        text = _FRAC.sub(r"(\1)/(\2)", text)
    for pattern, repl in _COMMANDS:
        text = re.sub(pattern, repl, text)
    text = _COMMAND_WORD.sub("", text)
    text = unicodedata.normalize("NFKC", text)
    for key, value in _UNICODE.items():
        text = text.replace(key, value)
    return "".join(ch for ch in text if not ch.isspace() and ch not in _DROP)


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


def gate_a1(baseline: dict[str, float], candidate: dict[str, float], *, target: str,
            min_gain: float = 0.02, max_loss: float = 0.01) -> dict:
    """Preregistered Stage A1 gate on skeleton CER per image."""
    if set(baseline) != set(candidate) or target not in baseline:
        raise ValueError("baseline/candidate images differ or target missing")
    gain = baseline[target] - candidate[target]
    worse = sorted(k for k in baseline if k != target and candidate[k] - baseline[k] > max_loss)
    return {"target_gain": gain, "worse_images": worse,
            "passed": gain >= min_gain and not worse}
