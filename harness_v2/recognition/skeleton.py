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
# Canonical roots can directly follow a coefficient or variable, including
# inside another root. Recognize them again without losing their scope.
_ROOT_HEAD = re.compile(r"\\sqrt(?![A-Za-z])|[√∛∜]|(?:sqrt|root)(?=\s*\()")
_SUPERSCRIPTS = re.compile(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+")
_SUPERSCRIPT_MAP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_BARRED = re.compile(r"(?:[^\s\u0304\u0305][\u0304\u0305]+)+")
_COMMANDS = [
    (r"\\(?:leqq|leq|le)(?![A-Za-z])", "<="),
    (r"\\(?:geqq|geq|ge)(?![A-Za-z])", ">="),
    (r"\\(?:neq|ne)(?![A-Za-z])", "!="),
    (r"\\lt(?![A-Za-z])", "<"),
    (r"\\gt(?![A-Za-z])", ">"),
    (r"\\minus(?![A-Za-z])", "-"),
    (r"\\(?:times|cdot)(?![A-Za-z])", "*"),
    (r"\\div(?![A-Za-z])", "/"),
    (r"\\pm(?![A-Za-z])", "±"),
]
_UNICODE = {"≤": "<=", "≦": "<=", "≥": ">=", "≧": ">=", "≠": "!=", "−": "-",
            "×": "*", "·": "*", "÷": "/"}
_SYMBOLS = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
    "zeta": "ζ", "eta": "η", "theta": "θ", "iota": "ι", "kappa": "κ",
    "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "omicron": "ο",
    "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ", "upsilon": "υ",
    "phi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ",
    "Pi": "Π", "Sigma": "Σ", "Upsilon": "Υ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
    # Match NFKC's canonical forms for Unicode Greek symbol variants.
    "varepsilon": "ε", "vartheta": "θ", "varkappa": "κ", "varpi": "π",
    "varrho": "ρ", "varsigma": "ς", "varphi": "φ", "circ": "°", "infty": "∞",
}
_COMMAND_WORD = re.compile(r"\\[A-Za-z]+")
_LAYOUT = re.compile(r"\\[,;:! ()\[\]]|\$+")
_ENVIRONMENTS = "pmatrix|bmatrix|vmatrix|Vmatrix|matrix|array|cases"
_CALL_HEAD = re.compile(r"(?:sqrt|root|" + _ENVIRONMENTS + r")(?=\()")
_MATH_SPAN = re.compile(r"\$\$(.*?)\$\$|\$(.*?)\$|\\\((.*?)\\\)|\\\[(.*?)\\\]", re.DOTALL)


def _layout(text: str) -> str:
    # Each math span is a separate content unit. Outer delimiters carry no content.
    text = _MATH_SPAN.sub(lambda m: ";" + next(g for g in m.groups() if g is not None).strip() + ";", text)
    text = re.sub(r"\\(begin|end)\s*\{([^{}]*)\}",
                  lambda m: (m[2] + "(" if m[1] == "begin" else ")")
                  if m[2] in _ENVIRONMENTS.split("|") else " ", text)
    text = text.replace(r"\\", ";").replace("&", "¦")
    out, depth = [], 0
    for ch in text:
        if ch in "{([":
            depth += 1
        elif ch in "})]":
            depth = max(0, depth - 1)
        out.append(";" if ch == "\n" and depth == 0 else " " if ch == "\n" else ch)
    text = re.sub(r"\s*;\s*(?:;\s*)*", ";", "".join(out)).strip("; ")
    return _LAYOUT.sub(" ", text)

_FORMATTING = {"left", "right", "overline", "bar", "mathrm", "mathbf", "mathit",
               "text", "displaystyle", "textstyle", "quad", "qquad"}
# Token boundaries must survive removing groups: numbers, commands, and roots
# cannot be manufactured by concatenating formerly separate operands.
_TOKEN = re.compile(r"!=|<=|>=|\\[A-Za-z]+|(?:sqrt|root)(?=\()|\d+(?:\.\d+)?|.", re.DOTALL)
_BOUNDARY = "\u2063"  # Canonical token boundary, never a mathematical parenthesis.
_DROP = { "\u0304", "\u0305"}
_PRECEDENCE = {";": -1, "¦": -1, ",": -1, "=": 0, "<": 0, ">": 0, "!=": 0, "<=": 0, ">=": 0, "!": 5, "+": 1, "-": 1, "±": 1,
               "*": 2, "/": 2, "^": 4, "_": 4}


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


def _expression_group(text: str, start: int) -> tuple[str, int, str, str] | None:
    """Read content brackets, including nested and half-open intervals."""
    brackets = re.compile(r"\\[{}]|[()[\]]")
    first = brackets.match(text, start)
    if not first or first.group() not in ("(", "[", r"\{"):
        return None
    depth = 0
    for token in brackets.finditer(text, start):
        depth += 1 if token.group() in ("(", "[", r"\{") else -1
        if depth == 0:
            return text[first.end():token.start()], token.end(), first.group(), token.group()
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
        out.append(f"(({_fractions(numerator[0])})/({_fractions(denominator[0])}))")
        pos = denominator[1]


def _radicand(text: str, start: int) -> tuple[str, int] | None:
    """Read a grouped expression, nested root, number, or single variable."""
    while start < len(text) and text[start].isspace():
        start += 1
    root = _ROOT_HEAD.match(text, start)
    if root:
        _, _, end = _root_argument(text, root)
        return text[start:end], end
    barred = _BARRED.match(text, start)
    if barred:
        return barred.group(), barred.end()
    number = _NUMBER.match(text, start)
    if number:
        return number.group(), number.end()
    return _group(text, start, delimiters="{([")


def _root_argument(text: str, root: re.Match) -> tuple[str | None, str, int]:
    """Read an optional LaTeX root index separately from its radicand."""
    start = root.end()
    index = {"∛": "3", "∜": "4"}.get(root.group())
    if root.group() == r"\sqrt":
        start = start + len(text[start:]) - len(text[start:].lstrip())
        if text[start:start + 1] == "[":
            group = _group(text, start, delimiters="[")
            if group:
                index, start = group
    argument = _radicand(text, start)
    inner, end = argument if argument else ("", start)
    return index, inner, end


def _content(text: str) -> str:
    """Strip formatting, retaining root scope and expression grouping."""
    out, pos = [], 0
    while pos < len(text):
        root = _ROOT_HEAD.match(text, pos)
        if root:
            index, inner, pos = _root_argument(text, root)
            head = "root" if index is not None or root.group() == "root" else "sqrt"
            argument = f"{_content(index)}," if index is not None else ""
            out.append(f"{head}({argument}{_content(inner)})")
        else:
            command = _COMMAND_WORD.match(text, pos)
            if command:
                # A dedicated boundary keeps command names separate from operands.
                out.append(command.group())
                if text[command.end():command.end() + 1] != _BOUNDARY:
                    out.append(_BOUNDARY)
                pos = command.end()
                continue
            ch = text[pos]
            if ch == "\\" and text[pos + 1:pos + 2] in ("{", "}"):
                # Literal braces are content, not groups. Keep their spelling
                # so a second pass cannot mistake them for math-mode markers.
                out.append(text[pos:pos + 2])
                pos += 2
                continue
            if not ch.isspace() and ch not in _DROP:
                if pos and text[pos - 1].isspace() and (ch.isalnum() or ch == "."):
                    # Preserve lexical boundaries without inventing mathematical groups.
                    token = _TOKEN.match(text, pos)
                    prefix = "".join(out)
                    if (_TOKEN.findall(prefix + text[pos:])
                            != _TOKEN.findall(prefix) + _TOKEN.findall(text[pos:])):
                        out.append(_BOUNDARY)
                    out.append(token.group())
                    pos = token.end()
                    continue
                out.append({"{": "(", "}": ")"}.get(ch, ch))
            pos += 1
    return "".join(out)


def _precedence(text: str) -> int:
    """Weakest top-level operation; adjacency is implicit multiplication."""
    weakest, operand, pos = 6, False, 0
    while pos < len(text):
        if text[pos] == _BOUNDARY:
            pos += 1
            continue
        ch = text[pos:pos + 2] if text[pos:pos + 2] in _PRECEDENCE else text[pos]
        if ch in _PRECEDENCE:
            unary = ch in "+-±" and not operand
            weakest = min(weakest, 3 if unary else _PRECEDENCE[ch])
            operand = operand if ch == "!" else False
            pos += len(ch)
            continue
        if operand:
            weakest = min(weakest, 2)
        root = _CALL_HEAD.match(text, pos)
        start = root.end() if root else pos
        group = _expression_group(text, start)
        number = _NUMBER.match(text, pos)
        command = _COMMAND_WORD.match(text, pos)
        pos = group[1] if group else command.end() if command else number.end() if number else pos + 1
        operand = True
    return weakest


def _grouping(text: str) -> str:
    """Remove only parentheses redundant under precedence and associativity.

    Keep right operands of subtraction/division (and other left-associative
    operators) grouped at equal precedence. Powers associate to the right.
    Root delimiters always remain. No algebraic simplification is performed.
    """
    out, pos = "", 0
    while pos < len(text):
        root = _CALL_HEAD.match(text, pos)
        start = root.end() if root else pos
        group = _expression_group(text, start)
        if not group:
            out += text[pos]
            pos += 1
            continue
        inner = _grouping(group[0])
        end = group[1]
        if root:
            out += f"{root.group()}({inner})"
        elif group[2:] != ("(", ")") or _precedence(inner) == -1:
            out += group[2] + inner + group[3]
        else:
            strength = _precedence(inner)
            left = out[-2:] if out[-2:] in _PRECEDENCE else out[-1:]
            right = text[end:end + 2] if text[end:end + 2] in _PRECEDENCE else text[end:end + 1]
            lp = _PRECEDENCE.get(left, 2) if left else -1
            if left in ("+", "-", "±") and (len(out) == 1 or out[-2] in "(=<>!+-±*/^_"):
                lp = 3  # unary sign
            rp = _PRECEDENCE.get(right, 2) if right else -1
            left_ok = strength > lp or (left in ("^", "_") and strength >= 3)
            right_ok = strength > rp if right in ("^", "_") else strength >= rp
            signed_factor = (left and left not in _PRECEDENCE and inner[:1] in ("+", "-", "±"))
            tail = text[end:]
            joins_tokens = (_TOKEN.findall(out + inner + tail)
                            != _TOKEN.findall(out) + _TOKEN.findall(inner) + _TOKEN.findall(tail))
            command_argument = re.search(r"\\[A-Za-z]+" + _BOUNDARY + r"$", out)
            out += (inner if inner and left_ok and right_ok and not signed_factor and not joins_tokens
                    and not command_argument
                    else f"({inner})")
        pos = end
    return out


def skeleton(text: str) -> str:
    # NFKC alone would flatten superscripts into ordinary baseline digits.
    text = _SUPERSCRIPTS.sub(lambda m: "^(" + m.group().translate(_SUPERSCRIPT_MAP) + ")",
                             _FENCE.sub("", text))
    text = unicodedata.normalize("NFKC", text)
    # Remove layout before parsing arguments: spacing can occur between a
    # fraction/root command and its operand. Keep separators until _content.
    text = _layout(text)
    text = _COMMAND_WORD.sub(lambda m: " " if m.group()[1:] in _FORMATTING
                             else _SYMBOLS.get(m.group()[1:], m.group()), text)
    text = _fractions(text)
    for pattern, repl in _COMMANDS:
        text = re.sub(pattern, repl, text)
    for key, value in _UNICODE.items():
        text = text.replace(key, value)
    return _grouping(_content(text))


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
