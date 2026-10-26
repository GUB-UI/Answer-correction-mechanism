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
_FRAC_HEAD = re.compile(r"\\[dt]?frac(?![A-Za-z\u2063])\s*")
# Canonical roots can directly follow a coefficient or variable, including
# inside another root. Recognize them again without losing their scope.
_ROOT_HEAD = re.compile(r"\\sqrt(?![A-Za-z])|[√∛∜]|(?:sqrt|root)(?=\s*\()")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_BARRED = re.compile(r"(?:[^\s\u0304\u0305][\u0304\u0305]+)+")
_COMMANDS = [
    (r"\\(?:leqslant|leqq|leq|le)(?![A-Za-z])", "<="),
    (r"\\(?:geqslant|geqq|geq|ge)(?![A-Za-z])", ">="),
    (r"\\(?:neq|ne)(?![A-Za-z])", "!="),
    (r"\\lt(?![A-Za-z])", "<"),
    (r"\\gt(?![A-Za-z])", ">"),
    (r"\\minus(?![A-Za-z])", "-"),
    (r"\\(?:times|cdot)(?![A-Za-z])", "*"),
    (r"\\div(?![A-Za-z])", "÷"),
    (r"\\pm(?![A-Za-z])", "±"),
    (r"\\mp(?![A-Za-z])", "∓"),
]
_UNICODE = {"⩽": "<=", "⩾": ">=", "≤": "<=", "≦": "<=", "≥": ">=", "≧": ">=", "≠": "!=", "−": "-",
            "×": "*", "·": "*", "÷": "÷"}
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
    "lbrack": "[", "rbrack": "]",
    "varrho": "ρ", "varsigma": "ς", "varphi": "φ", "circ": "°", "infty": "∞",
}
_COMMAND_WORD = re.compile(r"\\[A-Za-z]+")
_LAYOUT = re.compile(r"\\[,;:! ()\[\]]|\$+")
_ENVIRONMENTS = "pmatrix|bmatrix|vmatrix|Vmatrix|matrix|array|cases"
_STYLE_ABBREVIATIONS = {
    "SANS-SERIF": "sf",
    "DOUBLE-STRUCK": "bb",
    "FRAKTUR": "frak",
    "MONOSPACE": "tt",
    "SCRIPT": "scr",
    "BOLD": "bf",
    "ITALIC": "it",
}


def _mathematical_bold_form(ch: str) -> tuple[str, str] | None:
    """Return ``(command, base)`` for one bold mathematical alphanumeric glyph.

    Style words in the Unicode name become the command, in that order, so
    bold italic and sans-serif bold share one rule with their LaTeX names.
    Styles that do not contain bold are left for compatibility folding.
    """
    name = unicodedata.name(ch, "")
    if not name.startswith("MATHEMATICAL ") or "BOLD" not in name.split():
        return None
    abbreviations = []
    for word in name.split()[1:]:
        if word not in _STYLE_ABBREVIATIONS:
            break
        abbreviations.append(_STYLE_ABBREVIATIONS[word])
    if "bf" not in abbreviations:
        return None
    base = unicodedata.normalize("NFKC", ch)
    if len(base) != 1 or not base.isalnum():
        return None
    return "math" + "".join(abbreviations), base


def _bold_style_commands() -> list[str]:
    found = {
        parsed[0]
        for codepoint in range(0x1D400, 0x1D800)
        if (parsed := _mathematical_bold_form(chr(codepoint)))
    }
    return sorted(found, key=len, reverse=True)


# Longer commands first so mathbfit is not tokenized as mathbf.
_DECORATIONS = "overline|bar|vec|hat|dot|tilde|text|" + "|".join(_bold_style_commands())
_DECORATION_HEAD = re.compile(r"\\(?:" + _DECORATIONS + r")(?![A-Za-z\u2063])|(?:" + _DECORATIONS + r")(?=\()")
_CALL_HEAD = re.compile(r"(?:sqrt|root|" + _ENVIRONMENTS + "|" + _DECORATIONS + r")(?=\()")
_MATH_SPAN = re.compile(r"\$\$(.*?)\$\$|\$(.*?)\$|\\\((.*?)\\\)|\\\[(.*?)\\\]", re.DOTALL)


def _layout_arguments(text: str) -> str:
    """Consume balanced arguments belonging only to presentation commands."""
    head = re.compile(r"\\(begin|end)\s*\{([^{}]*)\}|\\(?:hspace|vspace|tag|label)\*?(?![A-Za-z])")
    out, pos = [], 0
    while match := head.search(text, pos):
        out.append(text[pos:match.start()])
        pos = match.end()
        if match[1]:
            environment = match[2]
            opening = match[1] == "begin"
            if opening:
                start = pos + len(text[pos:]) - len(text[pos:].lstrip())
                if text[start:start + 1] == "[":
                    group = _group(text, start, delimiters="[")
                    if group:
                        pos = group[1]
                if environment == "array":
                    start = pos + len(text[pos:]) - len(text[pos:].lstrip())
                    if text[start:start + 1] == "{":
                        group = _group(text, start)
                        if group:
                            pos = group[1]
            out.append((environment + "(" if opening else ")")
                       if environment in _ENVIRONMENTS.split("|") else " ")
        else:
            start = pos + len(text[pos:]) - len(text[pos:].lstrip())
            if text[start:start + 1] == "{":
                group = _group(text, start)
                if group:
                    pos = group[1]
            out.append(" ")
    out.append(text[pos:])
    return "".join(out)


def _layout(text: str) -> str:
    text = re.sub(r"\\\\(?:\s*\[[^\]]*\])?", ";", text)
    text = re.sub(r"\\(?:newline|linebreak|cr)(?![A-Za-z])(?:\s*\[[^\]]*\])?", ";", text)
    text = _layout_arguments(text)
    text = re.sub(r"\\(?:nonumber|notag|hline)(?![A-Za-z])", " ", text)
    text = re.sub(r"\\(?:left|right)\s*\.", " ", text)
    text = re.sub(r"\\(?:big|Big|bigg|Bigg)[lrm]?(?![A-Za-z])", " ", text)
    text = text.replace("~", " ")
    # Each math span is a separate content unit. Outer delimiters carry no content.
    text = _MATH_SPAN.sub(lambda m: ";" + next(g for g in m.groups() if g is not None).strip() + ";", text)
    text = text.replace("&", "¦")
    out, depth = [], 0
    for ch in text:
        if ch in "{([":
            depth += 1
        elif ch in "})]":
            depth = max(0, depth - 1)
        out.append(";" if ch == "\n" and depth == 0 else " " if ch == "\n" else ch)
    text = re.sub(r"\s*;\s*(?:;\s*)*", ";", "".join(out)).strip("; ")
    return re.sub(r"\s*;\s*(?:;\s*)*", ";", _LAYOUT.sub(" ", text)).strip("; ")

_FORMATTING = {"left", "right", "mathrm", "mathit", "displaystyle", "textstyle", "quad", "qquad"}
# Token boundaries must survive removing groups: numbers, commands, and roots
# cannot be manufactured by concatenating formerly separate operands.
_TOKEN = re.compile(r"!=|<=|>=|\\[A-Za-z]+|(?:sqrt|root|" + _ENVIRONMENTS + "|" + _DECORATIONS + r")(?=\()|\d+(?:\.\d+)?|.", re.DOTALL)
_BOUNDARY = "\u2063"  # Canonical token boundary, never a mathematical parenthesis.
_PRECEDENCE = {";": -1, "¦": -1, ",": -1, "=": 0, "<": 0, ">": 0, "!=": 0, "<=": 0, ">=": 0, "!": 5, "+": 1, "-": 1, "±": 1, "∓": 1,
               "*": 2, "/": 2, "÷": 2, "^": 4, "_": 4}


def _group(text: str, start: int, *, delimiters: str = "{") -> tuple[str, int] | None:
    """Balanced group (or a single token) starting at ``start``."""
    while start < len(text) and (text[start].isspace() or text[start] == _BOUNDARY):
        start += 1
    if start >= len(text) or text[start] in "})]":
        return None
    if text[start] not in delimiters:
        if text[start] in "{([":
            return None
        command = _COMMAND_WORD.match(text, start)
        return (command.group(), command.end()) if command else (text[start], start + 1)
    stack = []
    tokens = re.compile(r"\\[A-Za-z]+|\\.|.", re.DOTALL)
    for token in tokens.finditer(text, start):
        ch = token.group()
        if ch in ("{", "(", "["):
            stack.append((ch, token.start()))
        elif ch in ("}", ")", "]"):
            opening, begin = stack[-1]
            if {"{": "}", "(": ")", "[": "]"}[opening] != ch:
                if not (opening in "([" and ch in ")]" and "," in text[begin:token.start()]):
                    return None
            stack.pop()
            if not stack:
                return text[start + 1:token.start()], token.end()
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
    while start < len(text) and (text[start].isspace() or text[start] == _BOUNDARY):
        start += 1
    root = _ROOT_HEAD.match(text, start)
    if root:
        _, _, end = _root_argument(text, root)
        return text[start:end], end
    decoration = _DECORATION_HEAD.match(text, start)
    if decoration:
        argument = _group(text, decoration.end(), delimiters="{(")
        if argument:
            return text[start:argument[1]], argument[1]
    barred = _BARRED.match(text, start)
    if barred:
        return barred.group(), barred.end()
    number = _NUMBER.match(text, start)
    if number:
        return number.group(), number.end()
    # Square brackets are content, including after a root index. Curly braces
    # and parentheses still delimit a radicand and are not part of its value.
    if start < len(text) and text[start] == "[":
        group = _group(text, start, delimiters="[")
        if group:
            # Keep the bracket that actually closed the interval. [0,1) and
            # [0,1] are different endpoints.
            return "[" + group[0] + text[group[1] - 1], group[1]
    return _group(text, start, delimiters="{(")


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


def _root_call_parts(text: str) -> tuple[str, str] | None:
    """Split ``index, radicand`` at a top-level comma."""
    depth = 0
    for pos, ch in enumerate(text):
        if ch in "{([":
            depth += 1
        elif ch in "})]":
            if depth == 0:
                return None
            depth -= 1
        elif ch == "," and depth == 0:
            return text[:pos], text[pos + 1:]
    return None


def _content(text: str) -> str:
    """Strip formatting, retaining root scope and expression grouping."""
    out, pos = [], 0
    while pos < len(text):
        root = _ROOT_HEAD.match(text, pos)
        if root:
            index, inner, pos = _root_argument(text, root)
            if index is None and root.group() == "root":
                parts = _root_call_parts(inner)
                if parts:
                    index, inner = parts
            index_content = _content(index) if index is not None else ""
            # An explicit index of two is the ordinary square root.
            square = index_content == "2"
            if square:
                index = None
            elif index is not None and _precedence(index_content) == -1:
                index_content = f"({index_content})"
            head = "sqrt" if square or (index is None and root.group() != "root") else "root"
            argument = f"{index_content}," if index is not None else ""
            # Combining root strokes are removed before this point. An explicit
            # overline or bar is a decoration and stays inside the radicand.
            radicand = _content(inner)
            if index is not None and _precedence(radicand) == -1:
                radicand = f"({radicand})"
            out.append(f"{head}({argument}{radicand})")
        else:
            decoration = _DECORATION_HEAD.match(text, pos)
            if decoration:
                group = _group(text, decoration.end(), delimiters="{(")
                if group:
                    name = decoration.group().lstrip("\\")
                    inner = ("␠".join(group[0].split()) if name == "text" else _content(group[0]))
                    out.append(f"{name}({inner})")
                    pos = group[1]
                    continue
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
            if not ch.isspace():
                if pos and text[pos - 1].isspace():
                    # Preserve lexical boundaries without inventing mathematical groups.
                    token = _TOKEN.match(text, pos)
                    prefix = "".join(out)
                    if (_TOKEN.findall(prefix + text[pos:])
                            != _TOKEN.findall(prefix) + _TOKEN.findall(text[pos:])):
                        out.append(_BOUNDARY)
                    out.append({"{": "(", "}": ")"}.get(token.group(), token.group()))
                    pos = token.end()
                    continue
                out.append({"{": "(", "}": ")"}.get(ch, ch))
            pos += 1
    return "".join(out)


def _precedence(text: str, *, script: str | None = None) -> int:
    """Weakest top-level operation; adjacency is implicit multiplication."""
    weakest, operand, pos = 6, False, 0
    while pos < len(text):
        if text[pos] == _BOUNDARY:
            pos += 1
            continue
        ch = text[pos:pos + 2] if text[pos:pos + 2] in _PRECEDENCE else text[pos]
        if ch in _PRECEDENCE:
            unary = ch in "+-±∓" and not operand
            priority = _PRECEDENCE[ch]
            if script in ("^", "_") and ch in ("^", "_") and ch != script:
                priority = 2  # Mixed scripts must retain their nested scope.
            weakest = min(weakest, 3 if unary else priority)
            operand = operand if ch == "!" else False
            pos += len(ch)
            continue
        if operand:
            weakest = min(weakest, 3)
        root = _CALL_HEAD.match(text, pos)
        start = root.end() if root else pos
        group = _expression_group(text, start)
        number = _NUMBER.match(text, pos)
        command = _COMMAND_WORD.match(text, pos)
        pos = group[1] if group else command.end() if command else number.end() if number else pos + 1
        operand = True
    return weakest


def _function_tail(text: str) -> bool:
    """A letter identifier, optionally indexed, powered or primed."""
    if text[-1:].isalpha() or text.endswith(("′", "'")):
        return True
    script = re.search(r"[_^]\d+$", text)
    if script:
        return _function_tail(text[:script.start()])
    if text.endswith(")"):
        depth = 0
        for pos in range(len(text) - 1, -1, -1):
            depth += {"(": -1, ")": 1}.get(text[pos], 0)
            if depth == 0:
                return pos > 0 and text[pos - 1] in "^_" and _function_tail(text[:pos - 1])
    return False


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
        inner = group[0] if root and root.group() == "text" else _grouping(group[0])
        end = group[1]
        if root:
            out += f"{root.group()}({inner})"
        elif group[2:] != ("(", ")") or _precedence(inner) == -1:
            out += group[2] + inner + group[3]
        else:
            strength = _precedence(inner)
            left = out[-2:] if out[-2:] in _PRECEDENCE else out[-1:]
            right = text[end:end + 2] if text[end:end + 2] in _PRECEDENCE else text[end:end + 1]
            lp = _PRECEDENCE.get(left, 3) if left else -1
            if left in ("+", "-", "±", "∓") and (len(out) == 1 or out[-2] in "(=<>!+-±∓*/÷^_"):
                lp = 3  # unary sign
            rp = _PRECEDENCE.get(right, 3) if right else -1
            signed_power = (inner[:1] in ("+", "-", "±", "∓")
                            and _precedence(inner.lstrip("+-±∓"), script=left) >= 4)
            left_ok = strength > lp or (left in ("^", "_") and (_precedence(inner, script=left) >= 4 or signed_power))
            right_ok = strength > rp if right in ("^", "_") else strength >= rp
            signed_factor = (left and left not in _PRECEDENCE and inner[:1] in ("+", "-", "±", "∓"))
            tail = text[end:]
            joins_tokens = (_TOKEN.findall(out + inner + tail)
                            != _TOKEN.findall(out) + _TOKEN.findall(inner) + _TOKEN.findall(tail))
            command_argument = re.search(r"\\[A-Za-z]+" + _BOUNDARY + r"$", out)
            # A grouped numeric script followed by a number needs a lexical
            # boundary, just like whitespace after an ungrouped script.
            if (left in ("^", "_") and _NUMBER.fullmatch(inner)
                    and _NUMBER.match(tail) and left_ok and right_ok):
                out += inner + _BOUNDARY
                pos = end
                continue
            if inner.endswith("!") and tail.startswith("=") and left_ok and right_ok:
                out += inner + _BOUNDARY
                pos = end
                continue
            function_argument = _function_tail(out)
            out += (inner if inner and left_ok and right_ok and not signed_factor and not joins_tokens
                    and not command_argument and not function_argument
                    else f"({inner})")
        pos = end
    return out


_SPACING_MARK = {
    "\u00af": "\u0304",  # macron
    "\u203e": "\u0305",  # overline
    "\uffe3": "\u0304",  # fullwidth macron
    "\u02dc": "\u0303",  # spacing tilde
}


def _combining_notation(text: str, *, decompose: bool = True) -> str:
    # Decompose precomposed accents as well as explicitly combining accents.
    # A later pass can skip decomposition so compatibility folding cannot
    # turn a finished letter back into a mark.
    if decompose:
        text = unicodedata.normalize("NFD", text)
    # A trailing stroke on an ungrouped numeric radicand is its vinculum.
    text = re.sub(r"(\\sqrt(?![A-Za-z])|[√∛∜])\s*(\d+(?:\.\d+)?)[\u0304\u0305]+(?![^\s\u0304\u0305][\u0304\u0305])",
                  lambda m: m[1] + "{" + m[2] + "}", text)
    # A combining stroke that is the whole radicand is the root's own line.
    # Remove it before the generic overline conversion, so an explicit
    # \overline stays a different decoration.
    def _root_stroke(match: re.Match) -> str:
        body = match.group(3) if match.group(3) is not None else match.group(4)
        return match.group(1) + (match.group(2) or "") + "{" + re.sub(r"[\u0304\u0305]", "", body) + "}"

    text = re.sub(
        r"(\\sqrt(?![A-Za-z])|[√∛∜])(\s*\[[^\[\]]*\])?\s*"
        r"(?:\{((?:[^\s\u0304\u0305{}][\u0304\u0305]+)+)\}|((?:[^\s\u0304\u0305][\u0304\u0305]+)+))",
        _root_stroke,
        text,
    )
    text = _BARRED.sub(lambda m: r"\overline{" + re.sub("[\u0304\u0305]", "", m.group()) + "}", text)
    marks = {"\u20d7": "vec", "\u0302": "hat", "\u0307": "dot", "\u0303": "tilde"}

    def decorate(match):
        inner = match[1]
        for mark in match[2]:
            inner = "\\" + marks[mark] + "{" + inner + "}"
        return inner

    text = re.sub(r"([^\s])([\u20d7\u0302\u0307\u0303]+)", decorate, text)
    return re.sub(r"[\u0304\u0305\u20d7\u0302\u0307\u0303]",
                  lambda m: f"mark[{ord(m[0]):04x}]", text)


def _unicode_notation(text: str) -> str:
    """Protect scripts, bold letters, and enclosed numbers before compatibility folding."""
    out, plain, script, kind = [], [], [], ""

    def flush_script():
        if script:
            out.append(kind + "(" + unicodedata.normalize("NFKC", "".join(script)) + ")")
            script.clear()

    def flush_plain():
        out.append(unicodedata.normalize("NFKC", "".join(plain)))
        plain.clear()

    for ch in text:
        decomposition = unicodedata.decomposition(ch).split()
        tag = decomposition[0] if decomposition else ""
        operator = {"<super>": "^", "<sub>": "_"}.get(tag, "")
        if tag == "<fraction>":
            flush_script()
            flush_plain()
            numerator, _, denominator = "".join(chr(int(code, 16)) for code in decomposition[1:]).partition("⁄")
            out.append(f"(({numerator})/({denominator}))" if denominator else ch)
            continue
        if operator:
            flush_plain()
            if operator != kind:
                flush_script()
            kind = operator
            script.extend(chr(int(code, 16)) for code in decomposition[1:])
        else:
            flush_script()
            if ch in _SPACING_MARK:
                mark = _SPACING_MARK[ch]
                if not plain:
                    plain.append(mark)
                    continue
                base = unicodedata.normalize("NFKC", plain.pop())
                flush_plain()
                # Emit the decoration before NFKC can compose the mark into a
                # precomposed letter. That letter would not be idempotent.
                command = "tilde" if mark == "\u0303" else "overline"
                out.append("\\" + command + "{" + base + "}")
                continue
            styled = _mathematical_bold_form(ch)
            if styled:
                flush_plain()
                command, base = styled
                out.append(f"{command}({base})")
                continue
            name = unicodedata.name(ch, "")
            enclosed = (unicodedata.numeric(ch, None) is not None
                        and any(label in name for label in ("CIRCLED", "PARENTHESIZED", "FULL STOP", "DINGBAT")))
            if enclosed:
                flush_plain()
                out.append(ch)
            else:
                plain.append(ch)
    flush_script()
    flush_plain()
    return "".join(out)


def skeleton(text: str) -> str:
    text = _unicode_notation(_combining_notation(_FENCE.sub("", text)))
    # NFKC of spacing marks introduces combining marks. Attach those before
    # the result is returned, without decomposing characters already folded.
    text = _combining_notation(text, decompose=False)
    # Remove layout before parsing arguments: spacing can occur between a
    # fraction/root command and its operand. Keep separators until _content.
    text = _layout(text)
    text = _COMMAND_WORD.sub(lambda m: " " if m.group()[1:] in _FORMATTING
                             else _SYMBOLS.get(m.group()[1:], m.group()), text)
    text = re.sub(r"\^(?:\s*\{\s*°\s*\}|\s*\(\s*°\s*\)|\s*°)", "°", text)
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
