"""Deterministic checks on an OCR transcript.

The checks never change a score and never read a reference score or a problem
id. They only route a grade to review_required. The gate is off unless the
run sets transcript_quality_gate to "on". An error inside the gate records
gate_error and leaves the grade as the model wrote it.

A clear garble is transcript_garbled: a replacement or control character, an
unbalanced bracket or fraction, a stray symbol on a math line, or a Cyrillic
letter mixed into another script. A numeric contradiction between two
consecutive simple equations is transcript_inconsistent. Both sides are
compared only when a safe arithmetic evaluator, with no eval, can reduce them
to numbers. Superscript digits are exponents. Dollar delimiters, alignment
ampersands, absolute-value bars, percent signs, half-open intervals, and
\frac with a digit or letter argument are ordinary Math I/A notation. A
question mark on a math line, or a fullwidth Latin letter mixed with an
ASCII letter, is a flag. That flag routes to review only together with model
uncertainty of at least 0.5 or a model confidence of at most 0.5.
"""

from __future__ import annotations

import json
import math
import unicodedata
from typing import Any

from harness_v2.core.contracts import Check, GradingResult

TRANSCRIPT_GARBLED = "transcript_garbled"
TRANSCRIPT_INCONSISTENT = "transcript_inconsistent"
_CONFIDENCE_AT = 0.5
_STRAY = set("#@~`§¶†‡•◆■□●○※\ufffd")
_CLOSERS = {")": "([", "]": "[(", "}": "{"}
_FRAC = ("\\dfrac", "\\tfrac", "\\frac")
_SUPER_DIGITS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
_MAX_ABS_EXPONENT = 8
_MAX_ABS_POWER = 1e12


class _BadExpr(Exception):
    pass


def apply_transcript_gate(result: GradingResult, text: str, *, enabled: bool) -> None:
    """Route to review when the gate is on. Scores stay as the model wrote them."""
    if not enabled or result.status == "failed":
        return
    try:
        reasons = transcript_reasons(text, low_confidence=_low_confidence(result))
    except Exception as exc:
        result.checks.append(Check("gate_error", False, type(exc).__name__))
        return
    if not reasons:
        return
    if result.status == "validated":
        result.status = "review_required"
    parts = [part for part in result.failure_reason.split(",") if part]
    for code in reasons:
        if code not in parts:
            parts.append(code)
        result.checks.append(Check(code, False, code))
    result.failure_reason = ",".join(parts)


def transcript_reasons(text: str, *, low_confidence: bool = False) -> list[str]:
    """Reason codes for this transcript, in a stable order."""
    reasons: list[str] = []
    if _garbled(text) or (low_confidence and _soft_flag(text)):
        reasons.append(TRANSCRIPT_GARBLED)
    if _contradicts(text):
        reasons.append(TRANSCRIPT_INCONSISTENT)
    return reasons


def _low_confidence(result: GradingResult) -> bool:
    for item in result.items:
        value = item.uncertainty
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if value >= _CONFIDENCE_AT:
            return True
    payload = _object(result.raw_response)
    if payload is None:
        return False
    confidence = payload.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return False
    return confidence <= _CONFIDENCE_AT


def _object(raw: str) -> dict[str, Any] | None:
    try:
        loaded = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _garbled(text: str) -> bool:
    for char in text:
        if char in "\n\t\r":
            continue
        if char == "\ufffd" or unicodedata.category(char) in {"Cc", "Cs", "Co", "Cf"}:
            return True
    if _unbalanced(text):
        return True
    for line in text.splitlines():
        if not _math_line(line):
            continue
        if any(char in _STRAY for char in line):
            return True
    for token in text.split():
        if _hard_mix(token):
            return True
    return False


def _soft_flag(text: str) -> bool:
    for line in text.splitlines():
        if "?" in line and _math_line(line):
            return True
        for token in line.split():
            ascii_latin = any(("A" <= char <= "Z") or ("a" <= char <= "z") for char in token)
            fullwidth = any("\uff21" <= char <= "\uff3a" or "\uff41" <= char <= "\uff5a" for char in token)
            if ascii_latin and fullwidth:
                return True
    return False


def _math_line(line: str) -> bool:
    return "=" in line or "\\frac" in line or any(mark in line for mark in ("·", "×", "÷"))


def _hard_mix(token: str) -> bool:
    scripts: set[str] = set()
    for char in token:
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        if "CYRILLIC" in name:
            scripts.add("cyrillic")
        elif "GREEK" in name:
            scripts.add("greek")
        elif "CJK" in name or "HIRAGANA" in name or "KATAKANA" in name:
            scripts.add("cjk")
        elif "LATIN" in name:
            scripts.add("latin")
    return "cyrillic" in scripts and len(scripts) > 1


def _unbalanced(text: str) -> bool:
    stack: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        if text.startswith(_FRAC, index):
            command = next(item for item in _FRAC if text.startswith(item, index))
            index += len(command)
            for _group in range(2):
                end = _frac_arg(text, index)
                if end is None:
                    return True
                index = end
            continue
        char = text[index]
        if char == "\\" and index + 1 < length:
            index += 2
            continue
        if char == "$":
            index += 1
            continue
        if char in "([{":
            stack.append(char)
        elif char in _CLOSERS:
            if not stack or stack[-1] not in _CLOSERS[char]:
                return True
            stack.pop()
        index += 1
    return bool(stack)


def _frac_arg(text: str, index: int) -> int | None:
    length = len(text)
    while index < length and text[index].isspace():
        index += 1
    if index >= length:
        return None
    if text[index] == "{":
        return _group_end(text, index)
    if text[index].isascii() and text[index].isalnum():
        return index + 1
    return None


def _group_end(text: str, start: int) -> int | None:
    depth = 0
    index = start
    while index < len(text):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    return None


def _contradicts(text: str) -> bool:
    equations = [_equation(line.strip()) for line in text.splitlines() if line.strip()]
    for first, second in zip(equations, equations[1:]):
        if first is None or second is None:
            continue
        env: dict[str, float] = {}
        for equation in (first, second):
            _bind(equation, env)
        for equation in (first, second):
            left = _eval_arith(equation[0], env)
            right = _eval_arith(equation[1], env)
            if left is None or right is None:
                continue
            if abs(left - right) > 1e-6:
                return True
    return False


def _equation(line: str) -> tuple[Any, Any] | None:
    if line.count("=") != 1:
        return None
    left, right = line.split("=")
    try:
        return _parse_expr(left), _parse_expr(right)
    except _BadExpr:
        return None


def _bind(equation: tuple[Any, Any], env: dict[str, float]) -> None:
    left, right = equation
    if left[0] == "name":
        value = _eval_arith(right, env)
        if value is not None:
            env[left[1]] = value
    elif right[0] == "name":
        value = _eval_arith(left, env)
        if value is not None:
            env[right[1]] = value


def _parse_expr(text: str) -> Any:
    tokens = _tokenize(text)
    parser = _ExprParser(tokens)
    node = parser.expr()
    if parser.peek() is not None:
        raise _BadExpr()
    return node


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char.isspace():
            index += 1
            continue
        if char in "+-*/()":
            tokens.append(char)
            index += 1
            continue
        if char in "·×":
            tokens.append("*")
            index += 1
            continue
        if char == "÷":
            tokens.append("/")
            index += 1
            continue
        if ("0" <= char <= "9") or char == ".":
            start = index
            index += 1
            while index < length and (("0" <= text[index] <= "9") or text[index] == "."):
                index += 1
            token = text[start:index]
            if token.count(".") > 1 or not any("0" <= item <= "9" for item in token):
                raise _BadExpr()
            tokens.append(token)
            index = _attach_power(text, index, tokens)
            continue
        if char.isascii() and char.isalpha():
            start = index
            index += 1
            while index < length and text[index].isascii() and (text[index].isalnum() or text[index] == "_"):
                index += 1
            tokens.append(text[start:index])
            index = _attach_power(text, index, tokens)
            continue
        raise _BadExpr()
    return tokens


def _attach_power(text: str, index: int, tokens: list[str]) -> int:
    sign = ""
    if index < len(text) and text[index] in "⁺⁻":
        if index + 1 >= len(text) or text[index + 1] not in "⁰¹²³⁴⁵⁶⁷⁸⁹":
            raise _BadExpr()
        sign = "-" if text[index] == "⁻" else ""
        index += 1
    if index >= len(text) or text[index] not in "⁰¹²³⁴⁵⁶⁷⁸⁹":
        return index
    start = index
    while index < len(text) and text[index] in "⁰¹²³⁴⁵⁶⁷⁸⁹":
        index += 1
    tokens.append("**")
    if sign:
        tokens.append("-")
    tokens.append(text[start:index].translate(_SUPER_DIGITS))
    return index


class _ExprParser:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.index = 0

    def peek(self) -> str | None:
        if self.index >= len(self.tokens):
            return None
        return self.tokens[self.index]

    def eat(self, expected: str | None = None) -> str:
        token = self.peek()
        if token is None or (expected is not None and token != expected):
            raise _BadExpr()
        self.index += 1
        return token

    def expr(self) -> Any:
        node = self.term()
        while self.peek() in {"+", "-"}:
            op = self.eat()
            node = ("bin", op, node, self.term())
        return node

    def term(self) -> Any:
        node = self.factor()
        while self.peek() in {"*", "/"}:
            op = self.eat()
            node = ("bin", op, node, self.factor())
        return node

    def factor(self) -> Any:
        if self.peek() == "-":
            self.eat()
            return ("neg", self.factor())
        node = self._atom()
        if self.peek() == "**":
            self.eat("**")
            node = ("bin", "**", node, self.factor())
        return node

    def _atom(self) -> Any:
        token = self.peek()
        if token == "(":
            self.eat()
            node = self.expr()
            self.eat(")")
            return node
        if token is None:
            raise _BadExpr()
        self.eat()
        if token[0].isdigit() or token[0] == ".":
            try:
                return ("num", float(token))
            except (ValueError, OverflowError):
                raise _BadExpr() from None
        if token[0].isalpha():
            return ("name", token)
        raise _BadExpr()


def _eval_arith(node: Any, env: dict[str, float]) -> float | None:
    kind = node[0]
    if kind == "num":
        return float(node[1])
    if kind == "name":
        return env.get(node[1])
    if kind == "neg":
        value = _eval_arith(node[1], env)
        return None if value is None else -value
    if kind != "bin":
        return None
    left = _eval_arith(node[2], env)
    right = _eval_arith(node[3], env)
    if left is None or right is None:
        return None
    op = node[1]
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    if op == "*":
        return left * right
    if op == "/" and right != 0:
        return left / right
    if op == "**":
        if abs(right) > _MAX_ABS_EXPONENT or abs(left) > _MAX_ABS_POWER:
            return None
        try:
            value = left ** right
        except (OverflowError, ValueError, ArithmeticError):
            return None
        if isinstance(value, complex) or not math.isfinite(value) or abs(value) > _MAX_ABS_POWER:
            return None
        return float(value)
    return None
