"""Sympy equivalence for expressions that parse as arithmetic and sqrt only."""

from __future__ import annotations

import re

_SAFE = re.compile(r"^[0-9A-Za-z+\-*/^=().,\s]+$")
_MECHANICAL = re.compile(r"[0-9+\-*/^=]|sqrt")


def mechanical_equivalent(left: str, right: str) -> bool | None:
    """True or False when both sides are safe expressions. None otherwise."""
    left = left.strip().strip("$")
    right = right.strip().strip("$")
    if not left or not right:
        return None
    if not _SAFE.fullmatch(left) or not _SAFE.fullmatch(right):
        return None
    if not _MECHANICAL.search(left) or not _MECHANICAL.search(right):
        return None
    lowered = (left + right).lower()
    if "__" in lowered or "lambda" in lowered:
        return None
    try:
        import sympy
        from sympy.parsing.sympy_parser import (
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )
    except ImportError:
        return None
    transforms = standard_transformations + (implicit_multiplication_application,)
    try:
        allowed = {
            "Symbol": sympy.Symbol,
            "Integer": sympy.Integer,
            "Float": sympy.Float,
            "Rational": sympy.Rational,
            "Add": sympy.Add,
            "Mul": sympy.Mul,
            "Pow": sympy.Pow,
            "sqrt": sympy.sqrt,
        }
        a = parse_expr(
            left.replace("^", "**"),
            transformations=transforms,
            local_dict={},
            global_dict=allowed,
            evaluate=True,
        )
        b = parse_expr(
            right.replace("^", "**"),
            transformations=transforms,
            local_dict={},
            global_dict=allowed,
            evaluate=True,
        )
        functions = {func.func for func in a.atoms(sympy.Function)} | {
            func.func for func in b.atoms(sympy.Function)
        }
        if not functions <= {sympy.sqrt}:
            return None
        return bool(sympy.simplify(a - b) == 0)
    except Exception:
        return None
