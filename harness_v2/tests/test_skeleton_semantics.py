"""OCR notation equivalence must preserve radicals and exponent operators."""

import pytest

from harness_v2.recognition.skeleton import cer, skeleton


@pytest.mark.parametrize("notation", [
    r"\sqrt{13}", "√13", "√(13)", "√{13}", "√（１３）",
    r"\sqrt 13", r"\sqrt{\overline{13}}", "√1̅3̅", "√13̅",
    "√1̄3̄", "sqrt(13)", r"\sqrt\left(13\right)",
])
def test_root_notations_have_explicit_common_form(notation):
    assert skeleton(notation) == "sqrt(13)"
    assert cer(notation, r"\sqrt{13}") == 0


@pytest.mark.parametrize("latex,unicode,expected", [
    (r"\sqrt{x+3}", "√(x+3)", "sqrt(x+3)"),
    (r"\sqrt{x+3}", "√x̅+̅3̅", "sqrt(x+3)"),
    (r"\sqrt{\sqrt{13}}", "√(√13)", "sqrt(sqrt(13))"),
    (r"\sqrt{\sqrt{13}}", "√√13", "sqrt(sqrt(13))"),
    (r"\sqrt{\frac{x^{2}}{3}}", "√(x²/3)", "sqrt(x^2/3)"),
    (r"\frac{\sqrt{13}}{3}", "√13/3", "sqrt(13)/3"),
    (r"\sqrt{x}^{2}", "√(x)²", "sqrt(x)^2"),
])
def test_grouped_and_nested_roots(latex, unicode, expected):
    assert skeleton(latex) == skeleton(unicode) == expected
    assert skeleton(expected) == expected


@pytest.mark.parametrize("digits,superscript", list(zip("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")))
def test_superscript_digits_keep_exponent_operator(digits, superscript):
    expected = "x^" + digits
    assert skeleton("x" + superscript) == expected
    assert skeleton("x^{" + digits + "}") == expected
    assert skeleton(expected) == expected


@pytest.mark.parametrize("latex,unicode,expected", [
    (r"x^{123}", "x¹²³", "x^123"),
    (r"x^{-12}", "x⁻¹²", "x^-12"),
    (r"x^{+12}", "x⁺¹²", "x^+12"),
    (r"x^{n}", "x^n", "x^n"),
])
def test_exponent_notations_match(latex, unicode, expected):
    assert skeleton(latex) == skeleton(unicode) == expected
    assert cer(unicode, latex) == 0


@pytest.mark.parametrize("ocr,reference", [
    ("13", r"\sqrt{13}"), ("13", "√13"), ("13", "√(13)"),
    ("x+3", r"\sqrt{x+3}"), ("x2", "x²"), ("x2", "x^2"),
    ("x2", r"x^{2}"), ("x", "x²"), ("x123", "x¹²³"),
    ("x/3", r"\frac{\sqrt{x}}{3}"),
    ("x2/3", r"\frac{x^{2}}{3}"),
    ("sqrt(13)", r"\sqrt{\sqrt{13}}"),
    ("sqrt(x)+3", r"\sqrt{x+3}"),
    ("sqrt(x)^2", r"\sqrt{x^{2}}"),
])
def test_dropping_operators_or_changing_root_scope_increases_cer(ocr, reference):
    assert skeleton(ocr) != skeleton(reference)
    assert cer(ocr, reference) > 0
