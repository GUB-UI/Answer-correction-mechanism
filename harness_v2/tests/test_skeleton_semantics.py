"""OCR notation equivalence must preserve operators and expression scope."""

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


@pytest.mark.parametrize("prefix", ["2", "x", "ab", "2x"])
@pytest.mark.parametrize("notation", [r"\sqrt{13}", "√13", "sqrt(13)"])
def test_roots_after_coefficients_and_variables_match(prefix, notation):
    canonical = prefix + "sqrt(13)"
    assert skeleton(prefix + notation) == skeleton(canonical) == canonical
    assert cer(prefix + notation, canonical) == 0


@pytest.mark.parametrize("prefix", ["2", "x", "ab", "2x"])
@pytest.mark.parametrize("outside,inside", [
    ("sqrt(x)+3", "sqrt(x+3)"),
    ("sqrt(x)^2", "sqrt(x^2)"),
    ("sqrt(2sqrt(x))+3", "sqrt(2sqrt(x)+3)"),
])
def test_prefixed_canonical_roots_preserve_scope(prefix, outside, inside):
    assert skeleton(prefix + outside) == prefix + outside
    assert skeleton(prefix + inside) == prefix + inside
    assert cer(prefix + outside, prefix + inside) > 0


@pytest.mark.parametrize("text", [
    r"2\sqrt{13}", "2√13", "2sqrt(13)", "xsqrt(x+3)",
    "absqrt(x)+3", "2xsqrt(x^2)", r"\sqrt{2\sqrt{x}+3}",
    "2sqrt(xsqrt(x+3))+sqrt(y)", r"\frac{2\sqrt{x}}{3}",
    "２√（ｘ＋３）", "2√x²", r"$a \sqrt{x} + b \sqrt{y}$",
])
def test_prefixed_root_normalization_is_idempotent(text):
    normalized = skeleton(text)
    assert skeleton(normalized) == normalized


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


@pytest.mark.parametrize("left,right", [
    ("a+b/c", "(a+b)/c"),
    (r"\frac{a+b}{c}", r"a+\frac{b}{c}"),
    (r"\frac{a}{b+c}", r"\frac{a}{b}+c"),
    (r"\frac{a}{\frac{b}{c}}", r"\frac{\frac{a}{b}}{c}"),
    ("a/(b*c)", "a/b*c"),
    ("a/(bc)", "a/bc"),
    ("a*(b+c)", "a*b+c"),
    ("a-(b-c)", "a-b-c"),
    ("a+(b+c)", "(a+b)+c"),
    (r"x^{a+b}", "x^a+b"),
    ("(x^a)^b", "x^(a^b)"),
    ("(x+y)^2", "x+y^2"),
    (r"x^\frac{a}{b}", "x^a/b"),
    (r"\frac{a}{b}^c", "a/b^c"),
    (r"a/\frac{b}{c}", "a/b/c"),
    ("(12)(34)", "1234"),
    ("(12).34", "12.34"),
    (r"\sqrt{(a+b)/c}", r"\sqrt{a+b/c}"),
])
def test_expression_grouping_collisions_are_rejected(left, right):
    assert skeleton(left) != skeleton(right)
    assert cer(left, right) > 0


@pytest.mark.parametrize("latex,plain", [
    (r"\frac{a+b}{c}", "(a+b)/c"),
    (r"a+\frac{b}{c}", "a+b/c"),
    (r"\frac{a}{b+c}", "a/(b+c)"),
    (r"\frac{ab}{c}", "ab/c"),
    (r"\frac{a}{bc}", "a/(bc)"),
    (r"\frac{a}{\frac{b}{c}}", "a/(b/c)"),
    (r"\frac{\frac{a}{b}}{c}", "(a/b)/c"),
    (r"\frac{a+b}{\frac{c}{d+e}}", "(a+b)/(c/(d+e))"),
    (r"\left(a+b\right)\cdot c", "(a+b)*c"),
    (r"\frac{(a+b)c}{d}", "(a+b)c/d"),
    (r"x^{a+b}", "x^(a+b)"),
    (r"x^\frac{a}{b}", "x^(a/b)"),
    (r"\frac{a}{b}^c", "(a/b)^c"),
    (r"a/\frac{b}{c}", "a/(b/c)"),
    (r"\sqrt{\frac{a+b}{c}}", "sqrt((a+b)/c)"),
    (r"\frac{\sqrt{a+b}}{c}", "sqrt(a+b)/c"),
    ("（ａ＋ｂ）／ｃ", "(a+b)/c"),
])
def test_grouped_notations_match_and_are_idempotent(latex, plain):
    normalized = skeleton(latex)
    assert normalized == skeleton(plain)
    assert cer(latex, plain) == 0
    assert skeleton(normalized) == normalized


@pytest.mark.parametrize("text", [
    "((a+b)/c)", "a/(b/(c+d))", "a*(b+c)", "a-(b-c)",
    "(x^a)^b", "x^(a^b)", "x^{a+b}", "x^{-12}", "(12)(34)",
    r"\frac{\frac{a+b}{c}}{\frac{d}{e+f}}", r"-\frac{a+b}{c}",
    r"\sqrt{\frac{a+b}{\sqrt{c+d}}}^2",
])
def test_grouping_normalization_is_idempotent(text):
    normalized = skeleton(text)
    assert skeleton(normalized) == normalized
