"""OCR notation equivalence must preserve operators and expression scope."""

import pytest

from harness_v2.recognition.skeleton import cer, skeleton


@pytest.mark.parametrize("notation", [
    r"\sqrt{13}", "√13", "√(13)", "√{13}", "√（１３）",
    r"\sqrt 13", "√1̅3̅", "√13̅",
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


@pytest.mark.parametrize("factor,ungrouped", [
    ("a", "a"), (r"\frac{a}{b}", "a/b"), ("(a+b)", "(a+b)"),
    ("sqrt(d)", "sqrt(d)"), ("a^2", "a^2"),
])
@pytest.mark.parametrize("sign", ["+", "-", "±"])
@pytest.mark.parametrize("variable", ["b", "c"])
def test_signed_implicit_factors_keep_multiplication(factor, ungrouped, sign, variable):
    product = factor + "(" + sign + variable + ")"
    addition = ungrouped + sign + variable
    normalized = skeleton(product)
    assert normalized != skeleton(addition)
    assert cer(product, addition) > 0
    assert skeleton(normalized) == normalized


@pytest.mark.parametrize("superscript,exponent", [
    ("¹⁻²", "1-2"), ("¹⁺²", "1+2"), ("⁻¹⁺²", "-1+2"),
    ("⁺¹⁻²", "+1-2"), ("¹²⁻³⁺⁴", "12-3+4"),
])
def test_complete_superscript_run_is_one_exponent(superscript, exponent):
    normalized = skeleton("x" + superscript)
    assert normalized == skeleton("x^{" + exponent + "}")
    assert normalized != skeleton("x^" + exponent)
    assert cer("x" + superscript, "x^{" + exponent + "}") == 0
    assert skeleton(normalized) == normalized


@pytest.mark.parametrize("prefix", ["a", "a/b", "(a+b)"])
def test_plus_minus_is_distinct_from_addition_of_negative(prefix):
    normalized = skeleton(prefix + "±b")
    assert normalized == skeleton(prefix + r"\pm b")
    assert normalized != skeleton(prefix + "+(-b)")
    assert cer(prefix + "±b", prefix + "+(-b)") > 0
    assert skeleton(normalized) == normalized


@pytest.mark.parametrize("notation", [r"\div", "÷"])
@pytest.mark.parametrize("left,right", [("a", "b"), ("(a+b)", "(c-d)"), ("sqrt(x)", "y^2")])
def test_division_operator_notations_match_without_disappearing(notation, left, right):
    text = left + notation + " " + right
    normalized = skeleton(text)
    assert normalized == skeleton(left + "÷" + right)
    assert normalized != skeleton(left + right)
    assert skeleton(normalized) == normalized


@pytest.mark.parametrize("command", [r"\alphaextra", r"\sin", r"\unknown", r"\timesextra", r"\leqqextra"])
def test_unknown_semantic_commands_and_their_boundaries_survive(command):
    text = "a" + command + " b"
    normalized = skeleton(text)
    assert command in normalized
    assert normalized != skeleton("ab")
    assert normalized != skeleton("a" + command + "b")
    assert normalized != skeleton("a" + command[1:] + "b")
    assert skeleton(normalized) == normalized


@pytest.mark.parametrize("letters", [
    "(s)qrt", "s(q)rt", "sq(r)t", "sqr(t)", "(sq)(rt)", "(sqrt)", "((s))qrt",
])
@pytest.mark.parametrize("prefix", ["", "2", "a+"])
def test_removing_groups_cannot_manufacture_a_function_token(letters, prefix):
    text = prefix + letters + "(x+y)"
    normalized = skeleton(text)
    assert normalized != skeleton(prefix + "sqrt(x+y)")
    assert cer(text, prefix + "sqrt(x+y)") > 0
    assert skeleton(normalized) == normalized


def test_multiline_aligned_latex_matches_plain_text_and_is_idempotent():
    latex = r"""```latex
\begin{aligned}
\displaystyle \alpha & = \frac{a+b}{c} \\
\beta & = \left(x+y\right)^2 \, + \pi \\
\theta & = 30\circ \qquad \gamma = \sqrt{x+1}
\end{aligned}
```"""
    plain = "α ¦ = (a+b)/c\nβ ¦ = (x+y)^2 + π\nθ ¦ = 30° γ = sqrt(x+1)"
    expected = "α¦=(a+b)/c;β¦=(x+y)^2+π;θ¦=30°γ=sqrt(x+1)"
    assert skeleton(latex) == skeleton(plain) == expected
    assert cer(latex, plain) == 0
    for text in (latex, plain, expected):
        assert skeleton(skeleton(text)) == expected


@pytest.mark.parametrize("layout", [
    r"\begin{aligned}", r"\end{aligned}",
    r"\begin{gather*}", r"\end{gather*}",
    r"\,", r"\;", r"\:", r"\!", r"\quad", r"\qquad",
    r"\left", r"\right", r"\displaystyle",
])
@pytest.mark.parametrize("left,right", [("12", "34"), ("s", "qrt(x+y)"), (r"\unknown", "b")])
def test_layout_is_removed_without_joining_tokens(layout, left, right):
    latex = left + layout + " " + right
    plain = left + " " + right
    normalized = skeleton(latex)
    assert normalized == skeleton(plain)
    assert normalized != skeleton(left + right)
    assert skeleton(normalized) == normalized
    if right[0].isdigit() or not layout[-1].isalpha():
        # Control words need a terminator before letters, but other layout
        # constructs must also preserve boundaries with no surrounding spaces.
        assert skeleton(left + layout + right) == normalized


@pytest.mark.parametrize("command,symbol", [
    ("alpha", "α"), ("beta", "β"), ("gamma", "γ"), ("delta", "δ"),
    ("epsilon", "ε"), ("zeta", "ζ"), ("eta", "η"), ("theta", "θ"),
    ("iota", "ι"), ("kappa", "κ"), ("lambda", "λ"), ("mu", "μ"),
    ("nu", "ν"), ("xi", "ξ"), ("omicron", "ο"), ("pi", "π"),
    ("rho", "ρ"), ("sigma", "σ"), ("tau", "τ"), ("upsilon", "υ"),
    ("phi", "φ"), ("chi", "χ"), ("psi", "ψ"), ("omega", "ω"),
    ("Gamma", "Γ"), ("Delta", "Δ"), ("Theta", "Θ"), ("Lambda", "Λ"),
    ("Xi", "Ξ"), ("Pi", "Π"), ("Sigma", "Σ"), ("Upsilon", "Υ"),
    ("Phi", "Φ"), ("Psi", "Ψ"), ("Omega", "Ω"),
    ("varepsilon", "ϵ"), ("vartheta", "ϑ"), ("varkappa", "ϰ"),
    ("varpi", "ϖ"), ("varrho", "ϱ"), ("varsigma", "ς"), ("varphi", "ϕ"),
    ("circ", "°"), ("times", "×"), ("cdot", "·"), ("infty", "∞"),
    ("neq", "≠"), ("pm", "±"), ("div", "÷"),
])
def test_symbol_commands_match_unicode_and_are_idempotent(command, symbol):
    latex = "a\\" + command + " b"
    plain = "a" + symbol + "b"
    normalized = skeleton(latex)
    assert normalized == skeleton(plain)
    assert normalized != skeleton("ab")
    assert normalized != skeleton("a\\" + command + "extra b")
    assert cer(latex, plain) == 0
    for text in (latex, plain, normalized):
        assert skeleton(skeleton(text)) == normalized


@pytest.mark.parametrize("latex,plain", [
    (r"\frac\, {a+b}\; {c}", "(a+b)/c"),
    (r"\sqrt\quad {a+b}", "sqrt(a+b)"),
    (r"\sqrt\,\pi", "sqrt(π)"),
    (r"x^{\alpha+\beta}", "x^(α+β)"),
])
def test_layout_and_symbols_preserve_fraction_root_and_power_scope(latex, plain):
    normalized = skeleton(latex)
    assert normalized == skeleton(plain)
    assert skeleton(normalized) == normalized


@pytest.mark.parametrize("opening,closing", [
    (r"\(", r"\)"), (r"\[", r"\]"), ("$", "$"), ("$$", "$$"),
])
@pytest.mark.parametrize("content,expected", [
    ("x", "x"),
    (r"\frac{a+b}{c}", "(a+b)/c"),
    (r"\sqrt{x+1} + y^{a+b}", "sqrt(x+1)+y^(a+b)"),
    ("a/(b*c)", "a/(b*c)"),
    ("(12)(34)", "12(34)"),
])
def test_math_mode_delimiters_match_undelimited_content(opening, closing, content, expected):
    for text in (opening + content + closing, opening + "\n " + content + " \n" + closing):
        normalized = skeleton(text)
        assert normalized == skeleton(content) == expected
        assert cer(text, content) == 0
        assert skeleton(normalized) == normalized


@pytest.mark.parametrize("opening,closing", [
    (r"\(", r"\)"), (r"\[", r"\]"), ("$", "$"), ("$$", "$$"),
])
@pytest.mark.parametrize("left,right", [
    ("12", "34"), ("s", "qrt(x+y)"), (r"\unknown", "b"),
])
def test_math_mode_delimiters_preserve_token_boundaries(opening, closing, left, right):
    text = opening + left + closing + opening + right + closing
    normalized = skeleton(text)
    assert normalized == skeleton(left + ";" + right)
    assert normalized != skeleton(left + right)
    assert skeleton(normalized) == normalized


@pytest.mark.parametrize("text,expected", [
    (r"\{x\}", r"\{x\}"),
    (r"\lbrack x\rbrack", "[x]"),
    (r"\langle x\rangle", "\\langle\u2063x\\rangle\u2063"),
])
def test_escaped_content_brackets_survive_math_mode_removal(text, expected):
    for wrapped in (text, r"\(" + text + r"\)", r"\[" + text + r"\]", "$" + text + "$"):
        normalized = skeleton(wrapped)
        assert normalized == expected
        assert normalized != skeleton("x")
        assert skeleton(normalized) == normalized


def test_many_math_mode_pairs_match_plain_ocr_page():
    fragments = [r"x = \frac{a+b}{c}", r"y = \sqrt{x+1}", "z = (x+y)^2"] * 5
    pairs = [(r"\(", r"\)"), (r"\[", r"\]"), ("$", "$"), ("$$", "$$")]
    latex = "\n".join(pairs[i % 4][0] + text + pairs[i % 4][1]
                      for i, text in enumerate(fragments))
    plain = "\n".join(fragments)
    normalized = skeleton(latex)
    assert normalized == skeleton(plain)
    assert cer(latex, plain) == 0
    assert skeleton(normalized) == normalized
