"""Regressions for the independent PR #12 skeleton collision review."""

import pytest

from harness_v2.recognition.skeleton import cer, skeleton


def check_forms(forms, different):
    canonical = skeleton(forms[0])
    for text in forms:
        assert skeleton(text) == canonical
        assert cer(text, forms[0]) == 0
    for text in different:
        assert skeleton(text) != canonical
        assert cer(text, forms[0]) > 0
    for text in [*forms, *different, canonical]:
        assert skeleton(skeleton(text)) == skeleton(text)


@pytest.mark.parametrize('forms,different', [
    ([r'\sqrt[3]{x}', '∛x', 'root(3,x)'], [r'\sqrt{3}x', r'\sqrt{x}']),
    ([r'\sqrt[4]{x+1}', '∜(x+1)', 'root(4,x+1)'], [r'\sqrt{4}(x+1)']),
    ([r'\sqrt[n]{x+1}', 'root(n,x+1)'], [r'\sqrt{n}(x+1)']),
    ([r'\sqrt{\sqrt[3]{x}}', '√∛x', 'sqrt(root(3,x))'], ['sqrt(3)x']),
])
def test_fix_01_indexed_roots(forms, different):
    check_forms(forms, different)



@pytest.mark.parametrize('forms,different', [
    ([r'a_{n+1}', 'a_(n+1)'], ['a(n+1)', 'a_n+1']),
    ([r'x_{2}', 'x_2'], ['x2']),
    ([r'\log_{2} x', r'\log_2 x'], [r'\log 2x']),
    ([r'a_{b_c}', 'a_(b_c)'], ['(a_b)_c']),
])
def test_fix_02_subscripts(forms, different):
    check_forms(forms, different)


@pytest.mark.parametrize('forms,different', [
    (['(n+1)!', '{n+1}!'], ['n+1!']),
    (['(2n)!', '{2n}!'], ['2n!']),
    (['x!=y', 'x≠y', r'x\neq y'], ['x!y', '(x!)=y']),
    (['(x^2)!', '{x^2}!'], ['x^2!']),
])
def test_fix_03_factorial_and_inequality(forms, different):
    check_forms(forms, different)


@pytest.mark.parametrize('forms,different', [
    ([r'x\in[0,1]', r'x\in [0,1]'], [r'x\in(0,1)']),
    (['[a,b)', '[ a , b )'], ['(a,b)', '[a,b]']),
    (['[x]', '[ x ]'], ['x', '(x)']),
    (['(1,2)', '( 1 , 2 )'], ['1,2', '[1,2]']),
    ([r'\{a,b\}', r'\{ a , b \}'], ['(a,b)', '[a,b]', 'a,b']),
    (['([a,b],c)', '( [a,b] , c )'], ['[a,b],c']),
])
def test_fix_04_bracket_kinds_and_separators(forms, different):
    check_forms(forms, different)


@pytest.mark.parametrize('command', [r'\sin', r'\log', r'\cos', r'\unknown'])
def test_fix_05_synthetic_boundaries_are_not_parentheses(command):
    check_forms([command + ' x^2', command + '  x^{2}'],
                [command + '(x)^2', command + 'x^2'])
    check_forms([command + ' x', '(' + command + ') x'], [command + 'x'])


@pytest.mark.parametrize('forms,different', [
    ([r'x=1\\ y=2', 'x=1\ny=2', 'x=1;y=2'], ['x=1y=2']),
    (['$12$ $34$', '$12$$34$', r'\(12\)\(34\)', '12;34'], ['12 34', '1234']),
    (['a&b', 'a¦b'], ['ab', 'a;b']),
    ([r'\begin{pmatrix}a&b\\c&d\end{pmatrix}', 'pmatrix(a¦b;c¦d)'],
     [r'\begin{vmatrix}a&b\\c&d\end{vmatrix}', r'\begin{pmatrix}a&b&c&d\end{pmatrix}']),
])
def test_fix_06_structural_layout(forms, different):
    check_forms(forms, different)


@pytest.mark.parametrize('environment', ['pmatrix', 'bmatrix', 'vmatrix', 'Vmatrix', 'matrix', 'array', 'cases'])
def test_fix_06_environment_identity(environment):
    check_forms([rf'\begin{{{environment}}}a&b\\c&d\end{{{environment}}}',
                 environment + '(a¦b;c¦d)'],
                ['a¦b;c¦d', environment + '(a¦b¦c¦d)'])


@pytest.mark.parametrize('forms,different', [
    (['2ⁿ', '2^n'], ['2n', '2_n']),
    (['xⁿ', 'x^{n}'], ['x_n', 'xn']),
    (['xₙ₊₁', 'x_{n+1}'], ['x^(n+1)', 'xn+1']),
    (['x⁽ⁿ⁺¹⁾', 'x^{(n+1)}'], ['x(n+1)']),
    (['xᵝ', r'x^\beta'], ['xβ']),
    (['①2x=4', '① 2x=4'], ['12x=4', '1 2x=4']),
    (['⑴x', '⑴ x'], ['(1)x']),
])
def test_fix_07_unicode_scripts_and_enclosed_numbers(forms, different):
    check_forms(forms, different)


@pytest.mark.parametrize('glyph,operator,base', [
    ('ᵃ', '^', 'a'), ('ʰ', '^', 'h'), ('ᶜ', '^', 'c'), ('⁼', '^', '='),
    ('ᵞ', '^', 'γ'), ('ₐ', '_', 'a'), ('ᵢ', '_', 'i'), ('ᵦ', '_', 'β'), ('₌', '_', '='),
])
def test_fix_07_non_digit_script_characters(glyph, operator, base):
    check_forms(['x' + glyph, 'x' + operator + '(' + base + ')'], ['x' + base])


@pytest.mark.parametrize('forms,different', [
    ([r'\overline{AB}', 'A̅B̅', 'ĀB̄', 'overline(AB)'], ['AB']),
    ([r'\overline{z}', 'z̅', 'overline(z)'], ['z']),
    ([r'\bar{z}', 'bar(z)'], ['z', 'overline(z)']),
    ([r'\mathbf{v}', 'mathbf(v)'], ['v']),
    ([r'\text{or}', 'text(or)'], ['or']),
    ([r'\vec{v}', 'v⃗', 'vec(v)'], ['v', r'\mathbf{v}']),
    ([r'\hat{x}', 'x̂', 'hat(x)'], ['x']),
    ([r'\dot{x}', 'ẋ', 'dot(x)'], ['x']),
    ([r'\tilde{x}', 'x̃', 'tilde(x)'], ['x']),
    ([r'\sqrt{\overline{AB}}', '√A̅B̅', 'sqrt(AB)'], ['overline(sqrt(AB))']),
])
def test_fix_08_meaningful_decorations(forms, different):
    check_forms(forms, different)


def test_fix_08_text_content_keeps_word_boundaries():
    check_forms([r'\text{a b}', 'text(a␠b)'], [r'\text{ab}', 'ab'])


@pytest.mark.parametrize('forms,different', [
    ([r'\begin{array}{cc}a&b\\[4pt]c&d\end{array}', 'array(a¦b;c¦d)'], ['array(a¦b¦c¦d)']),
    ([r'\begin{aligned}[t]x&=1\\[4pt]y&=2\end{aligned}', 'x¦=1;y¦=2'], ['tx¦=1;y¦=2']),
    ([r'x=1\newline y=2', 'x=1;y=2'], ['x=1y=2']),
    ([r'x\hspace{1em}+y\tag{1}\nonumber\hline', 'x+y'], ['x+y1']),
    ([r'\big(x+y\Big)', r'\bigl(x+y\bigr)', r'\left(x+y\right)', '(x+y)'], ['x-y']),
    (['x~+~y', 'x+y', r'x\vspace{2pt}+y'], ['xy']),
    ([r'\begin{array}{*{2}{c}}a&b\end{array}', 'array(a¦b)'], ['array(ab)']),
])
def test_fix_09_pure_layout(forms, different):
    check_forms(forms, different)


@pytest.mark.parametrize('size', ['big', 'Big', 'bigg', 'Bigg', 'bigl', 'Bigl', 'biggl', 'Biggl'])
def test_fix_09_sizing_keeps_content_brackets(size):
    check_forms(['\\' + size + '[x]', '[x]'], ['x'])


def test_fix_09_invisible_sizing_delimiter():
    check_forms([r'\left. x\right)', 'x)'], ['x'])


@pytest.mark.parametrize('forms,different', [
    ([r'\sqrt\ln x', '√\\ln x', r'\sqrt{\ln}x'], [r'\sqrt{l}nx']),
    ([r'\sqrt\sin x', r'\sqrt{\sin}x'], [r'\sqrt{s}inx']),
    ([r'\sqrt[3]\ln x', r'\sqrt[3]{\ln}x'], [r'\sqrt{3}\ln x']),
])
def test_fix_10_command_radicands(forms, different):
    check_forms(forms, different)


def test_fix_10_seeded_notation_fuzz_is_idempotent():
    import random

    rng = random.Random(12010)
    tokens = ['a', '2', '13', '+', '-', '/', '*', '^', '_', '!', '=', ',', ' ', '\n',
              '(', ')', '[', ']', '{', '}', r'\sqrt', '√', '∛', r'\ln', r'\alpha',
              r'\frac', r'\overline', r'\text', r'\\', '&', '$', r'\(', r'\)',
              'ⁿ', '₁', '①', '̅', '\u00af', '\u02dc', '\uffe3', '\u203e']
    for _ in range(4000):
        text = ''.join(rng.choices(tokens, k=rng.randrange(1, 16)))
        normalized = skeleton(text)
        assert skeleton(normalized) == normalized, repr(text)


@pytest.mark.parametrize('forms,different', [
    ([r'6\div 2(1+2)', '6÷2(1+2)'], [r'\frac{6}{2}(1+2)', '6/2(1+2)']),
    (['a/(bc)', r'\frac{a}{bc}', 'a/bc'], ['(a/b)c', 'a/b*c']),
    (['a/(b*c)', r'\frac{a}{b*c}'], ['a/b*c', 'a/bc']),
    (['(ab)/c', r'\frac{ab}{c}', 'ab/c'], ['a(b/c)']),
])
def test_fix_11_implicit_multiplication_and_division(forms, different):
    check_forms(forms, different)


@pytest.mark.parametrize('forms,different', [
    ([r'30^\circ', r'30^{\circ}', '30°', '30^°'], ['30']),
    (['x^{2}3', 'x^2 3', 'x^(2)3'], ['x^23']),
    (['½', r'\frac12', '1/2'], ['12']),
    (['⅔x', r'\frac23 x', '(2/3)x'], ['2/3x']),
    ([r'a\mp b', 'a∓b'], ['a±b']),
    ([r'x\leqslant y', 'x≤y'], ['x<y']),
    ([r'x\geqslant y', 'x≥y'], ['x>y']),
    ([r'\lbrack x\rbrack', '[x]'], ['x', '(x)']),
    (['f(x)', 'f( x )'], ['fx']),
])
def test_fix_12_equivalent_notations(forms, different):
    check_forms(forms, different)


@pytest.mark.parametrize('glyph,fraction', [
    ('¼', r'\frac14'), ('¾', r'\frac34'), ('⅐', r'\frac17'), ('⅑', r'\frac19'),
    ('⅒', r'\frac{1}{10}'), ('⅓', r'\frac13'), ('⅕', r'\frac15'), ('⅖', r'\frac25'),
    ('⅗', r'\frac35'), ('⅘', r'\frac45'), ('⅙', r'\frac16'), ('⅚', r'\frac56'),
    ('⅛', r'\frac18'), ('⅜', r'\frac38'), ('⅝', r'\frac58'), ('⅞', r'\frac78'), ('↉', r'\frac03'),
])
def test_fix_12_vulgar_fractions(glyph, fraction):
    check_forms([glyph, fraction], ['1'])


@pytest.mark.parametrize('identifier', ['f', 'g', 'sin', 'log', 'φ'])
def test_fix_12_function_application_keeps_parentheses(identifier):
    check_forms([identifier + '(x)', identifier + '( x )'], [identifier + 'x'])


@pytest.mark.parametrize('operator', ['^', '_'])
def test_fix_12_script_product_scope(operator):
    check_forms(['x' + operator + '{ab}', 'x' + operator + '(ab)'], ['x' + operator + 'a b'])


@pytest.mark.parametrize('forms,different', [
    ([r'x^{a_b}', 'x^(a_b)'], ['x^a_b']),
    ([r'x_{a^b}', 'x_(a^b)'], ['x_a^b']),
    ([r'\text{(a)}', 'text((a))'], [r'\text{a}']),
    (['f_1(x)', r'f_{1}( x )'], ['f_1x']),
    ([r'f_{n+1}(x)', 'f_(n+1)(x)'], [r'f_{n+1}x']),
    (['f′(x)', 'f′( x )'], ['f′x']),
    (['x⩽y', r'x\leqslant y'], ['x<y']),
    (['x⩾y', r'x\geqslant y'], ['x>y']),
    ([r'\sqrt[1,2]{3}', 'root((1,2),3)'], [r'\sqrt[1]{2,3}']),
])
def test_fix_12_grouping_and_alias_interactions(forms, different):
    check_forms(forms, different)


def test_fix_12_factorial_equality_keeps_lexical_boundary():
    check_forms(['n! = 6', r'n!\,=6', r'n!\quad =6', '(n!)=6'], ['n!=6', 'n≠6'])


@pytest.mark.parametrize('forms,different', [
    ([r'\sqrt{[x]}', '√[x]', 'sqrt([x])'], [r'\sqrt{x}', '√x', 'sqrt(x)']),
    ([r'\sqrt{[x+1]}', '√[x+1]', 'sqrt([x+1])'], ['sqrt(x+1)']),
    ([r'\sqrt[3]{[x]}', r'\sqrt[3][x]', '∛[x]', 'root(3,[x])'], ['root(3,x)', '∛x']),
    ([r'\sqrt[n]{[x+1]}', r'\sqrt[n][x+1]', 'root(n,[x+1])'], ['root(n,x+1)']),
])
def test_rereview_fix_2_root_content_brackets(forms, different):
    check_forms(forms, different)


@pytest.mark.parametrize('forms,different', [
    ([r'\sqrt{\bar{z}}', r'√\bar{z}', 'sqrt(bar(z))'], [r'\sqrt{z}', r'\sqrt{\overline{z}}', 'sqrt(z)']),
    ([r'\sqrt{\overline{\bar{z}}}', 'sqrt(bar(z))'], [r'\sqrt{\bar{z}\bar{z}}']),
    ([r'\sqrt{x+\bar{z}}', 'sqrt(x+bar(z))'], ['sqrt(x+z)', 'sqrt(bar(x+z))']),
    ([r'\sqrt{\bar{\bar{z}}}', 'sqrt(bar(bar(z)))'], ['sqrt(z)', 'sqrt(bar(z))']),
])
def test_rereview_fix_3_explicit_bar_inside_roots(forms, different):
    check_forms(forms, different)


@pytest.mark.parametrize('glyph,command,plain', [
    ('\U0001D42F', r'\mathbf{v}', 'v'),
    ('\U0001D415', r'\mathbf{V}', 'V'),
    ('\U0001D6C2', r'\mathbf{\alpha}', 'α'),
    ('\U0001D7CF', r'\mathbf{1}', '1'),
])
def test_rereview_fix_4_unicode_bold_matches_mathbf(glyph, command, plain):
    check_forms([glyph, command, 'mathbf(' + plain + ')'], [plain])


@pytest.mark.parametrize('forms,different', [
    ([r'\sqrt[2]{x}', r'\sqrt{x}', '√x', 'root(2,x)', 'sqrt(x)'], [r'\sqrt[3]{x}', 'root(3,x)']),
    ([r'\sqrt[2]{x+1}', 'sqrt(x+1)', 'root(2,x+1)'], ['root(2,x)+1', r'\sqrt[4]{x+1}']),
    ([r'\sqrt[2]{[x]}', '√[x]', 'root(2,[x])', 'sqrt([x])'], ['sqrt(x)', 'root(3,[x])']),
])
def test_rereview_fix_7_explicit_square_root_index(forms, different):
    check_forms(forms, different)


@pytest.mark.parametrize('mark,canonical', [
    ('\u00af', 'overline(x)'),
    ('\uffe3', 'overline(x)'),
    ('\u203e', 'overline(x)'),
    ('\u02dc', 'tilde(x)'),
])
def test_rereview_fix_6_compatibility_marks_are_idempotent(mark, canonical):
    text = 'x' + mark
    assert skeleton(text) == canonical
    assert skeleton(canonical) == canonical
    assert skeleton(skeleton(text)) == canonical
