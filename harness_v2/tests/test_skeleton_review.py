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
