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
