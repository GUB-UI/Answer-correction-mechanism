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

