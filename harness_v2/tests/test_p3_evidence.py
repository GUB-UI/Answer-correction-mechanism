import json

import pytest
from jsonschema import Draft202012Validator

from harness_v2.core.contracts import RubricItem
from harness_v2.evaluation.review import REVIEW_SCHEMA, grading_review_contract
from harness_v2.grading.scorer import build_prompt, items_from_completion
from harness_v2.grading.validator import validate


def grade(row, answer='x'):
    rubric = [RubricItem('a', 4)]
    raw = json.dumps({'items': [{'item_id': 'a', 'score': 0, 'evidence_spans': [],
        'math_reason': 'not written', 'uncertainty': 0, 'review_reason': '', **row}],
        'total_score': row.get('score', 0)})
    items, total, *_ = items_from_completion(raw, rubric)
    return validate(rubric, items, total, answer), raw


def test_explicit_absence_is_preserved_and_accepted():
    result, raw = grade({'evidence_absent': True, 'absent_reason': 'No final solution'})
    assert result.status == 'validated'
    assert result.to_dict()['items'][0]['evidence_absent'] is True
    assert result.to_dict()['items'][0]['absent_reason'] == 'No final solution'
    assert Draft202012Validator(REVIEW_SCHEMA).is_valid(json.loads(raw))
    assert 'absent_reason' in build_prompt('', '', [RubricItem('a', 4)])
    assert 'evidence_absent' in grading_review_contract()


@pytest.mark.parametrize('row', [
    {}, {'evidence_absent': 1, 'absent_reason': 'missing'},
    {'evidence_absent': 'true', 'absent_reason': 'missing'},
    {'evidence_absent': False, 'absent_reason': 'missing'},
    {'evidence_absent': True},
    {'score': 1, 'evidence_absent': True, 'absent_reason': 'missing'},
    {'score': None, 'evidence_absent': True, 'absent_reason': 'missing'},
])
def test_invalid_absence_never_validates(row):
    result, _ = grade(row)
    assert result.status != 'validated'


@pytest.mark.parametrize('answer,span', [
    (r'$= \sqrt{9}-\sqrt{2}$', r'$\sqrt{9}-\sqrt{2}$'),
    (r'最小値 \(a\)', '最小値 a'),
    (r'$$ = x + 1 $$', r'\[x+1\]'),
    (r'\[ = a + b \]', r'\(a+b\)'),
])
def test_math_normalization_only_changes_comparison(answer, span):
    result, _ = grade({'score': 4, 'evidence_spans': [{'text': span}]}, answer)
    assert result.status == 'validated'
    assert result.items[0].evidence_spans[0].text == span


@pytest.mark.parametrize('answer,span', [('x+1', 'x+2'), ('x+1', '$=$'), ('x+1', '=x+1')])
def test_fabricated_evidence_still_fails(answer, span):
    result, _ = grade({'score': 4, 'evidence_spans': [{'text': span}]}, answer)
    assert result.status != 'validated'


@pytest.mark.parametrize('span', [r'= \sqrt{9}-\sqrt{2}', r'\sqrt{9}-\sqrt{2}', r'$= \sqrt{9}-\sqrt{2}$'])
def test_undelimited_span_with_leading_equals_matches_delimited_answer(span):
    from harness_v2.grading.validator import evidence_traceable
    answer = '$\\sqrt{11-6\\sqrt{2}}$\n\n$= \\sqrt{9}-\\sqrt{2}$\n\n$= 3\\sqrt{2}$'
    assert evidence_traceable(span, answer)
