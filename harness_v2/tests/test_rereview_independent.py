"""Finding 2: score mentions are blinded while answer math stays intact."""
import json

import pytest

from harness_v2.evaluation.independent import grading_allowlist, proposal_review_allowlist, scrub_score_text


@pytest.mark.parametrize("annotation", [
    "total_score: 8", "score: 4", "total score = 8", "AI score was 8",
    "reference score is 8", "8/10点", "８／１０点", "採点結果: 8点",
    '{"note":"score: 4", "total_score":8}',
])
def test_score_mentions_are_scrubbed_across_all_review_fields(annotation):
    view = grading_allowlist({
        "problem_text": annotation, "answer_text": annotation,
        "rubric": [{"item_id": "a", "max_points": 10, "description": annotation,
                    "acceptable_expression": annotation}],
    })
    assert "8" not in json.dumps(view, ensure_ascii=False).replace('10', '')
    assert "4" not in json.dumps(view, ensure_ascii=False)
    proposal = proposal_review_allowlist({"failure_case": {"errors": [annotation]}, "diff": annotation})
    assert annotation not in json.dumps(proposal, ensure_ascii=False)


@pytest.mark.parametrize("answer", [
    "[-1,1]", '[1,2,3]', '{"x":1,"y":[2,3]}',
    'x = 8/10; y=4; f(x)=x+1', '区間 [-2,2] で 4/5 を求める',
    '  [-1,1]\r\nx+1\r\n',
])
def test_blinding_preserves_math_and_format_verbatim(answer):
    assert scrub_score_text(answer) == answer
    assert grading_allowlist({"answer_text": answer})["answer_text"] == answer


def test_rubric_fields_cannot_smuggle_nested_scores():
    view = grading_allowlist({"rubric": [{"item_id": "a", "max_points": 5,
                            "description": {"note": "score: 4"}}]})
    assert "score" not in json.dumps(view)


def test_public_redaction_preserves_untouched_math_fragments():
    from harness_v2.storage.secrets import redact
    text = '[-1,1] {"api_key":"secret-value"}'
    cleaned = redact(text)
    assert cleaned.startswith('[-1,1] ')
    assert 'secret-value' not in cleaned


@pytest.mark.parametrize('answer', ['3点 A, B, C を結ぶ', '直線は2点で交わる', 'score = x+1; x=2'])
def test_blinding_does_not_erase_math_counts_or_symbolic_variables(answer):
    assert scrub_score_text(answer) == answer


@pytest.mark.parametrize('annotation', [
    'AI scored 8/10', 'The reference grade is 8 out of 10.', '前回AIの採点は8点でした',
    'teacher_score: -Infinity', 'Previous AI awarded 8 points',
])
def test_natural_language_prior_scores_are_blinded(annotation):
    assert scrub_score_text(annotation) != annotation
    assert '8' not in scrub_score_text(annotation)
    assert 'Infinity' not in scrub_score_text(annotation)


@pytest.mark.parametrize('answer', ['score = 2*x + 1', 'score = 2 + x', 'score = 2/3*x', 'score = 2**x'])
def test_numeric_prefix_of_symbolic_math_is_preserved(answer):
    assert scrub_score_text(answer) == answer


@pytest.mark.parametrize('annotation', ['score: 8点', 'total_score: 8点でした'])
def test_score_assignments_with_japanese_units_are_blinded(annotation):
    assert '8' not in scrub_score_text(annotation)


@pytest.mark.parametrize('annotation', ['total_score: "8/10"', "reference score: '8 points'"])
def test_quoted_score_ratios_and_units_are_blinded(annotation):
    assert '8' not in scrub_score_text(annotation)
