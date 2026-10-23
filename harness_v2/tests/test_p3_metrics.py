import copy
import json

import pytest


def metric_fixture(tmp_path):
    from harness_v2.evaluation.dataset import write_dataset, load_dataset
    cases = [{'problem_id': p, 'input_version': 'manual', 'problem_text': 'solve',
              'answer_text': 'x y', 'rubric_version': 'r',
              'rubric': [{'item_id': 'a', 'max_points': 3}, {'item_id': 'b', 'max_points': 3}]}
             for p in ['A', 'B']]
    ds = load_dataset(write_dataset(tmp_path / 'd', dataset_id='d', version='1', cases=cases,
         reference_scores={'A': 4, 'B': 2}, reference_items={'A': {'a': 2, 'b': 2}}))
    rows, expected = [], []
    data = [('A', 1, 4, [2, 2], 'validated'), ('A', 2, 4, [1, 3], 'review_required'),
            ('B', 1, 3, [3], 'validated'), ('B', 2, 1, [1], 'review_required')]
    for n, (p, rep, total, scores, status) in enumerate(data, 1):
        rid = str(n)
        job = {'request_id': rid, 'case_id': p+':manual', 'problem_id': p, 'repetition': rep}
        expected.append(job)
        items = [{'item_id': key, 'score': score, 'evidence_spans': [{'text': 'fake' if n == 2 and key == 'b' else 'x'}],
                  'uncertainty': 0, 'review_reason': ''} for key, score in zip(['a', 'b'], scores)]
        attempts = ([{'latency_ms': 20, 'status': 'error', 'error_class': 'ServerError'},
                     {'latency_ms': 80, 'status': 'ok', 'error_class': ''}] if n == 1 else
                    [{'latency_ms': n*100, 'status': 'ok', 'error_class': ''}])
        rows.append({'run_id': rid, 'request_id': rid, 'latency_ms': n*100, 'input_tokens': 10, 'output_tokens': 5,
                     'outputs': {'experiment_case': job, 'run_attempt': 2 if n == 4 else 1,
                         'attempt_count': len(attempts), 'attempts': attempts,
                         'grading': {'status': status, 'total_score': total, 'items': items}}})
    earlier = {'run_id': 'earlier', 'request_id': '4', 'latency_ms': 3, 'input_tokens': None, 'output_tokens': None,
               'outputs': {'run_attempt': 1, 'attempt_count': 3, 'attempts': [
                   {'latency_ms': 1, 'status': 'error', 'error_class': 'ServerError'}]*3,
                   'grading': {'status': 'failed', 'total_score': None, 'items': []}}}
    return {'runs': rows, 'attempt_runs': [earlier, *rows], 'expected_runs': expected}, ds


def test_reference_evidence_review_latency_and_retry_usage_metrics(tmp_path):
    from harness_v2.evaluation.metrics import compute_metrics
    payload, dataset = metric_fixture(tmp_path)
    metrics = compute_metrics(payload, dataset)
    assert metrics['mae'] == 0.5
    assert metrics['item_agreement'] == 0.5
    assert metrics['evidence_accuracy'] == pytest.approx(5/6)
    assert metrics['wrong_grading'] == {'count': 3, 'unjust_addition': 2, 'unjust_deduction': 2}
    assert metrics['review_detection']['flagged_wrong'] == 2
    assert metrics['review_detection']['wrong_count'] == 3
    assert metrics['review_detection']['recall'] == pytest.approx(2/3)
    assert metrics['review_detection']['false_flags'] == 0
    assert metrics['review_detection']['correct_count'] == 1
    assert metrics['success_rate'] == 1
    assert metrics['latency']['total_ms'] == {'mean': 250, 'median': 250, 'count': 4}
    assert metrics['latency']['per_attempt_ms']['count'] == 8
    assert metrics['usage']['requests'] == 8
    assert metrics['usage']['input_tokens'] == 40 and metrics['usage']['output_tokens'] == 20


def test_missing_scores_stay_null_and_incomplete_runs_stay_in_denominators(tmp_path):
    from harness_v2.evaluation.metrics import compute_metrics
    payload, ds = metric_fixture(tmp_path)
    payload['runs'] = []
    payload['attempt_runs'] = []
    before = copy.deepcopy(payload)
    metrics = compute_metrics(payload, ds)
    assert payload == before
    assert metrics['success_rate'] == 0
    assert metrics['mae'] is None
    assert metrics['item_agreement'] == 0
    assert metrics['evidence_accuracy'] is None
    assert metrics['unscored_totals'] == 4
    assert metrics['usage']['requests'] == 0
    assert metrics['latency']['total_ms']['mean'] is None


def test_false_review_flags_are_separate_from_wrong_detection(tmp_path):
    from harness_v2.evaluation.metrics import compute_metrics
    payload, ds = metric_fixture(tmp_path)
    payload['runs'][0]['outputs']['grading']['status'] = 'review_required'
    metrics = compute_metrics(payload, ds)
    assert metrics['review_detection']['false_flags'] == 1
    assert metrics['review_detection']['false_flag_rate'] == 1


def test_severe_regressions_block_improved_verdict(tmp_path):
    from harness_v2.evaluation.comparison import compare_experiments
    from harness_v2.tests.test_p3_comparison import result_pair
    before, after, _ = result_pair(tmp_path)
    bad = after['runs'][0]['outputs']['grading']
    bad['items'][0]['score'] = 3
    bad['total_score'] = 3
    result = compare_experiments(before, after)
    assert result['verdict'] == 'regressed'
    assert result['severe_regressions']['new_wrong_grading']
    # No score change: lower traceable-evidence accuracy is independently severe.
    bad['items'][0]['score'] = 4; bad['total_score'] = 4
    bad['items'][0]['evidence_spans'][0]['text'] = 'fabricated'
    result = compare_experiments(before, after)
    assert result['verdict'] == 'regressed'
    assert result['severe_regressions']['lower_evidence_accuracy'] is True


def test_improvement_requires_a_measured_gain_and_no_severe_regression(tmp_path):
    from harness_v2.evaluation.comparison import compare_experiments
    from harness_v2.tests.test_p3_comparison import result_pair
    before, after, _ = result_pair(tmp_path)
    assert compare_experiments(before, after)['verdict'] == 'not_improved'
    before['runs'][0]['outputs']['grading']['items'][0]['score'] = 2
    before['runs'][0]['outputs']['grading']['total_score'] = 2
    result = compare_experiments(before, after)
    assert result['verdict'] == 'improved'
    assert result['deltas']['mae'] == -2


def test_new_item_error_blocks_gain_even_when_other_item_error_disappears(tmp_path):
    from harness_v2.evaluation.comparison import compare_experiments
    from harness_v2.evaluation.dataset import load_dataset
    from harness_v2.tests.test_p3_comparison import result_pair
    before, after, _ = result_pair(tmp_path)
    # Baseline wrong -> candidate wrong at the same unit is not a new error.
    for payload in [before, after]:
        payload['runs'][0]['outputs']['grading']['items'][0]['score'] = 3
        payload['runs'][0]['outputs']['grading']['total_score'] = 3
    assert compare_experiments(before, after)['severe_regressions']['new_wrong_grading'] == []


def test_metrics_usage_exposes_unknown_retry_tokens(tmp_path):
    from harness_v2.evaluation.metrics import compute_metrics
    payload, ds = metric_fixture(tmp_path)
    metrics = compute_metrics(payload, ds)
    assert metrics['usage']['unknown_input_token_attempts'] >= 4


def test_lower_mae_cannot_hide_a_new_wrong_repetition(tmp_path):
    from harness_v2.run import run_experiment
    from harness_v2.evaluation.comparison import compare_experiments
    from harness_v2.tests.test_p3_dataset import make_dataset
    from harness_v2.tests.test_p3_runner import config_at
    from harness_v2.tests.test_pipeline import _valid
    manifest = make_dataset(tmp_path)
    def grade(score):
        obj = json.loads(_valid())
        obj['items'][0]['score'] = score
        obj['total_score'] = score
        return obj
    for name, scores in [('baseline', [3, 4]), ('candidate', [4, 3.5])]:
        config = config_at(tmp_path, name=name, fixture_completion=[grade(s) for s in scores])
        run_experiment(config, manifest, repeats=2, out=tmp_path / name)
    before = json.loads((tmp_path / 'baseline/results.json').read_text())
    after = json.loads((tmp_path / 'candidate/results.json').read_text())
    result = compare_experiments(before, after)
    assert result['candidate']['mae'] < result['baseline']['mae']
    assert result['candidate']['wrong_grading']['count'] == result['baseline']['wrong_grading']['count'] == 1
    assert result['verdict'] == 'regressed'
    assert result['severe_regressions']['new_wrong_grading']


@pytest.mark.parametrize('total,addition,deduction', [(5, 1, 0), (3, 0, 1)])
@pytest.mark.parametrize('partial_labels', [False, True])
def test_total_fallback_counts_direction_when_labeled_items_match(tmp_path, total, addition, deduction, partial_labels):
    from harness_v2.evaluation.metrics import compute_metrics
    payload, ds = metric_fixture(tmp_path)
    payload['runs'] = payload['runs'][:1]
    payload['expected_runs'] = payload['expected_runs'][:1]
    grade = payload['runs'][0]['outputs']['grading']
    grade['total_score'] = total
    if partial_labels:
        ds.references['reference_items']['A'] = {'a': 2}
        grade['items'][1]['score'] = total - 2
    original = copy.deepcopy(payload)
    metrics = compute_metrics(payload, ds)
    assert metrics['wrong_grading'] == {'count': 1, 'unjust_addition': addition, 'unjust_deduction': deduction}
    assert metrics['wrong_units'] == ['["A:manual",1,"__total__"]']
    assert metrics['item_agreement'] == 1
    assert payload == original


@pytest.mark.parametrize('total,addition,deduction', [(5, 1, 0), (3, 0, 1)])
def test_total_fallback_does_not_double_count_a_known_item_error(tmp_path, total, addition, deduction):
    from harness_v2.evaluation.metrics import compute_metrics
    payload, ds = metric_fixture(tmp_path)
    payload['runs'] = payload['runs'][:1]
    payload['expected_runs'] = payload['expected_runs'][:1]
    grade = payload['runs'][0]['outputs']['grading']
    grade['total_score'] = total
    grade['items'][1]['score'] = total - 2
    metrics = compute_metrics(payload, ds)
    assert metrics['wrong_grading'] == {'count': 1, 'unjust_addition': addition, 'unjust_deduction': deduction}
    assert metrics['wrong_units'] == ['["A:manual",1,"b"]']
