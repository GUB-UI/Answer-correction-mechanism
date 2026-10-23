import copy
import json
import subprocess
import sys

import pytest

from harness_v2.tests.test_p3_dataset import make_dataset
from harness_v2.tests.test_p3_runner import config_at


def result_pair(tmp_path):
    from harness_v2.run import run_experiment
    manifest = make_dataset(tmp_path)
    config = config_at(tmp_path)
    for name in ['baseline', 'candidate']:
        run_experiment(config, manifest, repeats=1, out=tmp_path / name)
    return (json.loads((tmp_path / 'baseline/results.json').read_text()),
            json.loads((tmp_path / 'candidate/results.json').read_text()), manifest)


def test_comparison_accepts_same_inputs_and_computes_metrics(tmp_path):
    from harness_v2.evaluation.comparison import compare_experiments
    baseline, candidate, _ = result_pair(tmp_path)
    result = compare_experiments(baseline, candidate)
    assert result['comparable'] and result['reasons'] == []
    assert result['baseline']['success_rate'] == result['candidate']['success_rate'] == 1
    assert result['deltas']['success_rate'] == 0


@pytest.mark.parametrize('mutation,reason', [
    (lambda c: c['config'].update(model='another'), 'model'),
    (lambda c: c['runs'][0].update(model_id='another-revision'), 'model'),
    (lambda c: c['config']['generation_settings'].update(seed=101), 'generation'),
    (lambda c: c['config'].update(rubric_version='other'), 'rubric'),
    (lambda c: c['dataset']['cases'][0].update(problem_text='different problem'), 'problem'),
    (lambda c: c['dataset']['cases'][0]['grading_input'].update(sha256='0'*64), 'input'),
    (lambda c: c['dataset']['cases'][0].update(ocr_metadata={'engine': 'other'}), 'OCR'),
    (lambda c: c['dataset']['cases'][0]['rubric'].update(sha256='1'*64), 'rubric'),
    (lambda c: c['expected_runs'].clear(), 'case'),
    (lambda c: c['runs'].clear(), 'incomplete'),
])
def test_comparison_refuses_changed_conditions_with_reasons(tmp_path, mutation, reason):
    from harness_v2.evaluation.comparison import compare_experiments
    baseline, candidate, _ = result_pair(tmp_path)
    mutation(candidate)
    result = compare_experiments(baseline, candidate)
    assert not result['comparable']
    assert result['verdict'] == 'incomparable'
    assert any(reason.lower() in r.lower() for r in result['reasons']), result


def test_comparison_cli_works_on_run_directories(tmp_path):
    result_pair(tmp_path)
    report = tmp_path / 'comparison.json'
    cmd = [sys.executable, '-m', 'harness_v2.evaluation.compare',
           '--baseline', str(tmp_path / 'baseline'), '--candidate', str(tmp_path / 'candidate'),
           '--out', str(report)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(report.read_text())['comparable'] is True


@pytest.mark.parametrize('mutation', [
    lambda c: c['expected_runs'][0].update(input_sha256='0'*64),
    lambda c: c['expected_runs'][0].update(case_id='other'),
    lambda c: c['runs'][0]['outputs'].update(ocr_sha256='0'*64),
    lambda c: c['runs'][0]['outputs']['experiment_case'].update(input_sha256='0'*64),
])
def test_comparison_rejects_inconsistent_run_provenance_even_on_both_sides(tmp_path, mutation):
    from harness_v2.evaluation.comparison import compare_experiments
    baseline, candidate, _ = result_pair(tmp_path)
    mutation(baseline); mutation(candidate)
    result = compare_experiments(baseline, candidate)
    assert result['verdict'] == 'incomparable'


def test_comparison_rejects_both_runs_claiming_another_derived_grading_input(tmp_path):
    from harness_v2.evaluation.comparison import compare_experiments
    baseline, candidate, _ = result_pair(tmp_path)
    for payload in [baseline, candidate]:
        payload['runs'][0]['outputs']['grading_input_sha256'] = '0'*64
    result = compare_experiments(baseline, candidate)
    assert result['verdict'] == 'incomparable'
