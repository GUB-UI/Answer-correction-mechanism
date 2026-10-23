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


@pytest.mark.parametrize('field,value', [
    ('engine', 'different-engine'), ('revision', 'different-revision'),
    ('dtype', 'bfloat16'), ('source_sha256', '0' * 64),
])
@pytest.mark.parametrize('both_sides', [False, True])
def test_comparison_checks_actual_ocr_metadata_against_manifest(tmp_path, field, value, both_sides):
    from harness_v2.evaluation.comparison import compare_experiments
    baseline, candidate, _ = result_pair(tmp_path)
    for payload in ([baseline, candidate] if both_sides else [candidate]):
        payload['runs'][0]['outputs']['ocr_metadata'][field] = value
    result = compare_experiments(baseline, candidate)
    assert result['verdict'] == 'incomparable'
    assert any('OCR' in reason and field in reason for reason in result['reasons'])


@pytest.mark.parametrize('field', ['engine', 'revision', 'dtype', 'source_sha256', None])
def test_comparison_refuses_missing_recorded_ocr_provenance_on_both_sides(tmp_path, field):
    from harness_v2.evaluation.comparison import compare_experiments
    baseline, candidate, _ = result_pair(tmp_path)
    for payload in [baseline, candidate]:
        outputs = payload['runs'][0]['outputs']
        if field is None:
            outputs.pop('ocr_metadata')
        else:
            outputs['ocr_metadata'].pop(field, None)
    assert compare_experiments(baseline, candidate)['verdict'] == 'incomparable'


def test_comparison_normalizes_manifest_revision_alias_and_ignores_elapsed_time(tmp_path):
    from harness_v2.evaluation.comparison import compare_experiments
    from harness_v2.evaluation.dataset import write_dataset
    from harness_v2.run import run_experiment
    from harness_v2.tests.test_p3_dataset import sample_cases
    cases = sample_cases()
    cases[0]['ocr_metadata'] = {'engine': 'Xiaomi-OCR-0', 'model_revision': 'revision-1',
                                'dtype': 'float32', 'source': 'cpu-batch'}
    manifest = write_dataset(tmp_path / 'dataset', dataset_id='d', version='1', cases=cases,
                             reference_scores={'P': 4})
    config = config_at(tmp_path)
    for name in ['baseline', 'candidate']:
        run_experiment(config, manifest, repeats=1, out=tmp_path / name)
    baseline = json.loads((tmp_path / 'baseline/results.json').read_text())
    candidate = json.loads((tmp_path / 'candidate/results.json').read_text())
    candidate['runs'][0]['outputs']['ocr_metadata']['elapsed_s'] = 99
    assert compare_experiments(baseline, candidate)['comparable']
    for payload in [baseline, candidate]:
        payload['runs'][0]['outputs']['ocr_metadata']['revision'] = 'wrong'
    assert compare_experiments(baseline, candidate)['verdict'] == 'incomparable'


def test_comparison_detects_changed_sidecar_even_with_same_text_and_effective_engine(tmp_path):
    from harness_v2.evaluation.comparison import compare_experiments
    from harness_v2.evaluation.dataset import load_dataset
    from harness_v2.run import run_experiment
    from pathlib import Path
    manifest = make_dataset(tmp_path)
    source = Path(load_dataset(manifest).cases[0]['ocr_import_path'])
    sidecar = source.with_name(source.name + '.meta.json')
    config = config_at(tmp_path)
    for name, elapsed in [('baseline', 1), ('candidate', 2)]:
        sidecar.write_text(json.dumps({'elapsed_s': elapsed}))
        run_experiment(config, manifest, repeats=1, out=tmp_path / name)
    baseline = json.loads((tmp_path / 'baseline/results.json').read_text())
    candidate = json.loads((tmp_path / 'candidate/results.json').read_text())
    assert baseline['runs'][0]['outputs']['ocr_sha256'] == candidate['runs'][0]['outputs']['ocr_sha256']
    result = compare_experiments(baseline, candidate)
    assert result['verdict'] == 'incomparable'
    assert any('sidecar' in reason for reason in result['reasons'])
