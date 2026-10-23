import copy
import json

import pytest

from harness_v2.evaluation.comparison import compare_experiments
from harness_v2.evaluation.dataset import load_dataset
from harness_v2.tests.test_p3_failed_settings import failed_pair
from harness_v2.tests.test_p3_historical_ocr import failed_attempt, resumed_pair


def without_ocr_evidence(outputs):
    for field in ('ocr_metadata', 'ocr_sha256', 'ocr_artifact', 'ocr_redacted',
                  'grading_input_sha256', 'grading_input_artifact', 'prompt_artifact',
                  'raw_response_artifact', 'exchange', 'provider_response_id'):
        outputs.pop(field, None)
    outputs.update(attempt_count=0, attempts=[])


@pytest.mark.parametrize('evidence', [
    {'attempt_count': 1}, {'attempts': [{'status': 'error'}]},
    {'prompt_artifact': 'prompts/recorded.txt'},
    {'grading_input_artifact': 'grading_private/recorded.txt'},
    {'attempt_count': None}, {'attempt_count': False}, {'attempt_count': 0.0},
    {'attempts': None}, {'attempts': {}},
    {'attempt_count': 'missing'}, {'attempts': 'missing'},
])
@pytest.mark.parametrize('historical', [False, True])
@pytest.mark.parametrize('both_sides', [False, True])
def test_missing_ocr_provenance_requires_affirmative_pre_ocr_evidence(tmp_path, evidence, historical, both_sides):
    baseline, candidate = resumed_pair(tmp_path) if historical else failed_pair(tmp_path, 'http')
    for payload in ([baseline, candidate] if both_sides else [candidate]):
        run = failed_attempt(payload) if historical else payload['runs'][0]
        outputs = run['outputs']
        without_ocr_evidence(outputs)
        for field, value in evidence.items():
            if value == 'missing':
                outputs.pop(field)
            else:
                outputs[field] = value
    before = copy.deepcopy([baseline, candidate])
    result = compare_experiments(baseline, candidate)
    assert result['verdict'] == 'incomparable'
    assert result['baseline'] is result['candidate'] is None
    assert any('OCR metadata' in reason for reason in result['reasons'])
    assert [baseline, candidate] == before
    run = failed_attempt(candidate) if historical else candidate['runs'][0]
    assert run['outputs']['grading']['total_score'] is None


@pytest.mark.parametrize('bad_value', [None, [], 'invalid', 0])
@pytest.mark.parametrize('field', ['outputs', 'grading'])
@pytest.mark.parametrize('historical', [False, True])
@pytest.mark.parametrize('supplied_dataset', [False, True])
def test_malformed_attempt_outputs_return_incomparable(tmp_path, bad_value, field, historical, supplied_dataset):
    baseline, candidate = resumed_pair(tmp_path)
    run = failed_attempt(candidate) if historical else candidate['runs'][0]
    if field == 'outputs':
        run['outputs'] = bad_value
    else:
        run['outputs']['grading'] = bad_value
    before = copy.deepcopy([baseline, candidate])
    dataset = load_dataset(tmp_path / 'dataset/manifest.json') if supplied_dataset else None
    result = compare_experiments(baseline, candidate, dataset=dataset)
    assert result['verdict'] == 'incomparable'
    assert result['reasons']
    assert result['baseline'] is result['candidate'] is None
    assert result['deltas'] == {}
    assert [baseline, candidate] == before


def test_cli_writes_private_incomparable_report_for_null_historical_outputs(tmp_path, capsys):
    from harness_v2.evaluation.compare import main
    _, candidate = resumed_pair(tmp_path)
    failed_attempt(candidate)['outputs'] = None
    (tmp_path / 'candidate/results.json').write_text(json.dumps(candidate))
    report = tmp_path / 'report/comparison.json'
    assert main(['--baseline', str(tmp_path / 'baseline'), '--candidate', str(tmp_path / 'candidate'),
                 '--dataset', str(tmp_path / 'dataset/manifest.json'), '--out', str(report)]) == 2
    result = json.loads(report.read_text())
    assert result['verdict'] == 'incomparable' and result['reasons']
    assert result['baseline'] is result['candidate'] is None
    assert json.loads(capsys.readouterr().out)['verdict'] == 'incomparable'
    assert report.stat().st_mode & 0o777 == 0o600
    assert report.parent.stat().st_mode & 0o777 == 0o700
