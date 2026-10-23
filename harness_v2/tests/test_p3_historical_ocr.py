import copy
import json
from pathlib import Path

import pytest

from harness_v2.evaluation.comparison import compare_experiments
from harness_v2.evaluation.dataset import load_dataset
from harness_v2.providers.gemma_ai_studio import ProviderSetupError
from harness_v2.run import run_experiment
from harness_v2.tests.test_p3_failed_settings import failed_pair
from harness_v2.tests.test_p3_dataset import make_dataset
from harness_v2.tests.test_p3_runner import config_at


def resumed_pair(tmp_path, *, sidecar=False):
    if sidecar:
        manifest = make_dataset(tmp_path)
        source = Path(load_dataset(manifest).cases[0]['ocr_import_path'])
        source.with_name(source.name + '.meta.json').write_text('{"elapsed_s": 1}')
    failed_pair(tmp_path, 'http')
    payloads = []
    for name in ('baseline', 'candidate'):
        run_experiment(tmp_path / 'config.json', tmp_path / 'dataset/manifest.json',
                       repeats=1, out=tmp_path / name)
        payloads.append(json.loads((tmp_path / name / 'results.json').read_text()))
    assert compare_experiments(*payloads)['comparable']
    return payloads


def failed_attempt(payload):
    return next(run for run in payload['attempt_runs'] if run['outputs']['grading']['status'] == 'failed')


@pytest.mark.parametrize('field,value', [
    ('engine', 'other'), ('revision', 'other'), ('dtype', 'bfloat16'),
    ('source_sha256', '0' * 64), ('sidecar_sha256', 'invalid'),
    ('sidecar_sha256', '0' * 64),
])
@pytest.mark.parametrize('both_sides', [False, True])
def test_historical_failed_ocr_provenance_cannot_hide_behind_success(tmp_path, field, value, both_sides):
    baseline, candidate = resumed_pair(tmp_path)
    for payload in ([baseline, candidate] if both_sides else [candidate]):
        failed_attempt(payload)['outputs']['ocr_metadata'][field] = value
    before = copy.deepcopy([baseline, candidate])
    result = compare_experiments(baseline, candidate)
    assert result['verdict'] == 'incomparable'
    assert result['baseline'] is result['candidate'] is None
    assert any('OCR' in reason and field in reason for reason in result['reasons'])
    assert [baseline, candidate] == before
    assert failed_attempt(candidate)['outputs']['grading']['total_score'] is None


@pytest.mark.parametrize('field', ['engine', 'revision', 'dtype', 'source_sha256', None])
def test_missing_historical_ocr_metadata_is_rejected(tmp_path, field):
    baseline, candidate = resumed_pair(tmp_path)
    for payload in (baseline, candidate):
        outputs = failed_attempt(payload)['outputs']
        if field is None:
            outputs.pop('ocr_metadata')
        else:
            outputs['ocr_metadata'].pop(field)
    result = compare_experiments(baseline, candidate)
    assert result['verdict'] == 'incomparable'
    assert any('OCR' in reason for reason in result['reasons'])


@pytest.mark.parametrize('mutation', [
    lambda outputs: outputs.update(ocr_sha256='0' * 64),
    lambda outputs: outputs['experiment_case'].update(input_sha256='0' * 64),
    lambda outputs: outputs.pop('experiment_case'),
    lambda outputs: outputs.update(ocr_metadata=None, ocr_sha256=None),
])
def test_historical_case_hash_and_ocr_evidence_are_checked(tmp_path, mutation):
    baseline, candidate = resumed_pair(tmp_path)
    mutation(failed_attempt(candidate)['outputs'])
    assert compare_experiments(baseline, candidate)['verdict'] == 'incomparable'


@pytest.mark.parametrize('mutation', ['changed', 'missing', 'declared'])
def test_historical_sidecar_identity_includes_presence_and_manifest_claims(tmp_path, mutation):
    baseline, candidate = resumed_pair(tmp_path, sidecar=True)
    for payload in (baseline, candidate):
        metadata = failed_attempt(payload)['outputs']['ocr_metadata']
        if mutation == 'changed':
            metadata['sidecar_sha256'] = '0' * 64
        elif mutation == 'missing':
            metadata.pop('sidecar_sha256')
        else:
            payload['dataset']['cases'][0]['ocr_metadata']['sidecar_sha256'] = '0' * 64
    result = compare_experiments(baseline, candidate)
    assert result['verdict'] == 'incomparable'
    assert any('sidecar_sha256' in reason for reason in result['reasons'])


def test_real_sidecar_change_between_failure_and_success_is_incomparable(tmp_path):
    failed_pair(tmp_path, 'http')
    source = Path(load_dataset(tmp_path / 'dataset/manifest.json').cases[0]['ocr_import_path'])
    source.with_name(source.name + '.meta.json').write_text('{"elapsed_s": 2}')
    payloads = []
    for name in ('baseline', 'candidate'):
        run_experiment(tmp_path / 'config.json', tmp_path / 'dataset/manifest.json',
                       repeats=1, out=tmp_path / name)
        payloads.append(json.loads((tmp_path / name / 'results.json').read_text()))
    result = compare_experiments(*payloads)
    assert result['verdict'] == 'incomparable'
    assert any('sidecar_sha256' in reason for reason in result['reasons'])


def test_historical_failure_before_ocr_is_exempt_without_score_fill(tmp_path):
    config, manifest = config_at(tmp_path), make_dataset(tmp_path)
    def before_ocr(self, case, artifact_id):
        raise ProviderSetupError('setup failed before OCR initialization')
    from harness_v2.core.pipeline import Pipeline
    payloads = []
    for name in ('baseline', 'candidate'):
        out = tmp_path / name
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(Pipeline, '_answer_text', before_ocr)
            run_experiment(config, manifest, repeats=1, out=out)
        run_experiment(config, manifest, repeats=1, out=out)
        payload = json.loads((out / 'results.json').read_text())
        outputs = failed_attempt(payload)['outputs']
        assert 'ocr_metadata' not in outputs and 'ocr_sha256' not in outputs
        payloads.append(payload)
    before = copy.deepcopy(payloads)
    result = compare_experiments(*payloads)
    assert result['comparable'], result['reasons']
    assert payloads == before
    assert failed_attempt(payloads[0])['outputs']['grading']['total_score'] is None


def test_comparison_report_preserves_numeric_usage_and_redacts_secret_values(tmp_path):
    from harness_v2.evaluation.compare import main
    from harness_v2.storage.secrets import redact
    baseline, candidate = resumed_pair(tmp_path)
    # Fixture exchanges have unknown token usage on both historical/current calls.
    report = tmp_path / 'report/comparison.json'
    assert main(['--baseline', str(tmp_path / 'baseline'), '--candidate', str(tmp_path / 'candidate'),
                 '--out', str(report)]) == 0
    result = json.loads(report.read_text())
    for section in ('baseline', 'candidate', 'deltas'):
        for name in ('unknown_input_token_runs', 'unknown_output_token_runs',
                     'unknown_input_token_attempts', 'unknown_output_token_attempts'):
            assert type(result[section]['usage'][name]) is int
    assert result['baseline']['usage']['unknown_input_token_runs'] == 2
    assert report.stat().st_mode & 0o777 == 0o600
    assert report.parent.stat().st_mode & 0o777 == 0o700
    assert redact({'unknown_input_token_runs': 'sk-FAKEKEY123456',
                   'authorization': 123, 'access_token': 456}) == {
        'unknown_input_token_runs': '[redacted]', 'authorization': '[redacted]', 'access_token': '[redacted]'}


def test_historical_derived_input_hash_is_checked(tmp_path):
    baseline, candidate = resumed_pair(tmp_path)
    for payload in (baseline, candidate):
        failed_attempt(payload)['outputs']['grading_input_sha256'] = '0' * 64
    result = compare_experiments(baseline, candidate)
    assert result['verdict'] == 'incomparable'
    assert any('derived grading input hash' in reason for reason in result['reasons'])


def test_sidecars_are_compared_between_histories_when_latest_failure_precedes_ocr(tmp_path):
    baseline, candidate = failed_pair(tmp_path, 'http')
    for payload, digest in ((baseline, 'a' * 64), (candidate, 'b' * 64)):
        old = payload['attempt_runs'][0]
        old['outputs']['ocr_metadata']['sidecar_sha256'] = digest
        latest = copy.deepcopy(old)
        latest['run_id'] = 'latest'
        latest['outputs'].update(run_attempt=2, attempt_count=0, attempts=[], error_class='ProviderSetupError')
        for field in ('ocr_metadata', 'ocr_sha256', 'ocr_artifact', 'ocr_redacted', 'grading_input_sha256',
                      'raw_response_artifact', 'exchange', 'provider_response_id'):
            latest['outputs'].pop(field, None)
        payload['runs'] = [latest]
        payload['attempt_runs'].append(copy.deepcopy(latest))
    result = compare_experiments(baseline, candidate)
    assert result['verdict'] == 'incomparable'
    assert any('sidecar_sha256 differs between counterparts' in reason for reason in result['reasons'])


@pytest.mark.parametrize('status', ['validated', 'review_required', 'failed'])
def test_provenance_validation_includes_every_historical_grade_status(tmp_path, status):
    baseline, candidate = resumed_pair(tmp_path)
    history = copy.deepcopy(candidate['runs'][0])
    history['run_id'] = 'another-historical-run'
    history['outputs']['grading']['status'] = status
    history['outputs']['ocr_metadata']['revision'] = 'wrong-revision'
    candidate['attempt_runs'].append(history)
    result = compare_experiments(baseline, candidate)
    assert result['verdict'] == 'incomparable'
    assert any('OCR revision' in reason for reason in result['reasons'])
