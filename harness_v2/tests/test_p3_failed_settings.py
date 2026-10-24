import copy
import json

import pytest

from harness_v2.core.contracts import ProviderExchange
from harness_v2.evaluation.comparison import compare_experiments
from harness_v2.providers.fixture import FixtureProvider
from harness_v2.providers.gemma_ai_studio import MissingAPIKey, ProviderSetupError, QuotaExceeded
from harness_v2.run import run_experiment
from harness_v2.tests.test_p3_dataset import make_dataset
from harness_v2.tests.test_p3_runner import config_at


def failed_pair(tmp_path, failure):
    config, manifest = config_at(tmp_path), make_dataset(tmp_path)
    class Failing(FixtureProvider):
        def generate(self, prompt, *, request_id, settings=None):
            assert 'reference_' not in prompt
            if failure == 'exception':
                raise RuntimeError('provider call failed before returning an exchange')
            exchange = ProviderExchange(request_id, 'fixture', 'fixture', dict(settings),
                                        None, None, 1, 'invalid JSON',
                                        'ok' if failure == 'parse' else 'error')
            if failure == 'quota':
                exchange.status = 'quota'
                raise QuotaExceeded(exchange)
            return exchange
    def factory(*args):
        if failure == 'setup':
            raise ProviderSetupError('before any model call')
        if failure == 'missing_key':
            raise MissingAPIKey('no configured key')
        return Failing('')
    payloads = []
    for name in ('baseline', 'candidate'):
        out = tmp_path / name
        run_experiment(config, manifest, repeats=1, out=out, provider_factory=factory)
        payload = json.loads((out / 'results.json').read_text())
        assert payload['runs'][0]['outputs']['grading']['status'] == 'failed'
        assert payload['runs'][0]['outputs']['grading'].get('total_score') is None
        payloads.append(payload)
    return payloads


@pytest.mark.parametrize('failure', ['parse', 'http', 'quota'])
@pytest.mark.parametrize('both_sides', [False, True])
@pytest.mark.parametrize('settings', [{'temperature': 1, 'seed': 7}, None, 'missing', []])
def test_failed_calls_require_matching_recorded_settings(tmp_path, failure, both_sides, settings):
    baseline, candidate = failed_pair(tmp_path, failure)
    for payload in ([baseline, candidate] if both_sides else [candidate]):
        outputs = payload['runs'][0]['outputs']
        if settings == 'missing':
            outputs.pop('settings')
        else:
            outputs['settings'] = settings
    before = copy.deepcopy([baseline, candidate])
    result = compare_experiments(baseline, candidate)
    assert result['verdict'] == 'incomparable'
    assert result['baseline'] is result['candidate'] is None
    assert any('effective generation settings' in reason for reason in result['reasons'])
    assert [baseline, candidate] == before


@pytest.mark.parametrize('failure', ['parse', 'http', 'quota', 'setup', 'missing_key'])
def test_matching_failed_calls_and_zero_attempt_setup_are_comparable(tmp_path, failure):
    baseline, candidate = failed_pair(tmp_path, failure)
    for payload in (baseline, candidate):
        outputs = payload['runs'][0]['outputs']
        if failure in {'setup', 'missing_key'}:
            assert outputs['attempt_count'] == 0 and outputs['attempts'] == []
            assert 'settings' not in outputs
        else:
            outputs['settings']['provider_policy'] = {'operator_note': 'metadata only'}
    before = copy.deepcopy([baseline, candidate])
    result = compare_experiments(baseline, candidate)
    assert result['comparable'], result['reasons']
    assert result['baseline']['success_rate'] == result['candidate']['success_rate'] == 0
    assert result['baseline']['unscored_totals'] == result['candidate']['unscored_totals'] == 1
    assert [baseline, candidate] == before


def test_unclassified_provider_exception_is_not_a_proven_setup_failure(tmp_path):
    baseline, candidate = failed_pair(tmp_path, 'exception')
    assert candidate['runs'][0]['outputs']['attempt_count'] == 0
    result = compare_experiments(baseline, candidate)
    assert result['verdict'] == 'incomparable'
    assert any('effective generation settings' in reason for reason in result['reasons'])


@pytest.mark.parametrize('count,attempts', [(None, []), (False, []), (0, None), (0, [{'status': 'error'}])])
def test_only_explicit_zero_call_setup_failures_are_exempt(tmp_path, count, attempts):
    baseline, candidate = failed_pair(tmp_path, 'setup')
    outputs = candidate['runs'][0]['outputs']
    if count is None:
        outputs.pop('attempt_count')
    else:
        outputs['attempt_count'] = count
    outputs['attempts'] = attempts
    result = compare_experiments(baseline, candidate)
    assert result['verdict'] == 'incomparable'
    assert any('effective generation settings' in reason for reason in result['reasons'])


def test_response_evidence_prevents_zero_count_from_bypassing_settings(tmp_path):
    baseline, candidate = failed_pair(tmp_path, 'http')
    for payload in (baseline, candidate):
        outputs = payload['runs'][0]['outputs']
        outputs.update(attempt_count=0, attempts=[], settings={'temperature': 1, 'seed': 7})
    result = compare_experiments(baseline, candidate)
    assert result['verdict'] == 'incomparable'
    assert any('effective generation settings' in reason for reason in result['reasons'])


def test_historical_failed_calls_validate_settings_after_successful_resume(tmp_path):
    baseline, candidate = failed_pair(tmp_path, 'http')
    config = tmp_path / 'config.json'
    manifest = tmp_path / 'dataset/manifest.json'
    for name in ('baseline', 'candidate'):
        run_experiment(config, manifest, repeats=1, out=tmp_path / name)
    baseline = json.loads((tmp_path / 'baseline/results.json').read_text())
    candidate = json.loads((tmp_path / 'candidate/results.json').read_text())
    assert compare_experiments(baseline, candidate)['comparable']
    for payload in (baseline, candidate):
        failed = next(run for run in payload['attempt_runs'] if run['outputs']['grading']['status'] == 'failed')
        failed['outputs']['settings']['seed'] = 99
    result = compare_experiments(baseline, candidate)
    assert result['verdict'] == 'incomparable'
    assert any('effective generation settings' in reason for reason in result['reasons'])
