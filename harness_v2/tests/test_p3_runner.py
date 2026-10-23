import json
import subprocess
import sys

import pytest

from harness_v2.core.contracts import ProviderExchange
from harness_v2.providers.fixture import FixtureProvider
from harness_v2.providers.gemma_ai_studio import QuotaExceeded, ProviderSetupError
from harness_v2.tests.test_p3_dataset import make_dataset
from harness_v2.tests.test_pipeline import _valid


def config_at(tmp_path, **changes):
    config = {'name': 'baseline', 'provider': 'fixture', 'model': 'fixture',
              'generation_settings': {'temperature': 0, 'seed': 7}, 'prompt_version': 'p1',
              'rubric_version': 'r1', 'fixture_completion': _valid(), **changes}
    path = tmp_path / 'config.json'; path.write_text(json.dumps(config))
    return path


def test_repeats_are_deterministic_resumable_and_blind(tmp_path):
    from harness_v2.run import run_experiment
    from harness_v2.storage.sqlite_store import RunStore
    from harness_v2.storage.artifacts import ArtifactStore
    manifest = make_dataset(tmp_path)
    config = config_at(tmp_path)
    out = tmp_path / 'runs'
    seen, delays = [], []
    class Provider(FixtureProvider):
        def generate(self, prompt, **kwargs):
            seen.append(prompt)
            assert 'reference_' not in prompt
            return super().generate(prompt, **kwargs)
    factory = lambda cfg, case, rep: Provider(_valid())
    first = run_experiment(config, manifest, repeats=2, out=out, delay_s=0.25,
                           sleep=delays.append, provider_factory=factory)
    assert first['completed'] == 2
    assert len(seen) == 2 and delays == [0.25]
    again = run_experiment(config, manifest, repeats=2, out=out, provider_factory=factory)
    assert again['skipped'] == 2 and len(seen) == 2
    other = run_experiment(config, manifest, repeats=2, out=tmp_path / 'other', provider_factory=factory)
    assert first['request_ids'] == other['request_ids']
    store = RunStore(out / 'runs.sqlite', ArtifactStore(out / 'artifacts'))
    assert len(store.list_runs()) == 2
    assert all(r['outputs']['experiment_case']['input_sha256'] for r in store.list_runs())
    assert len(list((out / 'failure_cases').glob('*.json'))) == 0


def test_quota_saves_state_stops_and_resume_retries_only_unfinished(tmp_path):
    from harness_v2.run import run_experiment
    calls = []
    class Limited(FixtureProvider):
        def generate(self, prompt, *, request_id, settings=None):
            calls.append(request_id)
            if len(calls) == 2:
                raise QuotaExceeded(ProviderExchange(request_id, 'fixture', '', {}, None, None, 3, '429', 'quota'))
            return super().generate(prompt, request_id=request_id, settings=settings)
    config = config_at(tmp_path); manifest = make_dataset(tmp_path); out = tmp_path / 'runs'
    result = run_experiment(config, manifest, repeats=3, out=out,
                            provider_factory=lambda *a: Limited(_valid()))
    assert result['stopped_reason'] == 'quota' and len(calls) == 2
    assert json.loads((out / 'state.json').read_text())['stopped_reason'] == 'quota'
    assert len(list((out / 'failure_cases').glob('*.json'))) == 1
    result = run_experiment(config, manifest, repeats=3, out=out,
                            provider_factory=lambda *a: Limited(_valid()))
    assert result['completed'] == 3 and result['skipped'] == 1
    assert len(calls) == 4
    assert calls[1] == calls[2]


def test_provider_construction_error_is_a_failed_record(tmp_path):
    from harness_v2.run import run_experiment
    def factory(*args):
        raise ProviderSetupError('setup failed')
    out = tmp_path / 'runs'
    result = run_experiment(config_at(tmp_path), make_dataset(tmp_path), repeats=1, out=out,
                            provider_factory=factory)
    assert result['failed'] == 1
    failure = json.loads(next((out / 'failure_cases').glob('*.json')).read_text())
    assert failure['predicted_score'] is None
    assert 'ProviderSetupError' in failure['errors'][0]


def test_cli_fixture_is_offline_and_applies_settings(tmp_path):
    manifest = make_dataset(tmp_path); config = config_at(tmp_path); out = tmp_path / 'runs'
    result = subprocess.run([sys.executable, '-m', 'harness_v2.run', '--config', str(config),
        '--dataset', str(manifest), '--repeats', '1', '--out', str(out)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['completed'] == 1
    runs = json.loads((out / 'results.json').read_text())['runs']
    assert runs[0]['outputs']['settings'] == {'temperature': 0, 'seed': 7}


def test_runner_rejects_changed_resume_identity_and_rubric_version(tmp_path):
    from harness_v2.run import run_experiment
    manifest = make_dataset(tmp_path); config = config_at(tmp_path); out = tmp_path / 'runs'
    run_experiment(config, manifest, repeats=1, out=out)
    changed = config_at(tmp_path, name='changed')
    with pytest.raises(ValueError, match='identity'):
        run_experiment(changed, manifest, repeats=1, out=out)
    with pytest.raises(ValueError, match='rubric_version'):
        run_experiment(config_at(tmp_path, rubric_version='wrong'), manifest, repeats=1, out=tmp_path / 'wrong')
