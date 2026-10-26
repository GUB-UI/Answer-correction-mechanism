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
              'generation_settings': {'temperature': 0, 'seed': 7}, 'prompt_version': 'harness_v2_default',
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
    store = RunStore(out / 'runs.sqlite', ArtifactStore(out / 'artifacts'), readonly=True)
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
        '--dataset', str(manifest), '--repeats', '1', '--out', str(out),
        '--commit', 'test-revision'], capture_output=True, text=True)
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


def test_dataset_input_mutation_between_repetitions_fails_before_provider(tmp_path):
    from harness_v2.run import run_experiment
    from harness_v2.evaluation.dataset import load_dataset
    manifest = make_dataset(tmp_path)
    source = load_dataset(manifest).cases[0]['ocr_import_path']
    calls = []
    class Mutating(FixtureProvider):
        def generate(self, prompt, **kwargs):
            calls.append(prompt)
            from pathlib import Path
            Path(source).write_text('changed')
            return super().generate(prompt, **kwargs)
    out = tmp_path / 'runs'
    result = run_experiment(config_at(tmp_path), manifest, repeats=2, out=out,
                            provider_factory=lambda *a: Mutating(_valid()))
    assert len(calls) == 1
    assert result['failed'] == 1
    assert 'hash' in json.loads((out / 'results.json').read_text())['runs'][1]['failure_reason']


@pytest.mark.parametrize('original,resumed', [('rev-a', 'rev-b'), ('rev-a', 'unknown')])
def test_resume_rejects_changed_implementation_without_mutating_results(tmp_path, original, resumed):
    from harness_v2.run import run_experiment
    manifest = make_dataset(tmp_path)
    config = config_at(tmp_path)
    out = tmp_path / 'runs'
    run_experiment(config, manifest, repeats=1, out=out, commit=original)
    before = {name: (out / name).read_bytes() for name in ['experiment.json', 'results.json', 'state.json']}
    def unexpected_provider(*args):
        pytest.fail('completed runs must not call a provider')
    with pytest.raises(ValueError, match='commit'):
        run_experiment(config, manifest, repeats=2, out=out, commit=resumed,
                       provider_factory=unexpected_provider)
    assert {name: (out / name).read_bytes() for name in before} == before


@pytest.mark.parametrize('commit', ['rev-a', 'rev-b'])
def test_experiment_persists_implementation_identity_and_resumes_same_commit(tmp_path, commit):
    from harness_v2.run import run_experiment
    manifest = make_dataset(tmp_path)
    config = config_at(tmp_path)
    out = tmp_path / 'runs'
    run_experiment(config, manifest, repeats=1, out=out, commit=commit)
    metadata = json.loads((out / 'experiment.json').read_text())
    assert metadata['commit'] == commit
    def unexpected_provider(*args):
        pytest.fail('completed runs must not call a provider')
    assert run_experiment(config, manifest, repeats=1, out=out, commit=commit,
                          provider_factory=unexpected_provider)['skipped'] == 1
    run_experiment(config, manifest, repeats=1, out=tmp_path / 'other', commit=commit + '-other')
    assert json.loads((tmp_path / 'other/experiment.json').read_text())['identity'] != metadata['identity']


def test_resume_after_initial_row_save_keeps_provenance_and_never_repeats_provider(tmp_path, monkeypatch):
    from harness_v2.run import run_experiment
    from harness_v2.storage.sqlite_store import RunStore
    from harness_v2.storage.artifacts import ArtifactStore
    manifest = make_dataset(tmp_path)
    config = config_at(tmp_path)
    out = tmp_path / 'runs'
    original_save = RunStore.save_run
    def interrupt_after_save(self, record):
        original_save(self, record)
        raise KeyboardInterrupt('after durable row')
    with monkeypatch.context() as patch:
        patch.setattr(RunStore, 'save_run', interrupt_after_save)
        with pytest.raises(KeyboardInterrupt):
            run_experiment(config, manifest, repeats=1, out=out)
    store = RunStore(out / 'runs.sqlite', ArtifactStore(out / 'artifacts'), readonly=True)
    row = store.list_runs()[0]
    assert row['outputs']['experiment_case']['input_sha256']
    assert row['outputs']['experiment_case']['repetition'] == 1
    def unexpected_provider(*args):
        pytest.fail('completed run was sent again')
    assert run_experiment(config, manifest, repeats=1, out=out,
                          provider_factory=unexpected_provider)['skipped'] == 1
    assert len(store.list_runs()) == 1


def test_resume_recovers_missing_failure_for_completed_run_with_durable_identity(tmp_path, monkeypatch):
    import harness_v2.run as runner
    from harness_v2.storage.sqlite_store import RunStore
    from harness_v2.storage.artifacts import ArtifactStore
    manifest = make_dataset(tmp_path)
    grade = json.loads(_valid())
    grade['items'][0]['score'] = grade['total_score'] = 3
    config = config_at(tmp_path, fixture_completion=grade)
    out = tmp_path / 'runs'
    original_write = runner.write_json
    def interrupt_failure_write(path, value):
        if path.parent.name == 'failure_cases':
            raise KeyboardInterrupt('before failure artifact')
        original_write(path, value)
    with monkeypatch.context() as patch:
        patch.setattr(runner, 'write_json', interrupt_failure_write)
        with pytest.raises(KeyboardInterrupt):
            runner.run_experiment(config, manifest, repeats=1, out=out)
    store = RunStore(out / 'runs.sqlite', ArtifactStore(out / 'artifacts'), readonly=True)
    row = store.list_runs()[0]
    # Recovery is safe because the full experiment identity survived in SQLite.
    assert row['outputs']['experiment_case']['experiment_identity']
    def unexpected_provider(*args):
        pytest.fail('completed mismatch was sent again')
    summary = runner.run_experiment(config, manifest, repeats=1, out=out,
                                    provider_factory=unexpected_provider)
    assert summary['skipped'] == summary['completed'] == 1
    results = json.loads((out / 'results.json').read_text())
    repaired = results['runs'][0]
    assert repaired['outputs']['experiment_case']['request_id'] == row['request_id']
    assert store.list_runs()[0]['outputs'] == repaired['outputs']
    assert repaired['outputs']['grading'] == row['outputs']['grading']
    assert len(store.list_runs()) == 1
    path = next((out / 'failure_cases').glob('*.json'))
    failure_bytes = path.read_bytes()
    assert 'reference_total_mismatch' in json.loads(failure_bytes)['errors']
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    runner.run_experiment(config, manifest, repeats=1, out=out, provider_factory=unexpected_provider)
    assert path.read_bytes() == failure_bytes


def test_resume_regenerates_missing_failures_for_earlier_attempts(tmp_path):
    from harness_v2.run import run_experiment
    manifest = make_dataset(tmp_path)
    config = config_at(tmp_path)
    out = tmp_path / 'runs'
    def broken(*args):
        raise ProviderSetupError('temporary failure')
    run_experiment(config, manifest, repeats=1, out=out, provider_factory=broken)
    run_experiment(config, manifest, repeats=1, out=out)
    failure_path = next((out / 'failure_cases').glob('*.json'))
    failure_path.unlink()
    def unexpected_provider(*args):
        pytest.fail('completed run was sent again')
    run_experiment(config, manifest, repeats=1, out=out, provider_factory=unexpected_provider)
    assert failure_path.exists()
    assert 'temporary failure' in json.loads(failure_path.read_text())['errors'][0]
    assert len(json.loads((out / 'results.json').read_text())['attempt_runs']) == 2


def test_failure_cases_directory_is_tightened_only_on_write(tmp_path):
    import os
    import stat
    from harness_v2.run import run_experiment
    previous = os.umask(0)
    try:
        manifest = make_dataset(tmp_path)
        grade = json.loads(_valid())
        grade['items'][0]['score'] = grade['total_score'] = 3
        config = config_at(tmp_path, fixture_completion=grade)
        out = tmp_path / 'runs'
        failure_dir = out / 'failure_cases'
        failure_dir.mkdir(parents=True)
        failure_dir.chmod(0o755)
        run_experiment(config, manifest, repeats=1, out=out)
        assert stat.S_IMODE(failure_dir.stat().st_mode) == 0o700
        assert len(list(failure_dir.glob('*.json'))) == 1
        failure_dir.chmod(0o755)

        def unexpected_provider(*args):
            pytest.fail('completed mismatch must not call a provider')

        run_experiment(config, manifest, repeats=1, out=out, provider_factory=unexpected_provider)
        assert stat.S_IMODE(failure_dir.stat().st_mode) == 0o755

        ok = tmp_path / 'ok'
        ok_dir = ok / 'failure_cases'
        ok_dir.mkdir(parents=True)
        ok_dir.chmod(0o755)
        run_experiment(config_at(tmp_path), manifest, repeats=1, out=ok)
        assert stat.S_IMODE(ok_dir.stat().st_mode) == 0o755
        assert list(ok_dir.glob('*.json')) == []
    finally:
        os.umask(previous)
