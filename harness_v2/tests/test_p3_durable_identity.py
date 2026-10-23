import json
import sqlite3

import pytest

import harness_v2.run as runner
from harness_v2.evaluation.dataset import load_dataset
from harness_v2.providers.gemma_ai_studio import ProviderSetupError
from harness_v2.tests.test_p3_dataset import make_dataset
from harness_v2.tests.test_p3_runner import config_at


def files_at(out):
    return {p.relative_to(out): p.read_bytes() for p in out.rglob('*') if p.is_file()}


def assert_refused(config, manifest, out, match='identity|commit', commit='rev-a'):
    before = files_at(out)
    def forbidden(*args):
        pytest.fail('unverified durable identity reached provider')
    with pytest.raises(ValueError, match=match):
        runner.run_experiment(config, manifest, repeats=2, out=out, commit=commit,
                              provider_factory=forbidden)
    assert files_at(out) == before


def edit_outputs(out, mutate):
    with sqlite3.connect(out / 'runs.sqlite') as conn:
        for run_id, raw in conn.execute('SELECT run_id, outputs_json FROM runs').fetchall():
            outputs = json.loads(raw)
            mutate(outputs)
            conn.execute('UPDATE runs SET outputs_json = ? WHERE run_id = ?',
                         (json.dumps(outputs), run_id))


@pytest.mark.parametrize('metadata', ['missing', 'rewritten'])
def test_sqlite_prevents_relabelling_commit_without_original_metadata(tmp_path, metadata):
    config, manifest, out = config_at(tmp_path), make_dataset(tmp_path), tmp_path / 'runs'
    runner.run_experiment(config, manifest, repeats=1, out=out, commit='rev-a')
    if metadata == 'missing':
        (out / 'experiment.json').unlink()
    else:
        stored = json.loads((out / 'experiment.json').read_text())
        stored['commit'] = 'rev-b'
        stored['identity'] = runner.identity({'config': stored['config'], 'dataset': stored['dataset'], 'commit': 'rev-b'})
        (out / 'experiment.json').write_text(json.dumps(stored))
    assert_refused(config, manifest, out, match='durable row implementation commit', commit='rev-b')


@pytest.mark.parametrize('commit', ['', 'unknown', ' UNKNOWN ', 'rev-b'])
def test_durable_commit_is_checked_even_with_matching_metadata(tmp_path, commit):
    config, manifest, out = config_at(tmp_path), make_dataset(tmp_path), tmp_path / 'runs'
    runner.run_experiment(config, manifest, repeats=1, out=out, commit='rev-a')
    with sqlite3.connect(out / 'runs.sqlite') as conn:
        conn.execute('UPDATE runs SET commit_sha = ?', (commit,))
    assert_refused(config, manifest, out, match='durable row implementation commit')


@pytest.mark.parametrize('field,value', [
    ('experiment_identity', 'different'), ('experiment_identity', None),
    ('config_id', 'different'), ('dataset_id', 'different'), ('dataset_version', 'different'),
    ('input_sha256', '0' * 64), ('rubric_sha256', '0' * 64), ('request_id', 'different'),
    ('case_id', 'different'), ('repetition', True),
])
def test_existing_durable_experiment_provenance_must_match(tmp_path, field, value):
    config, manifest, out = config_at(tmp_path), make_dataset(tmp_path), tmp_path / 'runs'
    runner.run_experiment(config, manifest, repeats=1, out=out, commit='rev-a')
    edit_outputs(out, lambda outputs: outputs['experiment_case'].update({field: value}))
    assert_refused(config, manifest, out, match='durable row experiment identity')


@pytest.mark.parametrize('name', ['', 'other-experiment'])
def test_durable_experiment_name_is_checked(tmp_path, name):
    config, manifest, out = config_at(tmp_path), make_dataset(tmp_path), tmp_path / 'runs'
    runner.run_experiment(config, manifest, repeats=1, out=out, commit='rev-a')
    with sqlite3.connect(out / 'runs.sqlite') as conn:
        conn.execute('UPDATE runs SET experiment_id = ?', (name,))
    assert_refused(config, manifest, out, match='durable row experiment identity')


@pytest.mark.parametrize('legacy', ['missing_identity', 'missing_provenance', 'invalid_provenance'])
def test_missing_metadata_and_unestablished_row_identity_refuses_resume(tmp_path, legacy):
    config, manifest, out = config_at(tmp_path), make_dataset(tmp_path), tmp_path / 'runs'
    runner.run_experiment(config, manifest, repeats=1, out=out, commit='rev-a')
    def mutate(outputs):
        if legacy == 'missing_identity':
            outputs['experiment_case'].pop('experiment_identity')
        elif legacy == 'missing_provenance':
            outputs.pop('experiment_case')
        else:
            outputs['experiment_case'] = []
    edit_outputs(out, mutate)
    (out / 'experiment.json').unlink()
    assert_refused(config, manifest, out, match='cannot be established')


def test_matching_rows_recover_missing_metadata_without_inference(tmp_path):
    config, manifest, out = config_at(tmp_path), make_dataset(tmp_path), tmp_path / 'runs'
    runner.run_experiment(config, manifest, repeats=1, out=out, commit='rev-a')
    expected = (out / 'experiment.json').read_bytes()
    (out / 'experiment.json').unlink()
    def forbidden(*args):
        pytest.fail('matching completed run reached provider')
    assert runner.run_experiment(config, manifest, repeats=1, out=out, commit='rev-a',
                                 provider_factory=forbidden)['skipped'] == 1
    assert (out / 'experiment.json').read_bytes() == expected
    for p in out.rglob('*'):
        assert p.stat().st_mode & 0o777 == (0o700 if p.is_dir() else 0o600)


def test_rewritten_metadata_cannot_hide_changed_config(tmp_path):
    config, manifest, out = config_at(tmp_path), make_dataset(tmp_path), tmp_path / 'runs'
    runner.run_experiment(config, manifest, repeats=1, out=out, commit='rev-a')
    config_at(tmp_path, prompt_version='p2')
    stored = json.loads((out / 'experiment.json').read_text())
    stored['config'] = runner.load_config(config)
    stored['identity'] = runner.identity({'config': stored['config'], 'dataset': load_dataset(manifest).manifest, 'commit': 'rev-a'})
    (out / 'experiment.json').write_text(json.dumps(stored))
    assert_refused(config, manifest, out, match='durable row experiment identity')


def test_commit_validation_includes_historical_failed_rows(tmp_path):
    config, manifest, out = config_at(tmp_path), make_dataset(tmp_path), tmp_path / 'runs'
    def broken(*args):
        raise ProviderSetupError('temporary failure')
    runner.run_experiment(config, manifest, repeats=1, out=out, commit='rev-a', provider_factory=broken)
    runner.run_experiment(config, manifest, repeats=1, out=out, commit='rev-a')
    with sqlite3.connect(out / 'runs.sqlite') as conn:
        conn.execute("UPDATE runs SET commit_sha = 'old' WHERE failure_reason != ''")
    assert_refused(config, manifest, out, match='durable row implementation commit')


@pytest.mark.parametrize('damage', ['corrupt_database', 'missing_commit_column', 'invalid_outputs'])
def test_unreadable_durable_identity_refuses_before_metadata_writes(tmp_path, damage):
    config, manifest, out = config_at(tmp_path), make_dataset(tmp_path), tmp_path / 'runs'
    runner.run_experiment(config, manifest, repeats=1, out=out, commit='rev-a')
    if damage == 'corrupt_database':
        (out / 'runs.sqlite').write_bytes(b'not a sqlite database')
    else:
        with sqlite3.connect(out / 'runs.sqlite') as conn:
            if damage == 'missing_commit_column':
                conn.execute('ALTER TABLE runs DROP COLUMN commit_sha')
            else:
                conn.execute("UPDATE runs SET outputs_json = 'not json'")
    assert_refused(config, manifest, out, match='cannot be established')
