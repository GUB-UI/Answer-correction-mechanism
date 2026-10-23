import json
import subprocess
from pathlib import Path

import pytest

import harness_v2.run as runner
from harness_v2.tests.test_p3_dataset import make_dataset
from harness_v2.tests.test_p3_runner import config_at


def test_default_commit_is_persisted_and_changed_resolution_refuses_resume(tmp_path, monkeypatch):
    config, manifest, out = config_at(tmp_path), make_dataset(tmp_path), tmp_path / 'runs'
    monkeypatch.setattr(runner, 'resolve_implementation_commit', lambda: 'a' * 40)
    runner.run_experiment(config, manifest, repeats=1, out=out)
    for name in ('experiment.json', 'results.json'):
        assert json.loads((out / name).read_text())['commit'] == 'a' * 40
    results = json.loads((out / 'results.json').read_text())
    assert results['runs'][0]['commit'] == 'a' * 40
    assert runner.run_experiment(config, manifest, repeats=1, out=out)['skipped'] == 1
    before = {p.relative_to(out): p.read_bytes() for p in out.rglob('*') if p.is_file()}
    monkeypatch.setattr(runner, 'resolve_implementation_commit', lambda: 'b' * 40)
    with pytest.raises(ValueError, match='commit'):
        runner.run_experiment(config, manifest, repeats=2, out=out)
    assert {p.relative_to(out): p.read_bytes() for p in out.rglob('*') if p.is_file()} == before


@pytest.mark.parametrize('commit', [None, 'unknown', '', ' UNKNOWN '])
@pytest.mark.parametrize('resume', [False, True])
def test_invalid_commit_refuses_before_writes_or_provider(tmp_path, monkeypatch, commit, resume):
    config, manifest, out = config_at(tmp_path), make_dataset(tmp_path), tmp_path / 'runs'
    if resume:
        runner.run_experiment(config, manifest, repeats=1, out=out, commit='rev-a')
    before = {p.relative_to(out): p.read_bytes() for p in out.rglob('*') if p.is_file()}
    monkeypatch.setattr(runner, 'resolve_implementation_commit', lambda: None)
    def forbidden(*args):
        pytest.fail('invalid identity reached provider')
    with pytest.raises(ValueError, match='commit'):
        runner.run_experiment(config, manifest, repeats=2, out=out,
                              commit=commit, provider_factory=forbidden)
    assert {p.relative_to(out): p.read_bytes() for p in out.rglob('*') if p.is_file()} == before
    assert out.exists() == resume


def test_explicit_commit_works_without_git_resolution(tmp_path, monkeypatch):
    def forbidden():
        pytest.fail('explicit commit must not require Git')
    monkeypatch.setattr(runner, 'resolve_implementation_commit', forbidden)
    out = tmp_path / 'runs'
    runner.run_experiment(config_at(tmp_path), make_dataset(tmp_path), repeats=1, out=out, commit='release-1')
    assert json.loads((out / 'experiment.json').read_text())['commit'] == 'release-1'


def test_legacy_unknown_commit_cannot_resume_or_be_relabelled(tmp_path):
    config, manifest, out = config_at(tmp_path), make_dataset(tmp_path), tmp_path / 'runs'
    runner.run_experiment(config, manifest, repeats=1, out=out, commit='rev-a')
    metadata = json.loads((out / 'experiment.json').read_text())
    metadata['commit'] = 'unknown'
    (out / 'experiment.json').write_text(json.dumps(metadata))
    before = (out / 'experiment.json').read_bytes()
    for commit in ('unknown', None, 'rev-a'):
        with pytest.raises(ValueError, match='commit'):
            runner.run_experiment(config, manifest, repeats=1, out=out, commit=commit)
    assert (out / 'experiment.json').read_bytes() == before


@pytest.mark.parametrize('resolved', ['a' * 40 + '+dirty', None])
def test_cli_default_resolves_or_reports_clear_error(tmp_path, monkeypatch, capsys, resolved):
    config, manifest, out = config_at(tmp_path), make_dataset(tmp_path), tmp_path / 'runs'
    monkeypatch.setattr(runner, 'resolve_implementation_commit', lambda: resolved)
    args = ['--config', str(config), '--dataset', str(manifest), '--repeats', '1', '--out', str(out)]
    if resolved is None:
        with pytest.raises(SystemExit) as exc:
            runner.main(args)
        assert exc.value.code == 2
        assert 'commit' in capsys.readouterr().err and not out.exists()
    else:
        assert runner.main(args) == 0
        assert json.loads((out / 'experiment.json').read_text())['commit'] == resolved


@pytest.mark.parametrize('status,expected', [('', 'a' * 40), (' M harness_v2/run.py\n', 'a' * 40 + '+dirty'),
    ('M  harness_v2/run.py\n', 'a' * 40 + '+dirty')])
def test_git_resolution_uses_source_checkout_and_tracked_status(monkeypatch, isolated_commit_resolver, status, expected):
    def git(args, *, cwd, **kwargs):
        assert cwd == Path(runner.__file__).resolve().parent
        assert kwargs == {'capture_output': True, 'text': True, 'check': True, 'timeout': 5}
        if args == ['git', 'rev-parse', 'HEAD']:
            return subprocess.CompletedProcess(args, 0, 'a' * 40 + '\n', '')
        assert args == ['git', 'status', '--porcelain', '--untracked-files=no']
        return subprocess.CompletedProcess(args, 0, status, '')
    monkeypatch.setattr(runner.subprocess, 'run', git)
    assert isolated_commit_resolver() == expected


@pytest.mark.parametrize('stage', ['HEAD', 'status'])
@pytest.mark.parametrize('error', [FileNotFoundError('git unavailable'),
    subprocess.CalledProcessError(128, ['git']), subprocess.TimeoutExpired(['git'], 5)])
def test_git_resolution_fails_closed(monkeypatch, isolated_commit_resolver, stage, error):
    def git(args, **kwargs):
        if (stage == 'HEAD') == (args[-1] == 'HEAD'):
            raise error
        return subprocess.CompletedProcess(args, 0, 'a' * 40 + '\n', '')
    monkeypatch.setattr(runner.subprocess, 'run', git)
    assert isolated_commit_resolver() is None


@pytest.mark.parametrize('head', ['', 'unknown\n', 'HEAD\n'])
def test_git_resolution_rejects_missing_revision(monkeypatch, isolated_commit_resolver, head):
    monkeypatch.setattr(runner.subprocess, 'run', lambda args, **kw: subprocess.CompletedProcess(args, 0, head, ''))
    assert isolated_commit_resolver() is None
