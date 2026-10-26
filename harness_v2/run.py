"""Repeatable dataset runs. Gemma is explicit; fixture mode is fully offline."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import subprocess
import time
import uuid
from contextlib import closing
from pathlib import Path
from typing import Any, Callable

from harness_v2.core.pipeline import Pipeline
from harness_v2.evaluation.dataset import Dataset, load_dataset
from harness_v2.evaluation.reference import failure_from_run
from harness_v2.providers.fixture import FixtureProvider
from harness_v2.providers.gemma_ai_studio import (
    DEADLINE_GROWTH,
    DEADLINE_RETRY_PAUSE_S,
    FAST_BACKOFF_MAX_S,
    FAST_BACKOFF_MIN_S,
    MAX_ATTEMPTS,
    REQUEST_TIMEOUT_CAP_S,
    REQUEST_TIMEOUT_S,
    TRANSPORT_CONFIG_KEYS,
    GemmaAIStudio,
    QuotaExceeded,
)
from harness_v2.storage.artifacts import ArtifactStore
from harness_v2.storage.permissions import private_directory, private_write_text
from harness_v2.storage.secrets import redact
from harness_v2.storage.sqlite_store import RunStore


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def identity(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def read_json(path: Path) -> Any:
    with Path(path).open(encoding='utf-8') as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    """Atomic private checkpoints; SQLite remains the source for resume."""
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        private_write_text(temporary, json.dumps(redact(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        os.replace(temporary, path)
        path.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)


def load_config(path: Path) -> dict:
    config = read_json(path)
    if not isinstance(config, dict):
        raise ValueError('config must be an object')
    for field in ('name', 'provider', 'model', 'prompt_version', 'rubric_version'):
        if not isinstance(config.get(field), str) or not config[field].strip():
            raise ValueError(f'config {field} must be a nonempty string')
    from harness_v2.grading.scorer import prompt_suffix
    prompt_suffix(config['prompt_version'])  # rejects unregistered versions before any call
    if config['provider'] not in {'gemma', 'fixture'}:
        raise ValueError('provider must be gemma or fixture; no fallback')
    if config['provider'] == 'gemma' and not config['model'].startswith('gemma-'):
        raise ValueError('gemma provider requires a Gemma model')
    settings = config.get('generation_settings')
    if not isinstance(settings, dict):
        raise ValueError('generation_settings must be an object')
    if set(settings) - {'temperature', 'seed', 'max_output_tokens', 'top_p', 'top_k'}:
        raise ValueError('unsupported generation settings')
    for name, value in settings.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'invalid generation setting: {name}')
        if name in {'seed', 'top_k', 'max_output_tokens'} and not isinstance(value, int):
            raise ValueError(f'{name} must be an integer')
        if name != 'seed' and value < 0:
            raise ValueError(f'{name} must be nonnegative')
    config = {**config, 'generation_settings': {'temperature': 0, **settings}}
    config = _with_transport_defaults(config)
    canonical(config)
    return config


def reusable_config(config: dict) -> dict:
    """Config that changes what the model is asked to generate.

    Timeout and retry policy does not change the prompt, model, or sampling.
    A successful grade stays reusable when only that policy changes.
    """
    return {key: value for key, value in config.items() if key not in TRANSPORT_CONFIG_KEYS}


def _with_transport_defaults(config: dict) -> dict:
    defaults = {
        'request_timeout_s': REQUEST_TIMEOUT_S,
        'request_timeout_cap_s': REQUEST_TIMEOUT_CAP_S,
        'deadline_growth': DEADLINE_GROWTH,
        'deadline_retry_pause_s': DEADLINE_RETRY_PAUSE_S,
        'max_attempts': MAX_ATTEMPTS,
        'fast_backoff_min_s': FAST_BACKOFF_MIN_S,
        'fast_backoff_max_s': FAST_BACKOFF_MAX_S,
    }
    merged = {**defaults, **config}
    # Validate by constructing the numeric policy. The provider applies the same rules.
    GemmaAIStudio(
        model_id='gemma-transport-check',
        request_timeout_s=merged['request_timeout_s'],
        request_timeout_cap_s=merged['request_timeout_cap_s'],
        deadline_growth=merged['deadline_growth'],
        deadline_retry_pause_s=merged['deadline_retry_pause_s'],
        max_attempts=merged['max_attempts'],
        fast_backoff_min_s=merged['fast_backoff_min_s'],
        fast_backoff_max_s=merged['fast_backoff_max_s'],
    )
    for key in defaults:
        value = merged[key]
        if isinstance(value, float) and value.is_integer():
            merged[key] = int(value)
    return merged


def request_id_for(dataset: Dataset, case: dict, config: dict, repetition: int) -> str:
    return 'h2-' + identity([dataset.manifest['dataset_id'], dataset.manifest['version'],
        case['case_id'], case['input_sha256'], case['rubric_sha256'],
        identity(reusable_config(config)), repetition])


def default_provider(config: dict, case: dict, repetition: int):
    if config['provider'] == 'gemma':
        return GemmaAIStudio(
            model_id=config['model'],
            request_timeout_s=config['request_timeout_s'],
            request_timeout_cap_s=config['request_timeout_cap_s'],
            deadline_growth=config['deadline_growth'],
            deadline_retry_pause_s=config['deadline_retry_pause_s'],
            max_attempts=config['max_attempts'],
            fast_backoff_min_s=config['fast_backoff_min_s'],
            fast_backoff_max_s=config['fast_backoff_max_s'],
        )
    completion = config.get('fixture_completions', {}).get(case['case_id'], config.get('fixture_completion'))
    if isinstance(completion, list):
        completion = completion[repetition - 1]
    if isinstance(completion, dict):
        completion = json.dumps(completion, ensure_ascii=False)
    if not isinstance(completion, str):
        raise ValueError('fixture provider requires fixture_completion or a case fixture_completions entry')
    return FixtureProvider(completion, model_id=config['model'])


def resolve_implementation_commit() -> str | None:
    """Identify the harness source checkout, independent of the caller's cwd."""
    source = Path(__file__).resolve().parent
    try:
        head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=source,
                              capture_output=True, text=True, check=True, timeout=5).stdout.strip()
        if len(head) not in {40, 64} or any(c not in '0123456789abcdef' for c in head):
            return None
        diff = subprocess.run(['git', 'diff', '--binary', '--no-ext-diff', '--no-textconv', 'HEAD', '--'],
                              cwd=source, capture_output=True, check=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return head + ('+dirty-' + hashlib.sha256(diff).hexdigest() if diff else '')


def validate_durable_identity(db_path: Path, *, commit: str, experiment_identity: str,
                              config: dict, dataset: Dataset) -> None:
    """Read existing rows without creating/updating the database or checkpoints."""
    if not db_path.exists():
        return
    try:
        with closing(sqlite3.connect(db_path.resolve().as_uri() + '?mode=ro', uri=True)) as conn:
            rows = conn.execute('SELECT commit_sha, experiment_id, request_id, outputs_json FROM runs').fetchall()
        cases = {case['case_id']: case for case in dataset.cases}
        for row_commit, experiment_id, request_id, raw_outputs in rows:
            if row_commit != commit:
                raise ValueError('durable row implementation commit differs or cannot be established')
            if experiment_id != config['name']:
                raise ValueError('durable row experiment identity differs or cannot be established')
            outputs = json.loads(raw_outputs)
            if not isinstance(outputs, dict):
                raise ValueError('durable row experiment identity cannot be established: invalid outputs')
            provenance = outputs.get('experiment_case')
            if not isinstance(provenance, dict):
                raise ValueError('durable row experiment identity cannot be established: missing provenance')
            # Mutable checkpoints and partial case/request hashes do not bind
            # the full dataset (including problem text and OCR conditions).
            # Without a durable full identity, legacy repair would relabel grades.
            if provenance.get('experiment_identity') != experiment_identity:
                raise ValueError('durable row experiment identity differs or cannot be established')
            case = cases.get(provenance.get('case_id'))
            rep = provenance.get('repetition')
            if case is None or type(rep) is not int or rep < 1:
                raise ValueError('durable row experiment identity cannot be established: invalid case/repetition')
            expected = {'request_id': request_id_for(dataset, case, config, rep),
                        'dataset_id': dataset.manifest['dataset_id'], 'dataset_version': dataset.manifest['version'],
                        'config_id': identity(reusable_config(config)), 'problem_id': case['problem_id'],
                        'input_version': case['input_version'], 'input_sha256': case['input_sha256'],
                        'rubric_sha256': case['rubric_sha256']}
            if request_id != expected['request_id'] or any(provenance.get(k) != v for k, v in expected.items()):
                raise ValueError('durable row experiment identity differs or cannot be established: case provenance')
    except (sqlite3.Error, TypeError, json.JSONDecodeError) as exc:
        raise ValueError('durable row commit/experiment identity cannot be established') from exc


class _LazyProvider:
    """Construction failures occur inside Pipeline's recorded exception boundary."""
    def __init__(self, config, case, repetition, factory):
        self.config, self.case, self.repetition, self.factory = config, case, repetition, factory
        self.model_id = config['model']

    def generate(self, prompt, *, request_id, settings=None):
        provider = self.factory(self.config, self.case, self.repetition)
        return provider.generate(prompt, request_id=request_id, settings=settings)


def run_experiment(config_path: Path, dataset_path: Path, *, repeats: int, out: Path,
                   delay_s: float = 0, sleep: Callable = time.sleep,
                   provider_factory: Callable | None = None, commit: str | None = None) -> dict:
    if commit is None:
        commit = resolve_implementation_commit()
    if not isinstance(commit, str) or not commit.strip() or commit.strip().lower() == 'unknown':
        raise ValueError('implementation commit is required: supply --commit (not unknown) '
                         'or run from a resolvable harness source checkout')
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError('repeats must be a positive integer')
    if isinstance(delay_s, bool) or not isinstance(delay_s, (int, float)) or not math.isfinite(delay_s) or delay_s < 0:
        raise ValueError('delay must be finite and nonnegative')
    config, dataset = load_config(Path(config_path)), load_dataset(Path(dataset_path))
    if any(case['rubric_version'] != config['rubric_version'] for case in dataset.cases):
        raise ValueError('config rubric_version differs from dataset rubric_version')
    out = private_directory(Path(out), managed=True)
    experiment_identity = identity({'config': reusable_config(config), 'dataset': dataset.manifest, 'commit': commit})
    experiment_path = out / 'experiment.json'
    metadata = {'schema_version': 1, 'identity': experiment_identity, 'config': config,
                'dataset': dataset.manifest, 'dataset_path': str(dataset.path), 'commit': commit}
    if experiment_path.exists():
        stored = read_json(experiment_path)
        if stored.get('commit') != commit:
            raise ValueError('output directory implementation commit differs; use a new output directory')
        if stored.get('identity') != experiment_identity:
            raise ValueError('output directory experiment identity differs; use a new output directory')
    validate_durable_identity(out / 'runs.sqlite', commit=commit, experiment_identity=experiment_identity,
                              config=config, dataset=dataset)
    write_json(experiment_path, metadata)
    store = RunStore(out / 'runs.sqlite', ArtifactStore(out / 'artifacts'))
    previous = store.list_runs()
    # ORDER BY timestamps is not enough when attempts finish in the same second.
    latest = {}
    for row in previous:
        rid = row['request_id']
        if rid not in latest or row['outputs'].get('run_attempt', 1) > latest[rid]['outputs'].get('run_attempt', 1):
            latest[rid] = row
    jobs = [(case, rep, request_id_for(dataset, case, config, rep))
            for rep in range(1, repeats + 1) for case in dataset.cases]
    expected = [{'request_id': rid, 'case_id': case['case_id'], 'repetition': rep,
                 'problem_id': case['problem_id'], 'input_version': case['input_version'],
                 'input_sha256': case['input_sha256'], 'rubric_sha256': case['rubric_sha256']}
                for case, rep, rid in jobs]
    provenance = {job['request_id']: {**job,
        'experiment_identity': experiment_identity,
        'dataset_id': dataset.manifest['dataset_id'], 'dataset_version': dataset.manifest['version'],
        'config_id': identity(reusable_config(config))} for job in expected}
    cases_by_request = {rid: case for case, _, rid in jobs}

    def save_failure(row, case):
        path = out / 'failure_cases' / f'{row["request_id"]}--attempt-{row["outputs"].get("run_attempt", 1)}.json'
        if path.exists():
            return
        failure = failure_from_run(row, **dataset.reference_for(case['problem_id']))
        if failure is None:
            return
        private_directory(path.parent, managed=True)
        write_json(path, failure.to_dict())

    # Recover evaluation artifacts only after every durable row's full identity
    # is established, including failures from earlier attempts.
    for row in previous:
        rid = row['request_id']
        if rid not in provenance:
            continue
        save_failure(row, cases_by_request[rid])
    skipped, sent, stopped = 0, 0, ''
    def checkpoint():
        selected = [latest[rid] for _, _, rid in jobs if rid in latest]
        completed = sum(row['outputs']['grading']['status'] != 'failed' for row in selected)
        summary = {'identity': experiment_identity, 'expected': len(jobs), 'completed': completed,
                   'failed': len(selected) - completed, 'pending': len(jobs) - len(selected),
                   'skipped': skipped, 'stopped_reason': stopped, 'request_ids': [rid for _, _, rid in jobs]}
        write_json(out / 'state.json', summary)
        write_json(out / 'results.json', {**metadata, 'expected_runs': expected, 'runs': selected,
                                         'attempt_runs': store.list_runs(), 'summary': summary})
        return summary
    checkpoint()
    for case, rep, rid in jobs:
        if rid in latest and latest[rid]['outputs']['grading']['status'] != 'failed':
            skipped += 1
            continue
        if sent and delay_s:
            sleep(delay_s)
        provider = _LazyProvider(config, case, rep, provider_factory or default_provider)
        # References are attached only as evaluation fields; Pipeline uses an explicit
        # model-input allowlist. No references are supplied to the provider factory.
        runtime_case = {**case, 'request_id': rid, 'prompt_version': config['prompt_version'],
                        'generation_settings': config['generation_settings'],
                        'experiment_case': provenance[rid],
                        **dataset.reference_for(case['problem_id'])}
        pipeline = Pipeline(store, provider)
        sent += 1
        try:
            record = pipeline.run(runtime_case, experiment_id=config['name'], commit=commit)
        except QuotaExceeded as exc:
            record = exc.record
            stopped = 'quota'
        latest[rid] = record.to_dict()
        save_failure(latest[rid], case)
        checkpoint()
        if stopped:
            break
    return checkpoint()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--repeats', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--delay', type=float, default=0, help='seconds between logical requests')
    parser.add_argument('--commit', help='implementation identity; default: harness source HEAD with +dirty-<sha256> for tracked changes')
    args = parser.parse_args(argv)
    try:
        result = run_experiment(args.config, args.dataset, repeats=args.repeats, out=args.out,
                                delay_s=args.delay, commit=args.commit)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False))
    return 3 if result['stopped_reason'] == 'quota' else 1 if result['failed'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
