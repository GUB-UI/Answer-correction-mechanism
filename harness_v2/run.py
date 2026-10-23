"""Repeatable dataset runs. Gemma is explicit; fixture mode is fully offline."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from harness_v2.core.pipeline import Pipeline
from harness_v2.evaluation.dataset import Dataset, load_dataset
from harness_v2.evaluation.reference import failure_from_run
from harness_v2.providers.fixture import FixtureProvider
from harness_v2.providers.gemma_ai_studio import GemmaAIStudio, QuotaExceeded
from harness_v2.storage.artifacts import ArtifactStore
from harness_v2.storage.permissions import private_directory, private_open, private_write_text
from harness_v2.storage.secrets import redact
from harness_v2.storage.sqlite_store import RunStore


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def identity(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def read_json(path: Path) -> Any:
    with private_open(path) as handle:
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
    canonical(config)
    return config


def request_id_for(dataset: Dataset, case: dict, config: dict, repetition: int) -> str:
    return 'h2-' + identity([dataset.manifest['dataset_id'], dataset.manifest['version'],
        case['case_id'], case['input_sha256'], case['rubric_sha256'],
        identity(config), repetition])


def default_provider(config: dict, case: dict, repetition: int):
    if config['provider'] == 'gemma':
        return GemmaAIStudio(model_id=config['model'])
    completion = config.get('fixture_completions', {}).get(case['case_id'], config.get('fixture_completion'))
    if isinstance(completion, list):
        completion = completion[repetition - 1]
    if isinstance(completion, dict):
        completion = json.dumps(completion, ensure_ascii=False)
    if not isinstance(completion, str):
        raise ValueError('fixture provider requires fixture_completion or a case fixture_completions entry')
    return FixtureProvider(completion, model_id=config['model'])


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
                   provider_factory: Callable | None = None, commit: str = 'unknown') -> dict:
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError('repeats must be a positive integer')
    if isinstance(delay_s, bool) or not isinstance(delay_s, (int, float)) or not math.isfinite(delay_s) or delay_s < 0:
        raise ValueError('delay must be finite and nonnegative')
    config, dataset = load_config(Path(config_path)), load_dataset(Path(dataset_path))
    if any(case['rubric_version'] != config['rubric_version'] for case in dataset.cases):
        raise ValueError('config rubric_version differs from dataset rubric_version')
    out = private_directory(Path(out))
    experiment_identity = identity({'config': config, 'dataset': dataset.manifest})
    experiment_path = out / 'experiment.json'
    metadata = {'schema_version': 1, 'identity': experiment_identity, 'config': config,
                'dataset': dataset.manifest, 'dataset_path': str(dataset.path)}
    if experiment_path.exists() and read_json(experiment_path).get('identity') != experiment_identity:
        raise ValueError('output directory experiment identity differs; use a new output directory')
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
                        **dataset.reference_for(case['problem_id'])}
        pipeline = Pipeline(store, provider)
        sent += 1
        try:
            record = pipeline.run(runtime_case, experiment_id=config['name'], commit=commit)
        except QuotaExceeded as exc:
            record = exc.record
            stopped = 'quota'
        record.outputs['experiment_case'] = {
            **next(item for item in expected if item['request_id'] == rid),
            'dataset_id': dataset.manifest['dataset_id'], 'dataset_version': dataset.manifest['version'],
            'config_id': identity(config)}
        # Persist experiment metadata in SQLite too, without creating another run.
        store.update_outputs(record.run_id, record.outputs)
        latest[rid] = record.to_dict()
        failure = failure_from_run(record, **dataset.reference_for(case['problem_id']))
        if failure is not None:
            write_json(out / 'failure_cases' / f'{rid}--attempt-{record.outputs["run_attempt"]}.json', failure.to_dict())
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
    parser.add_argument('--commit', default='unknown')
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
