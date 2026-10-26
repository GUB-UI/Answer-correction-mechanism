"""Versioned, hashed, owner-only datasets; reference labels stay separate."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness_v2.evaluation.reference import optional_score
from harness_v2.grading.rubric import rubric_from_data
from harness_v2.storage.permissions import private_directory, private_open, private_write_text


class DatasetError(ValueError):
    pass


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(data: bytes) -> Any:
    try:
        return json.loads(data.decode('utf-8'))
    except (ValueError, UnicodeError) as exc:
        raise DatasetError(f'invalid dataset JSON: {exc}') from exc


def _read(path: Path) -> bytes:
    with path.open('rb') as handle:
        return handle.read()


def _checked(root: Path, descriptor: dict) -> tuple[Path, bytes]:
    if not isinstance(descriptor, dict) or not isinstance(descriptor.get('path'), str):
        raise DatasetError('file descriptor requires path and sha256')
    path = (root / descriptor['path']).resolve()
    if root.resolve() not in path.parents:
        raise DatasetError('dataset path escapes dataset directory')
    try:
        data = _read(path)
    except OSError as exc:
        raise DatasetError(f'dataset file unavailable: {descriptor["path"]}') from exc
    if descriptor.get('sha256') != sha256(data):
        raise DatasetError(f'hash mismatch: {descriptor["path"]}')
    return path, data


@dataclass
class Dataset:
    path: Path
    manifest: dict[str, Any]
    cases: list[dict[str, Any]]
    references: dict[str, Any]

    def reference_for(self, problem_id: str) -> dict[str, Any]:
        return {'reference_score': self.references['reference_scores'].get(problem_id),
                'reference_items': self.references.get('reference_items', {}).get(problem_id)}


def _required_text(row: dict, key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DatasetError(f'{key} must be a nonempty string')
    return value


def load_dataset(manifest_path: Path) -> Dataset:
    path = Path(manifest_path).resolve()
    manifest = _json(_read(path))
    if not isinstance(manifest, dict) or manifest.get('schema_version') != 1:
        raise DatasetError('unsupported dataset schema_version')
    dataset_id = _required_text(manifest, 'dataset_id')
    version = _required_text(manifest, 'version')
    rows = manifest.get('cases')
    if not isinstance(rows, list) or not rows:
        raise DatasetError('dataset cases must be nonempty')
    cases, seen, seen_inputs, maxima = [], set(), set(), {}
    for row in rows:
        if not isinstance(row, dict):
            raise DatasetError('case must be an object')
        case_id = _required_text(row, 'case_id')
        problem_id = _required_text(row, 'problem_id')
        input_version = _required_text(row, 'input_version')
        if case_id in seen or (problem_id, input_version) in seen_inputs:
            raise DatasetError('duplicate dataset case')
        seen.add(case_id); seen_inputs.add((problem_id, input_version))
        problem_text = _required_text(row, 'problem_text')
        rubric_version = _required_text(row, 'rubric_version')
        input_path, raw = _checked(path.parent, row.get('grading_input'))
        try:
            raw.decode('utf-8')
        except UnicodeError as exc:
            raise DatasetError('grading input must be UTF-8') from exc
        _, rubric_raw = _checked(path.parent, row.get('rubric'))
        rubric_data = _json(rubric_raw)
        try:
            rubric = rubric_from_data(rubric_data)
        except (ValueError, TypeError) as exc:
            raise DatasetError(f'invalid rubric: {exc}') from exc
        allocations = {item.item_id: item.max_points for item in rubric}
        if problem_id in maxima and maxima[problem_id] != allocations:
            raise DatasetError('one problem has inconsistent rubric allocations')
        maxima[problem_id] = allocations
        metadata = row.get('ocr_metadata', {})
        if not isinstance(metadata, dict):
            raise DatasetError('ocr_metadata must be an object')
        # Explicit allowlist: the inference case never contains reference labels.
        cases.append({'case_id': case_id, 'input_id': case_id, 'problem_id': problem_id,
                      'input_version': input_version, 'problem_text': problem_text,
                      'rubric': rubric_data, 'rubric_version': rubric_version,
                      'ocr_import_path': str(input_path), 'ocr_metadata': metadata,
                      'input_sha256': row['grading_input']['sha256'],
                      'rubric_sha256': row['rubric']['sha256']})
    _, ref_raw = _checked(path.parent, manifest.get('references'))
    refs = _json(ref_raw)
    if not isinstance(refs, dict) or refs.get('dataset_id') != dataset_id or refs.get('version') != version:
        raise DatasetError('reference dataset identity/version mismatch')
    totals = refs.get('reference_scores')
    item_refs = refs.get('reference_items', {})
    if not isinstance(totals, dict) or not isinstance(item_refs, dict):
        raise DatasetError('reference_scores/reference_items must be objects')
    if set(totals) - set(maxima) or set(item_refs) - set(maxima):
        raise DatasetError('reference refers to unknown problem')
    for problem, total in totals.items():
        number = optional_score(total)
        if number is None or not 0 <= number <= sum(maxima[problem].values()):
            raise DatasetError('reference total outside rubric range')
    for problem, items in item_refs.items():
        if not isinstance(items, dict) or set(items) - set(maxima[problem]):
            raise DatasetError('invalid reference item ids')
        if any(optional_score(score) is None or not 0 <= score <= maxima[problem][key]
               for key, score in items.items()):
            raise DatasetError('reference item outside rubric range')
        if set(items) == set(maxima[problem]) and problem in totals and abs(sum(items.values()) - totals[problem]) > 1e-6:
            raise DatasetError('reference item sum differs from total')
    return Dataset(path, manifest, cases, refs)


def write_dataset(out: Path, *, dataset_id: str, version: str, cases: list[dict],
                  reference_scores: dict, reference_items: dict | None = None) -> Path:
    """Copy input bytes and rubrics into a portable manifest directory."""
    out = private_directory(Path(out), managed=True)
    for folder in ("inputs", "rubrics"):
        private_directory(out / folder, managed=True)
    def write(relative: str, data: bytes) -> dict:
        target = out / relative
        with private_open(target, 'wb') as handle:
            handle.write(data)
        return {'path': relative, 'sha256': sha256(data)}
    def encoded(value: Any) -> bytes:
        return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    rows = []
    for index, case in enumerate(cases):
        raw = case.get('grading_input_bytes')
        if raw is None:
            raw = case['answer_text'].encode('utf-8')
        if not isinstance(raw, bytes):
            raise DatasetError('grading_input_bytes must be bytes')
        rows.append({'case_id': case.get('case_id') or f"{case['problem_id']}:{case['input_version']}",
                     'problem_id': case['problem_id'], 'input_version': case['input_version'],
                     'problem_text': case['problem_text'], 'rubric_version': case['rubric_version'],
                     'rubric': write(f'rubrics/{index:04d}.json', encoded(case['rubric'])),
                     'grading_input': write(f'inputs/{index:04d}.txt', raw),
                     'ocr_metadata': case.get('ocr_metadata', {})})
    reference = write('references.json', encoded({'dataset_id': dataset_id, 'version': version,
        'reference_scores': reference_scores, 'reference_items': reference_items or {}}))
    manifest_path = out / 'manifest.json'
    private_write_text(manifest_path, encoded({'schema_version': 1, 'dataset_id': dataset_id,
        'version': version, 'cases': rows, 'references': reference}).decode('utf-8'))
    load_dataset(manifest_path)
    return manifest_path
