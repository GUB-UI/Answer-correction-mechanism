import hashlib
import json

import pytest

from harness_v2.core.pipeline import Pipeline
from harness_v2.providers.fixture import FixtureProvider
from harness_v2.recognition.ocr import XiaomiOCR
from harness_v2.tests.test_p3_setup import store_at
from harness_v2.tests.test_pipeline import _case, _valid


@pytest.mark.parametrize('mode', ['sidecar', 'case', 'explicit_path'])
def test_import_provenance_survives_exact_byte_copy_and_pipeline(tmp_path, mode):
    source = tmp_path / 'answer.md'
    raw = b'x+1\r\n'
    source.write_bytes(raw)
    meta = {'engine': 'Xiaomi-OCR-0', 'model_revision': 'ff990be3', 'dtype': 'float32',
            'timing': {'wall_s': 2.75}, 'source': 'cpu-batch'}
    case = _case('m', _valid())
    case.pop('answer_text')
    case['ocr_import_path'] = str(source)
    if mode == 'sidecar':
        source.with_name(source.name + '.meta.json').write_text(json.dumps(meta))
    elif mode == 'case':
        case['ocr_metadata'] = meta
    else:
        sidecar = tmp_path / 'meta.json'
        sidecar.write_text(json.dumps(meta))
        case['ocr_metadata_path'] = str(sidecar)
    store = store_at(tmp_path / 'out')
    rec = Pipeline(store, FixtureProvider(_valid())).run(case, experiment_id='e', commit='c')
    saved = rec.outputs['ocr_metadata']
    assert saved['engine'] == meta['engine']
    assert saved['revision'] == 'ff990be3'
    assert saved['dtype'] == 'float32'
    assert saved['elapsed_s'] == 2.75
    assert saved['source'] == 'cpu-batch'
    assert saved['timing'] == meta['timing']
    assert store.artifacts.read_bytes(rec.outputs['ocr_artifact']) == raw
    assert rec.outputs['ocr_sha256'] == hashlib.sha256(raw).hexdigest()


def test_case_metadata_overrides_sidecar(tmp_path):
    source = tmp_path / 'a.md'
    source.write_text('x')
    source.with_name('a.md.meta.json').write_text(json.dumps({'engine': 'old', 'dtype': 'float32'}))
    store = store_at(tmp_path / 'out')
    artifact = XiaomiOCR(store.artifacts).import_file(source, ocr_id='a', metadata={'engine': 'new'})
    assert artifact.engine == 'new'
    assert artifact.dtype == 'float32'


@pytest.mark.parametrize('explicit', [False, True])
def test_import_records_exact_sidecar_and_source_hashes_without_trusting_claims(tmp_path, explicit):
    source = tmp_path / 'answer.md'
    raw = b'x+1\r\n'
    source.write_bytes(raw)
    sidecar = tmp_path / 'chosen.json' if explicit else source.with_name(source.name + '.meta.json')
    sidecar_bytes = b'{\r\n "engine": "manual", "elapsed_s": 1\r\n}\r\n'
    sidecar.write_bytes(sidecar_bytes)
    artifact = XiaomiOCR(store_at(tmp_path / 'out').artifacts).import_file(
        source, ocr_id='import', metadata={'sidecar_sha256': 'forged', 'source_sha256': 'forged'},
        metadata_path=sidecar if explicit else None)
    assert artifact.metadata['sidecar_sha256'] == hashlib.sha256(sidecar_bytes).hexdigest()
    assert artifact.metadata['source_sha256'] == hashlib.sha256(raw).hexdigest()
    assert sidecar.read_bytes() == sidecar_bytes
    assert sidecar.stat().st_mode & 0o777 == 0o600


def test_import_without_sidecar_does_not_invent_a_sidecar_hash(tmp_path):
    source = tmp_path / 'answer.md'
    source.write_bytes(b'x')
    artifact = XiaomiOCR(store_at(tmp_path / 'out').artifacts).import_file(
        source, ocr_id='import', metadata={'sidecar_sha256': 'forged'})
    assert 'sidecar_sha256' not in artifact.metadata
    assert artifact.metadata['source_sha256'] == hashlib.sha256(b'x').hexdigest()
