import hashlib
import json
import os
import stat

import pytest


def sample_cases():
    return [{'problem_id': 'P', 'input_version': 'manual', 'problem_text': 'solve',
             'rubric_version': 'r1', 'rubric': [{'item_id': 'R1', 'max_points': 4}],
             'answer_text': 'x+1\r\n', 'ocr_metadata': {'engine': 'manual'}}]


def make_dataset(tmp_path):
    from harness_v2.evaluation.dataset import write_dataset
    return write_dataset(tmp_path / 'dataset', dataset_id='dev', version='1', cases=sample_cases(),
                         reference_scores={'P': 4}, reference_items={'P': {'R1': 4}})


def test_private_hashed_dataset_separates_references(tmp_path):
    from harness_v2.evaluation.dataset import load_dataset
    old = os.umask(0)
    try:
        path = make_dataset(tmp_path)
        dataset = load_dataset(path)
        assert dataset.manifest['dataset_id'] == 'dev'
        assert dataset.references['reference_scores'] == {'P': 4}
        case = dataset.cases[0]
        assert case['input_version'] == 'manual'
        assert case['rubric'][0]['item_id'] == 'R1'
        assert 'reference' not in json.dumps(case)
        assert open(case['ocr_import_path'], 'rb').read() == b'x+1\r\n'
        root = path.parent
        for f in root.rglob('*'):
            assert stat.S_IMODE(f.stat().st_mode) == (0o700 if f.is_dir() else 0o600)
        assert stat.S_IMODE(root.stat().st_mode) == 0o700
    finally:
        os.umask(old)


@pytest.mark.parametrize('target', ['input', 'rubric', 'references'])
def test_loader_rejects_hash_tampering(tmp_path, target):
    from harness_v2.evaluation.dataset import DatasetError, load_dataset
    path = make_dataset(tmp_path)
    manifest = json.loads(path.read_text())
    descriptor = manifest['references'] if target == 'references' else manifest['cases'][0][
        'grading_input' if target == 'input' else 'rubric']
    (path.parent / descriptor['path']).write_text('tampered')
    with pytest.raises(DatasetError, match='hash'):
        load_dataset(path)


def test_loader_rejects_duplicates_and_escaping_paths(tmp_path):
    from harness_v2.evaluation.dataset import DatasetError, load_dataset
    path = make_dataset(tmp_path)
    manifest = json.loads(path.read_text())
    manifest['cases'].append(manifest['cases'][0])
    path.write_text(json.dumps(manifest))
    with pytest.raises(DatasetError, match='duplicate'):
        load_dataset(path)
    manifest['cases'].pop()
    manifest['cases'][0]['grading_input']['path'] = '../outside.txt'
    path.write_text(json.dumps(manifest))
    with pytest.raises(DatasetError, match='escape'):
        load_dataset(path)


def test_dev_builder_with_temporary_sources(tmp_path):
    from scripts.build_dev_dataset import build_dev_dataset
    from harness_v2.evaluation.dataset import load_dataset
    kit = tmp_path / 'kit'; kit.mkdir()
    refs = {'dataset_id': 'dev', 'version': '1', 'reference_scores': {'P': 4},
            'reference_items': {'P': {'R1': 4}}}
    ref_path = tmp_path / 'references.json'; ref_path.write_text(json.dumps(refs))
    (kit / 'rubric.json').write_text(json.dumps({'P': [{'id': 'R1', 'max_score': 4, 'label': 'work', 'rule': 'match'}]}))
    (kit / 'questions.json').write_text(json.dumps({'P': {'question': 'solve', 'correct_answer': 'do not copy'}}))
    (kit / 'manual_transcriptions.json').write_text(json.dumps([{'question_id': 'P', 'answer_text': 'x+1'}]))
    c1 = tmp_path / 'c1.json'; c1.write_text(json.dumps({'P': {'answer_text': 'x + 1'}}))
    ocr = tmp_path / 'ocr'; ocr.mkdir()
    (ocr / 'ans_P.md').write_bytes(b'$=x+1$\r\n')
    (ocr / 'stats.jsonl').write_text(json.dumps({'image': 'ans_P', 'wall_s': 16, 'dtype': 'float32', 'revision': 'ff990be3'})+'\n')
    path = build_dev_dataset(tmp_path / 'built', kit=kit, c1_path=c1, ocr_dir=ocr,
                             references_path=ref_path, dataset_id='dev', version='1')
    dataset = load_dataset(path)
    assert {c['input_version'] for c in dataset.cases} == {'manual_C0', 'manual_C1', 'xiaomi_cpu_fp32'}
    assert 'do not copy' not in json.dumps(dataset.manifest)
    cpu = next(c for c in dataset.cases if c['input_version'] == 'xiaomi_cpu_fp32')
    assert cpu['ocr_metadata']['revision'] == 'ff990be3'
    assert cpu['ocr_metadata']['engine'] == 'Xiaomi-OCR-0'
    assert cpu['ocr_metadata']['dtype'] == 'float32'
    assert cpu['ocr_metadata']['elapsed_s'] == 16
