"""Build private development data offline; no source answers are added to git."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness_v2.evaluation.dataset import write_dataset


def build_dev_dataset(out: Path, *, kit: Path = Path('/workspace/experiment_v1_1/kit/data'),
                      c1_path: Path = Path('/workspace/experiment_v1_2/transcription_C1_v1.2.json'),
                      ocr_dir: Path = Path('/workspace/ocr_cpu/outputs_fp32'),
                      references_path: Path = Path(__file__).with_name('dev_dataset_references.json'),
                      dataset_id: str = 'harness-v2-dev', version: str = '1') -> Path:
    def read(path):
        return json.loads(Path(path).read_bytes().decode('utf-8'))
    rubrics = read(kit / 'rubric.json')
    questions = read(kit / 'questions.json')
    manual = {row['question_id']: row['answer_text'] for row in read(kit / 'manual_transcriptions.json')}
    c1 = read(c1_path)
    references = read(references_path)
    stats = {}
    stats_path = ocr_dir / 'stats.jsonl'
    if stats_path.exists():
        for line in stats_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                stats[Path(str(row['image'])).stem] = row
    cases = []
    for problem_id, rows in rubrics.items():
        rubric = [{'item_id': row['id'], 'max_points': row['max_score'],
                   'description': f"{row['label']}。{row['rule']}"} for row in rows]
        common = {'problem_id': problem_id, 'problem_text': questions[problem_id]['question'],
                  'rubric': rubric, 'rubric_version': 'v1.1'}
        has_c1 = isinstance(c1.get(problem_id), dict) and isinstance(c1[problem_id].get('answer_text'), str)
        cases.append({**common, 'input_version': 'manual_C0' if has_c1 else 'manual',
                      'answer_text': manual[problem_id], 'ocr_metadata': {
                          'engine': 'manual', 'source': str(kit / 'manual_transcriptions.json')}})
        if has_c1:
            cases.append({**common, 'input_version': 'manual_C1', 'answer_text': c1[problem_id]['answer_text'],
                          'ocr_metadata': {'engine': 'manual', 'source': str(c1_path)}})
        stem = f"ans_{problem_id.replace('-', '')}"
        source = ocr_dir / f'{stem}.md'
        meta = stats.get(stem, {})
        # The Phase 2 provenance records this revision; stats can supply a fuller one.
        cases.append({**common, 'input_version': 'xiaomi_cpu_fp32', 'grading_input_bytes': source.read_bytes(),
                      'ocr_metadata': {**meta, 'engine': 'Xiaomi-OCR-0',
                          'revision': meta.get('revision') or 'ff990be3', 'dtype': 'float32',
                          'elapsed_s': meta.get('wall_s'), 'source': str(source)}})
    return write_dataset(out, dataset_id=dataset_id, version=version, cases=cases,
                         reference_scores=references['reference_scores'],
                         reference_items=references.get('reference_items'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--kit', type=Path, default=Path('/workspace/experiment_v1_1/kit/data'))
    parser.add_argument('--c1', type=Path, default=Path('/workspace/experiment_v1_2/transcription_C1_v1.2.json'))
    parser.add_argument('--ocr-dir', type=Path, default=Path('/workspace/ocr_cpu/outputs_fp32'))
    parser.add_argument('--references', type=Path, default=Path(__file__).with_name('dev_dataset_references.json'))
    parser.add_argument('--dataset-id', default='harness-v2-dev')
    parser.add_argument('--version', default='1')
    args = parser.parse_args(argv)
    print(build_dev_dataset(args.out, kit=args.kit, c1_path=args.c1, ocr_dir=args.ocr_dir,
                            references_path=args.references, dataset_id=args.dataset_id, version=args.version))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
