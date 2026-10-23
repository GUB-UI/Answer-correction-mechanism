"""Compare two saved experiments under identical grading conditions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness_v2.evaluation.comparison import compare_experiments
from harness_v2.evaluation.dataset import load_dataset
from harness_v2.run import read_json, write_json


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, help='override saved manifest location')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    def load(path):
        return read_json(path / 'results.json' if path.is_dir() else path)
    try:
        baseline, candidate = load(args.baseline), load(args.candidate)
        dataset = load_dataset(args.dataset or Path(baseline['dataset_path']))
        result = compare_experiments(baseline, candidate, dataset=dataset)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        result = {'comparable': False, 'verdict': 'incomparable',
                  'reasons': [f'invalid experiment or dataset: {exc}'],
                  'baseline': None, 'candidate': None, 'deltas': {}}
    write_json(args.out, result)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['comparable'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
