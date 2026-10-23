"""Run the existing grader's JSON parser on one saved completion.

This process does not import the v2 pipeline. It only reads a completion that
was already produced by the old system.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "math-answer-diagnosis"))

from harness_v2.evaluation.independent import prepare_independent_input  # noqa: E402
from harness_v2.storage.permissions import private_directory, private_write_text  # noqa: E402
from src.grading.parse import GradingOutputError, parse_grading_text  # noqa: E402


def run(input_path: Path, out_dir: Path) -> dict:
    case = json.loads(input_path.read_text(encoding="utf-8"))
    public = prepare_independent_input(case)
    private_directory(out_dir, managed=True)
    private_write_text(out_dir / "input_used.json",
        json.dumps(public, ensure_ascii=False, indent=2),
    )
    completion = case.get("legacy_completion")
    if not isinstance(completion, str) or not completion.strip():
        payload = {
            "pipeline": "legacy",
            "input_id": case.get("input_id"),
            "status": "no_legacy_completion",
            "failure_reason": "old pipeline was not given a saved completion and no model was called",
        }
    else:
        try:
            parsed = parse_grading_text(completion)
        except GradingOutputError as exc:
            payload = {
                "pipeline": "legacy",
                "input_id": case.get("input_id"),
                "status": "parse_error",
                "failure_reason": str(exc),
            }
        else:
            payload = {
                "pipeline": "legacy",
                "input_id": case.get("input_id"),
                "status": "parsed",
                "fence_stripped": parsed.fence_stripped,
                "data": parsed.data,
                "failure_reason": "",
            }
    private_write_text(out_dir / "result.json", json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parse one saved legacy grading completion.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.input, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
