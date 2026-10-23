"""Run the old parser and the v2 harness in separate processes.

Each side gets its own output directory and its own copy of the input with
prior AI scores and reference scores removed. Neither side reads the other
side's output.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness_v2.evaluation.comparison import (  # noqa: E402
    evaluate_comparison,
    load_result,
    summarize_comparisons,
)


def compare(inputs: Path, out: Path) -> dict:
    rows = []
    for path in sorted(inputs.glob("*.json")):
        old_dir = out / path.stem / "old"
        new_dir = out / path.stem / "new"
        old = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "legacy_runner.py"), "--input", str(path), "--out", str(old_dir)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        new = subprocess.run(
            [sys.executable, "-m", "harness_v2", "--input", str(path), "--out", str(new_dir)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        old_result, old_error = load_result(old_dir / "result.json")
        new_result, new_error = load_result(new_dir / "result.json")
        case = json.loads(path.read_text(encoding="utf-8"))
        evaluation = evaluate_comparison(old_result, new_result, case)
        reasons = [evaluation["reason"]] if evaluation["reason"] else []
        for side, process, error in (("old", old, old_error), ("new", new, new_error)):
            if process.returncode:
                reasons.append(f"{side}: process exited {process.returncode}")
            if error:
                reasons.append(f"{side}: {error}")
        if reasons:
            evaluation = {"success": False, "status": "incomplete", "reason": "; ".join(reasons)}
        rows.append(
            {
                **evaluation,
                "input": path.name,
                "old_exit": old.returncode,
                "new_exit": new.returncode,
                "old_dir": str(old_dir),
                "new_dir": str(new_dir),
                "old_stderr": old.stderr[-500:],
                "new_stderr": new.stderr[-500:],
            }
        )
    summary = summarize_comparisons(rows)
    out.mkdir(mode=0o700, parents=True, exist_ok=True)
    out.chmod(0o700)
    # stderr and validation details can contain private student text.
    fd = os.open(out / "summary.json", os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.fchmod(fd, 0o600)
        os.ftruncate(fd, 0)
        with os.fdopen(fd, "w", encoding="utf-8", closefd=False) as handle:
            handle.write(json.dumps(summary, indent=2))
    finally:
        os.close(fd)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare old and new graders on the same inputs.")
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = compare(args.inputs, args.out)
    return 1 if not summary["case_count"] or summary["incomplete_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
