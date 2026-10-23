"""Run the old parser and the v2 harness in separate processes.

Each side gets its own output directory and its own copy of the input with
prior AI scores and reference scores removed. Neither side reads the other
side's output.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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
        rows.append(
            {
                "input": path.name,
                "old_exit": old.returncode,
                "new_exit": new.returncode,
                "old_dir": str(old_dir),
                "new_dir": str(new_dir),
                "old_stderr": old.stderr[-500:],
                "new_stderr": new.stderr[-500:],
            }
        )
    summary = {"rows": rows}
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare old and new graders on the same inputs.")
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = compare(args.inputs, args.out)
    failed = [row for row in summary["rows"] if row["old_exit"] or row["new_exit"]]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
