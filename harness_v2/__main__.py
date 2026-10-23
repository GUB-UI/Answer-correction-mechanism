"""Run one harness case from a JSON file. No network provider is selected here."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from harness_v2.core.pipeline import Pipeline
from harness_v2.providers.fixture import FixtureProvider
from harness_v2.storage.artifacts import ArtifactStore
from harness_v2.storage.sqlite_store import RunStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the v2 harness on one input file.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--experiment", default="compare")
    parser.add_argument("--commit", default="unknown")
    args = parser.parse_args(argv)
    case = json.loads(args.input.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    store = RunStore(args.out / "runs.sqlite", ArtifactStore(args.out / "artifacts"))
    completion = case.get("new_completion")
    if not isinstance(completion, str) or not completion:
        payload = {
            "pipeline": "harness_v2",
            "status": "no_new_completion",
            "failure_reason": "saved new_completion is required for an offline run",
        }
        (args.out / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return 0
    pipeline = Pipeline(store, FixtureProvider(completion))
    record = pipeline.run(case, experiment_id=args.experiment, commit=args.commit)
    (args.out / "result.json").write_text(
        json.dumps({"pipeline": "harness_v2", "status": "ran", "run": record.to_dict()}, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
