"""Label-free report for the robustness challenge set (characterization only).

Pairs every scheduled variant job with the original (O) job of the same
source and repetition. Needs no reference scores and reads none. Every
scheduled job is classified as ``ok``, ``missing`` (no run recorded),
``failed`` (grading status failed) or ``incomplete`` (missing, duplicate,
unexpected or non-finite item scores). Only ``ok`` pairs are compared;
nothing is filled in.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from harness_v2.storage.permissions import private_write_text

FLAGGED = "review_required"
SCORED = {"validated", FLAGGED}


def _cell(run: dict | None, expected_items: set | None) -> dict:
    if run is None:
        return {"outcome": "missing", "status": None, "scores": None}
    grading = (run.get("outputs") or {}).get("grading") or {}
    status = grading.get("status")
    if status not in SCORED:
        return {"outcome": "failed", "status": status, "scores": None}
    items = grading.get("items")
    scores = {}
    ok = isinstance(items, list) and bool(items)
    for item in items if ok else []:
        key = item.get("item_id") if isinstance(item, dict) else None
        score = item.get("score") if isinstance(item, dict) else None
        if (not isinstance(key, str) or key in scores or isinstance(score, bool)
                or not isinstance(score, (int, float)) or not math.isfinite(score)):
            ok = False
            break
        scores[key] = float(score)
    if ok and expected_items is not None and set(scores) != expected_items:
        ok = False
    return {"outcome": "ok" if ok else "incomplete", "status": status, "scores": scores if ok else None}


def _empty() -> dict:
    return {"scheduled": 0, "pairs": 0, "missing": 0, "failed": 0, "incomplete": 0,
            "items": 0, "items_equal": 0, "runs_all_items_equal": 0,
            "flagged": 0, "flag_denominator": 0, "unpaired_no_valid_O": 0,
            "newly_flagged_relative_to_O": 0, "unflagged_relative_to_O": 0}


def _finish(stats: dict) -> dict:
    stats["item_agreement"] = stats["items_equal"] / stats["items"] if stats["items"] else None
    stats["flag_rate"] = stats["flagged"] / stats["flag_denominator"] if stats["flag_denominator"] else None
    return stats


def challenge_report(results: dict, *, rubric_items: dict | None = None) -> dict:
    """``rubric_items`` maps problem_id -> rubric item ids (no labels)."""
    runs = {run.get("request_id"): run for run in results.get("runs", [])}
    cells = {}
    for job in results.get("expected_runs", []):
        source, _, variant = job["case_id"].rpartition(":")
        expected = set(rubric_items[job["problem_id"]]) if rubric_items and job.get("problem_id") in rubric_items else None
        cells[(source, variant, job["repetition"])] = _cell(runs.get(job["request_id"]), expected)
    totals = defaultdict(_empty)
    by_source = defaultdict(lambda: defaultdict(_empty))
    for (source, variant, rep), cell in sorted(cells.items()):
        for stats in (totals[variant], by_source[source][variant]):
            stats["scheduled"] += 1
            if cell["outcome"] != "ok":
                stats[cell["outcome"]] += 1
            else:
                stats["flag_denominator"] += 1
                stats["flagged"] += cell["status"] == FLAGGED
            if variant == "O" or cell["outcome"] != "ok":
                continue
            base = cells.get((source, "O", rep))
            if base is None or base["outcome"] != "ok" or set(base["scores"]) != set(cell["scores"]):
                stats["unpaired_no_valid_O"] += 1
                continue
            stats["pairs"] += 1
            equal = [cell["scores"][k] == base["scores"][k] for k in base["scores"]]
            stats["items"] += len(equal)
            stats["items_equal"] += sum(equal)
            stats["runs_all_items_equal"] += all(equal)
            stats["newly_flagged_relative_to_O"] += cell["status"] == FLAGGED and base["status"] != FLAGGED
            stats["unflagged_relative_to_O"] += cell["status"] != FLAGGED and base["status"] == FLAGGED
    return {"kind": "label-free robustness characterization (development data, not generalization)",
            "note": "flag_rate = flagged / scored runs; pairs need ok variant and ok O of the same repetition",
            "variants": {v: _finish(s) for v, s in sorted(totals.items())},
            "by_source": {src: {v: _finish(s) for v, s in sorted(vs.items())}
                          for src, vs in sorted(by_source.items())}}


def _rubric_items(manifest: Path) -> dict:
    from harness_v2.evaluation.dataset import load_dataset
    out = {}
    for case in load_dataset(manifest).cases:
        out[case["problem_id"]] = [item["item_id"] for item in json.loads(json.dumps(case["rubric"]))]
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, help="challenge manifest (rubric item ids only)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    items = _rubric_items(args.dataset) if args.dataset else None
    report = challenge_report(json.loads(args.results.read_text(encoding="utf-8")), rubric_items=items)
    private_write_text(args.out, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
