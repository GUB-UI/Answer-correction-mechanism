"""Label-free report for the robustness challenge set (characterization only).

Pairs every scheduled variant job with the original (O) job of the same
source and repetition. Needs no reference scores and reads none. Every
scheduled job is classified as ``ok``, ``missing`` (no run recorded),
``failed`` (grading status failed) or ``incomplete`` (missing, duplicate,
unexpected or non-finite item scores). Only ``ok`` pairs are compared;
nothing is filled in. Missing/empty plans, duplicate planned IDs, duplicate
planned slots, duplicate, missing or unexpected run IDs, and run provenance
that is missing, mismatched, or from more than one experiment make the report
non-comparable;
any remaining metrics are diagnostic only.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from harness_v2.storage.permissions import private_write_text

FLAGGED = "review_required"
SCORED = {"validated", FLAGGED}


def _cell(run: dict | None, expected_items: set) -> dict:
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
    if ok and set(scores) != expected_items:
        ok = False
    return {"outcome": "ok" if ok else "incomplete", "status": status, "scores": scores if ok else None}


def _empty() -> dict:
    return {"scheduled": 0, "pairs": 0, "missing": 0, "failed": 0, "incomplete": 0,
            "items": 0, "items_equal": 0, "runs_all_items_equal": 0,
            "flagged": 0, "flag_denominator": 0, "unpaired_no_valid_O": 0,
            "newly_flagged_relative_to_O": 0, "unflagged_relative_to_O": 0}


def _provenance_issue(job: dict, run: dict):
    """Return ``(reason, experiment_identity)`` for one planned run."""
    rid = job.get("request_id")
    outputs = run.get("outputs") if isinstance(run, dict) else None
    provenance = outputs.get("experiment_case") if isinstance(outputs, dict) else None
    if not isinstance(provenance, dict):
        return f"missing run provenance: {rid}", None
    if any(provenance.get(field) != job.get(field) for field in ("case_id", "repetition", "problem_id")):
        return f"run provenance differs from planned job: {rid}", None
    identity = provenance.get("experiment_identity")
    if not isinstance(identity, str) or not identity.strip():
        return f"missing run provenance: {rid}", None
    return None, identity


def _finish(stats: dict) -> dict:
    stats["item_agreement"] = stats["items_equal"] / stats["items"] if stats["items"] else None
    stats["flag_rate"] = stats["flagged"] / stats["flag_denominator"] if stats["flag_denominator"] else None
    return stats


def challenge_report(results: dict, *, rubric_items: dict) -> dict:
    """``rubric_items`` maps problem_id -> rubric item ids (no labels); required.

    Fails closed: a scheduled problem without rubric ids raises ValueError.
    """
    if not isinstance(rubric_items, dict):
        raise ValueError("rubric_items is required")
    reasons = []
    jobs = results.get("expected_runs")
    if not isinstance(jobs, list) or not jobs:
        reasons.append("expected_runs must be an explicit nonempty job plan")
        jobs = []
    planned_counts = Counter(job["request_id"] for job in jobs)
    duplicate_planned_ids = {rid for rid, count in planned_counts.items() if count > 1}
    planned_ids = set(planned_counts)
    runs = defaultdict(list)
    for run in results.get("runs", []):
        runs[run.get("request_id")].append(run)
    for label, ids in (
        ("duplicate planned", duplicate_planned_ids),
        ("duplicate", {rid for rid, records in runs.items() if len(records) > 1}),
        ("missing", planned_ids - runs.keys()),
        ("unexpected", runs.keys() - planned_ids),
    ):
        reasons.extend(f"{label} request_id: {rid}" for rid in sorted(ids, key=str))
    parsed = []
    slot_counts = Counter()
    for job in jobs:
        source, _, variant = job["case_id"].rpartition(":")
        slot = (source, variant, job["repetition"])
        slot_counts[slot] += 1
        parsed.append((job, source, variant, slot))
    duplicate_slots = {slot for slot, count in slot_counts.items() if count > 1}
    for source, variant, rep in sorted(duplicate_slots, key=lambda slot: (str(slot[0]), str(slot[1]), str(slot[2]))):
        reasons.append(f"duplicate planned slot: {source}:{variant}:{rep}")
    cells = {}
    identities = set()
    provenance_reasons = []
    for job, source, variant, slot in parsed:
        ids = rubric_items.get(job.get("problem_id"))
        if not ids:
            raise ValueError(f"no rubric item ids for problem {job.get('problem_id')!r}")
        if slot in duplicate_slots:
            # Two planned jobs share a slot. Keep neither score.
            cells[slot] = {"outcome": "incomplete", "status": None, "scores": None}
            continue
        expected = set(ids)
        records = runs.get(job["request_id"], [])
        if job["request_id"] in duplicate_planned_ids or len(records) > 1:
            cell = {"outcome": "incomplete", "status": None, "scores": None}
        elif not records:
            cell = _cell(None, expected)
        else:
            issue, identity = _provenance_issue(job, records[0])
            if issue:
                provenance_reasons.append(issue)
                cell = {"outcome": "incomplete", "status": None, "scores": None}
            else:
                identities.add(identity)
                cell = _cell(records[0], expected)
        cells[slot] = cell
    reasons.extend(sorted(provenance_reasons))
    if len(identities) > 1:
        reasons.append("multiple experiment identities: " + ", ".join(sorted(identities)))
        for slot, cell in list(cells.items()):
            if cell["outcome"] == "ok":
                cells[slot] = {"outcome": "incomplete", "status": None, "scores": None}
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
            "comparable": not reasons, "reasons": reasons,
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
    parser.add_argument("--dataset", type=Path, required=True, help="challenge manifest (rubric item ids only)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    items = _rubric_items(args.dataset)
    report = challenge_report(json.loads(args.results.read_text(encoding="utf-8")), rubric_items=items)
    private_write_text(args.out, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["comparable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
