"""Label-free report for the robustness challenge set (characterization only).

Pairs every variant run with the original (O) run of the same source and
repetition. Needs no reference scores and reads none. Only complete, finite
item scores are compared; failed or partial runs are counted separately.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from harness_v2.storage.permissions import private_write_text

FLAGGED = "review_required"


def _scores(grading: dict) -> dict | None:
    items = grading.get("items") if isinstance(grading, dict) else None
    if not isinstance(items, list) or not items:
        return None
    out = {}
    for item in items:
        score = item.get("score") if isinstance(item, dict) else None
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            return None
        out[item.get("item_id")] = float(score)
    return out


def challenge_report(results: dict) -> dict:
    by_request = {row["request_id"]: row for row in results.get("expected_runs", [])}
    cells = {}
    for run in results.get("runs", []):
        expected = by_request.get(run.get("request_id"))
        if expected is None:
            continue
        source, _, variant = expected["case_id"].rpartition(":")
        grading = (run.get("outputs") or {}).get("grading") or {}
        cells[(source, variant, expected["repetition"])] = {
            "status": grading.get("status"), "scores": _scores(grading)}
    per_variant = defaultdict(lambda: {"pairs": 0, "incomplete_pairs": 0, "items": 0,
                                       "items_equal": 0, "runs_all_items_equal": 0,
                                       "newly_flagged_relative_to_O": 0, "flagged": 0,
                                       "unflagged_relative_to_O": 0})
    sources = sorted({key[0] for key in cells})
    for (source, variant, rep), cell in sorted(cells.items()):
        if variant == "O":
            continue
        stats = per_variant[variant]
        base = cells.get((source, "O", rep))
        stats["flagged"] += cell["status"] == FLAGGED
        if base is None or cell["scores"] is None or base["scores"] is None \
                or set(cell["scores"]) != set(base["scores"]):
            stats["incomplete_pairs"] += 1
            continue
        stats["pairs"] += 1
        equal = [cell["scores"][k] == base["scores"][k] for k in base["scores"]]
        stats["items"] += len(equal)
        stats["items_equal"] += sum(equal)
        stats["runs_all_items_equal"] += all(equal)
        stats["newly_flagged_relative_to_O"] += cell["status"] == FLAGGED and base["status"] != FLAGGED
        stats["unflagged_relative_to_O"] += cell["status"] != FLAGGED and base["status"] == FLAGGED
    original_flagged = sum(c["status"] == FLAGGED for k, c in cells.items() if k[1] == "O")
    for stats in per_variant.values():
        stats["item_agreement"] = stats["items_equal"] / stats["items"] if stats["items"] else None
    return {"kind": "label-free robustness characterization (development data, not generalization)",
            "transcription_sources": sources, "original_runs": sum(k[1] == "O" for k in cells),
            "original_flagged": original_flagged, "variants": dict(sorted(per_variant.items()))}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = challenge_report(json.loads(args.results.read_text(encoding="utf-8")))
    private_write_text(args.out, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
