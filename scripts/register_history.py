"""Idempotently register the Phase 4 prompt cycle and Phase 5 OCR-preprocess cycle.

Reads run evidence from --runs (outside git); every record cites source sha256.
Phase 4: evaluated improved; decision left pending (adoption awaits human review).
Phase 5 OCR: evaluated by the skeleton-CER gate; rejected (gate failed).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness_v2.core.contracts import ImprovementProposal  # noqa: E402
from harness_v2.improvement.history import History, canonical, sha256  # noqa: E402


def _full(h: History, short: str) -> str:
    return h._git("rev-parse", f"{short}^{{commit}}").strip()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--repo", type=Path, default=Path("."))
    ap.add_argument("--runs", required=True, type=Path)
    args = ap.parse_args(argv)
    h, runs = History(args.root, repo=args.repo), args.runs
    known = h.state()["proposals"]
    p4 = "phase4-prompt-ambiguity-aware-v1"
    if p4 not in known:
        b = json.loads((runs / "phase3/baseline_final/experiment.json").read_text())["commit"]
        c = json.loads((runs / "phase4/candidate/experiment.json").read_text())["commit"]
        h.propose(ImprovementProposal(p4, b, c, "Garbled OCR passages were not surfaced to the teacher",
                                      ["prompt"], "Score-affecting ambiguity cited with uncertainty >= 0.5",
                                      {}, [], [], "pending"), change_id=None,
                  source={"phase": 4})
        h.record_evaluation(p4, baseline_experiment=runs / "phase3/baseline_final/experiment.json",
                            candidate_experiment=runs / "phase4/candidate/experiment.json",
                            comparison=runs / "phase4/comparison_vs_phase3.json", ocr_preprocess="none")
    p5 = "phase5-ocr-preprocess-gray-autocontrast-x1.5"
    if p5 not in known:
        gate = runs / "phase5/a1_gate_v3.json"
        stats = (runs / "phase5/ocr_gray_autocontrast_x1.5/stats.jsonl").read_text().splitlines()
        rows = [json.loads(x) for x in stats if x.strip()]
        base, cand = _full(h, "53adc0b"), _full(h, "27cb2af")
        h.propose(ImprovementProposal(p5, base, cand, "OCR misreads remain after the Phase 4 prompt",
                                      ["recognition"], "Generic grayscale+autocontrast+x1.5 lowers OCR error",
                                      {}, [], [], "pending"), source={"phase": 5})
        snapshot = {"baseline_commit": base, "candidate_commit": cand,
                    "config_sha256": sha256(canonical({"preprocess": rows[0]["preprocess"],
                                                       "model": rows[0]["model"], "revision": rows[0]["revision"]})),
                    "dataset_manifest_sha256": sha256(canonical(sorted(r["original_image_sha256"] for r in rows))),
                    "prompt_version": "n/a (OCR only)", "ocr_preprocess": rows[0]["preprocess"]["name"],
                    "evaluation_digest": hashlib.sha256(gate.read_bytes()).hexdigest()}
        g = json.loads(gate.read_text())["gate_a1"]
        h.record_external_evaluation(p5, snapshot=snapshot, verdict="regressed" if g["worse_images"] else "not_improved",
                                     evidence_path=gate, regressions=[{"image": i} for i in g["worse_images"]])
        h.decide(p5, "rejected", reason="A1 gate failed (worse images: " + ", ".join(g["worse_images"]) + ")")
    print(f"ok: {h.verify()} events")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
