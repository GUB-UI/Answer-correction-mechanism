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

    def rec(pid):
        return h.state()["proposals"].get(pid)

    # Phase 4 prompt cycle: comparison recomputed from both results.json files.
    p4 = "phase4-prompt-ambiguity-aware-v1"
    base_res, cand_res = runs / "phase3/baseline_final/results.json", runs / "phase4/candidate/results.json"
    if rec(p4) is None:
        b = json.loads(base_res.read_text())["commit"]
        c = json.loads(cand_res.read_text())["commit"]
        h.propose(ImprovementProposal(p4, b, "", "Garbled OCR passages were not surfaced to the teacher",
                                      ["prompt"], "Score-affecting ambiguity cited with uncertainty >= 0.5",
                                      {}, [], [], "pending"), source={"phase": 4})
        h.attach_candidate(p4, c)
    if not rec(p4)["evaluations"]:
        h.record_evaluation(p4, baseline_results=base_res, candidate_results=cand_res, ocr_preprocess="none")
    # decision intentionally left pending: adoption awaits the owner's review

    # Phase 5 OCR preprocess cycle: skeleton-CER gate evidence.
    p5 = "phase5-ocr-preprocess-gray-autocontrast-x1.5"
    gate = runs / "phase5/a1_gate_v3.json"
    if rec(p5) is None:
        h.propose(ImprovementProposal(p5, _full(h, "53adc0b"), "", "OCR misreads remain after the Phase 4 prompt",
                                      ["recognition"], "Generic grayscale+autocontrast+x1.5 lowers OCR error",
                                      {}, [], [], "pending"), source={"phase": 5})
        h.attach_candidate(p5, _full(h, "27cb2af"))
    if not rec(p5)["evaluations"]:
        rows = [json.loads(x) for x in
                (runs / "phase5/ocr_gray_autocontrast_x1.5/stats.jsonl").read_text().splitlines() if x.strip()]
        snapshot = {"baseline_commit": _full(h, "53adc0b"), "candidate_commit": _full(h, "27cb2af"),
                    "config_sha256": sha256(canonical({"preprocess": rows[0]["preprocess"],
                                                       "model": rows[0]["model"], "revision": rows[0]["revision"]})),
                    "dataset_manifest_sha256": sha256(canonical(sorted(r["original_image_sha256"] for r in rows))),
                    "prompt_version": "n/a (OCR only)", "ocr_preprocess": rows[0]["preprocess"]["name"],
                    "evaluation_digest": hashlib.sha256(gate.read_bytes()).hexdigest()}
        g = json.loads(gate.read_text())["gate_a1"]
        h.record_external_evaluation(p5, snapshot=snapshot, verdict="regressed" if g["worse_images"] else "not_improved",
                                     evidence_path=gate, regressions=[{"image": i} for i in g["worse_images"]])
    if rec(p5)["decision"] == "pending":
        g = json.loads(gate.read_text())["gate_a1"]
        h.decide(p5, "rejected", reason="A1 gate failed (worse images: " + ", ".join(g["worse_images"]) + ")")
    print(f"ok: {h.verify()} events")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
