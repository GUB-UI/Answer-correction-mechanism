"""CLI for the improvement history (design §8 Phase 5).

Subcommands: status, verify, approve, override, rollback, check-merge.
`approve` is the ONLY place an approval is written; it requires the approver
and a verbatim quote of the owner's instruction. `check-merge` is advisory and
read-only (gh pr view); real merge enforcement is GitHub branch protection.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness_v2.improvement.history import History, promotion_blockers  # noqa: E402


def pr_head(repository: str, number: int) -> dict:
    out = subprocess.run(["gh", "pr", "view", str(number), "--repo", repository, "--json",
                          "headRefOid,baseRefName,number"], check=True, capture_output=True, text=True)
    return json.loads(out.stdout)


def main(argv=None, *, pr_lookup=pr_head) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--repo", type=Path, default=Path("."))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status"); sub.add_parser("verify")
    a = sub.add_parser("approve")
    for k in ("proposal", "approver", "repository", "base", "quote"):
        a.add_argument(f"--{k}", required=True)
    a.add_argument("--pr", type=int, required=True)
    o = sub.add_parser("override")
    for k in ("proposal", "approver", "reason"):
        o.add_argument(f"--{k}", required=True)
    r = sub.add_parser("rollback")
    r.add_argument("--to", required=True); r.add_argument("--reason", required=True)
    c = sub.add_parser("check-merge")
    c.add_argument("--proposal", required=True); c.add_argument("--repository", required=True)
    c.add_argument("--pr", type=int, required=True)
    args = ap.parse_args(argv)
    h = History(args.root, repo=args.repo)
    if args.cmd == "verify":
        print(f"ok: {h.verify()} events")
    elif args.cmd == "status":
        st = h.state()
        print(json.dumps({"active": st["active"], "versions": sorted(st["versions"]),
                          "proposals": {k: {"decision": v["decision"], "flags": v["flags"],
                                            "evaluations": [e["verdict"] for e in v["evaluations"]],
                                            "approvals": len(v["approvals"]), "promoted": v["promoted"]}
                                        for k, v in st["proposals"].items()}}, indent=1))
    elif args.cmd == "approve":
        rec = h.state()["proposals"][args.proposal]
        h.approve(args.proposal, approver=args.approver, repository=args.repository, base_branch=args.base,
                  pr_number=args.pr, snapshot_id_=rec["evaluations"][-1]["snapshot_id"], quote=args.quote)
    elif args.cmd == "override":
        h.override(args.proposal, approver=args.approver, reason=args.reason)
    elif args.cmd == "rollback":
        h.rollback(args.to, reason=args.reason)
    elif args.cmd == "check-merge":
        pr = pr_lookup(args.repository, args.pr)
        problems = promotion_blockers(h.state(), args.proposal, repository=args.repository,
                                      base_branch=pr["baseRefName"], pr_number=args.pr,
                                      head_commit=pr["headRefOid"])
        print(json.dumps({"mergeable_per_history": not problems, "blockers": problems}, ensure_ascii=False))
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
