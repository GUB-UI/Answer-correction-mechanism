"""Content-derived identity of a proposed change (design §8 (b)).

``change_id`` is byte-sensitive: sha256 over the sorted (path, candidate blob id)
pairs of every path changed between baseline and candidate, so a whitespace edit
in a prompt or Python file is a different change. ``patch_id`` (git
``patch-id --stable``) is only a similarity hint: it ignores whitespace and so
also matches the same edit replayed on another baseline.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


def _git(repo: Path, *args: str, stdin: str | None = None) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                          text=True, input=stdin).stdout


def change_identity(repo: Path, baseline: str, candidate: str) -> dict:
    names = _git(repo, "diff", "--name-only", "--no-renames", f"{baseline}..{candidate}").split("\n")
    pairs = []
    for path in sorted(n for n in names if n):
        try:
            blob = _git(repo, "rev-parse", f"{candidate}:{path}").strip()
        except subprocess.CalledProcessError:
            blob = "deleted"
        pairs.append([path, blob])
    diff = _git(repo, "diff", "--no-renames", f"{baseline}..{candidate}")
    patch = _git(repo, "patch-id", "--stable", stdin=diff).split()
    return {"change_id": hashlib.sha256(json.dumps(pairs).encode()).hexdigest(),
            "patch_id": patch[0] if patch else "", "paths": [p for p, _ in pairs], "pairs": pairs}


def overlaps(a: dict | None, b: dict | None) -> bool:
    """True if the changes share any exact (path, blob) edit.

    Catches a rejected component edit resubmitted with other, differing edits, while
    staying byte-sensitive (a whitespace change yields a different blob id).
    """
    if not a or not b or not a.get("pairs") or not b.get("pairs"):
        return False
    pa, pb = {tuple(x) for x in a["pairs"]}, {tuple(x) for x in b["pairs"]}
    return bool(pa & pb)


def find_duplicates(state: dict, identity: dict, baseline: str) -> list[dict]:
    """Earlier proposals with the same change (exact) or the same patch (similar)."""
    hits = []
    for pid, rec in state["proposals"].items():
        other = rec.get("identity") or {}
        kind = ("exact" if rec.get("change_id") == identity["change_id"]
                else "contains" if overlaps(identity, other)
                else "similar" if identity.get("patch_id") and other.get("patch_id") == identity["patch_id"]
                else None)
        if kind:
            hits.append({"proposal_id": pid, "kind": kind, "decision": rec["decision"],
                         "same_baseline": rec["proposal"].get("baseline_commit") == baseline})
    return hits
