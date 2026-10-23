"""Append-only improvement history (design §8, Phase 5).

Scope of guarantees (deliberately limited):
- Tamper-*evidence* against accidental corruption or edits: every event is
  hash-chained and ``head.json`` records the last sequence and hash. A writer
  with access to the directory can still rewrite both; this is not a security
  boundary.
- Approval and promotion checks guard against accidental or automated
  promotion. The harness never records an approval by itself; approvals come
  only from the CLI with an explicit approver and a quote of the user's
  instruction. Real merge enforcement belongs to GitHub branch protection.

Nothing is ever rewritten or deleted: proposals that did not improve are kept,
versions stay registered, and rollback only appends an event.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import fcntl
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Iterator

from harness_v2.core.contracts import ImprovementProposal
from harness_v2.storage.permissions import private_directory, private_open

DECISIONS = {"accepted", "rejected", "inconclusive"}
VERDICTS = {"improved", "not_improved", "regressed", "incomparable"}
EVENT_TYPES = {"version_registered", "proposed", "candidate_attached", "evaluated", "decided",
               "teacher_review", "override", "approval", "promoted", "rolled_back"}
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class HistoryError(RuntimeError):
    pass


class HistoryCorrupt(HistoryError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def snapshot_id(snapshot: dict) -> str:
    required = {"baseline_commit", "candidate_commit", "config_sha256", "dataset_manifest_sha256",
                "prompt_version", "ocr_preprocess", "evaluation_digest"}
    if set(snapshot) != required:
        raise HistoryError(f"snapshot fields must be exactly {sorted(required)}")
    for key in ("baseline_commit", "candidate_commit"):
        if not _SHA40.match(str(snapshot[key])):
            raise HistoryError(f"{key} must be a full 40-hex commit")
    for key in ("config_sha256", "dataset_manifest_sha256", "evaluation_digest"):
        if not _SHA256.match(str(snapshot[key])):
            raise HistoryError(f"{key} must be sha256 hex")
    return sha256(canonical(snapshot))


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class History:
    def __init__(self, root: Path, *, repo: Path | None = None) -> None:
        self.root = Path(root)
        self.repo = Path(repo) if repo else None
        private_directory(self.root)
        private_directory(self.root / "objects")
        self.log = self.root / "history.jsonl"
        self.head = self.root / "head.json"

    # ---------- storage ----------
    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        with private_open(self.root / ".lock", "a") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _read_raw(self) -> list[dict]:
        if not self.log.exists():
            return []
        data = self.log.read_bytes()
        if data and not data.endswith(b"\n"):
            raise HistoryCorrupt("torn final record (no trailing newline)")
        events, prev = [], ""
        for number, line in enumerate(data.splitlines(), 1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise HistoryCorrupt(f"invalid record {number}") from exc
            body = {k: v for k, v in event.items() if k != "sha256"}
            if event.get("seq") != number or event.get("prev_sha256") != prev \
                    or sha256(canonical(body)) != event.get("sha256"):
                raise HistoryCorrupt(f"hash chain broken at record {number}")
            prev = event["sha256"]
            events.append(event)
        return events

    def _read_head(self) -> dict | None:
        if not self.head.exists():
            return None
        return json.loads(self.head.read_text(encoding="utf-8"))

    def _write_head(self, seq: int, digest: str) -> None:
        tmp = self.root / "head.json.tmp"
        with private_open(tmp, "w") as handle:
            handle.write(json.dumps({"seq": seq, "sha256": digest}) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self.head)
        _fsync_dir(self.root)

    def _checked_events(self) -> list[dict]:
        events = self._read_raw()
        head = self._read_head()
        if head is None:
            if events:
                raise HistoryCorrupt("head.json missing for non-empty history")
            return events
        last = events[-1] if events else None
        last_seq, last_sha = (last["seq"], last["sha256"]) if last else (0, "")
        if (head.get("seq"), head.get("sha256")) == (last_seq, last_sha):
            return events
        if len(events) >= 2 and head.get("seq") == last_seq - 1 \
                and head.get("sha256") == events[-2]["sha256"]:
            self._write_head(last_seq, last_sha)  # crash between log and head
            return events
        raise HistoryCorrupt("head.json does not match history (truncated or replaced)")

    def verify(self) -> int:
        with self._locked():
            events = self._checked_events()
            _replay(events)
            return len(events)

    def events(self) -> list[dict]:
        with self._locked():
            return self._checked_events()

    def repair_drop_torn_tail(self) -> None:
        with self._locked():
            data = self.log.read_bytes()
            if data.endswith(b"\n"):
                raise HistoryError("no torn tail")
            keep = data[: data.rfind(b"\n") + 1]
            tmp = self.root / "history.jsonl.tmp"
            with private_open(tmp, "wb") as handle:
                handle.write(keep)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.log)
            _fsync_dir(self.root)
            self._checked_events()

    def _append(self, kind: str, proposal_id: str | None, payload: dict) -> dict:
        if kind not in EVENT_TYPES:
            raise HistoryError(f"unknown event type {kind}")
        with self._locked():
            events = self._checked_events()
            event = {"seq": len(events) + 1,
                     "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                     "type": kind, "proposal_id": proposal_id, "payload": payload,
                     "prev_sha256": events[-1]["sha256"] if events else ""}
            _replay(events + [event])  # validate transition before writing
            event["sha256"] = sha256(canonical(event))
            with private_open(self.log, "ab") as handle:
                handle.write(canonical(event) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._write_head(event["seq"], event["sha256"])
            return event

    def state(self) -> dict:
        return _replay(self.events())

    # ---------- objects / versions ----------
    def _store_object(self, data: bytes) -> str:
        digest = sha256(data)
        target = self.root / "objects" / digest
        if not target.exists():
            with private_open(target, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        elif sha256(target.read_bytes()) != digest:
            raise HistoryCorrupt(f"object {digest} corrupted")
        return digest

    def _object(self, digest: str) -> bytes:
        data = (self.root / "objects" / digest).read_bytes()
        if sha256(data) != digest:
            raise HistoryCorrupt(f"object {digest} corrupted")
        return data

    def _git(self, *args: str) -> str:
        if self.repo is None:
            raise HistoryError("repository path required")
        return subprocess.run(["git", "-C", str(self.repo), *args], check=True,
                              capture_output=True, text=True).stdout

    def _check_commit(self, commit: str) -> None:
        if not _SHA40.match(commit):
            raise HistoryError("commit must be full 40-hex")
        if self.repo is not None:
            try:
                self._git("cat-file", "-e", f"{commit}^{{commit}}")
            except subprocess.CalledProcessError as exc:
                raise HistoryError(f"unknown commit {commit}") from exc

    def _store_tree(self, directory: Path) -> dict:
        tree = {}
        for path in sorted(Path(directory).rglob("*")):
            if path.is_file():
                tree[path.relative_to(directory).as_posix()] = self._store_object(path.read_bytes())
        return tree

    def register_version(self, version_id: str, *, commit: str, config_path: Path,
                         dataset_manifest: Path, ocr_preprocess: str, baseline: bool = False) -> dict:
        """Register a runnable version: commit + exact config + complete dataset tree."""
        from harness_v2.evaluation.dataset import load_dataset
        from harness_v2.run import load_config
        self._check_commit(commit)
        config = load_config(Path(config_path))
        dataset = load_dataset(Path(dataset_manifest))
        manifest = Path(dataset_manifest)
        return self._append("version_registered", None, {
            "version_id": version_id, "commit": commit, "baseline": baseline,
            "config_object": self._store_object(Path(config_path).read_bytes()),
            "config_sha256": sha256(canonical(config)),
            "dataset_manifest_sha256": sha256(canonical(dataset.manifest)),
            "dataset_tree": self._store_tree(manifest.parent), "manifest_name": manifest.name,
            "prompt_version": config["prompt_version"], "ocr_preprocess": ocr_preprocess})

    # ---------- proposals ----------
    def propose(self, proposal: ImprovementProposal, *, change_id: str | None = None,
                source: dict | None = None) -> dict:
        record = proposal.to_dict()
        return self._append("proposed", proposal.proposal_id, {
            "proposal": record, "content_sha256": sha256(canonical(record)),
            "change_id": change_id, "source": source or {}})

    def propose_change(self, proposal: ImprovementProposal) -> dict:
        """Propose with an identity derived from the actual git change; duplicates are flagged."""
        from harness_v2.improvement.duplicates import change_identity, find_duplicates
        self._check_commit(proposal.baseline_commit)
        self._check_commit(proposal.candidate_commit)
        identity = change_identity(self.repo, proposal.baseline_commit, proposal.candidate_commit)
        dups = find_duplicates(self.state(), identity, proposal.baseline_commit)
        return self.propose(proposal, change_id=identity["change_id"],
                            source={"identity": identity, "duplicates": dups})

    def import_draft(self, path: Path) -> dict | None:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        proposal = ImprovementProposal(**record)
        digest = sha256(canonical(proposal.to_dict()))
        prior = self.state()["proposals"].get(proposal.proposal_id)
        if prior is not None:
            if prior["content_sha256"] == digest:
                return None  # idempotent
            raise HistoryError("different content already registered under this proposal_id")
        return self.propose(proposal, source={"draft_path": str(path), "sha256": digest})

    def attach_candidate(self, proposal_id: str, candidate_commit: str, *,
                         change_id: str | None = None) -> dict:
        self._check_commit(candidate_commit)
        return self._append("candidate_attached", proposal_id,
                            {"candidate_commit": candidate_commit, "change_id": change_id})

    def record_evaluation(self, proposal_id: str, *, baseline_experiment: Path,
                          candidate_experiment: Path, comparison: Path, ocr_preprocess: str,
                          regressions: list | None = None) -> dict:
        """Derive the snapshot from run evidence; the verdict comes from the comparison file."""
        base = json.loads(Path(baseline_experiment).read_bytes())
        cand = json.loads(Path(candidate_experiment).read_bytes())
        comp_bytes = Path(comparison).read_bytes()
        comp = json.loads(comp_bytes)
        if canonical(base["dataset"]) != canonical(cand["dataset"]):
            raise HistoryError("baseline and candidate ran on different datasets")
        if comp.get("comparable") is not True:
            raise HistoryError("comparison is not comparable")
        for exp in (base, cand):
            if exp.get("commit") and len(exp["commit"]) == 40:
                self._check_commit(exp["commit"])
            else:
                raise HistoryError("experiment commit must be a full 40-hex sha")
        snapshot = {"baseline_commit": base["commit"], "candidate_commit": cand["commit"],
                    "config_sha256": sha256(canonical(cand["config"])),
                    "dataset_manifest_sha256": sha256(canonical(cand["dataset"])),
                    "prompt_version": cand["config"]["prompt_version"], "ocr_preprocess": ocr_preprocess,
                    "evaluation_digest": sha256(comp_bytes)}
        evidence = {"baseline_identity": base.get("identity"), "candidate_identity": cand.get("identity"),
                    "comparison_object": self._store_object(comp_bytes),
                    "baseline_experiment_sha256": sha256(Path(baseline_experiment).read_bytes()),
                    "candidate_experiment_sha256": sha256(Path(candidate_experiment).read_bytes())}
        verdict = {"improved": "improved", "regressed": "regressed"}.get(comp.get("verdict"), "not_improved")
        return self._append("evaluated", proposal_id, {
            "snapshot": snapshot, "snapshot_id": snapshot_id(snapshot), "verdict": verdict,
            "evaluation_results": {k: comp.get(k) for k in ("verdict", "reasons", "deltas")},
            "regressions": regressions or [], "evidence": evidence})

    def record_external_evaluation(self, proposal_id: str, *, snapshot: dict, verdict: str,
                                   evidence_path: Path, regressions: list | None = None) -> dict:
        """Evaluations that are not runner comparisons (e.g. the OCR skeleton-CER gate)."""
        data = Path(evidence_path).read_bytes()
        if snapshot.get("evaluation_digest") != sha256(data):
            raise HistoryError("evaluation_digest must be the sha256 of the evidence file")
        return self._append("evaluated", proposal_id, {
            "snapshot": snapshot, "snapshot_id": snapshot_id(snapshot), "verdict": verdict,
            "evaluation_results": json.loads(data), "regressions": regressions or [],
            "evidence": {"object": self._store_object(data)}})

    def decide(self, proposal_id: str, decision: str, *, reason: str) -> dict:
        return self._append("decided", proposal_id, {"decision": decision, "reason": reason})

    def override(self, proposal_id: str, *, approver: str, reason: str) -> dict:
        return self._append("override", proposal_id, {"approver": approver, "reason": reason})

    def approve(self, proposal_id: str, *, approver: str, repository: str, base_branch: str,
                pr_number: int, snapshot_id_: str, quote: str) -> dict:
        """Record a human approval. Only the CLI calls this, never the runner."""
        return self._append("approval", proposal_id, {
            "approver": approver, "repository": repository, "base_branch": base_branch,
            "pr_number": pr_number, "snapshot_id": snapshot_id_,
            "source": "user-instruction", "quote": quote})

    def promote(self, proposal_id: str, version_id: str) -> dict:
        return self._append("promoted", proposal_id, {"version_id": version_id})

    def rollback(self, to_version_id: str, *, reason: str) -> dict:
        return self._append("rolled_back", None, {"version_id": to_version_id, "reason": reason})

    def active_version(self) -> dict | None:
        state = self.state()
        return state["versions"].get(state["active"]) if state["active"] else None

    def materialize(self, version_id: str, dest: Path) -> dict:
        """Check out a version's commit into a new worktree plus its config/manifest."""
        version = self.state()["versions"].get(version_id)
        if version is None:
            raise HistoryError(f"unknown version {version_id}")
        dest = Path(dest)
        if dest.exists():
            raise HistoryError("destination exists")
        self._git("worktree", "add", "--detach", str(dest), version["commit"])
        if self._git("-C", str(dest), "rev-parse", "HEAD").strip() != version["commit"]:
            raise HistoryError("worktree commit mismatch")
        from harness_v2.evaluation.dataset import load_dataset
        inputs = private_directory(dest.parent / f"{dest.name}.inputs")
        out = {"config": str(inputs / "config.json")}
        with private_open(inputs / "config.json", "wb") as handle:
            handle.write(self._object(version["config_object"]))
        for rel, digest in version["dataset_tree"].items():
            target = inputs / "dataset" / rel
            private_directory(target.parent)
            with private_open(target, "wb") as handle:
                handle.write(self._object(digest))
        manifest = inputs / "dataset" / version["manifest_name"]
        if sha256(canonical(load_dataset(manifest).manifest)) != version["dataset_manifest_sha256"]:
            raise HistoryCorrupt("restored dataset differs from the registered version")
        out["dataset_manifest"] = str(manifest)
        out["worktree"] = str(dest)
        return out


def promotion_blockers(state: dict, proposal_id: str, *, repository: str, base_branch: str,
                       pr_number: int, head_commit: str) -> list[str]:
    """Shared eligibility predicate for promote() and the merge check (advisory)."""
    rec = state["proposals"].get(proposal_id)
    if rec is None:
        return ["unknown proposal"]
    problems = []
    if rec["decision"] != "accepted":
        problems.append(f"decision is {rec['decision']}, not accepted")
    if not rec["evaluations"]:
        return problems + ["no evaluation"]
    latest = rec["evaluations"][-1]
    if latest["verdict"] != "improved":
        problems.append("latest evaluation is not improved")
    if latest["snapshot"]["candidate_commit"] != head_commit:
        problems.append("PR head differs from the evaluated candidate commit")
    if not any(a["snapshot_id"] == latest["snapshot_id"] and a["repository"] == repository
               and a["base_branch"] == base_branch and a["pr_number"] == pr_number
               for a in rec["approvals"]):
        problems.append("no approval of the latest snapshot for this repository/base/PR")
    if rec["flags"] and not rec["overrides"]:
        problems.append(f"flags {rec['flags']} need an audited override")
    return problems


def _replay(events: list[dict]) -> dict:
    """Fold events into state, validating every transition (writes and replay)."""
    versions: dict[str, dict] = {}
    proposals: dict[str, dict] = {}
    active = None
    promoted_versions: set[str] = set()
    for event in events:
        kind, pid, p = event["type"], event.get("proposal_id"), event["payload"]
        if kind == "version_registered":
            if p["version_id"] in versions:
                raise HistoryError("duplicate version_id")
            if p.get("baseline"):
                if versions:
                    raise HistoryError("the baseline must be the first registered version (immutable)")
                active = p["version_id"]
            elif not versions:
                raise HistoryError("register the trusted baseline first")
            versions[p["version_id"]] = dict(p)
            continue
        if kind == "rolled_back":
            target = p["version_id"]
            if target not in versions:
                raise HistoryError("rollback to unknown version")
            if not (versions[target].get("baseline") or target in promoted_versions):
                raise HistoryError("rollback only to the baseline or a previously promoted version")
            active = target
            continue
        if kind == "proposed":
            if pid in proposals:
                raise HistoryError("duplicate proposal_id")
            change_id = p.get("change_id")
            base = p["proposal"].get("baseline_commit")
            prior = [q for q in proposals.values() if change_id and q["change_id"] == change_id]
            proposals[pid] = {"proposal": p["proposal"], "content_sha256": p["content_sha256"],
                              "change_id": change_id, "candidate_commit": p["proposal"].get("candidate_commit"),
                              "evaluations": [], "decision": "pending", "reviews": [],
                              "approvals": [], "overrides": [], "promoted": None, "flags": [],
                              "identity": (p.get("source") or {}).get("identity")}
            for q in prior:  # exact change seen before: any reuse needs an audited override
                proposals[pid]["flags"].append(
                    "retry_on_new_baseline" if q["proposal"].get("baseline_commit") != base
                    else "duplicate_of_" + q["decision"])
            continue
        if pid not in proposals:
            raise HistoryError(f"{kind} for unknown proposal")
        rec = proposals[pid]
        if rec["decision"] == "rejected" and kind not in {"teacher_review"}:
            raise HistoryError("proposal was rejected; file a new proposal")
        if kind == "override":
            if not p.get("approver") or not p.get("reason"):
                raise HistoryError("override needs approver and reason")
            rec["overrides"].append(p)
        elif kind == "candidate_attached":
            if rec["evaluations"]:
                raise HistoryError("candidate cannot change after evaluation; file a new proposal")
            rec["candidate_commit"] = p["candidate_commit"]
            if p.get("change_id"):
                rec["change_id"] = p["change_id"]
        elif kind == "evaluated":
            if p["verdict"] not in VERDICTS:
                raise HistoryError("unknown verdict")
            if rec["candidate_commit"] and p["snapshot"]["candidate_commit"] != rec["candidate_commit"]:
                raise HistoryError("evaluation snapshot is for a different candidate")
            rec["evaluations"].append(p)
        elif kind == "decided":
            if p["decision"] not in DECISIONS:
                raise HistoryError("unknown decision")
            if p["decision"] == "accepted":
                if not rec["evaluations"] or rec["evaluations"][-1]["verdict"] != "improved":
                    raise HistoryError("accepted requires the latest evaluation to be improved")
            rec["decision"] = p["decision"]
            rec["decision_snapshot"] = rec["evaluations"][-1]["snapshot_id"] if rec["evaluations"] else None
        elif kind == "teacher_review":
            rec["reviews"].append(p)
        elif kind == "approval":
            if rec["decision"] != "accepted":
                raise HistoryError("approval requires an accepted decision")
            if not p.get("approver") or not p.get("quote") or p.get("source") != "user-instruction":
                raise HistoryError("approval needs approver, quote and source=user-instruction")
            if not isinstance(p.get("pr_number"), int) or not p.get("repository") or not p.get("base_branch"):
                raise HistoryError("approval must name repository, base branch and PR number")
            if p["snapshot_id"] != rec["evaluations"][-1]["snapshot_id"]:
                raise HistoryError("approval snapshot is not the latest evaluated snapshot")
            rec["approvals"].append(p)
        elif kind == "promoted":
            latest = rec["evaluations"][-1]["snapshot_id"] if rec["evaluations"] else None
            if rec["decision"] != "accepted" or not any(a["snapshot_id"] == latest for a in rec["approvals"]):
                raise HistoryError("promotion requires an accepted decision and an approval of the latest snapshot")
            if rec["flags"] and not rec["overrides"]:
                raise HistoryError("duplicate/retry of an earlier proposal needs an audited override")
            version = versions.get(p["version_id"])
            snap = rec["evaluations"][-1]["snapshot"]
            if version is None or (version["commit"], version["config_sha256"],
                                   version["dataset_manifest_sha256"], version["ocr_preprocess"]) != (
                    snap["candidate_commit"], snap["config_sha256"], snap["dataset_manifest_sha256"],
                    snap["ocr_preprocess"]):
                raise HistoryError("promoted version must equal the approved snapshot exactly")
            rec["promoted"] = p["version_id"]
            promoted_versions.add(p["version_id"])
            active = p["version_id"]
    return {"versions": versions, "proposals": proposals, "active": active}
