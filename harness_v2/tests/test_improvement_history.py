import json
import os
import subprocess

import pytest

from harness_v2.core.contracts import ImprovementProposal
from harness_v2.improvement.history import (History, HistoryCorrupt, HistoryError, promotion_blockers,
                                            sha256, snapshot_id)
from harness_v2.tests.test_p3_dataset import make_dataset

REPO_ARGS = dict(repository="o/r", base_branch="main", pr_number=1)


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q")
    _git(r, "config", "user.email", "t@example.invalid")
    _git(r, "config", "user.name", "t")
    commits = []
    for i in range(3):
        (r / "prompt.txt").write_text(f"prompt v{i}\n")
        _git(r, "add", ".")
        _git(r, "commit", "-qm", f"v{i}")
        commits.append(_git(r, "rev-parse", "HEAD"))
    return r, commits


@pytest.fixture
def inputs(tmp_path):
    manifest = make_dataset(tmp_path)
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"name": "x", "provider": "fixture", "model": "fixture",
                               "prompt_version": "harness_v2_default", "rubric_version": "r1",
                               "generation_settings": {}}))
    return cfg, manifest


def _proposal(pid, base, cand=""):
    return ImprovementProposal(pid, base, cand, "cause", ["prompt"], "fewer misses", {}, [], [], "pending")


def _experiments(tmp_path, h, base, cand, inputs, verdict="improved", tag="e"):
    from harness_v2.evaluation.dataset import load_dataset
    from harness_v2.run import load_config
    cfg, manifest = inputs
    config, ds = load_config(cfg), load_dataset(manifest).manifest
    paths = []
    for name, commit in (("base", base), ("cand", cand)):
        p = tmp_path / f"{tag}_{name}.json"
        p.write_text(json.dumps({"identity": name, "config": config, "dataset": ds, "commit": commit}))
        paths.append(p)
    comp = tmp_path / f"{tag}_comp.json"
    comp.write_text(json.dumps({"comparable": True, "verdict": verdict, "reasons": [], "deltas": {"tag": tag}}))
    return dict(baseline_experiment=paths[0], candidate_experiment=paths[1], comparison=comp, ocr_preprocess="none")


def _cycle(tmp_path, h, inputs, pid, base, cand, version, verdict="improved"):
    h.propose(_proposal(pid, base))
    h.attach_candidate(pid, cand)
    ev = h.record_evaluation(pid, **_experiments(tmp_path, h, base, cand, inputs, verdict, tag=pid))
    if verdict != "improved":
        h.decide(pid, "rejected", reason="no gain")
        return ev
    h.decide(pid, "accepted", reason="gain")
    h.register_version(version, commit=cand, config_path=inputs[0], dataset_manifest=inputs[1], ocr_preprocess="none")
    h.approve(pid, approver="owner", snapshot_id_=ev["payload"]["snapshot_id"], quote="merge it", **REPO_ARGS)
    h.promote(pid, version)
    return ev


def _base(h, c0, inputs):
    h.register_version("v0", commit=c0, config_path=inputs[0], dataset_manifest=inputs[1],
                       ocr_preprocess="none", baseline=True)


def test_multi_cycle_rollback_keeps_everything(tmp_path, repo, inputs):
    from harness_v2.evaluation.dataset import load_dataset
    r, (c0, c1, c2) = repo
    h = History(tmp_path / "hist", repo=r)
    _base(h, c0, inputs)
    _cycle(tmp_path, h, inputs, "P1", c0, c1, "v1")
    _cycle(tmp_path, h, inputs, "P2", c1, c2, "v2", verdict="no_change")  # stored, not promoted
    assert h.state()["active"] == "v1"
    n_before = len(h.events())
    h.rollback("v0", reason="demo")
    st = h.state()
    assert st["active"] == "v0" and st["proposals"]["P2"]["decision"] == "rejected"
    assert st["proposals"]["P2"]["evaluations"][-1]["verdict"] == "not_improved"
    assert len(h.events()) == n_before + 1 and h.verify() == n_before + 1
    out = h.materialize("v0", tmp_path / "wt_v0")
    assert (tmp_path / "wt_v0" / "prompt.txt").read_text() == "prompt v0\n"
    assert open(out["config"], "rb").read() == inputs[0].read_bytes()
    assert load_dataset(out["dataset_manifest"]).manifest == load_dataset(inputs[1]).manifest
    assert oct(os.stat(out["config"]).st_mode & 0o777) == "0o600"
    h.rollback("v1", reason="forward again")
    assert h.state()["active"] == "v1"
    with pytest.raises(HistoryError):
        h.rollback("nope", reason="x")


def test_baseline_is_first_and_immutable(tmp_path, repo, inputs):
    r, (c0, c1, _) = repo
    h = History(tmp_path / "h", repo=r)
    with pytest.raises(HistoryError, match="baseline first"):
        h.register_version("vx", commit=c1, config_path=inputs[0], dataset_manifest=inputs[1], ocr_preprocess="none")
    _base(h, c0, inputs)
    with pytest.raises(HistoryError, match="immutable"):
        h.register_version("vb", commit=c1, config_path=inputs[0], dataset_manifest=inputs[1],
                           ocr_preprocess="none", baseline=True)
    with pytest.raises(HistoryError):
        h.register_version("v0", commit=c1, config_path=inputs[0], dataset_manifest=inputs[1], ocr_preprocess="none")
    h.register_version("vx", commit=c1, config_path=inputs[0], dataset_manifest=inputs[1], ocr_preprocess="none")
    with pytest.raises(HistoryError, match="previously promoted"):
        h.rollback("vx", reason="x")


def test_promotion_requires_approval_of_latest_snapshot(tmp_path, repo, inputs):
    r, (c0, c1, _) = repo
    h = History(tmp_path / "h", repo=r)
    _base(h, c0, inputs)
    h.propose(_proposal("P", c0))
    h.attach_candidate("P", c1)
    e1 = h.record_evaluation("P", **_experiments(tmp_path, h, c0, c1, inputs, tag="a"))
    s1 = e1["payload"]["snapshot_id"]
    h.register_version("v1", commit=c1, config_path=inputs[0], dataset_manifest=inputs[1], ocr_preprocess="none")
    with pytest.raises(HistoryError):
        h.promote("P", "v1")  # not accepted
    h.decide("P", "accepted", reason="ok")
    with pytest.raises(HistoryError):
        h.promote("P", "v1")  # no approval
    with pytest.raises(HistoryError):
        h.approve("P", approver="", snapshot_id_=s1, quote="q", **REPO_ARGS)
    with pytest.raises(HistoryError):
        h.approve("P", approver="o", snapshot_id_=s1, quote="q", repository="", base_branch="main", pr_number=1)
    h.approve("P", approver="owner", snapshot_id_=s1, quote="q", **REPO_ARGS)
    assert promotion_blockers(h.state(), "P", head_commit=c1, **REPO_ARGS) == []
    assert promotion_blockers(h.state(), "P", head_commit=c0, **REPO_ARGS)
    assert promotion_blockers(h.state(), "P", head_commit=c1, repository="o/r", base_branch="dev", pr_number=1)
    h.record_evaluation("P", **_experiments(tmp_path, h, c0, c1, inputs, tag="b"))
    assert promotion_blockers(h.state(), "P", head_commit=c1, **REPO_ARGS)  # stale approval
    with pytest.raises(HistoryError):
        h.promote("P", "v1")
    with pytest.raises(HistoryError):
        h.approve("P", approver="o", snapshot_id_=s1, quote="q", **REPO_ARGS)


def test_promoted_version_must_match_snapshot(tmp_path, repo, inputs):
    r, (c0, c1, _) = repo
    h = History(tmp_path / "h", repo=r)
    _base(h, c0, inputs)
    h.propose(_proposal("P", c0))
    ev = h.record_evaluation("P", **_experiments(tmp_path, h, c0, c1, inputs))
    h.decide("P", "accepted", reason="ok")
    h.approve("P", approver="o", snapshot_id_=ev["payload"]["snapshot_id"], quote="q", **REPO_ARGS)
    other = tmp_path / "other.json"
    cfg = json.loads(inputs[0].read_text())
    cfg["generation_settings"] = {"temperature": 1}
    other.write_text(json.dumps(cfg))
    h.register_version("vbad", commit=c1, config_path=other, dataset_manifest=inputs[1], ocr_preprocess="none")
    with pytest.raises(HistoryError, match="snapshot exactly"):
        h.promote("P", "vbad")


def test_evaluation_evidence_binding(tmp_path, repo, inputs):
    r, (c0, c1, _) = repo
    h = History(tmp_path / "h", repo=r)
    h.propose(_proposal("P", c0))
    kw = _experiments(tmp_path, h, c0, c1, inputs)
    cand = json.loads(kw["candidate_experiment"].read_text())
    cand["dataset"] = {"different": True}
    kw["candidate_experiment"].write_text(json.dumps(cand))
    with pytest.raises(HistoryError, match="different datasets"):
        h.record_evaluation("P", **kw)
    kw = _experiments(tmp_path, h, c0, "f" * 40, inputs, tag="z")
    with pytest.raises(HistoryError, match="unknown commit"):
        h.record_evaluation("P", **kw)


def test_accept_requires_improved_and_rejected_is_final(tmp_path, repo, inputs):
    r, (c0, c1, _) = repo
    h = History(tmp_path / "h", repo=r)
    h.propose(_proposal("P", c0))
    with pytest.raises(HistoryError):
        h.decide("P", "accepted", reason="x")
    h.record_evaluation("P", **_experiments(tmp_path, h, c0, c1, inputs, verdict="regressed"))
    with pytest.raises(HistoryError):
        h.decide("P", "accepted", reason="x")
    h.decide("P", "rejected", reason="regressed")
    with pytest.raises(HistoryError):
        h.decide("P", "accepted", reason="x")
    with pytest.raises(HistoryError):
        h.propose(_proposal("P", c0))  # unique ids


def test_corruption_and_crash_recovery(tmp_path, repo):
    r, (c0, c1, _) = repo
    root = tmp_path / "h"
    h = History(root, repo=r)
    h.propose(_proposal("A", c0))
    h.propose(_proposal("B", c0))
    assert oct(os.stat(root / "history.jsonl").st_mode & 0o777) == "0o600"
    assert oct(os.stat(root).st_mode & 0o777) == "0o700"
    # crash between log append and head update -> head advanced
    head = json.loads((root / "head.json").read_text())
    first = json.loads((root / "history.jsonl").read_text().splitlines()[0])
    (root / "head.json").write_text(json.dumps({"seq": 1, "sha256": first["sha256"]}))
    assert h.verify() == 2
    assert json.loads((root / "head.json").read_text()) == head
    # torn tail
    with open(root / "history.jsonl", "ab") as f:
        f.write(b'{"seq": 3, "par')
    with pytest.raises(HistoryCorrupt, match="torn"):
        h.verify()
    h.repair_drop_torn_tail()
    assert h.verify() == 2
    # edited record
    lines = (root / "history.jsonl").read_text().splitlines()
    lines[0] = lines[0].replace('"cause"', '"CAUSE"')
    (root / "history.jsonl").write_text("\n".join(lines) + "\n")
    with pytest.raises(HistoryCorrupt):
        h.verify()


def test_missing_head_or_truncation_detected(tmp_path, repo):
    r, (c0, _, _) = repo
    root = tmp_path / "h"
    h = History(root, repo=r)
    for p in "ABC":
        h.propose(_proposal(p, c0))
    lines = (root / "history.jsonl").read_text().splitlines()
    (root / "history.jsonl").write_text(lines[0] + "\n")
    with pytest.raises(HistoryCorrupt):
        h.verify()
    (root / "head.json").unlink()
    with pytest.raises(HistoryCorrupt, match="missing"):
        h.verify()
    assert not (root / "head.json").exists()


def test_import_draft_idempotent(tmp_path, repo):
    r, (c0, _, _) = repo
    h = History(tmp_path / "h", repo=r)
    d = tmp_path / "draft.json"
    d.write_text(json.dumps(_proposal("D", c0).to_dict()))
    assert h.import_draft(d) is not None
    assert h.import_draft(d) is None
    other = _proposal("D", c0)
    other.cause = "different"
    d.write_text(json.dumps(other.to_dict()))
    with pytest.raises(HistoryError):
        h.import_draft(d)


def test_runner_never_records_approvals():
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    hits = [p for p in root.rglob("*.py") if "tests" not in p.parts and p.name != "history.py"
            and ".approve(" in p.read_text(encoding="utf-8")]
    assert all(p.parent.name == "scripts" for p in hits), hits
