"""Regression tests for Sol diff review round 2 of the Phase 5 loop."""
import json

import pytest

from harness_v2.improvement.duplicates import overlaps
from harness_v2.improvement.history import History, HistoryError
from harness_v2.tests.test_improvement_history import _experiments, _git, _proposal, inputs, repo  # noqa: F401


def test_foreign_run_in_results_is_rejected(tmp_path, repo, inputs):
    r, (c0, c1, _) = repo
    h = History(tmp_path / "h", repo=r)
    h.propose(_proposal("P", c0)); h.attach_candidate("P", c1)
    kw = _experiments(tmp_path, h, c0, c1, inputs)
    cand = json.loads(kw["candidate_results"].read_text())
    cand["runs"] = [{"commit": c0, "experiment_id": "other", "prompt_version": "x", "model_id": "m",
                     "rubric_version": "r1"}]
    kw["candidate_results"].write_text(json.dumps(cand))
    with pytest.raises(HistoryError, match="does not belong"):
        h.record_evaluation("P", **kw)
    cand["runs"], cand["identity"] = [], "forged"
    kw["candidate_results"].write_text(json.dumps(cand))
    with pytest.raises(HistoryError, match="identity does not match"):
        h.record_evaluation("P", **kw)


def test_evaluation_must_use_proposal_baseline(tmp_path, repo, inputs):
    r, (c0, c1, c2) = repo
    h = History(tmp_path / "h", repo=r)
    h.propose(_proposal("P", c0)); h.attach_candidate("P", c2)
    with pytest.raises(HistoryError, match="baseline differs"):
        h.record_evaluation("P", **_experiments(tmp_path, h, c1, c2, inputs))


def test_draft_import_and_unattached_evaluation(tmp_path, repo, inputs):
    r, (c0, c1, _) = repo
    h = History(tmp_path / "h", repo=r)
    h.propose(_proposal("A", c0)); h.attach_candidate("A", c1)
    h.decide("A", "rejected", reason="no gain")
    draft = tmp_path / "d.json"
    draft.write_text(json.dumps(_proposal("B", c0, c1).to_dict()))
    h.import_draft(draft)
    b = h.state()["proposals"]["B"]
    assert b["identity"] and b["flags"]
    h.propose(_proposal("C", c0))
    with pytest.raises(HistoryError, match="attached"):
        h.record_evaluation("C", **_experiments(tmp_path, h, c0, c1, inputs))


def test_shared_component_edit_overlaps_despite_unrelated_differences():
    a = {"pairs": [["prompt.txt", "p1"], ["README", "r1"]]}
    b = {"pairs": [["prompt.txt", "p1"], ["README", "r2"]]}
    assert overlaps(a, b)
    assert not overlaps(a, {"pairs": [["prompt.txt", "p2"]]})


def test_unsafe_proposal_id_rejected(tmp_path):
    h = History(tmp_path / "h")
    with pytest.raises(HistoryError, match="proposal_id"):
        h.propose(_proposal("sk-abcdefghijklmnopqrstuvwxyz0123456789", "0" * 40))
    with pytest.raises(HistoryError, match="proposal_id"):
        h.propose(_proposal("a b; rm", "0" * 40))
