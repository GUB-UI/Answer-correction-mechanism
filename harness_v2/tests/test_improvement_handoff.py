import json
import os

import pytest

from harness_v2.improvement.handoff import record_review, write_review_request
from harness_v2.improvement.history import History, HistoryError
from harness_v2.providers.grokbot_exchange import SchemaError
from harness_v2.tests.test_improvement_history import _experiments, _proposal, inputs, repo  # noqa: F401


def _setup(tmp_path, repo, inputs):
    r, (c0, c1, _) = repo
    h = History(tmp_path / "h", repo=r)
    h.propose(_proposal("P", c0))
    h.record_evaluation("P", **_experiments(tmp_path, h, c0, c1, inputs))
    return h


def _result(req, **kw):
    base = {"request_id": req["request_id"], "proposal_id": "P", "snapshot_id": req["snapshot_id"],
            "reviewer": "grokbot", "verdict": "approve_for_human", "comments": "ok"}
    base.update(kw)
    return base


def test_handoff_files_and_bound_review(tmp_path, repo, inputs):
    h = _setup(tmp_path, repo, inputs)
    folder = write_review_request(h, "P", tmp_path / "out")
    for name in ("request.json", "issue.md", "pr.md", "branch.txt", "commands.txt"):
        mode = os.stat(folder / name).st_mode
        assert mode & 0o777 == 0o600 and not mode & 0o111
    req = json.loads((folder / "request.json").read_text())
    with pytest.raises(SchemaError):
        record_review(h, _result(req, verdict="approve"))  # enum enforced
    with pytest.raises(HistoryError):
        record_review(h, _result(req, snapshot_id="0" * 64))
    with pytest.raises(HistoryError):
        record_review(h, _result(req, request_id="pr-unknown"))
    record_review(h, _result(req))
    with pytest.raises(HistoryError, match="replay"):
        record_review(h, _result(req))
    st = h.state()["proposals"]["P"]
    assert len(st["reviews"]) == 1 and st["approvals"] == []


def test_handoff_redacts_secrets(tmp_path, repo, inputs):
    r, (c0, c1, _) = repo
    h = History(tmp_path / "h", repo=r)
    p = _proposal("P", c0)
    p.cause = "leaked key sk-abcdefghijklmnopqrstuvwxyz0123456789 in config"
    h.propose(p)
    h.record_evaluation("P", **_experiments(tmp_path, h, c0, c1, inputs))
    folder = write_review_request(h, "P", tmp_path / "out")
    for name in ("request.json", "issue.md", "pr.md", "commands.txt"):
        assert "sk-abcdefghijklmnop" not in (folder / name).read_text()


def test_enum_checked_on_strings():
    from harness_v2.providers.grokbot_exchange import _check
    assert _check("x", {"type": "string", "enum": ["a"]}, "v")
