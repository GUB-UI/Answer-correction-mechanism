import subprocess

import pytest

from harness_v2.improvement.duplicates import change_identity
from harness_v2.improvement.history import History, HistoryError
from harness_v2.tests.test_improvement_history import _proposal


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "r"
    r.mkdir()
    _git(r, "init", "-q"); _git(r, "config", "user.email", "t@e.invalid"); _git(r, "config", "user.name", "t")
    out = {}
    def commit(name, text, parent=None, path="p.txt"):
        if parent:
            _git(r, "checkout", "-q", "--detach", parent)
        (r / path).write_text(text)
        _git(r, "add", "."); _git(r, "commit", "-qm", name)
        out[name] = _git(r, "rev-parse", "HEAD")
    commit("base", "a\nb\n")
    commit("cand", "a\nB\n", out["base"])
    commit("ws", "a\nB \n", out["base"])     # whitespace-only difference
    commit("base2", "other\n", out["base"], path="q.txt")  # new baseline, same p.txt
    commit("cand2", "a\nB\n", out["base2"])
    return r, out


def test_change_id_is_byte_sensitive(repo):
    r, c = repo
    a, b = change_identity(r, c["base"], c["cand"]), change_identity(r, c["base"], c["ws"])
    assert a["change_id"] != b["change_id"]
    assert change_identity(r, c["base"], c["cand"]) == a


def test_duplicate_of_rejected_needs_override(tmp_path, repo):
    r, c = repo
    h = History(tmp_path / "h", repo=r)
    h.propose_change(_proposal("P1", c["base"], c["cand"]))
    h.decide("P1", "rejected", reason="no gain")
    ev = h.propose_change(_proposal("P2", c["base"], c["cand"]))
    assert ev["payload"]["source"]["duplicates"][0]["kind"] == "exact"
    assert h.state()["proposals"]["P2"]["flags"] == ["duplicate_of_rejected"]
    with pytest.raises(HistoryError):
        h.override("P2", approver="", reason="")
    h.override("P2", approver="owner", reason="re-test after OCR fix")
    assert h.state()["proposals"]["P2"]["overrides"]
    # whitespace variant: not exact, so no flag
    h.propose_change(_proposal("P3", c["base"], c["ws"]))
    assert h.state()["proposals"]["P3"]["flags"] == []


def test_same_edit_on_new_baseline_is_similar(tmp_path, repo):
    r, c = repo
    h = History(tmp_path / "h", repo=r)
    h.propose_change(_proposal("P1", c["base"], c["cand"]))
    ev = h.propose_change(_proposal("P2", c["base2"], c["cand2"]))
    dups = ev["payload"]["source"]["duplicates"]
    assert dups and not dups[0]["same_baseline"]
    assert h.state()["proposals"]["P2"]["flags"] == ["retry_on_new_baseline"]
