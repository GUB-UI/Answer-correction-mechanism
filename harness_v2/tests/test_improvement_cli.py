import importlib.util
import json
from pathlib import Path

import pytest

from harness_v2.improvement.history import History
from harness_v2.tests.test_improvement_history import _cycle, _base, inputs, repo  # noqa: F401

SPEC = importlib.util.spec_from_file_location("ih", Path(__file__).resolve().parents[2] / "scripts/improvement_history.py")
ih = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(ih)


def test_check_merge_and_rollback_demo(tmp_path, repo, inputs, capsys):
    r, (c0, c1, c2) = repo
    root = tmp_path / "h"
    h = History(root, repo=r)
    _base(h, c0, inputs)
    from harness_v2.tests.test_improvement_history import _proposal, _experiments
    h.propose(_proposal("P", c0)); h.attach_candidate("P", c1)
    h.record_evaluation("P", **_experiments(tmp_path, h, c0, c1, inputs))
    h.decide("P", "accepted", reason="ok")
    look = lambda repo_, n: {"headRefOid": c1, "baseRefName": "main", "number": n}
    common = ["--root", str(root), "--repo", str(r)]
    assert ih.main(common + ["check-merge", "--proposal", "P", "--repository", "o/r", "--pr", "7"], pr_lookup=look) == 1
    ih.main(common + ["approve", "--proposal", "P", "--approver", "owner", "--repository", "o/r",
                      "--base", "main", "--pr", "7", "--quote", "PR7をマージしてよい"])
    assert ih.main(common + ["check-merge", "--proposal", "P", "--repository", "o/r", "--pr", "7"], pr_lookup=look) == 0
    moved = lambda repo_, n: {"headRefOid": c2, "baseRefName": "main", "number": n}
    assert ih.main(common + ["check-merge", "--proposal", "P", "--repository", "o/r", "--pr", "7"], pr_lookup=moved) == 1
    h.register_version("v1", commit=c1, config_path=inputs[0], dataset_manifest=inputs[1], ocr_preprocess="none")
    h.promote("P", "v1")
    assert ih.main(common + ["rollback", "--to", "v0", "--reason", "demo"]) == 0
    assert ih.main(common + ["verify"]) == 0
    capsys.readouterr()
    ih.main(common + ["status"])
    st = json.loads(capsys.readouterr().out)
    assert st["active"] == "v0" and st["versions"] == ["v0", "v1"] and st["proposals"]["P"]["promoted"] == "v1"
