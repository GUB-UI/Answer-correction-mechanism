import importlib.util
import json
from pathlib import Path

from harness_v2.evaluation.challenge import challenge_report
from harness_v2.evaluation.dataset import load_dataset

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("bcs", ROOT / "scripts" / "build_challenge_set.py")
bcs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bcs)


def test_transforms():
    assert bcs.full_width_digits("x=12") == "x=１２"
    assert bcs.space_operators("2x−4<−x+5") == "2x−4 < −x + 5"
    assert bcs.space_operators("a = b") == "a = b"


def test_variants_deterministic_and_zero_eligible():
    a, inv = bcs.variants("p:manual", "√9−√2")
    assert [label for label, _ in a] == ["O", "F"]
    assert inv["N_dropped"] == "zero eligible" and inv["S_dropped"] == "no-op"
    b1, _ = bcs.variants("p:x", "x<2 のとき 最小値 a")
    b2, _ = bcs.variants("p:x", "x<2 のとき 最小値 a")
    assert b1 == b2 and [label for label, _ in b1] == ["O", "F", "S", "N"]
    noisy = dict(b1)["N"]
    assert len(noisy) == len("x<2 のとき 最小値 a") - 1


def _kit(tmp_path):
    kit = tmp_path / "kit"
    kit.mkdir()
    (kit / "rubric.json").write_text(json.dumps({"P": [{"id": "R1", "max_score": 2, "label": "L", "rule": "r"}]}))
    (kit / "questions.json").write_text(json.dumps({"P": {"question": "q"}}))
    (kit / "manual_transcriptions.json").write_text(json.dumps([{"question_id": "P", "answer_text": "x=1 のとき"}]))
    c1 = tmp_path / "c1.json"
    c1.write_text("{}")
    return kit, c1


def test_builder_writes_label_free_private_dataset(tmp_path):
    kit, c1 = _kit(tmp_path)
    manifest = bcs.build_challenge_set(tmp_path / "out", kit=kit, c1_path=c1)
    ds = load_dataset(manifest)
    assert ds.references["reference_scores"] == {} and ds.references["reference_items"] == {}
    assert [c["input_version"] for c in ds.cases] == ["manual:O", "manual:F", "manual:S", "manual:N"]
    assert (manifest.parent.stat().st_mode & 0o777) == 0o700
    assert (manifest.stat().st_mode & 0o777) == 0o600
    assert ((tmp_path / "out" / "inventory.json").stat().st_mode & 0o777) == 0o600


def _run(rid, status, scores):
    items = [{"item_id": k, "score": v} for k, v in scores] if scores is not None else []
    return {"request_id": rid, "outputs": {"grading": {"status": status, "items": items}}}


R = {"P": ["R1"]}


def _jobs(sources=("P:manual",), variants=("O", "F", "N"), reps=(1, 2)):
    return [{"request_id": f"{s}{v}{r}", "case_id": f"{s}:{v}", "problem_id": "P", "repetition": r}
            for s in sources for v in variants for r in reps]


def test_report_pairs_by_repetition_and_classifies_outcomes():
    runs = [_run("P:manualO1", "validated", [("R1", 2)]), _run("P:manualO2", "validated", [("R1", 1)]),
            _run("P:manualF1", "validated", [("R1", 2)]), _run("P:manualF2", "review_required", [("R1", 2)]),
            _run("P:manualN1", "failed", None), _run("P:manualN2", "review_required", [("R1", 1)])]
    report = challenge_report({"expected_runs": _jobs(), "runs": runs}, rubric_items=R)
    f, n = report["variants"]["F"], report["variants"]["N"]
    assert f["pairs"] == 2 and f["items_equal"] == 1 and f["item_agreement"] == 0.5
    assert f["newly_flagged_relative_to_O"] == 1 and f["flag_rate"] == 0.5
    assert n["failed"] == 1 and n["pairs"] == 1 and n["flagged"] == 1 and n["flag_denominator"] == 1
    assert "reference" not in json.dumps(report)


def test_missing_runs_are_counted_not_dropped():
    runs = [_run("P:manualO1", "validated", [("R1", 2)]), _run("P:manualF1", "validated", [("R1", 2)])]
    report = challenge_report({"expected_runs": _jobs(), "runs": runs}, rubric_items=R)
    f, o, n = (report["variants"][k] for k in "FON")
    assert f["scheduled"] == 2 and f["missing"] == 1 and f["pairs"] == 1
    assert o["missing"] == 1 and n["missing"] == 2 and n["item_agreement"] is None and n["flag_rate"] is None


def test_variant_without_valid_original_is_unpaired():
    runs = [_run("P:manualO1", "failed", None), _run("P:manualF1", "validated", [("R1", 2)])]
    report = challenge_report({"expected_runs": _jobs(variants=("O", "F"), reps=(1,)), "runs": runs}, rubric_items=R)
    assert report["variants"]["F"]["unpaired_no_valid_O"] == 1 and report["variants"]["F"]["pairs"] == 0


def test_failed_duplicate_nonfinite_and_unexpected_items_never_agree():
    runs = [_run("P:manualO1", "validated", [("R1", 1)]),
            _run("P:manualF1", "failed", [("R1", 0), ("R1", 1)]),
            _run("P:manualG1", "validated", [("R1", 0), ("R1", 1)]),
            _run("P:manualH1", "validated", [("R1", float("nan"))]),
            _run("P:manualI1", "validated", [("R1", 1), ("R9", 0)])]
    report = challenge_report({"expected_runs": _jobs(variants=("O", "F", "G", "H", "I"), reps=(1,)),
                               "runs": runs}, rubric_items=R)
    v = report["variants"]
    assert v["F"]["failed"] == 1
    assert v["G"]["incomplete"] == v["H"]["incomplete"] == v["I"]["incomplete"] == 1
    assert all(v[k]["pairs"] == 0 for k in "FGHI")


def test_by_source_keeps_transcriptions_separate():
    jobs = _jobs(sources=("P:manual_C0", "P:manual_C1"), variants=("O", "F"), reps=(1,))
    runs = [_run("P:manual_C0O1", "validated", [("R1", 1)]), _run("P:manual_C0F1", "validated", [("R1", 1)]),
            _run("P:manual_C1O1", "validated", [("R1", 1)]), _run("P:manual_C1F1", "review_required", [("R1", 0)])]
    report = challenge_report({"expected_runs": jobs, "runs": runs}, rubric_items=R)
    assert report["by_source"]["P:manual_C0"]["F"]["item_agreement"] == 1.0
    assert report["by_source"]["P:manual_C1"]["F"]["item_agreement"] == 0.0
    assert report["by_source"]["P:manual_C1"]["F"]["newly_flagged_relative_to_O"] == 1
    assert report["variants"]["F"]["pairs"] == 2


def test_rubric_ids_are_required_and_fail_closed():
    import pytest
    jobs = _jobs(variants=("O", "F"), reps=(1,))
    with pytest.raises(TypeError):
        challenge_report({"expected_runs": jobs, "runs": []})
    with pytest.raises(ValueError):
        challenge_report({"expected_runs": jobs, "runs": []}, rubric_items={"Q": ["R1"]})


def test_equally_truncated_or_unexpected_scores_are_incomplete():
    runs = [_run("P:manualO1", "validated", [("R1", 1)]), _run("P:manualF1", "validated", [("R1", 1)])]
    report = challenge_report({"expected_runs": _jobs(variants=("O", "F"), reps=(1,)), "runs": runs},
                              rubric_items={"P": ["R1", "R2"]})
    assert report["variants"]["F"]["pairs"] == 0 and report["variants"]["O"]["incomplete"] == 1
    runs = [_run("P:manualO1", "validated", [("X", 1)]), _run("P:manualF1", "validated", [("X", 1)])]
    report = challenge_report({"expected_runs": _jobs(variants=("O", "F"), reps=(1,)), "runs": runs}, rubric_items=R)
    assert report["variants"]["F"]["incomplete"] == 1 and report["variants"]["F"]["pairs"] == 0
