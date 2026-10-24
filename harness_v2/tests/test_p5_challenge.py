import copy
import importlib.util
import json
from pathlib import Path

import pytest

from harness_v2.evaluation import challenge
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


def _bind(jobs, runs, identity="experiment-A"):
    """Attach provenance that matches each planned job. Tests of bad provenance skip this."""
    planned = {}
    if isinstance(jobs, list):
        for job in jobs:
            if isinstance(job, dict):
                planned.setdefault(job.get("request_id"), job)
    bound = []
    for run in runs:
        copied = copy.deepcopy(run)
        job = planned.get(copied.get("request_id")) if isinstance(copied, dict) else None
        outputs = copied.setdefault("outputs", {}) if isinstance(copied, dict) else None
        if job is not None and isinstance(outputs, dict) and not isinstance(outputs.get("experiment_case"), dict):
            outputs["experiment_case"] = {
                "case_id": job["case_id"], "repetition": job["repetition"],
                "problem_id": job["problem_id"], "experiment_identity": identity,
            }
        bound.append(copied)
    return bound


R = {"P": ["R1"]}


def _jobs(sources=("P:manual",), variants=("O", "F", "N"), reps=(1, 2)):
    return [{"request_id": f"{s}{v}{r}", "case_id": f"{s}:{v}", "problem_id": "P", "repetition": r}
            for s in sources for v in variants for r in reps]


def test_report_pairs_by_repetition_and_classifies_outcomes():
    runs = [_run("P:manualO1", "validated", [("R1", 2)]), _run("P:manualO2", "validated", [("R1", 1)]),
            _run("P:manualF1", "validated", [("R1", 2)]), _run("P:manualF2", "review_required", [("R1", 2)]),
            _run("P:manualN1", "failed", None), _run("P:manualN2", "review_required", [("R1", 1)])]
    jobs = _jobs()
    report = challenge_report({"expected_runs": jobs, "runs": _bind(jobs, runs)}, rubric_items=R)
    assert report["comparable"] is True and report["reasons"] == []
    f, n = report["variants"]["F"], report["variants"]["N"]
    assert f["pairs"] == 2 and f["items_equal"] == 1 and f["item_agreement"] == 0.5
    assert f["newly_flagged_relative_to_O"] == 1 and f["flag_rate"] == 0.5
    assert n["failed"] == 1 and n["pairs"] == 1 and n["flagged"] == 1 and n["flag_denominator"] == 1
    assert "reference" not in json.dumps(report)


def test_missing_runs_are_counted_not_dropped():
    jobs = _jobs()
    runs = [_run("P:manualO1", "validated", [("R1", 2)]), _run("P:manualF1", "validated", [("R1", 2)])]
    report = challenge_report({"expected_runs": jobs, "runs": _bind(jobs, runs)}, rubric_items=R)
    f, o, n = (report["variants"][k] for k in "FON")
    assert f["scheduled"] == 2 and f["missing"] == 1 and f["pairs"] == 1
    assert o["missing"] == 1 and n["missing"] == 2 and n["item_agreement"] is None and n["flag_rate"] is None
    assert report["comparable"] is False
    assert report["reasons"] == [f"missing request_id: P:manual{suffix}" for suffix in ("F2", "N1", "N2", "O2")]


def test_variant_without_valid_original_is_unpaired():
    jobs = _jobs(variants=("O", "F"), reps=(1,))
    runs = [_run("P:manualO1", "failed", None), _run("P:manualF1", "validated", [("R1", 2)])]
    report = challenge_report({"expected_runs": jobs, "runs": _bind(jobs, runs)}, rubric_items=R)
    assert report["variants"]["F"]["unpaired_no_valid_O"] == 1 and report["variants"]["F"]["pairs"] == 0


def test_failed_duplicate_nonfinite_and_unexpected_items_never_agree():
    runs = [_run("P:manualO1", "validated", [("R1", 1)]),
            _run("P:manualF1", "failed", [("R1", 0), ("R1", 1)]),
            _run("P:manualG1", "validated", [("R1", 0), ("R1", 1)]),
            _run("P:manualH1", "validated", [("R1", float("nan"))]),
            _run("P:manualI1", "validated", [("R1", 1), ("R9", 0)])]
    jobs = _jobs(variants=("O", "F", "G", "H", "I"), reps=(1,))
    report = challenge_report({"expected_runs": jobs, "runs": _bind(jobs, runs)}, rubric_items=R)
    v = report["variants"]
    assert v["F"]["failed"] == 1
    assert v["G"]["incomplete"] == v["H"]["incomplete"] == v["I"]["incomplete"] == 1
    assert all(v[k]["pairs"] == 0 for k in "FGHI")


def test_by_source_keeps_transcriptions_separate():
    jobs = _jobs(sources=("P:manual_C0", "P:manual_C1"), variants=("O", "F"), reps=(1,))
    runs = [_run("P:manual_C0O1", "validated", [("R1", 1)]), _run("P:manual_C0F1", "validated", [("R1", 1)]),
            _run("P:manual_C1O1", "validated", [("R1", 1)]), _run("P:manual_C1F1", "review_required", [("R1", 0)])]
    report = challenge_report({"expected_runs": jobs, "runs": _bind(jobs, runs)}, rubric_items=R)
    assert report["by_source"]["P:manual_C0"]["F"]["item_agreement"] == 1.0
    assert report["by_source"]["P:manual_C1"]["F"]["item_agreement"] == 0.0
    assert report["by_source"]["P:manual_C1"]["F"]["newly_flagged_relative_to_O"] == 1
    assert report["variants"]["F"]["pairs"] == 2


def test_rubric_ids_are_required_and_fail_closed():
    jobs = _jobs(variants=("O", "F"), reps=(1,))
    with pytest.raises(TypeError):
        challenge_report({"expected_runs": jobs, "runs": []})
    with pytest.raises(ValueError):
        challenge_report({"expected_runs": jobs, "runs": []}, rubric_items={"Q": ["R1"]})


def test_equally_truncated_or_unexpected_scores_are_incomplete():
    jobs = _jobs(variants=("O", "F"), reps=(1,))
    runs = [_run("P:manualO1", "validated", [("R1", 1)]), _run("P:manualF1", "validated", [("R1", 1)])]
    report = challenge_report({"expected_runs": jobs, "runs": _bind(jobs, runs)},
                              rubric_items={"P": ["R1", "R2"]})
    assert report["variants"]["F"]["pairs"] == 0 and report["variants"]["O"]["incomplete"] == 1
    runs = [_run("P:manualO1", "validated", [("X", 1)]), _run("P:manualF1", "validated", [("X", 1)])]
    report = challenge_report({"expected_runs": jobs, "runs": _bind(jobs, runs)}, rubric_items=R)
    assert report["variants"]["F"]["incomplete"] == 1 and report["variants"]["F"]["pairs"] == 0


@pytest.mark.parametrize("variant", ["O", "F"])
@pytest.mark.parametrize("duplicate_first", [False, True])
def test_duplicate_request_ids_are_invalid_and_neither_record_is_used(variant, duplicate_first):
    jobs = _jobs(variants=("O", "F"), reps=(1,))
    runs = [_run(job["request_id"], "validated", [("R1", 1)]) for job in jobs]
    rid = f"P:manual{variant}1"
    duplicate = _run(rid, "review_required", [("R1", 0)])
    runs.insert(0 if duplicate_first else len(runs), duplicate)
    report = challenge_report({"expected_runs": jobs, "runs": _bind(jobs, runs)}, rubric_items=R)
    assert report["comparable"] is False
    assert report["reasons"] == [f"duplicate request_id: {rid}"]
    assert report["variants"][variant]["incomplete"] == 1
    assert report["variants"][variant]["flag_denominator"] == 0
    assert report["variants"]["F"]["pairs"] == 0
    assert report["variants"]["F"]["item_agreement"] is None


def test_unexpected_request_id_invalidates_an_otherwise_complete_report():
    jobs = _jobs(variants=("O", "F"), reps=(1,))
    runs = [_run(job["request_id"], "validated", [("R1", 1)]) for job in jobs]
    runs.append(_run("unscheduled", "validated", [("R1", 0)]))
    report = challenge_report({"expected_runs": jobs, "runs": _bind(jobs, runs)}, rubric_items=R)
    assert report["comparable"] is False
    assert report["reasons"] == ["unexpected request_id: unscheduled"]
    assert report["variants"]["F"]["scheduled"] == 1


def test_all_request_id_problems_are_reported_together():
    jobs = _jobs(variants=("O", "F"), reps=(1,))
    runs = [_run("P:manualO1", "validated", [("R1", 1)])] * 2
    runs += [_run("unscheduled", "failed", None)] * 2
    report = challenge_report({"expected_runs": jobs, "runs": _bind(jobs, runs)}, rubric_items=R)
    assert report["comparable"] is False
    assert report["reasons"] == [
        "duplicate request_id: P:manualO1", "duplicate request_id: unscheduled",
        "missing request_id: P:manualF1", "unexpected request_id: unscheduled",
    ]


@pytest.mark.parametrize("record_count", [0, 1, 2])
@pytest.mark.parametrize("reverse_plan", [False, True])
@pytest.mark.parametrize("axis", ["variant", "source", "repetition"])
def test_duplicate_planned_ids_cannot_reuse_a_record(axis, reverse_plan, record_count):
    jobs = _jobs(variants=("O", "F"), reps=(1,))
    if axis == "source":
        jobs[1]["case_id"] = "P:other:F"
    elif axis == "repetition":
        jobs[1]["repetition"] = 2
    for job in jobs:
        job["request_id"] = "same"
    runs = [_run("same", "validated", [("R1", 1)]) for _ in range(record_count)]
    report = challenge_report({"expected_runs": jobs[::-1] if reverse_plan else jobs, "runs": runs},
                              rubric_items=R)
    assert report["comparable"] is False
    assert "duplicate planned request_id: same" in report["reasons"]
    for variant in ("O", "F"):
        stats = report["variants"][variant]
        assert stats["scheduled"] == stats["incomplete"] == 1
        assert stats["flag_denominator"] == stats["pairs"] == stats["items"] == 0
        assert stats["item_agreement"] is None and stats["flag_rate"] is None


@pytest.mark.parametrize("results", [
    {}, {"runs": []}, {"expected_runs": []}, {"expected_runs": [], "runs": []},
    {"expected_runs": None, "runs": []}, {"expected_runs": {}, "runs": []},
])
def test_absent_empty_or_invalid_job_plan_is_not_comparable(results):
    report = challenge_report(results, rubric_items=R)
    assert report["comparable"] is False
    assert report["reasons"] == ["expected_runs must be an explicit nonempty job plan"]
    assert report["variants"] == report["by_source"] == {}


def test_duplicate_planned_ids_leave_only_unambiguous_pairs_as_diagnostics():
    jobs = _jobs(variants=("O", "F"), reps=(1, 2))
    for job in jobs:
        if job["repetition"] == 1:
            job["request_id"] = "same"
    runs = [_run(rid, "validated", [("R1", 1)]) for rid in {job["request_id"] for job in jobs}]
    report = challenge_report({"expected_runs": jobs, "runs": _bind(jobs, runs)}, rubric_items=R)
    assert report["comparable"] is False
    assert report["reasons"] == ["duplicate planned request_id: same"]
    assert report["variants"]["F"]["pairs"] == 1
    assert report["variants"]["F"]["items"] == 1
    assert report["variants"]["F"]["incomplete"] == 1


@pytest.mark.parametrize("problem", [None, "duplicate", "missing", "unexpected", "duplicate planned",
                                     "absent plan", "empty plan"])
def test_cli_marks_request_id_integrity_in_report_and_exit_code(tmp_path, monkeypatch, capsys, problem):
    jobs = _jobs(variants=("O", "F"), reps=(1,))
    runs = [_run(job["request_id"], "validated", [("R1", 1)]) for job in reversed(jobs)]
    if problem == "duplicate":
        runs.append(runs[0])
    elif problem == "missing":
        runs.pop()
    elif problem == "unexpected":
        runs.append(_run("unscheduled", "validated", [("R1", 1)]))
    elif problem == "duplicate planned":
        for job in jobs:
            job["request_id"] = "same"
        runs = [_run("same", "validated", [("R1", 1)])]
    payload = {"expected_runs": jobs, "runs": _bind(jobs, runs)}
    if problem == "absent plan":
        payload = {}
    elif problem == "empty plan":
        payload = {"expected_runs": [], "runs": []}
    results_path, out = tmp_path / "results.json", tmp_path / "report.json"
    results_path.write_text(json.dumps(payload))
    monkeypatch.setattr(challenge, "_rubric_items", lambda manifest: R)
    code = challenge.main(["--results", str(results_path), "--dataset", "unused", "--out", str(out)])
    report = json.loads(out.read_text())
    assert json.loads(capsys.readouterr().out) == report
    assert code == (0 if problem is None else 2)
    assert report["comparable"] is (problem is None)
    if problem is None:
        assert report["reasons"] == []
        assert report["variants"]["F"]["item_agreement"] == 1
    else:
        assert len(report["reasons"]) == 1
        if problem in ("absent plan", "empty plan"):
            assert report["reasons"] == ["expected_runs must be an explicit nonempty job plan"]
        else:
            assert report["reasons"][0].startswith(f"{problem} request_id:")
        if problem == "duplicate planned":
            assert report["variants"]["F"]["pairs"] == 0
            assert report["variants"]["F"]["item_agreement"] is None


@pytest.mark.parametrize("reverse", [False, True])
def test_duplicate_planned_slots_do_not_overwrite(reverse):
    """Distinct request IDs that share a slot must not collapse into one agreement."""
    jobs = []
    runs = []
    # cfgA disagrees; cfgB agrees. Whichever is written last must not become the result.
    for cfg, variant_score in (("cfgA", 0), ("cfgB", 1)):
        for variant, score in (("O", 1), ("F", variant_score)):
            rid = f"{cfg}-{variant}"
            jobs.append({"request_id": rid, "case_id": f"P:manual:{variant}",
                         "problem_id": "P", "repetition": 1})
            runs.append(_run(rid, "validated", [("R1", score)]))
    if reverse:
        jobs.reverse()
        runs.reverse()
    report = challenge_report({"expected_runs": jobs, "runs": _bind(jobs, runs)}, rubric_items=R)
    assert report["comparable"] is False
    assert report["reasons"] == [
        "duplicate planned slot: P:manual:F:1",
        "duplicate planned slot: P:manual:O:1",
    ]
    assert report["variants"]["F"]["scheduled"] == 1
    assert report["variants"]["F"]["pairs"] == 0
    assert report["variants"]["F"]["items"] == 0
    assert report["variants"]["F"]["item_agreement"] is None
    assert report["variants"]["O"]["incomplete"] == report["variants"]["F"]["incomplete"] == 1


def test_duplicate_planned_slot_cli_exits_2_without_changing_inputs(tmp_path, monkeypatch):
    jobs = [
        {"request_id": "cfgA-O", "case_id": "P:manual:O", "problem_id": "P", "repetition": 1},
        {"request_id": "cfgB-O", "case_id": "P:manual:O", "problem_id": "P", "repetition": 1},
        {"request_id": "cfgA-F", "case_id": "P:manual:F", "problem_id": "P", "repetition": 1},
        {"request_id": "cfgB-F", "case_id": "P:manual:F", "problem_id": "P", "repetition": 1},
    ]
    runs = [_run(job["request_id"], "validated", [("R1", 1)]) for job in jobs]
    results_path = tmp_path / "results.json"
    out = tmp_path / "nested" / "report.json"
    tmp_path.chmod(0o755)
    results_path.write_text(json.dumps({"expected_runs": jobs, "runs": _bind(jobs, runs)}))
    results_path.chmod(0o644)
    monkeypatch.setattr(challenge, "_rubric_items", lambda manifest: R)
    code = challenge.main(["--results", str(results_path), "--dataset", "unused", "--out", str(out)])
    report = json.loads(out.read_text())
    assert code == 2 and report["comparable"] is False
    assert any(reason.startswith("duplicate planned slot:") for reason in report["reasons"])
    assert (out.stat().st_mode & 0o777) == 0o600
    assert (out.parent.stat().st_mode & 0o777) == 0o700
    assert (results_path.stat().st_mode & 0o777) == 0o644
    assert (tmp_path.stat().st_mode & 0o777) == 0o755


def _pair(identity="experiment-A", *, o_case=None, f_case=None):
    jobs = _jobs(variants=("O", "F"), reps=(1,))
    runs = _bind(jobs, [_run(job["request_id"], "validated", [("R1", 1)]) for job in jobs], identity=identity)
    if o_case is not None:
        runs[0]["outputs"]["experiment_case"].update(o_case)
    if f_case is not None:
        runs[1]["outputs"]["experiment_case"].update(f_case)
    return jobs, runs


@pytest.mark.parametrize("change", ["missing", "case_id", "repetition", "problem_id", "identity_blank"])
def test_run_provenance_must_match_the_planned_job(change):
    jobs, runs = _pair()
    if change == "missing":
        runs[1]["outputs"].pop("experiment_case")
        expected = "missing run provenance: P:manualF1"
    elif change == "identity_blank":
        runs[1]["outputs"]["experiment_case"]["experiment_identity"] = " "
        expected = "missing run provenance: P:manualF1"
    else:
        runs[1]["outputs"]["experiment_case"][change] = {"case_id": "other:F", "repetition": 9, "problem_id": "Q"}[change]
        expected = "run provenance differs from planned job: P:manualF1"
    report = challenge_report({"expected_runs": jobs, "runs": runs}, rubric_items=R)
    assert report["comparable"] is False
    assert expected in report["reasons"]
    assert report["variants"]["F"]["pairs"] == 0
    assert report["variants"]["F"]["item_agreement"] is None


def test_different_experiment_identities_are_not_compared():
    jobs = _jobs(variants=("O", "F"), reps=(1,))
    runs = _bind(jobs, [_run(job["request_id"], "validated", [("R1", 1)]) for job in jobs])
    runs[1]["outputs"]["experiment_case"]["experiment_identity"] = "experiment-B"
    report = challenge_report({"expected_runs": jobs, "runs": runs}, rubric_items=R)
    assert report["comparable"] is False
    assert report["reasons"] == ["multiple experiment identities: experiment-A, experiment-B"]
    assert report["variants"]["F"]["pairs"] == 0
    assert report["variants"]["F"]["items"] == 0
    assert report["variants"]["F"]["item_agreement"] is None


def test_matching_provenance_remains_comparable():
    jobs, runs = _pair()
    report = challenge_report({"expected_runs": jobs, "runs": runs}, rubric_items=R)
    assert report["comparable"] is True and report["reasons"] == []
    assert report["variants"]["F"]["item_agreement"] == 1


def test_missing_provenance_cli_exits_2_without_changing_inputs(tmp_path, monkeypatch):
    jobs = _jobs(variants=("O", "F"), reps=(1,))
    runs = [_run(job["request_id"], "validated", [("R1", 1)]) for job in jobs]
    results_path = tmp_path / "results.json"
    out = tmp_path / "out" / "report.json"
    tmp_path.chmod(0o755)
    results_path.write_text(json.dumps({"expected_runs": jobs, "runs": runs}))
    results_path.chmod(0o644)
    monkeypatch.setattr(challenge, "_rubric_items", lambda manifest: R)
    code = challenge.main(["--results", str(results_path), "--dataset", "unused", "--out", str(out)])
    report = json.loads(out.read_text())
    assert code == 2 and report["comparable"] is False
    assert any(reason.startswith("missing run provenance:") for reason in report["reasons"])
    assert report["variants"]["F"]["item_agreement"] is None
    assert (out.stat().st_mode & 0o777) == 0o600
    assert (out.parent.stat().st_mode & 0o777) == 0o700
    assert (results_path.stat().st_mode & 0o777) == 0o644
    assert (tmp_path.stat().st_mode & 0o777) == 0o755


@pytest.mark.parametrize("bad_id", [None, "", "  ", 1])
def test_non_string_request_ids_do_not_match(bad_id):
    jobs = [
        {"request_id": bad_id, "case_id": "P:manual:O", "problem_id": "P", "repetition": 1},
        {"request_id": bad_id, "case_id": "P:manual:F", "problem_id": "P", "repetition": 1},
    ]
    runs = [_run(bad_id, "validated", [("R1", 1)]), _run(bad_id, "validated", [("R1", 1)])]
    report = challenge_report({"expected_runs": jobs, "runs": runs}, rubric_items=R)
    assert report["comparable"] is False
    assert report["reasons"].count(f"invalid request_id: {bad_id!r}") == 4
    assert report["variants"].get("F", {}).get("item_agreement") is None
    assert report["variants"].get("F", {}).get("pairs", 0) == 0


def test_invalid_request_id_cli_exits_2_without_changing_inputs(tmp_path, monkeypatch):
    jobs = _jobs(variants=("O", "F"), reps=(1,))
    jobs[1]["request_id"] = None
    runs = _bind(jobs[:1], [_run(jobs[0]["request_id"], "validated", [("R1", 1)])])
    runs.append(_run(None, "validated", [("R1", 1)]))
    results_path = tmp_path / "results.json"
    out = tmp_path / "out" / "report.json"
    tmp_path.chmod(0o755)
    results_path.write_text(json.dumps({"expected_runs": jobs, "runs": runs}))
    results_path.chmod(0o644)
    monkeypatch.setattr(challenge, "_rubric_items", lambda manifest: R)
    code = challenge.main(["--results", str(results_path), "--dataset", "unused", "--out", str(out)])
    report = json.loads(out.read_text())
    assert code == 2 and report["comparable"] is False
    assert "invalid request_id: None" in report["reasons"]
    assert (out.stat().st_mode & 0o777) == 0o600
    assert (results_path.stat().st_mode & 0o777) == 0o644
    assert (tmp_path.stat().st_mode & 0o777) == 0o755


@pytest.mark.parametrize("missing", ["O", "F"])
def test_planned_slot_without_original_or_variant_is_not_comparable(missing):
    variant = "F" if missing == "O" else "O"
    jobs = _jobs(variants=(variant,), reps=(1,))
    runs = _bind(jobs, [_run(job["request_id"], "validated", [("R1", 1)]) for job in jobs])
    report = challenge_report({"expected_runs": jobs, "runs": runs}, rubric_items=R)
    assert report["comparable"] is False
    reason = "missing planned original: P:manual:1" if missing == "O" else "missing planned variant: P:manual:1"
    assert reason in report["reasons"]
    assert report["variants"][variant]["pairs"] == 0
    assert report["variants"][variant]["item_agreement"] is None


def test_missing_original_cli_exits_2_without_changing_inputs(tmp_path, monkeypatch):
    jobs = _jobs(variants=("F",), reps=(1,))
    runs = _bind(jobs, [_run(jobs[0]["request_id"], "validated", [("R1", 1)])])
    results_path = tmp_path / "results.json"
    out = tmp_path / "out" / "report.json"
    tmp_path.chmod(0o755)
    results_path.write_text(json.dumps({"expected_runs": jobs, "runs": runs}))
    results_path.chmod(0o644)
    monkeypatch.setattr(challenge, "_rubric_items", lambda manifest: R)
    code = challenge.main(["--results", str(results_path), "--dataset", "unused", "--out", str(out)])
    report = json.loads(out.read_text())
    assert code == 2 and report["comparable"] is False
    assert "missing planned original: P:manual:1" in report["reasons"]
    assert (out.stat().st_mode & 0o777) == 0o600
    assert (results_path.stat().st_mode & 0o777) == 0o644
    assert (tmp_path.stat().st_mode & 0o777) == 0o755


def test_malformed_challenge_input_invalidates_instead_of_crashing():
    valid = _jobs(variants=("O", "F"), reps=(1,))
    bound = _bind(valid, [_run(job["request_id"], "validated", [("R1", 1)]) for job in valid])
    report = challenge_report({"expected_runs": valid, "runs": {"kept": bound[0]}}, rubric_items=R)
    assert report["comparable"] is False
    assert "runs must be a list" in report["reasons"]
    assert report["variants"]["F"]["pairs"] == 0
    assert report["variants"]["F"]["item_agreement"] is None

    jobs = [*valid, {"case_id": "P:manual:O", "problem_id": "P", "repetition": 1}]
    report = challenge_report({"expected_runs": jobs, "runs": bound}, rubric_items=R)
    assert report["comparable"] is False
    assert "malformed planned job: missing request_id" in report["reasons"]
    assert report["variants"]["O"]["scheduled"] == 1

    bad = ["not-a-key"]
    jobs = [
        {"request_id": bad, "case_id": "P:manual:O", "problem_id": "P", "repetition": 1},
        {"request_id": bad, "case_id": "P:manual:F", "problem_id": "P", "repetition": 1},
    ]
    runs = [_run(bad, "validated", [("R1", 1)]), _run(bad, "validated", [("R1", 0)])]
    report = challenge_report({"expected_runs": jobs, "runs": runs}, rubric_items=R)
    assert report["comparable"] is False
    assert report["reasons"].count(f"unhashable request_id: {bad!r}") == 4
    assert report["variants"].get("F", {}).get("pairs", 0) == 0
    assert report["variants"].get("F", {}).get("item_agreement") is None

    report = challenge_report(["not-an-object"], rubric_items=R)
    assert report["comparable"] is False
    assert report["reasons"] == ["results must be an object"]
    assert report["variants"] == {}

    report = challenge_report({"expected_runs": valid, "runs": ["not-a-record"]}, rubric_items=R)
    assert report["comparable"] is False
    assert "malformed run record" in report["reasons"]
    assert report["variants"]["F"]["item_agreement"] is None

    jobs = [
        {"request_id": "o", "case_id": None, "problem_id": "P", "repetition": 1},
        {"request_id": "f", "problem_id": "P", "repetition": ["unhashable"]},
    ]
    report = challenge_report({"expected_runs": jobs, "runs": []}, rubric_items=R)
    assert report["comparable"] is False
    assert "malformed planned job: o" in report["reasons"]
    assert "malformed planned job: f" in report["reasons"]
    assert report["variants"] == {}


@pytest.mark.parametrize("payload,reason", [
    ({"expected_runs": [
        {"request_id": "o", "case_id": "P:manual:O", "problem_id": "P", "repetition": 1},
        {"request_id": "f", "case_id": "P:manual:F", "problem_id": "P", "repetition": 1},
    ], "runs": {"o": {"request_id": "o"}}}, "runs must be a list"),
    ({"expected_runs": [
        {"case_id": "P:manual:O", "problem_id": "P", "repetition": 1},
        {"request_id": "f", "case_id": "P:manual:F", "problem_id": "P", "repetition": 1},
    ], "runs": []}, "malformed planned job: missing request_id"),
    ({"expected_runs": [
        {"request_id": ["k"], "case_id": "P:manual:O", "problem_id": "P", "repetition": 1},
        {"request_id": "f", "case_id": "P:manual:F", "problem_id": "P", "repetition": 1},
    ], "runs": [{"request_id": {"k": 1}}]}, "unhashable request_id:"),
])
def test_malformed_input_cli_exits_2_without_changing_inputs(tmp_path, monkeypatch, payload, reason):
    results_path = tmp_path / "results.json"
    out = tmp_path / "out" / "report.json"
    tmp_path.chmod(0o755)
    results_path.write_text(json.dumps(payload))
    results_path.chmod(0o644)
    monkeypatch.setattr(challenge, "_rubric_items", lambda manifest: R)
    code = challenge.main(["--results", str(results_path), "--dataset", "unused", "--out", str(out)])
    report = json.loads(out.read_text())
    assert code == 2 and report["comparable"] is False
    assert any(item.startswith(reason) or item == reason for item in report["reasons"])
    assert report["variants"].get("F", {}).get("item_agreement") is None
    assert (out.stat().st_mode & 0o777) == 0o600
    assert (out.parent.stat().st_mode & 0o777) == 0o700
    assert (results_path.stat().st_mode & 0o777) == 0o644
    assert (tmp_path.stat().st_mode & 0o777) == 0o755
