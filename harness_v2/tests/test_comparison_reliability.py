"""A zero process exit is not evidence that either grader produced a grade."""
import copy
import json

import pytest

from scripts.compare_old_vs_new import compare, main


def case():
    return {"request_id": "r", "problem_text": "solve", "answer_text": "x",
            "rubric": [{"item_id": "a", "max_points": 3}],
            "legacy_completion": json.dumps({"score": 2, "max_score": 3, "rubric_scores": [
                {"item_id": "a", "score": 2, "max_score": 3, "reason": "shown", "evidence": "x"}]}),
            "new_completion": json.dumps({"items": [
                {"item_id": "a", "score": 2, "evidence_spans": [{"text": "x"}], "math_reason": "shown"}],
                "total_score": 2})}


@pytest.mark.parametrize("side,completion", [
    ("old", None), ("new", None), ("old", "not json"), ("new", "not json"),
    ("old", "{}"), ("new", "{}"),
    ("old", '{"score": 2}'),
    ("new", '{"items": [], "total_score": 0}'),
    ("old", '{"score": NaN, "max_score": 3, "rubric_scores": []}'),
])
def test_zero_exit_without_valid_grades_is_incomplete(tmp_path, side, completion):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    data = case()
    data[{"old": "legacy_completion", "new": "new_completion"}[side]] = completion
    (inputs / "case.json").write_text(json.dumps(data))
    out = tmp_path / "out"
    assert main(["--inputs", str(inputs), "--out", str(out)]) == 1
    summary = json.loads((out / "summary.json").read_text())
    row = summary["rows"][0]
    assert row["old_exit"] == row["new_exit"] == 0
    assert row["success"] is False
    assert row["status"] == "incomplete"
    assert side in row["reason"]
    assert summary["success_count"] == 0
    assert summary["incomplete_count"] == 1
    assert summary["success_rate"] == 0


def test_both_valid_results_are_required_for_success(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "case.json").write_text(json.dumps(case()))
    summary = compare(inputs, tmp_path / "out")
    assert summary["rows"][0]["success"] is True
    assert summary["rows"][0]["reason"] == ""
    assert summary["success_count"] == 1
    assert summary["success_rate"] == 1


@pytest.mark.parametrize("side,mutation", [
    ("old", "missing"), ("new", "missing"), ("old", "malformed"), ("new", "malformed"),
    ("old", "null"), ("new", "null"), ("old", "boolean_score"), ("new", "boolean_score"),
    ("old", "range"), ("new", "range"), ("old", "sum"), ("new", "sum"),
    ("old", "missing_item"), ("new", "missing_item"), ("new", "review_required"),
])
def test_evaluation_rejects_missing_or_invalid_side(side, mutation):
    from harness_v2.evaluation.comparison import evaluate_comparison, summarize_comparisons

    data = case()
    old = {"status": "parsed", "data": json.loads(data["legacy_completion"])}
    new = {"status": "ran", "run": {"outputs": {"grading": {
        "status": "validated", **json.loads(data["new_completion"])}}}}
    results = {"old": old, "new": new}
    if mutation == "missing":
        results[side] = None
    elif mutation == "malformed":
        results[side] = []
    elif mutation == "null":
        results[side] = {"status": "parsed", "data": None} if side == "old" else {"status": "ran", "run": None}
    else:
        grade = old["data"] if side == "old" else new["run"]["outputs"]["grading"]
        rows = grade["rubric_scores" if side == "old" else "items"]
        if mutation == "boolean_score":
            rows[0]["score"] = True
        elif mutation == "range":
            rows[0]["score"] = 4
        elif mutation == "sum":
            grade["score" if side == "old" else "total_score"] = 1
        elif mutation == "missing_item":
            rows.clear()
        elif mutation == "review_required":
            grade["status"] = "review_required"
    result = evaluate_comparison(results["old"], results["new"], data)
    assert result["success"] is False
    assert side in result["reason"]
    summary = summarize_comparisons([result, copy.deepcopy(result)])
    assert summary["success_count"] == 0
    assert summary["incomplete_count"] == 2
    assert summary["success_rate"] == 0


def test_missing_or_malformed_result_files_are_reported(tmp_path, monkeypatch):
    import scripts.compare_old_vs_new as script
    from types import SimpleNamespace

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "case.json").write_text(json.dumps(case()))

    def fake_process(command, **kwargs):
        output = tmp_path / "out" / "case" / "new"
        output.mkdir(parents=True, exist_ok=True)
        (output / "result.json").write_text("broken json")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(script.subprocess, "run", fake_process)
    summary = compare(inputs, tmp_path / "out")
    assert summary["success_count"] == 0
    reason = summary["rows"][0]["reason"]
    assert "old" in reason and "missing" in reason
    assert "new" in reason and "invalid" in reason


def test_empty_benchmark_has_no_success_rate_and_cli_fails(tmp_path):
    from harness_v2.evaluation.comparison import summarize_comparisons
    assert summarize_comparisons([])["success_rate"] is None
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    assert main(["--inputs", str(inputs), "--out", str(tmp_path / "out")]) == 1


@pytest.mark.parametrize("side,mutation", [
    ("old", "total_alias"), ("old", "items_alias"), ("old", "item_max_alias"),
    ("new", "total_alias"), ("new", "items_alias"),
])
def test_schema_aliases_cannot_hide_invalid_results(side, mutation):
    from harness_v2.evaluation.comparison import evaluate_comparison
    data = case()
    old = {"status": "parsed", "data": json.loads(data["legacy_completion"])}
    new = {"status": "ran", "run": {"outputs": {"grading": {
        "status": "validated", **json.loads(data["new_completion"])}}}}
    grade = old["data"] if side == "old" else new["run"]["outputs"]["grading"]
    if mutation == "total_alias":
        if side == "old":
            grade["total_score"], grade["score"] = 2, None
        else:
            grade["score"] = grade.pop("total_score")
    elif mutation == "items_alias":
        if side == "old":
            grade["items"], grade["rubric_scores"] = grade["rubric_scores"], []
        else:
            grade["rubric_scores"] = grade.pop("items")
    else:
        grade["rubric_scores"][0]["max_points"] = 3
        grade["rubric_scores"][0]["max_score"] = -1
    result = evaluate_comparison(old, new, data)
    assert result["success"] is False
    assert side in result["reason"]


@pytest.mark.parametrize("mutation", ["missing_max", "reason_alias", "evidence_alias"])
def test_legacy_required_item_fields_cannot_be_replaced_by_new_fields(mutation):
    from harness_v2.evaluation.comparison import evaluate_comparison
    data = case()
    grade = json.loads(data["legacy_completion"])
    row = grade["rubric_scores"][0]
    if mutation == "missing_max":
        del row["max_score"]
    elif mutation == "reason_alias":
        row["math_reason"] = row.pop("reason")
    else:
        row["evidence_spans"] = [{"text": row.pop("evidence")}]
    new = {"status": "ran", "run": {"outputs": {"grading": {
        "status": "validated", **json.loads(data["new_completion"])}}}}
    result = evaluate_comparison({"status": "parsed", "data": grade}, new, data)
    assert result["success"] is False
    assert "old" in result["reason"]


def test_comparison_summary_permissions_on_create_and_rewrite(tmp_path):
    import os
    import stat
    previous = os.umask(0o000)
    try:
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        (inputs / "case.json").write_text(json.dumps(case()))
        out = tmp_path / "out"
        compare(inputs, out)
        path = out / "summary.json"
        assert stat.S_IMODE(os.stat(out).st_mode) == 0o700
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
        path.chmod(0o666)
        out.chmod(0o777)
        # Empty rerun avoids reusing the immutable grading artifacts.
        (inputs / "case.json").unlink()
        compare(inputs, out)
        assert stat.S_IMODE(os.stat(out).st_mode) == 0o700
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    finally:
        os.umask(previous)


@pytest.mark.parametrize("flag,value", [("review_reason", "teacher verification required"), ("uncertainty", 0.9)])
def test_legacy_item_review_flags_survive_normalization(flag, value):
    from harness_v2.evaluation.comparison import evaluate_comparison
    data = case()
    grade = json.loads(data["legacy_completion"])
    grade["rubric_scores"][0][flag] = value
    new = {"status": "ran", "run": {"outputs": {"grading": {
        "status": "validated", **json.loads(data["new_completion"])}}}}
    result = evaluate_comparison({"status": "parsed", "data": grade}, new, data)
    assert result["success"] is False
    assert "old" in result["reason"]
