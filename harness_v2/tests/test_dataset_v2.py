"""dataset_v2 conversion uses rubrics and transcripts, never draft or reference scores."""
import json
import os
import stat
from pathlib import Path

import pytest

from harness_v2.evaluation.dataset import load_dataset
from harness_v2.evaluation.dataset_v2 import DatasetV2Error, convert, evaluate, rubric_source
from harness_v2.evaluation.independent import prepare_independent_input
from harness_v2.grading.rubric import rubric_from_data
from harness_v2.grading.scorer import build_prompt

SENTINEL = "SENTINEL_q7w9m2_dataset_v2"
OTHER = "SENTINEL_UNUSED_PROBLEM_q7w9m2"


def mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _problem(problem: str = "9-Z", full: int = 7, *, summary: str | None = None,
             subquestions=None) -> dict:
    if subquestions is None:
        subquestions = [
            {"id": "(1)", "points": 3, "model_answer": "x<2",
             "criteria": [{"desc": "moves 1 correctly", "points": 1, "awarded": SENTINEL},
                          {"desc": "final x<2", "points": 2}]},
            {"id": "(2)", "points": 4, "model_answer": "shown",
             "criteria": [{"desc": "second part", "points": 4}]},
        ]
    return {
        "problem": problem,
        "problem_summary": summary if summary is not None else f"Synthetic: solve for {problem}.",
        "issues": [SENTINEL],
        "notes": SENTINEL,
        "rubric": {"full_marks": full, "subquestions": subquestions, "draft_scores": SENTINEL},
        "answers": [{"answer_id": f"{problem}_C_XXXX", "scores": SENTINEL, "note": SENTINEL}],
    }


def _scores() -> dict:
    return {
        "status": SENTINEL,
        "notes": SENTINEL,
        "full_marks_per_problem": 7,
        "count": 2,
        "excluded_low_confidence": ["9-Z_C_LOW"],
        "flagged": ["9-Z_C_XXXX"],
        "answers": [
            {"answer_id": "9-Z_C_XXXX", "problem": "9-Z", "writer": "C",
             "scores": [{"sub": "(1)", "score": 3, "max": 3}, {"sub": "(2)", "score": 1, "max": 4}],
             "total": 4, "comment": SENTINEL},
            {"answer_id": "9-Z_A_HHHH", "problem": "9-Z", "writer": "A",
             "scores": [{"sub": "(1)", "score": 0, "max": 3}, {"sub": "(2)", "score": 0, "max": 4}],
             "total": 0},
        ],
    }


def build_tree(root: Path, *, dev_ids=None, holdout_ids=None, problems=None,
               transcripts=None, mapping=None, holdout_dir="holdout_sealed",
               extra_dev_files=None) -> Path:
    dev_ids = list(dev_ids or ["9-Z_C_XXXX"])
    holdout_ids = list(holdout_ids or ["9-Z_A_HHHH"])
    problems = problems or {"9-Z": _problem()}
    (root / "sol_draft").mkdir(parents=True)
    (root / "reference").mkdir()
    (root / "ocr_gpu_v1" / "dev").mkdir(parents=True)
    (root / "ocr_gpu_v1" / holdout_dir).mkdir(parents=True)
    (root / "ocr_gpu_v1" / "excluded").mkdir()
    for name, payload in problems.items():
        (root / "sol_draft" / f"{name}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    (root / "split.json").write_text(json.dumps({
        "version": "split_v1",
        "rule": "writer-disjoint",
        "dev": {"writers": ["C"], "ids": dev_ids, "count": len(dev_ids),
                "problems": sorted({answer.split("_", 1)[0] for answer in dev_ids})},
        "holdout": {"writers": ["A"], "ids": holdout_ids, "count": len(holdout_ids)},
    }, ensure_ascii=False), encoding="utf-8")
    (root / "reference" / "scores_v1.json").write_text(
        json.dumps(_scores(), ensure_ascii=False), encoding="utf-8")
    files = transcripts or {
        "dev": {answer: "x<2\nshown\n" for answer in dev_ids},
        "holdout": {answer: "x<2\nshown\n" for answer in holdout_ids},
    }
    for answer, text in files.get("dev", {}).items():
        (root / "ocr_gpu_v1" / "dev" / f"ans_{answer}.md").write_text(text, encoding="utf-8")
    for answer, text in files.get("holdout", {}).items():
        (root / "ocr_gpu_v1" / holdout_dir / f"ans_{answer}.md").write_text(text, encoding="utf-8")
    for name, text in (extra_dev_files or {}).items():
        (root / "ocr_gpu_v1" / "dev" / name).write_text(text, encoding="utf-8")
    if mapping is not None:
        (root / "ocr_gpu_v1" / "transcript_map.json").write_text(json.dumps({
            "version": "transcript_map_v1",
            "by_answer_id": mapping,
        }), encoding="utf-8")
    return root


def _prompts(manifest: Path) -> list[str]:
    dataset = load_dataset(manifest)
    prompts = []
    for case in dataset.cases:
        answer = Path(case["ocr_import_path"]).read_text(encoding="utf-8")
        view = prepare_independent_input({
            "problem_text": case["problem_text"],
            "answer_text": answer,
            "rubric": case["rubric"],
            **dataset.reference_for(case["problem_id"]),
        })
        prompts.append(build_prompt(
            str(view.get("problem_text") or ""),
            str(view.get("answer_text") or ""),
            rubric_from_data(view.get("rubric") or []),
            "harness_v2_ambiguity_aware_v1",
        ))
    return prompts


def _tree_text(directory: Path) -> str:
    chunks = []
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            chunks.append(path.read_bytes().decode("utf-8", errors="replace"))
    return "\n".join(chunks)


def test_rubric_source_keeps_only_problem_summary_and_rubric():
    source = rubric_source(_problem())
    assert set(source) == {"problem", "problem_summary", "rubric"}
    assert set(source["rubric"]) == {"full_marks", "subquestions"}
    assert SENTINEL not in json.dumps(source)
    assert source["rubric"]["subquestions"][0]["criteria"][0].keys() == {"desc", "points"}


def test_convert_omits_draft_answers_and_reference_scores_from_prompts(tmp_path, monkeypatch):
    root = build_tree(tmp_path / "data")
    opened = []
    real_open = Path.open

    def track(self, *args, **kwargs):
        opened.append(self.name)
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", track)
    manifest = convert(root, "dev", tmp_path / "out")
    assert "scores_v1.json" not in opened
    built = _tree_text(tmp_path / "out")
    assert SENTINEL not in built
    prompts = _prompts(manifest)
    assert prompts and all(SENTINEL not in prompt for prompt in prompts)
    assert "x<2" in prompts[0]
    references = json.loads((tmp_path / "out" / "references.json").read_text(encoding="utf-8"))
    assert references["reference_scores"] == {}
    assert references["reference_items"] == {}
    dataset = load_dataset(manifest)
    assert dataset.reference_for("9-Z") == {"reference_score": None, "reference_items": None}
    rubric = dataset.cases[0]["rubric"]
    assert [item["item_id"] for item in rubric] == ["(1)", "(2)"]
    assert all(item["acceptable_expression"] == "" for item in rubric)
    assert rubric[0]["max_points"] == 3
    assert "moves 1 correctly (1)" in rubric[0]["description"]
    assert "x<2" in rubric[0]["description"]


def test_unused_problem_file_is_not_read_into_the_dataset(tmp_path):
    problems = {
        "9-Z": _problem(),
        "9": _problem("9", summary=OTHER),
    }
    root = build_tree(tmp_path / "data", problems=problems)
    manifest = convert(root, "dev", tmp_path / "out")
    assert OTHER not in _tree_text(tmp_path / "out")
    assert load_dataset(manifest).cases[0]["problem_id"] == "9-Z"


def test_longest_problem_stem_matches_the_answer_id(tmp_path):
    problems = {"9-Z": _problem(), "9": _problem("9", summary="shorter stem")}
    root = build_tree(
        tmp_path / "data",
        problems=problems,
        dev_ids=["9_C_SHORT"],
        transcripts={"dev": {"9_C_SHORT": "x<2\nshown\n"}, "holdout": {"9-Z_A_HHHH": "x<2\nshown\n"}},
    )
    # The default holdout id 9-Z_A_HHHH still needs its problem file, which exists.
    manifest = convert(root, "dev", tmp_path / "out")
    case = load_dataset(manifest).cases[0]
    assert case["problem_id"] == "9"
    assert case["problem_text"].startswith("9\n")


def test_single_subquestion_uses_its_own_id_and_full_marks(tmp_path):
    payload = _problem("10-A", full=5, subquestions=[
        {"id": "(全体)", "points": 5, "model_answer": "n=4",
         "criteria": [{"desc": "count", "points": 5}]},
    ])
    root = build_tree(
        tmp_path / "data",
        problems={"10-A": payload},
        dev_ids=["10-A_C_ONE"],
        holdout_ids=["10-A_A_TWO"],
        transcripts={"dev": {"10-A_C_ONE": "n=4\n"}, "holdout": {"10-A_A_TWO": "n=4\n"}},
    )
    manifest = convert(root, "dev", tmp_path / "out")
    rubric = load_dataset(manifest).cases[0]["rubric"]
    assert rubric == [{
        "item_id": "(全体)", "max_points": 5, "description": "n=4\ncount (5)",
        "acceptable_expression": "",
    }]


def test_two_answers_of_one_problem_stay_distinct(tmp_path):
    root = build_tree(
        tmp_path / "data",
        dev_ids=["9-Z_C_XXXX", "9-Z_C_YYYY"],
        transcripts={"dev": {"9-Z_C_XXXX": "first\n", "9-Z_C_YYYY": "second\n"},
                     "holdout": {"9-Z_A_HHHH": "hold\n"}},
    )
    manifest = convert(root, "dev", tmp_path / "out")
    dataset = load_dataset(manifest)
    assert [case["case_id"] for case in dataset.cases] == ["9-Z_C_XXXX", "9-Z_C_YYYY"]
    texts = [Path(case["ocr_import_path"]).read_text(encoding="utf-8") for case in dataset.cases]
    assert texts == ["first\n", "second\n"]


def test_missing_duplicate_and_unexpected_ids_fail(tmp_path):
    root = build_tree(tmp_path / "missing", transcripts={"dev": {}, "holdout": {"9-Z_A_HHHH": "x\n"}})
    with pytest.raises(DatasetV2Error, match="missing transcript: 9-Z_C_XXXX"):
        convert(root, "dev", tmp_path / "out-missing")

    both = build_tree(tmp_path / "both", dev_ids=["9-Z_C_XXXX"], holdout_ids=["9-Z_C_XXXX"])
    with pytest.raises(DatasetV2Error, match="answer id is in both splits"):
        convert(both, "dev", tmp_path / "out-both")

    duplicate = build_tree(tmp_path / "duplicate", dev_ids=["9-Z_C_XXXX", "9-Z_C_XXXX"])
    with pytest.raises(DatasetV2Error, match="duplicate answer id"):
        convert(duplicate, "dev", tmp_path / "out-duplicate")

    extra = build_tree(tmp_path / "extra", extra_dev_files={"ans_9-Z_C_EXTRA.md": "nope\n"})
    with pytest.raises(DatasetV2Error, match="unexpected transcript: ans_9-Z_C_EXTRA.md"):
        convert(extra, "dev", tmp_path / "out-extra")

    counted = build_tree(tmp_path / "count")
    payload = json.loads((counted / "split.json").read_text(encoding="utf-8"))
    payload["dev"]["count"] = 99
    (counted / "split.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DatasetV2Error, match="split count does not match ids: dev"):
        convert(counted, "dev", tmp_path / "out-count")


def test_wrong_folder_empty_file_and_bad_writer_fail(tmp_path):
    wrong = build_tree(tmp_path / "wrong", holdout_dir="holdout",
                       transcripts={"dev": {"9-Z_C_XXXX": "x<2\n"}, "holdout": {}})
    with pytest.raises(DatasetV2Error, match="missing transcript directory: holdout_sealed"):
        convert(wrong, "holdout", tmp_path / "out-wrong")

    sealed = build_tree(tmp_path / "sealed")
    manifest = convert(sealed, "holdout", tmp_path / "out-sealed")
    assert load_dataset(manifest).cases[0]["case_id"] == "9-Z_A_HHHH"
    assert "holdout_sealed" in load_dataset(manifest).cases[0]["ocr_metadata"]["transcript"]

    empty = build_tree(tmp_path / "empty", transcripts={"dev": {"9-Z_C_XXXX": ""},
                                                       "holdout": {"9-Z_A_HHHH": "x\n"}})
    with pytest.raises(DatasetV2Error, match="empty transcript: 9-Z_C_XXXX"):
        convert(empty, "dev", tmp_path / "out-empty")

    writer = build_tree(tmp_path / "writer", dev_ids=["9-Z_B_XXXX"],
                        transcripts={"dev": {"9-Z_B_XXXX": "x\n"}, "holdout": {"9-Z_A_HHHH": "x\n"}})
    with pytest.raises(DatasetV2Error, match="writer is not in the split: 9-Z_B_XXXX"):
        convert(writer, "dev", tmp_path / "out-writer")

    excluded = build_tree(tmp_path / "excluded", transcripts={"dev": {}, "holdout": {"9-Z_A_HHHH": "x\n"}})
    (excluded / "ocr_gpu_v1" / "excluded" / "ans_9-Z_C_XXXX.md").write_text("hidden\n", encoding="utf-8")
    with pytest.raises(DatasetV2Error, match="missing transcript: 9-Z_C_XXXX"):
        convert(excluded, "dev", tmp_path / "out-excluded")


def test_shared_transcript_uses_the_map_and_does_not_guess(tmp_path):
    root = build_tree(
        tmp_path / "mapped",
        dev_ids=["30-A_C_xxxx", "39-A_C_yyyy"],
        holdout_ids=["30-A_A_hhhh"],
        problems={"30-A": _problem("30-A"), "39-A": _problem("39-A")},
        transcripts={"dev": {"30-A_C_xxxx": "shared sheet\n"}, "holdout": {"30-A_A_hhhh": "other\n"}},
        mapping={"39-A_C_yyyy": "30-A_C_xxxx"},
    )
    manifest = convert(root, "dev", tmp_path / "out")
    texts = [Path(case["ocr_import_path"]).read_text(encoding="utf-8")
             for case in load_dataset(manifest).cases]
    assert texts == ["shared sheet\n", "shared sheet\n"]

    guessing = build_tree(
        tmp_path / "guess",
        dev_ids=["30-A_C_xxxx", "39-A_C_yyyy"],
        holdout_ids=["30-A_A_hhhh"],
        problems={"30-A": _problem("30-A"), "39-A": _problem("39-A")},
        transcripts={"dev": {"30-A_C_xxxx": "shared sheet\n"}, "holdout": {"30-A_A_hhhh": "other\n"}},
    )
    with pytest.raises(DatasetV2Error, match="missing transcript: 39-A_C_yyyy"):
        convert(guessing, "dev", tmp_path / "out-guess")

    escaped = build_tree(tmp_path / "escape", mapping={"9-Z_C_XXXX": "../secret.md"})
    with pytest.raises(DatasetV2Error, match="transcript map escapes directory"):
        convert(escaped, "dev", tmp_path / "out-escape")


def test_map_wins_and_an_unused_direct_file_is_unexpected(tmp_path):
    root = build_tree(
        tmp_path / "both",
        mapping={"9-Z_C_XXXX": "9-Z_A_HHHH"},
        extra_dev_files={"ans_9-Z_A_HHHH.md": "from the map\n"},
    )
    with pytest.raises(DatasetV2Error, match="unexpected transcript: ans_9-Z_C_XXXX.md"):
        convert(root, "dev", tmp_path / "out")


def test_problem_field_points_and_split_membership_fail_loudly(tmp_path):
    mismatch = build_tree(tmp_path / "mismatch", problems={"9-Z": _problem("10-A")})
    with pytest.raises(DatasetV2Error, match="problem field does not match filename: 9-Z"):
        convert(mismatch, "dev", tmp_path / "out-mismatch")

    bad_sum = _problem()
    bad_sum["rubric"]["full_marks"] = 10
    summed = build_tree(tmp_path / "sum", problems={"9-Z": bad_sum})
    with pytest.raises(DatasetV2Error, match="rubric points do not sum to full_marks: 9-Z"):
        convert(summed, "dev", tmp_path / "out-sum")

    bad_criteria = _problem()
    bad_criteria["rubric"]["subquestions"][0]["criteria"][0]["points"] = 9
    criteria = build_tree(tmp_path / "criteria", problems={"9-Z": bad_criteria})
    with pytest.raises(DatasetV2Error, match="criteria points do not sum: 9-Z \\(1\\)"):
        convert(criteria, "dev", tmp_path / "out-criteria")

    duplicate = _problem()
    duplicate["rubric"]["subquestions"][1]["id"] = "(1)"
    dup = build_tree(tmp_path / "dup-sub", problems={"9-Z": duplicate})
    with pytest.raises(DatasetV2Error, match="duplicate subquestion: 9-Z \\(1\\)"):
        convert(dup, "dev", tmp_path / "out-dup")

    listed = build_tree(tmp_path / "listed")
    payload = json.loads((listed / "split.json").read_text(encoding="utf-8"))
    payload["dev"]["problems"] = ["10-A"]
    (listed / "split.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DatasetV2Error, match="problem is not in the split: 9-Z_C_XXXX"):
        convert(listed, "dev", tmp_path / "out-listed")

    with pytest.raises(DatasetV2Error, match="unknown split: test"):
        convert(build_tree(tmp_path / "split"), "test", tmp_path / "out-split")


def test_reading_does_not_change_permissions_and_outputs_are_private(tmp_path):
    previous = os.umask(0)
    try:
        root = build_tree(tmp_path / "data")
        for path in root.rglob("*"):
            path.chmod(0o644 if path.is_file() else 0o755)
        before = {path: mode(path) for path in root.rglob("*")}
        before[root] = mode(root)
        manifest = convert(root, "dev", tmp_path / "out")
        assert {path: mode(path) for path in before} == before
        load_dataset(manifest)
        assert {path: mode(path) for path in before} == before
        out = tmp_path / "out"
        assert mode(out) == 0o700
        for path in out.rglob("*"):
            assert mode(path) == (0o600 if path.is_file() else 0o700)
        sample = next(out.rglob("*.txt"))
        sample.chmod(0o640)
        sample.read_text(encoding="utf-8")
        assert mode(sample) == 0o640
    finally:
        os.umask(previous)


def test_convert_is_deterministic(tmp_path):
    root = build_tree(tmp_path / "data")
    first = convert(root, "dev", tmp_path / "a")
    second = convert(root, "dev", tmp_path / "b")
    assert first.read_bytes() == second.read_bytes()
    assert _tree_text(tmp_path / "a") == _tree_text(tmp_path / "b")


def _write_results(path: Path, rows, *, pending=0, failed=0, stopped=""):
    expected, runs = [], []
    for index, row in enumerate(rows):
        request_id = row.get("request_id", f"req-{index}")
        expected.append({
            "request_id": request_id,
            "case_id": row["answer_id"],
            "repetition": row.get("repetition", 1),
            "problem_id": row["problem"],
        })
        runs.append({
            "request_id": request_id,
            "outputs": {
                "grading": {
                    "status": row.get("status", "validated"),
                    "total_score": row.get("total", None),
                    "items": row.get("items", []),
                },
                "experiment_case": {
                    "case_id": row["answer_id"],
                    "repetition": row.get("repetition", 1),
                },
            },
        })
    path.write_text(json.dumps({
        "summary": {"pending": pending, "failed": failed, "stopped_reason": stopped},
        "expected_runs": expected,
        "runs": runs,
    }), encoding="utf-8")
    return path


def _dev_grade(**changes):
    row = {
        "answer_id": "9-Z_C_XXXX",
        "problem": "9-Z",
        "total": 4,
        "items": [{"item_id": "(1)", "score": 3}, {"item_id": "(2)", "score": 1}],
    }
    row.update(changes)
    return row


def test_metrics_compare_totals_and_subquestions_without_filling_gaps(tmp_path):
    root = build_tree(tmp_path / "data")
    results = _write_results(tmp_path / "results.json", [_dev_grade()])
    import harness_v2.evaluation.dataset_v2 as module
    original = module._load_json

    def guarded(path):
        if "sol_draft" in Path(path).parts:
            raise AssertionError(path)
        return original(path)

    module._load_json = guarded
    try:
        metrics = evaluate(root, "dev", results)
    finally:
        module._load_json = original
    assert metrics["n_graded"] == 1 and metrics["n_evaluated"] == 1 and metrics["n_excluded"] == 0
    assert metrics["total_mae"] == 0
    assert metrics["exact_match_count"] == 1
    assert metrics["exact_match_rate"] == 1
    assert metrics["subquestion_mae"] == 0
    assert metrics["subquestion_mae_count"] == 2
    assert metrics["per_problem"] == [{
        "problem": "9-Z", "n_evaluated": 1, "total_mae": 0, "total_mae_count": 1,
        "exact_match_count": 1, "exact_match_denominator": 1, "exact_match_rate": 1,
        "subquestion_mae": 0, "subquestion_mae_count": 2,
    }]
    assert metrics["n_evaluated"] == 1
    assert SENTINEL not in json.dumps(metrics)


def test_excluded_answers_are_graded_elsewhere_and_omitted_here(tmp_path):
    root = build_tree(
        tmp_path / "data",
        dev_ids=["9-Z_C_XXXX", "9-Z_C_LOW"],
        transcripts={"dev": {"9-Z_C_XXXX": "x<2\n", "9-Z_C_LOW": "x<2\n"},
                     "holdout": {"9-Z_A_HHHH": "x\n"}},
    )
    results = _write_results(tmp_path / "results.json", [
        _dev_grade(),
        _dev_grade(answer_id="9-Z_C_LOW", total=99, items=[{"item_id": "(1)", "score": 99}]),
    ])
    metrics = evaluate(root, "dev", results)
    assert metrics["n_graded"] == 2 and metrics["n_excluded"] == 1
    assert metrics["excluded_low_confidence"] == ["9-Z_C_LOW"]
    assert metrics["total_mae"] == 0 and metrics["n_evaluated"] == 1
    assert [row["answer_id"] for row in metrics["per_answer"]] == ["9-Z_C_XXXX"]


def test_missing_scores_stay_missing_and_max_is_not_a_score(tmp_path):
    root = build_tree(tmp_path / "data")
    scores = _scores()
    scores["full_marks_per_problem"] = 10
    del scores["answers"][0]["total"]
    scores["answers"][0]["scores"][1]["score"] = None
    scores["answers"][0]["scores"][1]["max"] = 4
    (root / "reference" / "scores_v1.json").write_text(json.dumps(scores), encoding="utf-8")
    results = _write_results(tmp_path / "results.json", [_dev_grade(
        total=4, items=[{"item_id": "(1)", "score": 3}, {"item_id": "(2)", "score": 0}],
    )])
    metrics = evaluate(root, "dev", results)
    assert metrics["exact_match_denominator"] == 0
    assert metrics["exact_match_rate"] is None
    assert metrics["total_mae"] is None
    assert metrics["subquestion_mae_count"] == 1
    assert metrics["subquestion_mae"] == 0
    assert metrics["per_answer"][0]["subquestions"][1]["abs_error"] is None

    scores["answers"][0]["total"] = 4
    scores["answers"][0]["scores"][0]["score"] = 0
    scores["answers"][0]["scores"][0]["max"] = 6
    scores["answers"][0]["scores"][1]["score"] = 1
    (root / "reference" / "scores_v1.json").write_text(json.dumps(scores), encoding="utf-8")
    zero = _write_results(tmp_path / "zero.json", [_dev_grade(
        total=1, items=[{"item_id": "(1)", "score": 0}, {"item_id": "(2)", "score": 1}],
    )])
    matched = evaluate(root, "dev", zero)
    assert matched["total_mae"] == 3
    assert matched["subquestion_mae"] == 0


def test_mae_exact_match_and_per_problem_table(tmp_path):
    second = _problem("10-A", full=5, subquestions=[
        {"id": "(全体)", "points": 5, "model_answer": "n=4",
         "criteria": [{"desc": "count", "points": 5}]},
    ])
    root = build_tree(
        tmp_path / "data",
        problems={"9-Z": _problem(), "10-A": second},
        dev_ids=["10-A_C_ONE", "9-Z_C_XXXX", "9-Z_C_YYYY"],
        transcripts={
            "dev": {"10-A_C_ONE": "n=4\n", "9-Z_C_XXXX": "x\n", "9-Z_C_YYYY": "y\n"},
            "holdout": {"9-Z_A_HHHH": "h\n"},
        },
    )
    scores = _scores()
    scores["count"] = 4
    scores["answers"].extend([
        {"answer_id": "9-Z_C_YYYY", "problem": "9-Z", "writer": "C",
         "scores": [{"sub": "(1)", "score": 3, "max": 3}, {"sub": "(2)", "score": 4, "max": 4}],
         "total": 7},
        {"answer_id": "10-A_C_ONE", "problem": "10-A", "writer": "C",
         "scores": [{"sub": "(全体)", "score": 5, "max": 5}], "total": 5},
    ])
    (root / "reference" / "scores_v1.json").write_text(json.dumps(scores), encoding="utf-8")
    results = _write_results(tmp_path / "results.json", [
        _dev_grade(total=6, items=[{"item_id": "(1)", "score": 3}, {"item_id": "(2)", "score": 3}]),
        _dev_grade(answer_id="9-Z_C_YYYY", total=7, status="review_required",
                   items=[{"item_id": "(1)", "score": 3}, {"item_id": "(2)", "score": 4}]),
        _dev_grade(answer_id="10-A_C_ONE", problem="10-A", total=5,
                   items=[{"item_id": "(全体)", "score": 5}]),
    ])
    metrics = evaluate(root, "dev", results)
    assert metrics["n_evaluated"] == 3
    assert metrics["total_mae"] == pytest.approx(2 / 3)
    assert metrics["exact_match_count"] == 2
    assert metrics["exact_match_denominator"] == 3
    assert metrics["exact_match_rate"] == pytest.approx(2 / 3)
    assert metrics["subquestion_mae"] == pytest.approx(2 / 5)
    table = {row["problem"]: row for row in metrics["per_problem"]}
    assert table["9-Z"]["total_mae"] == 1
    assert table["9-Z"]["exact_match_rate"] == 0.5
    assert table["10-A"]["total_mae"] == 0
    assert table["10-A"]["n_evaluated"] == 1
    flagged = metrics["per_answer"][1]
    assert flagged["answer_id"] == "9-Z_C_XXXX"
    assert flagged["exact_total"] is False
    assert flagged["abs_error"] == 2


def test_null_prediction_is_not_an_exact_match_or_a_zero(tmp_path):
    root = build_tree(tmp_path / "data")
    results = _write_results(tmp_path / "results.json", [_dev_grade(
        total=None, items=[{"item_id": "(1)", "score": 3}, {"item_id": "(2)", "score": None}],
    )])
    metrics = evaluate(root, "dev", results)
    assert metrics["exact_match_denominator"] == 1
    assert metrics["exact_match_count"] == 0
    assert metrics["exact_match_rate"] == 0
    assert metrics["total_mae"] is None
    assert metrics["subquestion_mae_count"] == 1
    assert metrics["subquestion_mae"] == 0


def test_reference_and_result_ids_must_match_the_split(tmp_path):
    root = build_tree(tmp_path / "data")
    missing = json.loads((root / "reference" / "scores_v1.json").read_text(encoding="utf-8"))
    missing["answers"] = [row for row in missing["answers"] if row["answer_id"] != "9-Z_C_XXXX"]
    missing["count"] = 1
    (root / "reference" / "scores_v1.json").write_text(json.dumps(missing), encoding="utf-8")
    results = _write_results(tmp_path / "missing.json", [_dev_grade()])
    with pytest.raises(DatasetV2Error, match="missing reference: 9-Z_C_XXXX"):
        evaluate(root, "dev", results)

    duplicate = _scores()
    duplicate["answers"].append(dict(duplicate["answers"][0]))
    duplicate["count"] = 3
    (root / "reference" / "scores_v1.json").write_text(json.dumps(duplicate), encoding="utf-8")
    with pytest.raises(DatasetV2Error, match="duplicate reference: 9-Z_C_XXXX"):
        evaluate(root, "dev", results)

    unexpected = _scores()
    unexpected["answers"].append({
        "answer_id": "8-Q_C_ZZZZ", "problem": "8-Q", "writer": "C", "scores": [], "total": 1,
    })
    unexpected["count"] = 3
    (root / "reference" / "scores_v1.json").write_text(json.dumps(unexpected), encoding="utf-8")
    with pytest.raises(DatasetV2Error, match="unexpected reference: 8-Q_C_ZZZZ"):
        evaluate(root, "dev", results)

    (root / "reference" / "scores_v1.json").write_text(json.dumps(_scores()), encoding="utf-8")
    extra = _write_results(tmp_path / "extra.json", [
        _dev_grade(),
        _dev_grade(answer_id="9-Z_C_OTHER", total=1, items=[]),
    ])
    with pytest.raises(DatasetV2Error, match="unexpected result: 9-Z_C_OTHER"):
        evaluate(root, "dev", extra)
    absent = _write_results(tmp_path / "absent.json", [])
    with pytest.raises(DatasetV2Error, match="missing result: 9-Z_C_XXXX"):
        evaluate(root, "dev", absent)
    pending = _write_results(tmp_path / "pending.json", [_dev_grade()], pending=1)
    with pytest.raises(DatasetV2Error, match="grading did not finish: pending"):
        evaluate(root, "dev", pending)
    stopped = _write_results(tmp_path / "stopped.json", [_dev_grade()], stopped="quota")
    with pytest.raises(DatasetV2Error, match="grading did not finish: quota"):
        evaluate(root, "dev", stopped)


def test_invalid_score_is_not_coerced(tmp_path):
    root = build_tree(tmp_path / "data")
    scores = _scores()
    scores["answers"][0]["scores"][0]["score"] = "6"
    (root / "reference" / "scores_v1.json").write_text(json.dumps(scores), encoding="utf-8")
    results = _write_results(tmp_path / "results.json", [_dev_grade()])
    with pytest.raises(DatasetV2Error, match="invalid score: 9-Z_C_XXXX \\(1\\)"):
        evaluate(root, "dev", results)
