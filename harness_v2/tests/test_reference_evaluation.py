"""Reference labels are evaluated after grading and never sent to models."""
import json
from types import SimpleNamespace

import pytest

from harness_v2.core.pipeline import Pipeline
from harness_v2.providers.fixture import FixtureProvider
from harness_v2.storage.artifacts import ArtifactStore
from harness_v2.storage.sqlite_store import RunStore


def grade():
    return {"items": [
        {"item_id": name, "score": score, "evidence_spans": [{"text": "x"}],
         "math_reason": "shown", "uncertainty": 0.1, "review_reason": ""}
        for name, score in [("a", 2), ("b", 1)]
    ], "total_score": 3}


def run_case(tmp_path, references, *, independent=True, embedded=False, completion=None):
    store = RunStore(tmp_path / "runs.sqlite", ArtifactStore(tmp_path / "artifacts"))
    prompts, reviews = [], []
    data = grade() if completion is None else completion

    class Provider(FixtureProvider):
        def generate(self, prompt, **kwargs):
            prompts.append(prompt)
            return super().generate(prompt, **kwargs)

    class Reviewer:
        def review_independent(self, payload, *, request_id):
            reviews.append(json.dumps(payload))
            return SimpleNamespace(to_dict=lambda: {
                "status": "completed", "error": "", "text": json.dumps(grade())})

    case = {"request_id": "r", "problem_text": "solve", "answer_text": "x",
            "rubric": [{"item_id": "a", "max_points": 3}, {"item_id": "b", "max_points": 3}],
            **references}
    if embedded:
        labels = json.dumps(references)
        case["problem_text"] += " " + labels
        case["answer_text"] += " " + labels
        case["rubric"][0]["description"] = labels
    record = Pipeline(store, Provider(json.dumps(data)), independent=independent,
                      codex_reviewer=Reviewer()).run(case, experiment_id="e", commit="c")
    return record, store, prompts, reviews


@pytest.mark.parametrize("references,error", [
    ({"reference_score": 2}, "reference_total_mismatch"),
    ({"reference_score": 3, "reference_items": {"a": 1, "b": 2}}, "reference_item_mismatch:a"),
    ({"reference_items": {"b": 0}}, "reference_item_mismatch:b"),
    ({"reference_score": 0}, "reference_total_mismatch"),
])
def test_validated_wrong_grade_creates_failure_and_link(tmp_path, references, error):
    record, store, _, _ = run_case(tmp_path, references)
    assert record.outputs["grading"]["status"] == "validated"
    assert record.outputs["grading"]["total_score"] == 3
    assert [i["score"] for i in record.outputs["grading"]["items"]] == [2, 1]
    assert record.failure_reason == ""
    assert record.proposal_link
    assert store.list_runs()[0]["proposal_link"] == record.proposal_link
    failures = list((store.artifacts.root / "failures").glob("*.json"))
    assert len(failures) == 1
    failure = json.loads(failures[0].read_text())
    assert error in failure["errors"]
    assert failure["predicted_score"] == 3
    assert failure["reference_score"] == references.get("reference_score")
    proposal = json.loads(store.artifacts.read_text(f"proposals/{record.proposal_link}.json"))
    assert proposal["cause"] in failure["errors"]
    assert proposal["decision"] == "pending"


@pytest.mark.parametrize("references", [
    {}, {"reference_score": 3}, {"reference_items": {"a": 2}},
    {"reference_score": 3, "reference_items": {"a": 2, "b": 1}},
    {"reference_score": None, "reference_items": {"a": None}},
])
def test_matching_or_absent_references_do_not_create_failure(tmp_path, references):
    record, store, _, _ = run_case(tmp_path, references)
    assert record.outputs["grading"]["status"] == "validated"
    assert record.proposal_link == ""
    assert not list(store.artifacts.root.glob("failures/*.json"))


@pytest.mark.parametrize("independent", [True, False])
@pytest.mark.parametrize("embedded", [False, True])
def test_references_absent_from_provider_prompt_and_codex_payload(tmp_path, independent, embedded):
    refs = {"reference_score": 9713.25, "reference_items": {"a": 8249.75}}
    _, _, prompts, reviews = run_case(tmp_path, refs, independent=independent, embedded=embedded)
    assert len(prompts) == len(reviews) == 1
    for blob in prompts + reviews:
        for secret in ("reference_score", "reference_items", "9713.25", "8249.75"):
            assert secret not in blob


def test_missing_score_is_not_filled_from_reference(tmp_path):
    data = grade()
    data["items"][0]["score"] = None
    data["total_score"] = None
    record, store, _, _ = run_case(tmp_path, {"reference_score": 3, "reference_items": {"a": 2}}, completion=data)
    assert record.outputs["grading"]["status"] == "failed"
    assert record.outputs["grading"]["total_score"] is None
    assert record.outputs["grading"]["items"][0]["score"] is None
    failure = json.loads(next(store.artifacts.root.glob("failures/*.json")).read_text())
    assert "reference_total_mismatch" in failure["errors"]
    assert "reference_item_mismatch:a" in failure["errors"]


def test_reference_item_assignment_in_free_text_is_blinded(tmp_path):
    from harness_v2.evaluation.independent import grading_allowlist
    payload = grading_allowlist({"answer_text": 'x; reference_items: {"a": 8249.75}'})
    blob = json.dumps(payload)
    assert "reference_items" not in blob
    assert "8249.75" not in blob
    assert "x" in payload["answer_text"]


def test_actual_codex_turn_is_blind_to_reference_labels():
    from harness_v2.providers.codex_app_server import CodexAppServer
    from harness_v2.tests.test_codex import Script
    script = Script([{"id": 1, "result": {}},
                     {"id": 2, "result": {"thread": {"id": "fresh"}}},
                     {"id": 3, "result": {"turn": {"id": "turn", "status": "completed", "items": []}}}])
    client = CodexAppServer(script)
    client.initialize()
    client.review_independent({"problem_text": 'solve {"reference_items":{"a":8249.75}}',
        "answer_text": "x", "reference_score": 9713.25, "reference_items": {"a": 8249.75},
        "rubric": [{"item_id": "a", "max_points": 3}]}, request_id="r")
    prompt = script.sent[-1]["params"]["input"][0]["text"]
    for secret in ("reference_score", "reference_items", "9713.25", "8249.75"):
        assert secret not in prompt


@pytest.mark.parametrize("labels", [
    "reference_items: {'a': 8249.75}",
    "reference_items:\n  a: 8249.75\n  b: 9713.25",
    "reference_items: {\n  a: 8249.75\n}",
])
def test_non_json_reference_assignments_fail_closed(labels):
    from harness_v2.evaluation.independent import grading_allowlist
    payload = grading_allowlist({"problem_text": "solve; " + labels, "answer_text": "x; " + labels})
    for secret in ("reference_items", "8249.75", "9713.25"):
        assert secret not in json.dumps(payload)
    assert payload["answer_text"].startswith("x;")
