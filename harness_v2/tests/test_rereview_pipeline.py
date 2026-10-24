"""Finding 13 and all new findings: auditable, fail-closed review."""
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from harness_v2.core.pipeline import Pipeline
from harness_v2.providers.fixture import FixtureProvider
from harness_v2.recognition.ocr import XiaomiOCR
from harness_v2.recognition.run_ocr_cpu import MIN_FREE_BYTES, REVISION
from harness_v2.storage.artifacts import ArtifactStore
from harness_v2.storage.sqlite_store import RunStore


def grade():
    return {"items": [
        {"item_id": item_id, "score": score, "evidence_spans": [{"text": "x"}],
         "math_reason": "shown", "uncertainty": 0.1, "review_reason": ""}
        for item_id, score in [("a", 2), ("b", 1)]
    ], "total_score": 3}


def case(**updates):
    data = {"request_id": "r", "problem_text": "solve", "answer_text": "x",
            "rubric": [{"item_id": "a", "max_points": 3}, {"item_id": "b", "max_points": 3}]}
    data.update(updates)
    return data


def store_at(tmp_path):
    return RunStore(tmp_path / "runs.sqlite", ArtifactStore(tmp_path / "artifacts"))


class Reviewer:
    def __init__(self, *, status="completed", error="", data=None, exception=None):
        self.saved = {"status": status, "error": error, "text": json.dumps(data if data is not None else grade())}
        self.exception = exception
        self.seen = []

    def review_independent(self, payload, *, request_id):
        self.seen.append(copy.deepcopy(payload))
        if self.exception:
            raise self.exception
        return SimpleNamespace(to_dict=lambda: self.saved)


def run(tmp_path, reviewer):
    store = store_at(tmp_path)
    record = Pipeline(store, FixtureProvider(json.dumps(grade())), codex_reviewer=reviewer).run(
        case(), experiment_id="e", commit="c")
    return record, store


@pytest.mark.parametrize("status,error", [("failed", ""), ("interrupted", ""), ("inProgress", ""),
                                          ("", ""), ("completed", "backend error")])
def test_unsuccessful_review_cannot_approve_matching_scores(tmp_path, status, error):
    record, store = run(tmp_path, Reviewer(status=status, error=error))
    assert record.outputs["grading"]["status"] == "review_required"
    assert record.outputs["grading"]["total_score"] == 3
    assert store.list_runs()[0]["outputs"]["independent_review"]["status"] == status


def test_matching_valid_completed_review_keeps_validation(tmp_path):
    reviewer = Reviewer()
    record, _ = run(tmp_path, reviewer)
    assert record.outputs["grading"]["status"] == "validated"
    assert "score" not in json.dumps(reviewer.seen)


@pytest.mark.parametrize("mutation", [
    "swapped_scores", "missing_items", "duplicate", "unknown_item", "missing_score", "missing_contract_field",
    "item_review_reason", "uncertain", "bad_uncertainty", "top_status", "top_errors", "top_review_reason",
    "top_review_flag", "untraceable_evidence", "malformed_row", "nonfinite_total", "nonfinite_uncertainty",
])
def test_review_contract_and_item_disagreement_fail_closed(tmp_path, mutation):
    data = grade()
    if mutation == "swapped_scores":
        data["items"][0]["score"], data["items"][1]["score"] = 1, 2
    elif mutation == "missing_items":
        del data["items"]
    elif mutation == "duplicate":
        data["items"].append(copy.deepcopy(data["items"][0]))
    elif mutation == "unknown_item":
        data["items"][0]["item_id"] = "unknown"
    elif mutation == "missing_score":
        data["items"][0]["score"] = None
    elif mutation == "missing_contract_field":
        del data["items"][0]["uncertainty"]
    elif mutation == "item_review_reason":
        data["items"][0]["review_reason"] = "unclear"
    elif mutation in {"uncertain", "bad_uncertainty", "nonfinite_uncertainty"}:
        data["items"][0]["uncertainty"] = {"uncertain": 0.7, "bad_uncertainty": "low", "nonfinite_uncertainty": float("nan")}[mutation]
    elif mutation == "top_status":
        data["status"] = "failed"
    elif mutation == "top_errors":
        data["errors"] = ["failure"]
    elif mutation == "top_review_reason":
        data["review_reason"] = "needs checking"
    elif mutation == "top_review_flag":
        data["review_required"] = True
    elif mutation == "untraceable_evidence":
        data["items"][0]["evidence_spans"] = [{"text": "absent"}]
    elif mutation == "malformed_row":
        data["items"].append(None)
    elif mutation == "nonfinite_total":
        data["total_score"] = float("nan")
    record, _ = run(tmp_path, Reviewer(data=data))
    assert record.outputs["grading"]["status"] == "review_required"
    assert [item["score"] for item in record.outputs["grading"]["items"]] == [2, 1]
    assert record.outputs["independent_review"]["text"]


@pytest.mark.parametrize("exception", [TimeoutError("deadline exceeded"), RuntimeError("reviewer broke")])
def test_reviewer_exceptions_preserve_completed_grade_and_raw_artifact(tmp_path, exception):
    record, store = run(tmp_path, Reviewer(exception=exception))
    saved = store.list_runs()[0]
    assert saved["outputs"]["grading"]["status"] == "review_required"
    assert saved["outputs"]["grading"]["total_score"] == 3
    assert saved["outputs"]["independent_review"]["status"] == "failed"
    assert type(exception).__name__ in saved["outputs"]["independent_review"]["error_type"]
    assert store.artifacts.read_text(record.outputs["raw_response_artifact"]) == json.dumps(grade())


def test_imported_bytes_are_exact_grading_input(tmp_path):
    store = store_at(tmp_path)
    raw = b'  [-1,1]\r\nx = 8/10\r\n'
    source = tmp_path / "ocr.txt"
    source.write_bytes(raw)
    seen = []

    class Provider(FixtureProvider):
        def generate(self, prompt, **kwargs):
            seen.append(json.loads(prompt)["answer_text"])
            return super().generate(prompt, **kwargs)

    record = Pipeline(store, Provider(json.dumps(grade()))).run(
        case(ocr_import_path=str(source)), experiment_id="e", commit="c")
    assert seen == [raw.decode()]
    assert store.artifacts.read_bytes(record.outputs["ocr_artifact"]) == raw
    assert record.outputs["ocr_sha256"] == hashlib.sha256(raw).hexdigest()
    assert record.outputs["grading_input_sha256"] == record.outputs["ocr_sha256"]


def test_blinded_derived_answer_is_saved_with_both_hashes(tmp_path):
    store = store_at(tmp_path)
    raw = "[-1,1]\r\nscore: 4\r\nx"
    seen = []

    class Provider(FixtureProvider):
        def generate(self, prompt, **kwargs):
            seen.append(json.loads(prompt)["answer_text"])
            return super().generate(prompt, **kwargs)

    reviewer = Reviewer()
    record = Pipeline(store, Provider(json.dumps(grade())), codex_reviewer=reviewer).run(
        case(answer_text=raw), experiment_id="e", commit="c")
    outputs = record.outputs
    assert outputs["ocr_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    derived = store.artifacts.read_bytes(outputs["grading_input_artifact"])
    assert derived.decode() == seen[0] == reviewer.seen[0]["answer_text"]
    assert "score: 4" not in seen[0]
    assert seen[0].startswith("[-1,1]\r\n")
    assert outputs["grading_input_sha256"] == hashlib.sha256(derived).hexdigest()
    assert outputs["grading_input_sha256"] != outputs["ocr_sha256"]


def test_cpu_ocr_provenance_is_persisted_with_hash_and_path(tmp_path):
    store = store_at(tmp_path)
    pipeline = Pipeline(store, FixtureProvider(json.dumps(grade())))
    pipeline.ocr = XiaomiOCR(store.artifacts, runner=lambda _: (
        "x", {"revision": REVISION, "dtype": "float32", "wall_s": 2.5}), free_bytes=lambda: MIN_FREE_BYTES)
    record = pipeline.run(case(answer_text="", image_path="page.png"), experiment_id="e", commit="c")
    saved = store.list_runs()[0]["outputs"]
    assert saved["ocr_metadata"]["revision"] == REVISION
    assert saved["ocr_metadata"]["dtype"] == "float32"
    assert saved["ocr_metadata"]["elapsed_s"] == 2.5
    assert saved["ocr_metadata"]["engine"]
    assert saved["ocr_sha256"] == hashlib.sha256(b"x").hexdigest()
    assert store.artifacts.read_bytes(record.outputs["ocr_artifact"]) == b"x"
