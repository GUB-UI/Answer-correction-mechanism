from __future__ import annotations

import json

from harness_v2.core.contracts import ProviderExchange
from harness_v2.core.orchestrator import Orchestrator
from harness_v2.core.pipeline import Pipeline
from harness_v2.improvement.proposals import build_proposal
from harness_v2.providers.gemma_ai_studio import QuotaExceeded
from harness_v2.storage.artifacts import ArtifactStore
from harness_v2.storage.sqlite_store import RunStore


def _case(input_id: str, raw: str) -> dict:
    return {
        "input_id": input_id,
        "request_id": input_id,
        "problem_text": "solve",
        "answer_text": "x+1",
        "rubric_version": "rubric-v0",
        "prompt_version": "prompt-v0",
        "reference_score": 99,
        "prior_ai_score": 1,
        "rubric": [{"item_id": "R1", "max_points": 4, "description": "work", "acceptable_expression": "1+x"}],
        "new_completion": raw,
        "image_path": "/tmp/student.png",
    }


def _valid() -> str:
    return json.dumps(
        {
            "items": [
                {
                    "item_id": "R1",
                    "max_points": 4,
                    "score": 4,
                    "evidence_spans": [{"text": "x+1"}],
                    "math_reason": "matches",
                    "uncertainty": 0.2,
                    "review_reason": "",
                }
            ],
            "total_score": 4,
        }
    )


class _Scripted:
    def __init__(self, raw: str) -> None:
        self.raw = raw
        self.calls = 0

    def generate(self, prompt: str, *, request_id: str, settings=None):
        del settings
        self.calls += 1
        if self.calls >= 2:
            raise QuotaExceeded(
                ProviderExchange(
                    request_id=request_id,
                    model_requested="gemma-4-31b-it",
                    model_returned="",
                    settings={},
                    input_tokens=2,
                    output_tokens=0,
                    latency_ms=5,
                    raw_response="quota body",
                    status="quota",
                    error="429",
                    attempt_count=1,
                )
            )
        assert "reference_score" not in prompt
        assert "99" not in prompt
        return ProviderExchange(
            request_id=request_id,
            model_requested="gemma-4-31b-it",
            model_returned="gemma-4-31b-it",
            settings={"temperature": 0},
            input_tokens=8,
            output_tokens=4,
            latency_ms=12,
            raw_response=self.raw,
            status="ok",
        )


def test_quota_persists_partial_results_and_stops(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "unique-test-key-value")
    store = RunStore(tmp_path / "runs.sqlite", ArtifactStore(tmp_path / "artifacts"))
    provider = _Scripted(_valid())
    result = Orchestrator(Pipeline(store, provider)).run(
        [_case("a", _valid()), _case("b", _valid()), _case("c", _valid())],
        experiment_id="exp-1",
        commit="a383b29",
    )
    assert result.stopped_reason == "quota"
    assert [record.request_id for record in result.records] == ["a", "b"]
    assert provider.calls == 2
    saved = {row["request_id"]: row for row in store.list_runs()}
    assert saved["a"]["outputs"]["grading"]["status"] == "validated"
    assert saved["a"]["failure_reason"] == ""
    assert saved["a"]["input_tokens"] == 8
    assert saved["b"]["failure_reason"] == "quota"
    blob = (tmp_path / "runs.sqlite").read_bytes()
    assert b"unique-test-key-value" not in blob
    assert b"student.png" not in blob
    assert not list((tmp_path / "artifacts").rglob("*.png"))
    prompt = (tmp_path / "artifacts" / "prompts" / "a.txt").read_text(encoding="utf-8")
    assert "reference_score" not in prompt


def test_parse_failure_links_a_proposal_and_does_not_apply_it(tmp_path) -> None:
    store = RunStore(tmp_path / "runs.sqlite", ArtifactStore(tmp_path / "artifacts"))
    provider = _Scripted("not json at all")
    record = Pipeline(store, provider).run(_case("a", "not json"), experiment_id="exp", commit="abc")
    assert record.outputs["grading"]["status"] == "failed"
    assert record.proposal_link
    proposal = json.loads(
        (tmp_path / "artifacts" / "proposals" / f"{record.proposal_link}.json").read_text(encoding="utf-8")
    )
    assert proposal["status"] == "proposed"
    assert "person reviews" in proposal["recommended_action"]
    built = build_proposal([])
    assert built is None
