from __future__ import annotations

import json

import pytest

from harness_v2.providers.grokbot_exchange import FileExchange, GitHubHandoff, SchemaError


def _request() -> dict:
    return {
        "request_id": "req-1",
        "kind": "grade",
        "experiment_id": "exp-1",
        "commit": "abc123",
        "rubric_version": "v0",
        "prompt_version": "v0",
        "input_ids": ["input-1"],
        "payload": {"answer_text": "x+1", "api_key": "secret-value"},
        "created_at": "2026-10-23T00:00:00Z",
        "independent_evaluation": True,
    }


def test_file_exchange_shares_request_id(tmp_path) -> None:
    exchange = FileExchange(tmp_path)
    exchange.publish_request(_request())
    written = json.loads((tmp_path / "requests" / "req-1.json").read_text(encoding="utf-8"))
    assert written["payload"]["api_key"] == "[redacted]"
    exchange.write_result(
        {
            "request_id": "req-1",
            "status": "completed",
            "outputs": {"total_score": 4},
            "failure_reason": "",
        }
    )
    assert exchange.read_result("req-1")["request_id"] == "req-1"
    with pytest.raises(SchemaError):
        exchange.write_result(
            {
                "request_id": "missing",
                "status": "failed",
                "outputs": {},
                "failure_reason": "no request",
            }
        )


def test_request_id_cannot_escape_and_results_are_validated(tmp_path) -> None:
    exchange = FileExchange(tmp_path)
    escaped = _request()
    escaped["request_id"] = "../../escaped"
    with pytest.raises(SchemaError):
        exchange.publish_request(escaped)
    assert not (tmp_path.parent / "escaped.json").exists()
    assert list(tmp_path.rglob("*escaped*")) == []
    exchange.publish_request(_request())
    (tmp_path / "results" / "req-1.json").write_text('{"request_id":"req-1"}', encoding="utf-8")
    with pytest.raises(SchemaError):
        exchange.read_result("req-1")


def test_image_data_uris_are_stripped_from_committed_files(tmp_path) -> None:
    payload = _request()
    payload["payload"] = {"answer_text": "see data:image/png;base64,AAAA data"}
    path = FileExchange(tmp_path).publish_request(payload)
    text = path.read_text(encoding="utf-8")
    assert "data:image" not in text
    assert "AAAA" not in text
    assert "[stripped-image]" in text
    folder = GitHubHandoff(tmp_path / "gh").write(payload)
    assert "data:image" not in (folder / "request.json").read_text(encoding="utf-8")


def test_github_handoff_is_files_only(tmp_path) -> None:
    folder = GitHubHandoff(tmp_path).write(_request())
    assert (folder / "issue.md").is_file()
    assert (folder / "pr.md").is_file()
    assert (folder / "branch.txt").read_text(encoding="utf-8").strip() == "harness/req-1"
    assert "secret-value" not in (folder / "request.json").read_text(encoding="utf-8")
