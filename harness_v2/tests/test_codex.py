from __future__ import annotations

import json
import sys
import time

import pytest

from harness_v2.providers.codex_app_server import (
    DEFAULT_MODEL,
    SANDBOX_SCHEMA,
    CodexAppServer,
    CodexRPCError,
    ModelUnavailable,
    sandbox_modes,
    start_app_server,
    validate_thread_start_params,
)
from harness_v2.storage.artifacts import ArtifactStore
from harness_v2.storage.sqlite_store import RunStore


class Script:
    def __init__(self, replies: list[dict]) -> None:
        self.replies = list(replies)
        self.sent: list[dict] = []

    def send(self, message: dict) -> None:
        self.sent.append(message)

    def read(self, timeout: float | None = None) -> dict:
        del timeout
        return self.replies.pop(0)


def _ready(model: str = DEFAULT_MODEL) -> list[dict]:
    return [
        {"id": 1, "result": {"userAgent": "codex"}},
        {"id": 2, "result": {"data": [{"model": model}], "nextCursor": None}},
        {"id": 3, "result": {"thread": {"id": "thr_1"}}},
    ]


def test_turn_completed_saves_text(tmp_path) -> None:
    script = Script(
        _ready()
        + [
            {"id": 4, "result": {"thread": {"id": "thr_fresh"}}},
            {"id": 5, "result": {"turn": {"id": "turn_1", "status": "inProgress", "items": []}}},
            {
                "method": "turn/completed",
                "params": {
                    "threadId": "thr_fresh",
                    "turn": {
                        "id": "turn_1",
                        "status": "completed",
                        "items": [{"type": "agentMessage", "text": "{\"ok\":true}"}],
                    }
                },
            },
        ]
    )
    client = CodexAppServer(script)
    client.require_model()
    thread_id = client.start_thread()
    result = client.start_turn(
        thread_id,
        "reference_score: 99\nprior_ai_score: 3\ngrade the work",
        request_id="req",
        independent=True,
    )
    assert result.status == "completed"
    assert result.text == "{\"ok\":true}"
    assert result.thread_id == "thr_fresh"
    sent_text = script.sent[-1]["params"]["input"][0]["text"]
    assert "99" not in sent_text
    assert "reference_score" not in sent_text
    assert "prior_ai_score" not in sent_text
    assert "[redacted-score]" in sent_text
    methods = [message["method"] for message in script.sent]
    assert methods[:6] == [
        "initialize",
        "initialized",
        "model/list",
        "thread/start",
        "thread/start",
        "turn/start",
    ]
    store = RunStore(tmp_path / "runs.sqlite", ArtifactStore(tmp_path / "artifacts"))
    name = client.save_result(store, result)
    assert "req" in store.artifacts.read_text(name)


def test_failed_and_interrupt_notifications() -> None:
    script = Script(
        _ready()
        + [
            {
                "id": 4,
                "result": {
                    "turn": {
                        "id": "turn_9",
                        "status": "failed",
                        "items": [],
                        "error": {"message": "model failed"},
                    }
                },
            },
            {"id": 5, "result": {}},
        ]
    )
    client = CodexAppServer(script)
    client.require_model()
    thread_id = client.start_thread()
    failed = client.start_turn(thread_id, "grade", request_id="req", independent=False)
    assert failed.status == "failed"
    assert failed.error == "model failed"
    client.interrupt(thread_id, failed.turn_id)
    assert script.sent[-1]["method"] == "turn/interrupt"
    assert script.sent[-1]["params"] == {"threadId": thread_id, "turnId": "turn_9"}


def test_sandbox_is_read_only_per_schema() -> None:
    schema = json.loads(SANDBOX_SCHEMA.read_text(encoding="utf-8"))
    assert schema["enum"] == ["read-only", "workspace-write", "danger-full-access"]
    assert sandbox_modes() == schema["enum"]
    assert "readOnly" not in schema["enum"]
    script = Script(_ready())
    client = CodexAppServer(script)
    client.require_model()
    client.start_thread()
    params = script.sent[-1]["params"]
    assert params["sandbox"] == "read-only"
    assert params["sandbox"] in schema["enum"]
    with pytest.raises(CodexRPCError):
        validate_thread_start_params({**params, "sandbox": "readOnly"})


def test_notification_arriving_before_response_is_not_lost() -> None:
    script = Script(
        _ready()
        + [
            {
                "method": "turn/completed",
                "params": {
                    "threadId": "thr_1",
                    "turn": {
                        "id": "turn_1",
                        "status": "completed",
                        "items": [{"type": "agentMessage", "text": "early"}],
                    }
                },
            },
            {"id": 4, "result": {"turn": {"id": "turn_1", "status": "inProgress", "items": []}}},
        ]
    )
    client = CodexAppServer(script)
    client.require_model()
    result = client.start_turn(client.start_thread(), "grade", request_id="req", independent=False)
    assert result.status == "completed"
    assert result.text == "early"
    assert script.replies == []


def test_server_request_is_not_treated_as_our_response() -> None:
    script = Script(
        _ready()
        + [
            {"id": 4, "method": "item/commandExecution/requestApproval", "params": {"command": "rm"}},
            {
                "id": 4,
                "result": {
                    "turn": {
                        "id": "turn_1",
                        "status": "completed",
                        "items": [{"type": "agentMessage", "text": "ok"}],
                    }
                },
            },
        ]
    )
    client = CodexAppServer(script)
    client.require_model()
    result = client.start_turn(client.start_thread(), "grade", request_id="req", independent=False)
    assert result.text == "ok"
    replies = [message for message in script.sent if message.get("id") == 4 and "method" not in message]
    assert replies == [{"id": 4, "result": {"decision": "decline"}}]


def test_stdio_times_out_drains_stderr_and_kills_the_child() -> None:
    transport = start_app_server(
        [
            sys.executable,
            "-c",
            "import sys,time; sys.stderr.write('err-marker\\n'); sys.stderr.flush(); time.sleep(60)",
        ]
    )
    try:
        with pytest.raises(CodexRPCError, match="timed out"):
            transport.read(timeout=0.3)
        deadline = time.time() + 2
        while time.time() < deadline and not any("err-marker" in line for line in transport.stderr_lines):
            time.sleep(0.05)
        assert any("err-marker" in line for line in transport.stderr_lines)
    finally:
        transport.close()
    assert transport._process.poll() is not None


def test_stdio_context_manager_closes() -> None:
    with start_app_server([sys.executable, "-c", "import time; time.sleep(60)"]) as transport:
        assert transport._process.poll() is None
    assert transport._process.poll() is not None


def test_model_must_be_listed() -> None:
    script = Script(
        [
            {"id": 1, "result": {"userAgent": "codex"}},
            {"id": 2, "result": {"data": [{"model": "other"}], "nextCursor": None}},
        ]
    )
    client = CodexAppServer(script)
    with pytest.raises(ModelUnavailable):
        client.require_model()
