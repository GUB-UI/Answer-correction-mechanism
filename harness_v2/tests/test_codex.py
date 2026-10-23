from __future__ import annotations

import pytest

from harness_v2.providers.codex_app_server import (
    DEFAULT_MODEL,
    CodexAppServer,
    ModelUnavailable,
)
from harness_v2.storage.artifacts import ArtifactStore
from harness_v2.storage.sqlite_store import RunStore


class Script:
    def __init__(self, replies: list[dict]) -> None:
        self.replies = list(replies)
        self.sent: list[dict] = []

    def send(self, message: dict) -> None:
        self.sent.append(message)

    def read(self) -> dict:
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
            {"id": 4, "result": {"turn": {"id": "turn_1", "status": "inProgress", "items": []}}},
            {
                "method": "turn/completed",
                "params": {
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
    sent_text = script.sent[-1]["params"]["input"][0]["text"]
    assert "99" not in sent_text
    assert "reference_score" not in sent_text
    assert "[redacted-score]" in sent_text
    methods = [message["method"] for message in script.sent]
    assert methods[:5] == ["initialize", "initialized", "model/list", "thread/start", "turn/start"]
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
