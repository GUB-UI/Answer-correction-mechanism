"""JSON-RPC client for `codex app-server` over stdio.

ChatGPT login lives in the Codex CLI. This client does not take an API key.
Independent evaluation removes prior AI scores and reference scores from the
turn text before it is sent.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from typing import Any, Protocol

from harness_v2.evaluation.independent import scrub_score_text

DEFAULT_MODEL = "gpt-6.1-sol"


class CodexRPCError(RuntimeError):
    def __init__(self, message: str, *, payload: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.payload = payload or {}


class ModelUnavailable(CodexRPCError):
    pass


class Transport(Protocol):
    def send(self, message: dict[str, Any]) -> None: ...

    def read(self) -> dict[str, Any]: ...


@dataclass
class CodexTurnResult:
    request_id: str
    thread_id: str
    turn_id: str
    status: str
    model_id: str
    text: str
    error: str = ""
    notifications: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "thread_id": self.thread_id,
            "turn_id": self.turn_id,
            "status": self.status,
            "model_id": self.model_id,
            "text": self.text,
            "error": self.error,
            "notifications": self.notifications,
        }


class StdioTransport:
    """Newline-delimited JSON to a `codex app-server` process."""

    def __init__(self, process: subprocess.Popen[str]) -> None:
        self._process = process

    def send(self, message: dict[str, Any]) -> None:
        assert self._process.stdin is not None
        self._process.stdin.write(json.dumps(message) + "\n")
        self._process.stdin.flush()

    def read(self) -> dict[str, Any]:
        assert self._process.stdout is not None
        line = self._process.stdout.readline()
        if not line:
            raise CodexRPCError("codex app-server closed stdout")
        loaded = json.loads(line)
        if not isinstance(loaded, dict):
            raise CodexRPCError("codex app-server sent a non-object message")
        return loaded


def start_app_server(command: list[str] | None = None) -> StdioTransport:
    argv = command or ["codex", "app-server", "--listen", "stdio://"]
    process = subprocess.Popen(
        argv,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return StdioTransport(process)


class CodexAppServer:
    def __init__(self, transport: Transport, *, model: str = DEFAULT_MODEL, cwd: str = "/tmp") -> None:
        self.transport = transport
        self.model = model
        self.cwd = cwd
        self._next_id = 0
        self._queued: list[dict[str, Any]] = []
        self.initialized = False

    def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._next_id += 1
        request_id = self._next_id
        self.transport.send({"method": method, "id": request_id, "params": params})
        while True:
            message = self.transport.read()
            if message.get("id") != request_id:
                self._queued.append(message)
                continue
            if "error" in message:
                error = message["error"] or {}
                raise CodexRPCError(str(error.get("message") or error), payload=message)
            result = message.get("result")
            return result if isinstance(result, dict) else {"value": result}

    def initialize(self) -> dict[str, Any]:
        result = self._request(
            "initialize",
            {
                "clientInfo": {
                    "name": "answer_correction_harness",
                    "title": "Answer Correction Harness",
                    "version": "0.1.0",
                }
            },
        )
        self.transport.send({"method": "initialized", "params": {}})
        self.initialized = True
        return result

    def list_models(self) -> list[dict[str, Any]]:
        self._ensure_initialized()
        found: list[dict[str, Any]] = []
        cursor: str | None = None
        for _ in range(10):
            params: dict[str, Any] = {"limit": 50, "includeHidden": False}
            if cursor:
                params["cursor"] = cursor
            result = self._request("model/list", params)
            found.extend(result.get("data") or [])
            cursor = result.get("nextCursor")
            if not cursor:
                break
        return found

    def require_model(self) -> str:
        models = self.list_models()
        ids = [str(row.get("model") or row.get("id") or "") for row in models]
        if self.model not in ids:
            raise ModelUnavailable(
                f"{self.model} is not in the Codex model list",
                payload={"available": ids},
            )
        return self.model

    def start_thread(self) -> str:
        self._ensure_initialized()
        result = self._request(
            "thread/start",
            {
                "model": self.model,
                "cwd": self.cwd,
                "approvalPolicy": "never",
                "sandbox": "readOnly",
                "serviceName": "answer_correction_harness",
            },
        )
        thread = result.get("thread") or {}
        thread_id = str(thread.get("id") or "")
        if not thread_id:
            raise CodexRPCError("thread/start returned no thread id", payload=result)
        return thread_id

    def start_turn(
        self,
        thread_id: str,
        text: str,
        *,
        request_id: str,
        independent: bool = False,
    ) -> CodexTurnResult:
        prompt = scrub_score_text(text) if independent else text
        result = self._request(
            "turn/start",
            {
                "threadId": thread_id,
                "input": [{"type": "text", "text": prompt}],
                "model": self.model,
            },
        )
        turn = result.get("turn") or {}
        turn_id = str(turn.get("id") or "")
        if turn.get("status") in {"completed", "failed", "interrupted"}:
            final = result
        else:
            final = self._read_until_terminal(turn_id)
        final_turn = final.get("turn") or turn
        status = str(final_turn.get("status") or "")
        error = ""
        error_obj = final_turn.get("error") or {}
        if isinstance(error_obj, dict):
            error = str(error_obj.get("message") or "")
        text_out = _agent_text(final_turn) or _agent_text_from_notes(self._queued)
        notes = list(self._queued)
        self._queued.clear()
        return CodexTurnResult(
            request_id=request_id,
            thread_id=thread_id,
            turn_id=str(final_turn.get("id") or turn_id),
            status=status,
            model_id=self.model,
            text=text_out,
            error=error,
            notifications=notes,
        )

    def interrupt(self, thread_id: str, turn_id: str) -> dict[str, Any]:
        return self._request("turn/interrupt", {"threadId": thread_id, "turnId": turn_id})

    def save_result(self, store: Any, result: CodexTurnResult) -> str:
        name = f"codex/{result.request_id}.json"
        store.artifacts.write_json(name, result.to_dict())
        return name

    def _ensure_initialized(self) -> None:
        if not self.initialized:
            self.initialize()

    def _read_until_terminal(self, turn_id: str) -> dict[str, Any]:
        while True:
            message = self.transport.read()
            method = message.get("method")
            if method == "turn/completed":
                turn = (message.get("params") or {}).get("turn") or {}
                if not turn_id or turn.get("id") == turn_id:
                    self._queued.append(message)
                    return {"turn": turn}
            self._queued.append(message)


def _agent_text(turn: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in turn.get("items") or []:
        if isinstance(item, dict) and item.get("type") == "agentMessage":
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)


def _agent_text_from_notes(notes: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in notes:
        if message.get("method") != "item/completed":
            continue
        item = (message.get("params") or {}).get("item") or {}
        if item.get("type") == "agentMessage" and isinstance(item.get("text"), str):
            parts.append(item["text"])
    return "\n".join(parts)
