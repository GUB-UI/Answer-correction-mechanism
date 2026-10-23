"""JSON-RPC client for `codex app-server` over stdio.

ChatGPT login lives in the Codex CLI. This client does not take an API key.
An independent evaluation starts a new thread and sends only the allowlisted
text.
"""

from __future__ import annotations

import json
import select
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from harness_v2.evaluation.independent import independent_text

DEFAULT_MODEL = "gpt-6.1-sol"
SANDBOX_SCHEMA = Path(__file__).resolve().parent / "schemas" / "codex_sandbox_mode.schema.json"
DEFAULT_READ_TIMEOUT = 60.0


class CodexRPCError(RuntimeError):
    def __init__(self, message: str, *, payload: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.payload = payload or {}


class ModelUnavailable(CodexRPCError):
    pass


class Transport(Protocol):
    def send(self, message: dict[str, Any]) -> None: ...

    def read(self, timeout: float | None = None) -> dict[str, Any]: ...


def sandbox_modes() -> list[str]:
    schema = json.loads(SANDBOX_SCHEMA.read_text(encoding="utf-8"))
    modes = schema.get("enum")
    if not isinstance(modes, list) or not modes:
        raise CodexRPCError("sandbox schema has no enum")
    return [str(mode) for mode in modes]


def validate_thread_start_params(params: dict[str, Any]) -> None:
    mode = params.get("sandbox")
    allowed = sandbox_modes()
    if mode not in allowed:
        raise CodexRPCError(
            f"sandbox {mode!r} is not in {allowed}",
            payload={"sandbox": mode, "allowed": allowed},
        )


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
        self.stderr_lines: list[str] = []
        self._stderr_thread: threading.Thread | None = None
        if process.stderr is not None:
            self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
            self._stderr_thread.start()

    def send(self, message: dict[str, Any]) -> None:
        assert self._process.stdin is not None
        self._process.stdin.write(json.dumps(message) + "\n")
        self._process.stdin.flush()

    def read(self, timeout: float | None = None) -> dict[str, Any]:
        assert self._process.stdout is not None
        if timeout is not None:
            ready, _, _ = select.select([self._process.stdout], [], [], timeout)
            if not ready:
                raise CodexRPCError("codex app-server read timed out")
        line = self._process.stdout.readline()
        if not line:
            raise CodexRPCError("codex app-server closed stdout")
        loaded = json.loads(line)
        if not isinstance(loaded, dict):
            raise CodexRPCError("codex app-server sent a non-object message")
        return loaded

    def close(self) -> None:
        process = self._process
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=1)

    def __enter__(self) -> StdioTransport:
        return self

    def __exit__(self, *_exc: object) -> bool:
        self.close()
        return False

    def _drain_stderr(self) -> None:
        stderr = self._process.stderr
        if stderr is None:
            return
        for line in stderr:
            self.stderr_lines.append(line)


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
    def __init__(
        self,
        transport: Transport,
        *,
        model: str = DEFAULT_MODEL,
        cwd: str = "/tmp",
        read_timeout: float | None = DEFAULT_READ_TIMEOUT,
    ) -> None:
        self.transport = transport
        self.model = model
        self.cwd = cwd
        self.read_timeout = read_timeout
        self._next_id = 0
        self._responses: dict[Any, dict[str, Any]] = {}
        self._notes: list[dict[str, Any]] = []
        self.initialized = False

    def __enter__(self) -> CodexAppServer:
        return self

    def __exit__(self, *_exc: object) -> bool:
        self.close()
        return False

    def close(self) -> None:
        closer = getattr(self.transport, "close", None)
        if closer is not None:
            closer()

    def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._next_id += 1
        request_id = self._next_id
        self.transport.send({"method": method, "id": request_id, "params": params})
        message = self._take_response(request_id)
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
        params = {
            "model": self.model,
            "cwd": self.cwd,
            "approvalPolicy": "never",
            "sandbox": "read-only",
            "serviceName": "answer_correction_harness",
        }
        validate_thread_start_params(params)
        result = self._request("thread/start", params)
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
        active_thread = thread_id
        prompt = text
        if independent:
            active_thread = self.start_thread()
            prompt = independent_text(text)
        result = self._request(
            "turn/start",
            {
                "threadId": active_thread,
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
        text_out = _agent_text(final_turn) or _agent_text_from_notes(self._notes)
        notes = list(self._notes)
        self._notes.clear()
        return CodexTurnResult(
            request_id=request_id,
            thread_id=active_thread,
            turn_id=str(final_turn.get("id") or turn_id),
            status=status,
            model_id=self.model,
            text=text_out,
            error=error,
            notifications=notes,
        )

    def review_independent(self, payload: dict[str, Any], *, request_id: str) -> CodexTurnResult:
        """One fresh thread. The payload is allowlisted again inside the turn."""
        return self.start_turn(
            "",
            json.dumps(payload, ensure_ascii=False),
            request_id=request_id,
            independent=True,
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

    def _accept(self, message: dict[str, Any]) -> None:
        method = message.get("method")
        has_method = isinstance(method, str) and method != ""
        has_id = "id" in message and message["id"] is not None
        if has_method and has_id:
            self._notes.append(message)
            self.transport.send({"id": message["id"], "result": {}})
            return
        if has_method or not has_id:
            self._notes.append(message)
            return
        if "result" in message or "error" in message:
            self._responses[message["id"]] = message
            return
        self._notes.append(message)

    def _take_response(self, request_id: int) -> dict[str, Any]:
        while request_id not in self._responses:
            self._accept(self.transport.read(timeout=self.read_timeout))
        return self._responses.pop(request_id)

    def _completed_turn(self, turn_id: str) -> dict[str, Any] | None:
        for message in self._notes:
            if message.get("method") != "turn/completed":
                continue
            turn = (message.get("params") or {}).get("turn") or {}
            if not turn_id or turn.get("id") == turn_id:
                return turn
        return None

    def _read_until_terminal(self, turn_id: str) -> dict[str, Any]:
        while True:
            found = self._completed_turn(turn_id)
            if found is not None:
                return {"turn": found}
            self._accept(self.transport.read(timeout=self.read_timeout))


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
