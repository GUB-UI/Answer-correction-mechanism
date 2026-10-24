"""OpenAI 互換 /v1/chat/completions の最小 HTTP 層（標準ライブラリのみ）。

glmocr_server.py / llm_server.py が共有する。モデル固有の処理は Backend 側に置き、
ここでは JSON の入出力・ヘルスチェック・逐次実行の排他だけを扱う。
"""

from __future__ import annotations

import base64
import io
import json
import re
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Protocol

THINK_PREFILL_RE = re.compile(r"^\s*<think>\s*</think>\s*$")
THINK_BLOCK_RE = re.compile(r"^\s*<think>[\s\S]*?</think>\s*")


class Backend(Protocol):
    model_name: str

    def load(self) -> None: ...

    def generate(self, request: dict[str, Any]) -> tuple[str, dict[str, int]]: ...


def content_text(content: Any) -> str:
    """content が文字列でも parts 配列でもテキスト部分を連結して返す。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                texts.append(str(part.get("text", "")))
        return "\n".join(texts)
    return str(content)


def image_urls(content: Any) -> list[str]:
    urls: list[str] = []
    if not isinstance(content, list):
        return urls
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "image_url":
            continue
        value = part.get("image_url")
        if isinstance(value, dict):
            value = value.get("url")
        if isinstance(value, str) and value:
            urls.append(value)
    return urls


def decode_image_url(url: str):
    """data URL またはローカルファイルパスを PIL.Image にする。"""
    from PIL import Image

    if url.startswith("data:"):
        _, _, encoded = url.partition(",")
        return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
    if url.startswith("file://"):
        url = url[len("file://") :]
    return Image.open(url).convert("RGB")


def split_assistant_prefill(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str | None, bool]:
    """末尾の assistant メッセージ（プレフィル）を分離する。

    戻り値: (残りのメッセージ, プレフィル文字列, プレフィルが空の思考ブロックだけか)
    クライアントは Qwen の思考モードを切るため ``<think>\\n\\n</think>\\n\\n`` を
    assistant として末尾に付けてくる。この場合はチャットテンプレートの
    enable_thinking=False で同じ効果を出せるので、プレフィルは捨ててよい。
    """
    if not messages or messages[-1].get("role") != "assistant":
        return list(messages), None, False
    prefill = content_text(messages[-1].get("content"))
    is_empty_think = bool(THINK_PREFILL_RE.match(prefill))
    return list(messages[:-1]), prefill, is_empty_think


def strip_think(text: str) -> str:
    return THINK_BLOCK_RE.sub("", text, count=1).strip()


def chat_completion_response(
    model: str,
    content: str,
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    usage = usage or {}
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": usage.get("finish_reason", "stop"),
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


class ServerState:
    def __init__(self, backend: Backend) -> None:
        self.backend = backend
        self.ready = False
        self.load_error: str | None = None
        self.generate_lock = threading.Lock()

    def load_in_background(self) -> threading.Thread:
        def _load() -> None:
            try:
                self.backend.load()
                self.ready = True
                print(f"[server] model ready: {self.backend.model_name}", flush=True)
            except Exception:
                self.load_error = traceback.format_exc()
                print(f"[server] model load failed:\n{self.load_error}", flush=True)

        thread = threading.Thread(target=_load, daemon=True)
        thread.start()
        return thread


def make_handler(state: ServerState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            print("[%s] %s" % (self.log_date_time_string(), format % args), flush=True)

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0].rstrip("/")
            if path in {"/health", "/v1/health"}:
                payload: dict[str, Any] = {"ok": state.ready, "model": state.backend.model_name}
                if state.ready:
                    try:
                        fingerprint = getattr(state.backend, "backend_fingerprint", None)
                        value = fingerprint() if callable(fingerprint) else None
                        # Use the same UTF-8 encoding as _send_json. ASCII escaping
                        # accepts lone surrogates that the response body cannot encode.
                        json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
                        payload["backend_fingerprint"] = value
                        if value is None:
                            payload["backend_fingerprint_error"] = "unavailable"
                    except Exception as exc:
                        payload["backend_fingerprint"] = None
                        payload["backend_fingerprint_error"] = type(exc).__name__
                if state.load_error:
                    payload["error"] = state.load_error.splitlines()[-1]
                self._send_json(payload, 200)
                return
            if path in {"/models", "/v1/models"}:
                self._send_json(
                    {
                        "object": "list",
                        "data": [
                            {"id": state.backend.model_name, "object": "model", "owned_by": "local"}
                        ],
                    }
                )
                return
            self._send_json({"error": "not found"}, 404)

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0].rstrip("/")
            if path not in {"/chat/completions", "/v1/chat/completions"}:
                self._send_json({"error": "not found"}, 404)
                return
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                request = json.loads(raw.decode("utf-8") or "{}")
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                self._send_json({"error": {"message": f"invalid JSON: {exc}"}}, 400)
                return
            if not state.ready:
                message = "model is loading" if state.load_error is None else "model load failed"
                self._send_json({"error": {"message": message}}, 503)
                return
            if not isinstance(request.get("messages"), list) or not request["messages"]:
                self._send_json({"error": {"message": "messages is required"}}, 400)
                return
            try:
                with state.generate_lock:
                    text, usage = state.backend.generate(request)
            except Exception as exc:
                traceback.print_exc()
                self._send_json({"error": {"message": str(exc)}}, 500)
                return
            model = str(request.get("model") or state.backend.model_name)
            self._send_json(chat_completion_response(model, text, usage))

    return Handler


def serve(backend: Backend, host: str, port: int) -> None:
    state = ServerState(backend)
    server = ThreadingHTTPServer((host, port), make_handler(state))
    print(f"[server] listening on http://{host}:{port} (loading {backend.model_name})", flush=True)
    state.load_in_background()
    try:
        server.serve_forever()
    finally:
        server.server_close()


def pick_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def pick_dtype(device: str, requested: str = "auto"):
    import torch

    if requested != "auto":
        return getattr(torch, requested)
    if device.startswith("cuda"):
        # T4 (sm75) は bf16 非対応なので fp16 に落とす。
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32
