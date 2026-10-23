"""Backend verification without sockets, API calls, or model downloads."""
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.adapters.lmstudio_client import LMStudioClient
from src.config import load_config


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Backend identity tests must not access the network")

    monkeypatch.setattr("socket.socket.connect", forbidden)


@pytest.mark.parametrize("payload", [
    {}, {"ok": False}, {"ok": True, "model": "alias"},
    {"ok": True, "backend_fingerprint": {}},
    {"ok": True, "backend_fingerprint": {"model_id": "weights"}},
    {"ok": True, "backend_fingerprint": {"model_id": "weights", "settings": {}}},
    {"ok": True, "backend_fingerprint": {"model_id": " ", "settings": {"cap": 512}}},
    {"ok": True, "error": "loading", "backend_fingerprint": {
        "model_id": "weights", "settings": {"cap": 512},
    }},
    [], "invalid json", OSError("offline"),
])
def test_health_without_verified_loaded_identity_disables_reuse(tmp_path, monkeypatch, payload):
    client = LMStudioClient(load_config(tmp_path))

    def health(*args, **kwargs):
        if isinstance(payload, Exception):
            raise payload
        body = payload if payload == "invalid json" else json.dumps(payload)
        return io.BytesIO(body.encode())

    monkeypatch.setattr("urllib.request.urlopen", health)
    assert client.backend_fingerprint() is None


def test_fingerprint_comes_from_actual_client_endpoint(tmp_path, monkeypatch):
    client = LMStudioClient(load_config(tmp_path))
    client._client.base_url = "http://actual-backend:4321/v1"
    fingerprint = {"model_id": "weights-A", "settings": {"max_new_tokens_cap": 512}}
    health = MagicMock(return_value=io.BytesIO(json.dumps({
        "ok": True, "model": "alias", "backend_fingerprint": fingerprint,
    }).encode()))
    monkeypatch.setattr("urllib.request.urlopen", health)
    assert client.backend_fingerprint() == fingerprint
    assert health.call_args.args[0].full_url == "http://actual-backend:4321/v1/health"


def test_managed_health_reports_loaded_weights_and_effective_settings(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    from llm_server import TransformersChatBackend
    from _openai_compat import ServerState, make_handler

    backend = TransformersChatBackend("weights-A", served_model_name="alias", max_new_tokens_cap=512)
    assert backend.backend_fingerprint() is None
    backend.model = SimpleNamespace(
        config=SimpleNamespace(_commit_hash="revision-A", quantization_config={"load_in_4bit": True}),
        generation_config=SimpleNamespace(to_dict=lambda: {"top_k": 20}),
        dtype="float16", device="cuda:0",
    )
    backend.tokenizer = SimpleNamespace(chat_template="template")
    state = ServerState(backend)
    handler_class = make_handler(state)
    handler = object.__new__(handler_class)
    handler.path = "/v1/health"
    handler._send_json = MagicMock()
    handler.do_GET()
    assert "backend_fingerprint" not in handler._send_json.call_args.args[0]
    state.ready = True
    handler.do_GET()
    payload = handler._send_json.call_args.args[0]
    assert payload["model"] == "alias"
    assert payload["backend_fingerprint"] == {
        "model_id": "weights-A",
        "settings": {
            "model_revision": "revision-A", "generation_config": {"top_k": 20},
            "max_new_tokens_cap": 512, "quantization_config": {"load_in_4bit": True},
            "dtype": "float16", "device": "cuda:0", "chat_template": "template",
        },
    }
