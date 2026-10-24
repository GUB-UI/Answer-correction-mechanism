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
    fingerprint = {"model_id": "weights-A", "settings": {"max_new_tokens_cap": 512, "model_revision": "revision-A"},
                   "server_source_sha256": "a" * 64,
                   "library_versions": dict.fromkeys(("torch", "transformers", "bitsandbytes", "tokenizers", "accelerate"), "test-version")}
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
        **backend._server_provenance,
        "model_id": "weights-A",
        "settings": {
            "model_revision": "revision-A", "weights_sha256": None, "generation_config": {"top_k": 20},
            "max_new_tokens_cap": 512, "quantization_config": {"load_in_4bit": True},
            "dtype": "float16", "device": "cuda:0", "chat_template": "template",
        },
    }


@pytest.mark.parametrize("disable_thinking", [True, False])
@pytest.mark.parametrize("assistant_last", [True, False])
def test_final_request_messages_match_the_request_sent_to_sdk(tmp_path, disable_thinking, assistant_last):
    client = LMStudioClient(load_config(tmp_path))
    create = MagicMock(return_value=SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content="response"),
    )]))
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    messages = [{"role": "user", "content": "question"}]
    if assistant_last:
        messages.append({"role": "assistant", "content": "existing prefill"})
    original = json.loads(json.dumps(messages))
    kwargs = dict(messages=messages, disable_thinking=disable_thinking, model="alias")
    prepared = client.prepare_chat_request(**kwargs)
    assert client.chat_completion(**kwargs) == "response"
    assert create.call_args.kwargs == prepared
    assert messages == original
    assert len(prepared["messages"]) == len(messages) + (disable_thinking and not assistant_last)


def _loaded_backend(monkeypatch, model_id="weights-A"):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from llm_server import TransformersChatBackend
    backend = TransformersChatBackend(model_id)
    backend.model = SimpleNamespace(
        config=SimpleNamespace(_commit_hash="revision-A", quantization_config=None),
        generation_config=SimpleNamespace(to_dict=lambda: {}), dtype="float16", device="cpu",
    )
    backend.tokenizer = SimpleNamespace(chat_template="template")
    return backend


@pytest.mark.parametrize("missing", ["revision", "server_source_sha256", "library_versions", "torch"])
def test_missing_weights_or_server_provenance_is_rejected(monkeypatch, missing):
    from src.adapters.diagnosis_helpers import valid_backend_fingerprint
    fingerprint = _loaded_backend(monkeypatch).backend_fingerprint()
    if missing == "revision":
        fingerprint["settings"]["model_revision"] = None
    elif missing == "torch":
        fingerprint.setdefault("library_versions", {}).pop("torch", None)
    else:
        fingerprint.pop(missing, None)
    assert not valid_backend_fingerprint(fingerprint)


@pytest.mark.parametrize("filename", ["model.safetensors", "config.json", "tokenizer.json"])
def test_local_checkpoint_overwrite_changes_fingerprint(tmp_path, monkeypatch, filename):
    (tmp_path / "model.safetensors").write_bytes(b"weights version A")
    (tmp_path / "config.json").write_text('{}')
    (tmp_path / "tokenizer.json").write_text('{}')
    first = _loaded_backend(monkeypatch, str(tmp_path)).backend_fingerprint()
    (tmp_path / filename).write_bytes(b"version B")
    second = _loaded_backend(monkeypatch, str(tmp_path)).backend_fingerprint()
    assert first != second
    from src.adapters.diagnosis_helpers import valid_backend_fingerprint
    assert valid_backend_fingerprint(first)
    assert valid_backend_fingerprint(second)


def test_server_source_and_library_changes_invalidate_fingerprint(monkeypatch):
    first = _loaded_backend(monkeypatch).backend_fingerprint()
    import importlib.metadata
    original = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "changed" if name == "torch" else original(name))
    second = _loaded_backend(monkeypatch).backend_fingerprint()
    assert first != second
    original_read = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda path: original_read(path) + b"\n# source change" if path.name == "_openai_compat.py" else original_read(path))
    third = _loaded_backend(monkeypatch).backend_fingerprint()
    assert second != third


def test_health_timeout_has_explicit_reason(tmp_path, monkeypatch):
    client = LMStudioClient(load_config(tmp_path))
    monkeypatch.setattr("urllib.request.urlopen", MagicMock(side_effect=TimeoutError()))
    assert client.backend_fingerprint() is None
    assert client.fingerprint_unavailable_reason == "backend_health_timeout"


@pytest.mark.parametrize("failure", ["raise", "unserializable", "circular", "surrogate"])
def test_health_stays_live_when_fingerprint_is_unavailable(monkeypatch, failure):
    backend = _loaded_backend(monkeypatch)
    from _openai_compat import ServerState, make_handler
    if failure == "raise":
        backend.backend_fingerprint = MagicMock(side_effect=RuntimeError("fingerprint failed"))
    elif failure == "unserializable":
        backend.backend_fingerprint = lambda: {"bad": object()}
    elif failure == "surrogate":
        backend.backend_fingerprint = lambda: {"model_id": "weights\udcff"}
    else:
        circular = {}
        circular["self"] = circular
        backend.backend_fingerprint = lambda: circular
    state = ServerState(backend)
    state.ready = True
    handler = object.__new__(make_handler(state))
    handler.path = "/v1/health"
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler.wfile = io.BytesIO()
    handler.do_GET()
    payload = json.loads(handler.wfile.getvalue())
    assert handler.send_response.call_args.args == (200,)
    assert payload["ok"] is True
    assert payload["backend_fingerprint"] is None
    assert payload["backend_fingerprint_error"]
    if failure == "surrogate":
        assert payload["backend_fingerprint_error"] == "UnicodeEncodeError"
    from src.adapters.diagnosis_helpers import valid_backend_fingerprint
    assert not valid_backend_fingerprint(payload["backend_fingerprint"])


def test_health_encodes_surrogate_default_served_name(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    from llm_server import TransformersChatBackend
    from _openai_compat import ServerState, make_handler

    backend = TransformersChatBackend("weights\udcff")
    assert backend.model_name == "weights\udcff"
    state = ServerState(backend)
    state.ready = True
    handler = object.__new__(make_handler(state))
    handler.path = "/health"
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler.wfile = io.BytesIO()
    handler.do_GET()
    raw = handler.wfile.getvalue()
    payload = json.loads(raw)
    assert handler.send_response.call_args.args == (200,)
    assert raw.startswith(b"{")
    assert payload["ok"] is True
    assert payload["model"] == "weights\udcff"
    assert payload["backend_fingerprint"] is None
    assert payload["backend_fingerprint_error"] == "unavailable"


def test_health_preserves_non_ascii_fingerprint(monkeypatch):
    backend = _loaded_backend(monkeypatch)
    fingerprint = backend.backend_fingerprint()
    fingerprint["settings"]["chat_template"] = "テンプレート"
    backend.backend_fingerprint = lambda: fingerprint
    from _openai_compat import ServerState, make_handler
    state = ServerState(backend)
    state.ready = True
    handler = object.__new__(make_handler(state))
    handler.path = "/health"
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler.wfile = io.BytesIO()
    handler.do_GET()
    payload = json.loads(handler.wfile.getvalue())
    assert payload["ok"] is True
    assert payload["backend_fingerprint"]["settings"]["chat_template"] == "テンプレート"
    assert "backend_fingerprint_error" not in payload
