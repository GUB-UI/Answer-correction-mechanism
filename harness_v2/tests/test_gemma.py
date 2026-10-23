from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from harness_v2.providers.gemma_ai_studio import (
    MODEL_ID,
    GemmaAIStudio,
    MissingAPIKey,
    ProviderSetupError,
    QuotaExceeded,
)


class _Models:
    def __init__(self, behavior) -> None:
        self.behavior = behavior
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return self.behavior(kwargs)


@pytest.fixture(autouse=True)
def _offline_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "offline-test-key")
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEX_AI", raising=False)


def _client(behavior):
    models = _Models(behavior)
    return SimpleNamespace(models=models), models


def test_missing_key_does_not_build_a_client(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "other-project-key")
    with pytest.raises(MissingAPIKey):
        GemmaAIStudio(free_tier_project=True).generate("grade this", request_id="r1")


def test_non_json_response_is_kept(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "should-not-be-read")

    def behavior(_kwargs):
        return SimpleNamespace(
            text="this is not json",
            model_version="gemma-4-31b-it-2026",
            response_id="resp-9",
            usage_metadata=SimpleNamespace(prompt_token_count=11, candidates_token_count=7),
        )

    client, models = _client(behavior)
    exchange = GemmaAIStudio(free_tier_project=True, client_factory=lambda **_kwargs: client).generate(
        "prompt",
        request_id="r2",
        settings={
            "temperature": 0,
            "max_output_tokens": 128,
            "model": "gemini-2.5-pro",
            "api_key": "secret-value",
        },
    )
    assert exchange.raw_response == "this is not json"
    assert exchange.model_requested == MODEL_ID
    assert exchange.model_returned == "gemma-4-31b-it-2026"
    assert exchange.provider_response_id == "resp-9"
    assert exchange.input_tokens == 11
    assert exchange.output_tokens == 7
    assert models.calls[0]["model"] == MODEL_ID
    assert "api_key" not in models.calls[0]["config"]
    assert "model" not in models.calls[0]["config"]
    assert models.calls[0]["config"]["max_output_tokens"] == 128
    assert exchange.settings["max_output_tokens"] == 128
    assert exchange.provider_response_id == "resp-9"
    assert exchange.attempt_count == 1
    assert "secret-value" not in json.dumps(exchange.to_dict())


def test_quota_stops_without_retry() -> None:
    slept: list[float] = []

    class QuotaError(Exception):
        code = 429

    def behavior(_kwargs):
        raise QuotaError("RESOURCE_EXHAUSTED quota")

    client, models = _client(behavior)
    with pytest.raises(QuotaExceeded) as caught:
        GemmaAIStudio(free_tier_project=True, client_factory=lambda **_kwargs: client, sleep=slept.append).generate("prompt", request_id="r3")
    assert slept == []
    assert len(models.calls) == 1
    assert caught.value.exchange.status == "quota"
    assert caught.value.exchange.attempt_count == 1
    assert "RESOURCE_EXHAUSTED" in caught.value.exchange.raw_response


def test_transient_errors_retry_then_stop() -> None:
    slept: list[float] = []

    class ServerError(Exception):
        status_code = 503

    def behavior(_kwargs):
        raise ServerError("unavailable")

    client, models = _client(behavior)
    exchange = GemmaAIStudio(free_tier_project=True, client_factory=lambda **_kwargs: client, sleep=slept.append, max_attempts=3).generate(
        "prompt",
        request_id="r4",
    )
    assert exchange.status == "error"
    assert exchange.attempt_count == 3
    assert len(models.calls) == 3
    assert slept == [0.5, 1.0]


def test_transient_error_then_success() -> None:
    slept: list[float] = []
    state = {"n": 0}

    def behavior(_kwargs):
        state["n"] += 1
        if state["n"] == 1:
            raise TimeoutError("timeout")
        return SimpleNamespace(text="{}", model_version=MODEL_ID, response_id="ok")

    client, _models = _client(behavior)
    exchange = GemmaAIStudio(free_tier_project=True, client_factory=lambda **_kwargs: client, sleep=slept.append).generate("prompt", request_id="r5")
    assert exchange.status == "ok"
    assert exchange.raw_response == "{}"
    assert exchange.attempt_count == 2
    assert slept == [0.5]


def test_sdk_missing_does_not_switch_provider(monkeypatch) -> None:
    import builtins

    monkeypatch.setenv("GEMINI_API_KEY", "unique-test-key-value")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "paid-project")
    for name in [key for key in list(__import__("sys").modules) if key == "google" or key.startswith("google.")]:
        monkeypatch.delitem(__import__("sys").modules, name, raising=False)
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "google" or name.startswith("google."):
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ProviderSetupError, match="google-genai") as caught:
        GemmaAIStudio(free_tier_project=True).generate("prompt", request_id="r6")
    assert "unique-test-key-value" not in str(caught.value)
    assert "paid-project" not in str(caught.value)


def test_client_is_constructed_for_the_gemini_api(monkeypatch) -> None:
    import sys
    from types import ModuleType, SimpleNamespace

    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "false")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "paid-project")
    captured: list[dict] = []

    class _Client:
        def __init__(self, **kwargs):
            captured.append(kwargs)
            self.models = SimpleNamespace(
                generate_content=lambda **_kwargs: SimpleNamespace(text="{}", response_id="resp-api")
            )

    google = ModuleType("google")
    genai = ModuleType("google.genai")
    genai.Client = _Client
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    exchange = GemmaAIStudio(free_tier_project=True).generate("prompt", request_id="vertex-off")
    assert captured == [{"vertexai": False, "api_key": "gemini-test-key", "http_options": {"base_url": "https://generativelanguage.googleapis.com"}}]
    assert "project" not in captured[0]
    assert "location" not in captured[0]
    assert exchange.status == "ok"
    assert exchange.provider_response_id == "resp-api"


def test_vertex_env_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    with pytest.raises(ProviderSetupError, match="Vertex"):
        GemmaAIStudio(free_tier_project=True).generate("prompt", request_id="vertex-on")
