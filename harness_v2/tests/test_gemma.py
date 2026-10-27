from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from harness_v2.providers.gemma_ai_studio import (
    MODEL_ID,
    WALL_CLOCK_LIMIT_MS,
    GemmaAIStudio,
    MissingAPIKey,
    ProviderSetupError,
    QuotaExceeded,
)


class _Models:
    def __init__(self, behavior) -> None:
        self.behavior = behavior
        self.calls: list[dict] = []

    def generate_content_stream(self, **kwargs):
        self.calls.append(kwargs)
        result = self.behavior(kwargs)
        if isinstance(result, (list, tuple)):
            return iter(result)
        return iter((result,))


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
            finish_reason="STOP",
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
    exchange = GemmaAIStudio(
        free_tier_project=True, client_factory=lambda **_kwargs: client, sleep=slept.append,
        rng=lambda: 0, max_attempts=3).generate(
        "prompt",
        request_id="r4",
    )
    assert exchange.status == "error"
    assert exchange.attempt_count == 3
    assert len(models.calls) == 3
    assert slept == [5.0, 5.0]


def test_transient_error_then_success() -> None:
    slept: list[float] = []
    state = {"n": 0}

    class ServerError(Exception):
        status_code = 503

    def behavior(_kwargs):
        state["n"] += 1
        if state["n"] == 1:
            raise ServerError("unavailable")
        return SimpleNamespace(text="{}", model_version=MODEL_ID, response_id="ok", finish_reason="STOP")

    client, _models = _client(behavior)
    exchange = GemmaAIStudio(
        free_tier_project=True, client_factory=lambda **_kwargs: client, sleep=slept.append, rng=lambda: 0,
    ).generate("prompt", request_id="r5")
    assert exchange.status == "ok"
    assert exchange.raw_response == "{}"
    assert exchange.attempt_count == 2
    assert slept == [5.0]


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
                generate_content_stream=lambda **_kwargs: iter(
                    (SimpleNamespace(text="{}", response_id="resp-api", finish_reason="STOP"),))
            )

    google = ModuleType("google")
    genai = ModuleType("google.genai")
    genai.Client = _Client
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    exchange = GemmaAIStudio(free_tier_project=True).generate("prompt", request_id="vertex-off")
    assert captured == [{"vertexai": False, "api_key": "gemini-test-key", "http_options": {"base_url": "https://generativelanguage.googleapis.com", "timeout": WALL_CLOCK_LIMIT_MS}}]
    assert "project" not in captured[0]
    assert "location" not in captured[0]
    assert exchange.status == "ok"
    assert exchange.provider_response_id == "resp-api"


def test_vertex_env_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    with pytest.raises(ProviderSetupError, match="Vertex"):
        GemmaAIStudio(free_tier_project=True).generate("prompt", request_id="vertex-on")


def test_timeout_is_terminal_and_records_no_retry() -> None:
    slept: list[float] = []
    built: list[int] = []

    class Deadline(Exception):
        code = 504

    class RequestTimeout(Exception):
        status_code = 408

    for exc in (Deadline("504 DEADLINE_EXCEEDED"), TimeoutError("read timeout"), RequestTimeout("408")):
        built.clear()
        slept.clear()

        def fail(_kwargs, exc=exc):
            raise exc

        def build(**kwargs):
            built.append(kwargs["http_options"]["timeout"])
            client, _models = _client(fail)
            return client

        exchange = GemmaAIStudio(
            client_factory=build, sleep=slept.append, rng=lambda: 0, max_attempts=3,
            request_timeout_s=600, request_timeout_cap_s=1800, deadline_growth=2,
        ).generate("prompt", request_id="deadline")
        assert built == [WALL_CLOCK_LIMIT_MS], type(exc).__name__
        assert slept == [], type(exc).__name__
        assert exchange.status == "error"
        assert exchange.attempt_count == 1
        assert exchange.attempts[0]["timeout_s"] == 900
        assert exchange.attempts[0]["thoughts_token_count"] is None


def test_success_records_usage_finish_reason_and_timestamps() -> None:
    def behavior(_kwargs):
        return SimpleNamespace(
            text="{}",
            model_version=MODEL_ID,
            response_id="ok",
            finish_reason="STOP",
            usage_metadata=SimpleNamespace(
                prompt_token_count=3, candidates_token_count=4, thoughts_token_count=11),
        )

    stamps = iter([
        datetime(2026, 10, 26, 1, 2, 3, tzinfo=timezone.utc),
        datetime(2026, 10, 26, 1, 2, 4, tzinfo=timezone.utc),
    ])
    client, _models = _client(behavior)
    exchange = GemmaAIStudio(
        client_factory=lambda **_kwargs: client, wall_time=lambda: next(stamps),
    ).generate("prompt", request_id="usage")
    row = exchange.attempts[0]
    assert exchange.status == "ok"
    assert row["finish_reason"] == "STOP"
    assert row["prompt_token_count"] == 3
    assert row["candidates_token_count"] == 4
    assert row["thoughts_token_count"] == 11
    assert row["input_tokens"] == 3
    assert row["output_tokens"] == 4
    assert row["started_at"] == "2026-10-26T01:02:03+00:00"
    assert row["ended_at"] == "2026-10-26T01:02:04+00:00"
    assert row["elapsed_s"] >= 0
    assert row["latency_ms"] == pytest.approx(row["elapsed_s"] * 1000)


def test_fast_server_errors_backoff_with_jitter_and_reuse_the_client() -> None:
    slept: list[float] = []
    built: list[int] = []

    class ServerError(Exception):
        status_code = 500

    def behavior(_kwargs):
        raise ServerError("internal")

    def factory(**kwargs):
        built.append(kwargs["http_options"]["timeout"])
        client, _models = _client(behavior)
        return client

    exchange = GemmaAIStudio(
        client_factory=factory, sleep=slept.append, rng=lambda: 1, monotonic=lambda: 0.0,
    ).generate("prompt", request_id="fast")
    assert exchange.status == "error"
    assert exchange.attempt_count == 3
    assert built == [WALL_CLOCK_LIMIT_MS]
    assert slept == [5.0, 10.0]
    assert all(row["timeout_s"] == 900 for row in exchange.attempts)
    assert all("elapsed_s" in row and "started_at" in row and "ended_at" in row for row in exchange.attempts)

    slept.clear()
    built.clear()
    capped = GemmaAIStudio(
        client_factory=factory, sleep=slept.append, rng=lambda: 1, max_attempts=6,
        monotonic=lambda: 0.0,
    ).generate("prompt", request_id="fast-cap")
    assert capped.attempt_count == 6
    assert slept == [5.0, 10.0, 20.0, 30.0, 30.0]
    assert built == [WALL_CLOCK_LIMIT_MS]


def _run_clocked(behavior, *, rng, max_attempts, advance_on_call: float):
    slept: list[float] = []
    built: list[int] = []
    clock = {"t": 0.0}

    def timed(kwargs):
        clock["t"] += advance_on_call
        return behavior(kwargs)

    def factory(**kwargs):
        built.append(kwargs["http_options"]["timeout"])
        client, _models = _client(timed)
        return client

    def sleep(seconds):
        slept.append(seconds)
        clock["t"] += seconds

    provider = GemmaAIStudio(
        client_factory=factory, sleep=sleep, monotonic=lambda: clock["t"], rng=rng,
        max_attempts=max_attempts,
    )
    return provider.generate("prompt", request_id="disconnect"), slept, built


def test_transport_disconnect_retries_with_a_fresh_client_and_the_remaining_budget() -> None:
    class RemoteProtocolError(Exception):
        pass

    class ReadError(Exception):
        pass

    class ConnectError(Exception):
        pass

    class ClientError(Exception):
        pass

    def one_failure(exc):
        state = {"n": 0}

        def behavior(_kwargs):
            state["n"] += 1
            if state["n"] == 1:
                raise exc
            return SimpleNamespace(text="{}", model_version=MODEL_ID, response_id="ok", finish_reason="STOP")

        return behavior

    wrapped = ClientError("request failed")
    wrapped.__cause__ = RemoteProtocolError("Server disconnected")
    failures = [
        RemoteProtocolError("Server disconnected"),
        ReadError("connection dropped"),
        ConnectError("connection reset"),
        wrapped,
    ]
    for exc in failures:
        exchange, slept, built = _run_clocked(
            one_failure(exc), rng=lambda: 0, max_attempts=3, advance_on_call=300)
        assert built == [WALL_CLOCK_LIMIT_MS, 595_000], type(exc).__name__
        assert slept == [5.0], type(exc).__name__
        assert exchange.status == "ok"
        assert exchange.attempt_count == 2
        assert exchange.attempts[0]["elapsed_s"] == pytest.approx(300)
        assert exchange.attempts[0]["timeout_s"] == 900
        assert exchange.attempts[1]["timeout_s"] == pytest.approx(595)

    def always(_kwargs):
        raise RemoteProtocolError("Server disconnected")

    stopped, slept, built = _run_clocked(always, rng=lambda: 0, max_attempts=2, advance_on_call=400)
    assert stopped.status == "error"
    assert stopped.attempt_count == 2
    assert built == [WALL_CLOCK_LIMIT_MS, 495_000]
    assert slept == [5.0]
    assert stopped.attempts[1]["timeout_s"] == pytest.approx(495)
    assert [row["error_class"] for row in stopped.attempts] == [
        "RemoteProtocolError", "RemoteProtocolError"]


def test_fast_transport_disconnect_uses_fast_backoff() -> None:
    class RemoteProtocolError(Exception):
        pass

    class ReadError(Exception):
        pass

    class ConnectError(Exception):
        pass

    class APIError(Exception):
        pass

    for exc_type, message in (
        (RemoteProtocolError, "Server disconnected"),
        (ReadError, "read failed"),
        (ConnectError, "connect failed"),
        (APIError, "Server disconnected"),
    ):
        def behavior(_kwargs, exc_type=exc_type, message=message):
            raise exc_type(message)

        client, models = _client(behavior)
        slept: list[float] = []
        exchange = GemmaAIStudio(
            client_factory=lambda **_kwargs: client, sleep=slept.append, rng=lambda: 0,
            max_attempts=3,
        ).generate("prompt", request_id="fast-disconnect")
        assert exchange.status == "error", exc_type.__name__
        assert exchange.attempt_count == 3
        assert len(models.calls) == 3
        assert slept == [5.0, 5.0]
        assert all(row["timeout_s"] == 900 for row in exchange.attempts)


def test_unrelated_provider_error_is_not_retried() -> None:
    slept: list[float] = []

    def behavior(_kwargs):
        raise ValueError("bad prompt")

    client, models = _client(behavior)
    exchange = GemmaAIStudio(
        client_factory=lambda **_kwargs: client, sleep=slept.append,
    ).generate("prompt", request_id="not-transport")
    assert exchange.status == "error"
    assert exchange.attempt_count == 1
    assert len(models.calls) == 1
    assert slept == []


def test_thinking_level_and_max_output_tokens_are_recorded_and_sent() -> None:
    def behavior(_kwargs):
        return SimpleNamespace(text="{}", finish_reason="STOP")

    client, models = _client(behavior)
    provider = GemmaAIStudio(client_factory=lambda **_kwargs: client)
    minimal = provider.generate(
        "prompt", request_id="minimal",
        settings={"temperature": 0, "max_output_tokens": 8192, "thinking_level": "minimal"},
    )
    assert models.calls[0]["config"]["max_output_tokens"] == 8192
    assert models.calls[0]["config"]["thinking_config"] == {"thinking_level": "MINIMAL"}
    assert "thinking_level" not in models.calls[0]["config"]
    assert minimal.settings["thinking_level"] == "minimal"
    assert minimal.settings["max_output_tokens"] == 8192
    high = provider.generate("prompt", request_id="high", settings={"thinking_level": "high"})
    assert models.calls[1]["config"]["thinking_config"] == {"thinking_level": "HIGH"}
    assert high.settings["thinking_level"] == "high"
    unset = provider.generate(
        "prompt", request_id="unset", settings={"temperature": 0, "thinking_level": "unset"})
    assert "thinking_config" not in models.calls[2]["config"]
    assert "thinking_level" not in models.calls[2]["config"]
    assert unset.settings["thinking_level"] == "unset"
    absent = provider.generate("prompt", request_id="absent", settings={"temperature": 0})
    assert "thinking_config" not in models.calls[3]["config"]
    assert "thinking_level" not in absent.settings
    with pytest.raises(ValueError, match="thinking_level"):
        provider.generate("prompt", request_id="bad", settings={"thinking_level": "low"})


def test_thinking_config_uses_sdk_types_when_importable(monkeypatch) -> None:
    import sys
    from types import ModuleType

    class ThinkingLevel:
        MINIMAL = "MINIMAL"
        HIGH = "HIGH"

    class ThinkingConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class GenerateContentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    types_mod = ModuleType("google.genai.types")
    types_mod.ThinkingLevel = ThinkingLevel
    types_mod.ThinkingConfig = ThinkingConfig
    types_mod.GenerateContentConfig = GenerateContentConfig
    google = ModuleType("google")
    genai = ModuleType("google.genai")
    genai.types = types_mod
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_mod)

    def behavior(_kwargs):
        return SimpleNamespace(text="{}", finish_reason="STOP")

    client, models = _client(behavior)
    GemmaAIStudio(client_factory=lambda **_kwargs: client).generate(
        "prompt", request_id="sdk",
        settings={"temperature": 0, "max_output_tokens": 8192, "thinking_level": "minimal"},
    )
    config = models.calls[0]["config"]
    assert isinstance(config, GenerateContentConfig)
    assert config.kwargs["max_output_tokens"] == 8192
    assert isinstance(config.kwargs["thinking_config"], ThinkingConfig)
    assert config.kwargs["thinking_config"].kwargs["thinking_level"] == "MINIMAL"
    assert "thinking_level" not in config.kwargs


def test_max_tokens_without_a_parseable_answer_is_terminal() -> None:
    slept: list[float] = []
    usage = SimpleNamespace(prompt_token_count=20, candidates_token_count=0, thoughts_token_count=4000)

    def empty(_kwargs):
        return SimpleNamespace(text="", finish_reason="MAX_TOKENS", usage_metadata=usage)

    def thoughts_only(_kwargs):
        return SimpleNamespace(
            candidates=[SimpleNamespace(
                finish_reason="FinishReason.MAX_TOKENS",
                content=SimpleNamespace(parts=[SimpleNamespace(thought=True, text="still thinking")]),
            )],
            usage_metadata=usage,
        )

    def prose(_kwargs):
        return SimpleNamespace(text="not json", finish_reason="MAX_TOKENS", usage_metadata=usage)

    for behavior in (empty, thoughts_only, prose):
        slept.clear()
        client, models = _client(behavior)
        exchange = GemmaAIStudio(
            client_factory=lambda **_kwargs: client, sleep=slept.append, max_attempts=3,
        ).generate("prompt", request_id="max")
        assert exchange.status == "error"
        assert exchange.error == "MAX_TOKENS with no parseable answer"
        assert exchange.attempt_count == 1
        assert len(models.calls) == 1
        assert slept == []
        assert exchange.attempts[0]["finish_reason"] == "MAX_TOKENS"
        assert exchange.attempts[0]["error_class"] == "MAX_TOKENS"
        assert exchange.attempts[0]["prompt_token_count"] == 20
        assert exchange.attempts[0]["thoughts_token_count"] == 4000
        assert exchange.attempts[0]["candidates_token_count"] == 0
        assert exchange.raw_response == ("" if behavior is not prose else "not json")


def test_max_tokens_with_a_parseable_object_is_accepted() -> None:
    def behavior(_kwargs):
        return [
            SimpleNamespace(
                candidates=[SimpleNamespace(
                    finish_reason=None,
                    content=SimpleNamespace(parts=[
                        SimpleNamespace(thought=True, text="hidden"),
                        SimpleNamespace(thought=False, text='{"a": 1}'),
                    ]),
                )],
            ),
            SimpleNamespace(
                finish_reason="MAX_TOKENS",
                usage_metadata=SimpleNamespace(
                    prompt_token_count=4, candidates_token_count=6, thoughts_token_count=2),
            ),
        ]

    client, models = _client(behavior)
    slept: list[float] = []
    exchange = GemmaAIStudio(
        client_factory=lambda **_kwargs: client, sleep=slept.append, max_attempts=3,
    ).generate("prompt", request_id="kept")
    assert exchange.status == "error"
    assert exchange.error == "finish_reason=MAX_TOKENS"
    assert exchange.attempt_count == 1
    assert len(models.calls) == 1
    assert slept == []
    assert exchange.raw_response == '{"a": 1}'
    assert exchange.attempts[0]["finish_reason"] == "MAX_TOKENS"
    assert exchange.output_tokens == 6
    assert exchange.attempts[0]["thoughts_token_count"] == 2


def test_backoff_that_would_pass_the_wall_clock_does_not_start_another_attempt() -> None:
    clock = {"t": 0.0}

    class ServerError(Exception):
        status_code = 502

    def behavior(_kwargs):
        clock["t"] += 896
        raise ServerError("bad gateway")

    client, models = _client(behavior)
    slept: list[float] = []

    def sleep(seconds):
        slept.append(seconds)
        clock["t"] += seconds

    exchange = GemmaAIStudio(
        client_factory=lambda **_kwargs: client, sleep=sleep, monotonic=lambda: clock["t"],
        rng=lambda: 0, wall_clock_limit_s=900, max_attempts=3,
    ).generate("prompt", request_id="budget")
    assert exchange.status == "error"
    assert exchange.attempt_count == 1
    assert len(models.calls) == 1
    assert slept == [4.0]
    assert clock["t"] == pytest.approx(900)
    assert clock["t"] <= 900 + 0.001


def test_read_wait_stays_inside_the_remaining_case_budget() -> None:
    clock = {"t": 0.0}
    built: list[int] = []
    calls = {"n": 0}

    class RemoteProtocolError(Exception):
        pass

    def factory(**kwargs):
        timeout_ms = kwargs["http_options"]["timeout"]
        built.append(timeout_ms)

        def generate_content_stream(**_kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                clock["t"] += 800
                raise RemoteProtocolError("Server disconnected")
            clock["t"] += timeout_ms / 1000
            raise TimeoutError("read timeout")

        return SimpleNamespace(models=SimpleNamespace(generate_content_stream=generate_content_stream))

    def sleep(seconds):
        clock["t"] += seconds

    exchange = GemmaAIStudio(
        client_factory=factory, sleep=sleep, monotonic=lambda: clock["t"], rng=lambda: 0,
        wall_clock_limit_s=900, max_attempts=3,
    ).generate("prompt", request_id="budget-read")
    assert built == [900_000, 95_000]
    assert calls["n"] == 2
    assert exchange.status == "error"
    assert exchange.attempt_count == 2
    assert exchange.attempts[1]["error_class"] == "TimeoutError"
    assert exchange.attempts[1]["timeout_s"] == pytest.approx(95)
    assert clock["t"] == pytest.approx(900)
    assert clock["t"] <= 900 + 0.001


@pytest.mark.parametrize("reason", ["SAFETY", "RECITATION", "OTHER"])
def test_non_stop_finish_is_terminal_even_when_the_text_is_json(reason: str) -> None:
    slept: list[float] = []

    def behavior(_kwargs):
        return SimpleNamespace(text='{"a": 1}', finish_reason=reason)

    client, models = _client(behavior)
    exchange = GemmaAIStudio(
        client_factory=lambda **_kwargs: client, sleep=slept.append, max_attempts=3,
    ).generate("prompt", request_id="finish")
    assert exchange.status == "error"
    assert exchange.error == f"finish_reason={reason}"
    assert exchange.attempt_count == 1
    assert len(models.calls) == 1
    assert slept == []
    assert exchange.raw_response == '{"a": 1}'
    assert exchange.attempts[0]["finish_reason"] == reason


def test_stream_without_finish_reason_retries_even_when_json_parses() -> None:
    built: list[int] = []
    state = {"n": 0}

    def behavior(_kwargs):
        state["n"] += 1
        if state["n"] == 1:
            return SimpleNamespace(
                text='{"a": 1}',
                usage_metadata=SimpleNamespace(
                    prompt_token_count=3, candidates_token_count=2, thoughts_token_count=1),
            )
        return SimpleNamespace(text='{"a": 2}', finish_reason="STOP")

    def factory(**kwargs):
        built.append(kwargs["http_options"]["timeout"])
        client, _models = _client(behavior)
        return client

    slept: list[float] = []
    exchange = GemmaAIStudio(
        client_factory=factory, sleep=slept.append, rng=lambda: 0, monotonic=lambda: 0.0,
    ).generate("prompt", request_id="open-stream")
    assert exchange.status == "ok"
    assert exchange.raw_response == '{"a": 2}'
    assert exchange.attempt_count == 2
    assert slept == [5.0]
    assert built == [WALL_CLOCK_LIMIT_MS, WALL_CLOCK_LIMIT_MS]
    assert exchange.attempts[0]["error_class"] == "StreamEnded"
    assert exchange.attempts[0]["finish_reason"] is None
    assert exchange.attempts[0]["prompt_token_count"] == 3
    assert exchange.attempts[1]["finish_reason"] == "STOP"


def test_safety_finish_is_not_validated(tmp_path, monkeypatch) -> None:
    from harness_v2.core.pipeline import Pipeline
    from harness_v2.tests.test_p3_setup import store_at
    from harness_v2.tests.test_pipeline import _case, _valid

    monkeypatch.setenv("GEMINI_API_KEY", "offline-key")

    def generate_content_stream(**_kwargs):
        return iter((SimpleNamespace(text=_valid(), finish_reason="SAFETY"),))

    provider = GemmaAIStudio(client_factory=lambda **_kwargs: SimpleNamespace(
        models=SimpleNamespace(generate_content_stream=generate_content_stream)))
    record = Pipeline(store_at(tmp_path), provider).run(
        _case("safety", _valid()), experiment_id="e", commit="c")
    grade = record.outputs["grading"]
    assert grade["status"] == "failed"
    assert grade["items"] == []
    assert "SAFETY" in grade["failure_reason"]
