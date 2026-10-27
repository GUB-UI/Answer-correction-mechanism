"""Google AI Studio client for Gemma. One key, one model, no paid fallback."""

from __future__ import annotations

import math
import os
import random
import time
from datetime import datetime, timezone
from typing import Any, Callable

from harness_v2.core.contracts import ProviderExchange
from harness_v2.grading.scorer import parse_completion
from harness_v2.storage.secrets import redact

MODEL_ID = "gemma-4-31b-it"
API_KEY_ENV = "GEMINI_API_KEY"
AI_STUDIO_ENDPOINT = "https://generativelanguage.googleapis.com"
MAX_ATTEMPTS = 3
# Older configs still name a per-call deadline. generate() does not grow it.
# The HTTP timeout is the case wall clock, once, for every attempt.
REQUEST_TIMEOUT_S = 600
REQUEST_TIMEOUT_CAP_S = 1800
DEADLINE_GROWTH = 2
DEADLINE_RETRY_PAUSE_S = 1
REQUEST_TIMEOUT_MS = REQUEST_TIMEOUT_S * 1000
FAST_BACKOFF_MIN_S = 5
FAST_BACKOFF_MAX_S = 30
WALL_CLOCK_LIMIT_S = 900
WALL_CLOCK_LIMIT_MS = WALL_CLOCK_LIMIT_S * 1000
# A read or a backoff may run this far past the case budget when the clock
# and the millisecond timeout do not land on the same instant.
_WALL_EPSILON_S = 0.001
DEFAULT_MAX_OUTPUT_TOKENS = 8192
THINKING_LEVELS = ("minimal", "high", "unset")
_THINKING_API = {"minimal": "MINIMAL", "high": "HIGH"}
TRANSPORT_CONFIG_KEYS = (
    "request_timeout_s",
    "request_timeout_cap_s",
    "deadline_growth",
    "deadline_retry_pause_s",
    "max_attempts",
    "fast_backoff_min_s",
    "fast_backoff_max_s",
    "wall_clock_limit_s",
)
_FAST_HTTP = {500, 502, 503}
_DEADLINE_HTTP = {408, 504}
# httpx transport drops. Matched by class name so a test double and a wrapped
# cause are recognized without importing httpx.
_DISCONNECT_NAMES = frozenset({
    "remoteprotocolerror",
    "localprotocolerror",
    "protocolerror",
    "readerror",
    "writeerror",
    "connecterror",
    "networkerror",
    "closeerror",
})
_VERTEX_ENV = ("GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_GENAI_USE_VERTEX_AI")
_SETTINGS_KEYS = ("temperature", "max_output_tokens", "top_p", "top_k", "seed", "thinking_level")


def _transport_number(name: str, value: Any, *, minimum: float, exclusive: bool = False) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if exclusive:
        if value <= minimum:
            raise ValueError(f"{name} must be greater than {minimum}")
    elif value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _vertex_requested() -> bool:
    for name in _VERTEX_ENV:
        value = os.environ.get(name, "").strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
    return False


class MissingAPIKey(RuntimeError):
    pass


class ProviderSetupError(RuntimeError):
    pass


class _PartialStream(Exception):
    """A stream read failed after some chunks had already arrived."""

    def __init__(
        self,
        cause: BaseException,
        *,
        visible: str,
        finish: str | None,
        usage: tuple[int | None, int | None, int | None],
        model: str,
        response_id: str,
    ) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.visible = visible
        self.finish = finish
        self.usage = usage
        self.model = model
        self.response_id = response_id


class QuotaExceeded(RuntimeError):
    """429 or a quota error. Callers stop and keep the partial exchange."""

    def __init__(self, exchange: ProviderExchange) -> None:
        super().__init__("gemini quota or rate limit; no fallback provider is configured")
        self.exchange = exchange


def _status_code(exc: BaseException) -> int | None:
    for attr in ("code", "status_code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    return None


def _is_quota(exc: BaseException, code: int | None) -> bool:
    if code == 429:
        return True
    text = str(exc).lower()
    return "resource_exhausted" in text or "quota" in text or "rate limit" in text


def _is_deadline(exc: BaseException, code: int | None) -> bool:
    """A timeout ends the case. The next attempt does not get a longer deadline."""
    if code in _FAST_HTTP:
        return False
    if code in _DEADLINE_HTTP:
        return True
    if isinstance(exc, TimeoutError):
        return True
    name = type(exc).__name__.lower()
    if "timeout" in name:
        return True
    text = str(exc).lower()
    return "deadline_exceeded" in text or "deadline exceeded" in text or "read timeout" in text


def _is_connection(exc: BaseException) -> bool:
    """Transport disconnect, including httpx RemoteProtocolError and ReadError.

    A timeout is not a disconnect. A name containing "timeout" is not a
    disconnect here.
    """
    if isinstance(exc, TimeoutError):
        return False
    name = type(exc).__name__.lower()
    if "timeout" in name:
        return False
    if isinstance(exc, ConnectionError):
        return True
    if name in _DISCONNECT_NAMES or "protocolerror" in name or name.endswith("readerror"):
        return True
    if "server disconnected" in str(exc).lower():
        return True
    return "connection" in name or "connect" in name


def _is_transport_disconnect(exc: BaseException) -> bool:
    for item in (exc, exc.__cause__, exc.__context__):
        if isinstance(item, BaseException) and _is_connection(item):
            return True
    return False


def _raw_from_exception(exc: BaseException) -> str:
    response = getattr(exc, "response", None)
    body = getattr(response, "text", None)
    if isinstance(body, str) and body:
        return body
    details = getattr(exc, "details", None)
    if isinstance(details, str) and details:
        return details
    return str(exc)


def _visible_text(response: Any) -> str:
    """Visible answer text. Thought parts are not part of the completion."""
    candidates = getattr(response, "candidates", None)
    if candidates:
        parts_out: list[str] = []
        saw_part = False
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                saw_part = True
                if getattr(part, "thought", False) is True:
                    continue
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str):
                    parts_out.append(part_text)
        if saw_part:
            return "".join(parts_out)
    text = getattr(response, "text", None)
    return text if isinstance(text, str) else ""


def _finish_name(value: Any) -> str | None:
    if value is None or value == "":
        return None
    name = getattr(value, "name", None)
    text = name if isinstance(name, str) and name else str(value)
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    text = text.strip()
    return text or None


def _finish_reason(response: Any) -> str | None:
    for candidate in getattr(response, "candidates", None) or []:
        name = _finish_name(getattr(candidate, "finish_reason", None))
        if name:
            return name
    return _finish_name(getattr(response, "finish_reason", None))


def _is_max_tokens(reason: str | None) -> bool:
    return (reason or "").upper() == "MAX_TOKENS"


def _is_stop(reason: str | None) -> bool:
    return (reason or "").upper() == "STOP"


def _parseable_visible_answer(text: str) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    try:
        parsed, *_rest = parse_completion(text)
    except Exception:
        return False
    return isinstance(parsed, dict)


def _retry_kind(exc: BaseException, code: int | None) -> str | None:
    """Fast 5xx and transport disconnects can be retried. A timeout cannot."""
    if _is_deadline(exc, code):
        return None
    if _is_transport_disconnect(exc):
        return "disconnect"
    if code in _FAST_HTTP:
        return "fast"
    return None


def _thinking_api_value(types: Any, name: str) -> Any:
    enum = getattr(types, "ThinkingLevel", None)
    if enum is not None and hasattr(enum, name):
        return getattr(enum, name)
    return name


def _generation_config(public: dict[str, Any]) -> Any:
    """Build the SDK config. thinking_level stays on the recorded settings."""
    level = public.get("thinking_level", "unset")
    if level is None:
        level = "unset"
    if not isinstance(level, str) or level not in THINKING_LEVELS:
        raise ValueError("thinking_level must be minimal, high, or unset")
    body = {key: value for key, value in public.items() if key != "thinking_level"}
    api_level = _THINKING_API.get(level)
    try:
        from google.genai import types
    except ImportError:
        if api_level is not None:
            body["thinking_config"] = {"thinking_level": api_level}
        return body
    kwargs = dict(body)
    if api_level is not None:
        kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level=_thinking_api_value(types, api_level))
    if hasattr(types, "GenerateContentConfig"):
        return types.GenerateContentConfig(**kwargs)
    return kwargs


def _usage(response: Any) -> tuple[int | None, int | None, int | None]:
    usage = getattr(response, "usage_metadata", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage_metadata")
    if usage is None:
        return None, None, None
    if isinstance(usage, dict):
        prompt = usage.get("prompt_token_count")
        output = usage.get("candidates_token_count")
        thoughts = usage.get("thoughts_token_count")
    else:
        prompt = getattr(usage, "prompt_token_count", None)
        output = getattr(usage, "candidates_token_count", None)
        thoughts = getattr(usage, "thoughts_token_count", None)
    return (
        prompt if isinstance(prompt, int) and not isinstance(prompt, bool) else None,
        output if isinstance(output, int) and not isinstance(output, bool) else None,
        thoughts if isinstance(thoughts, int) and not isinstance(thoughts, bool) else None,
    )


def _usage_from_exception(exc: BaseException) -> tuple[int | None, int | None, int | None]:
    for attr in ("response", "result"):
        response = getattr(exc, attr, None)
        if response is not None:
            return _usage(response)
    return None, None, None


def _model_returned(response: Any) -> str:
    for attr in ("model_version", "model"):
        value = getattr(response, attr, None)
        if isinstance(value, str) and value:
            return value
    return ""


def _response_id(response: Any) -> str:
    for attr in ("response_id", "id"):
        value = getattr(response, attr, None)
        if isinstance(value, str) and value:
            return value
    return ""


class GemmaAIStudio:
    """AI Studio only (vertexai=False, pinned endpoint). No paid fallback.

    free_tier_project is an optional operator note recorded with each run;
    it is not required (the project has no billing configured).

    One wall-clock limit covers every attempt of a case. Each read and each
    backoff uses only the time still left, so the case stays inside that
    limit. Streaming keeps the connection active.
    429 stops immediately. A timeout is terminal. A fast 500/502/503, or a
    transport disconnect, waits on exponential backoff with jitter, and a
    disconnect builds a new client. thinking_level and max_output_tokens are
    generation settings. Wall-clock and retry counts are transport policy.
    The model id is fixed. No second project, key, or API. client_factory
    is a test seam; it receives the same pinned SDK options.
    """

    def __init__(
        self,
        *,
        free_tier_project: bool | None = None,
        model_id: str = MODEL_ID,
        client_factory: Callable[..., Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.perf_counter,
        rng: Callable[[], float] = random.random,
        wall_time: Callable[[], datetime] | None = None,
        max_attempts: int = MAX_ATTEMPTS,
        wall_clock_limit_s: float = WALL_CLOCK_LIMIT_S,
        request_timeout_s: float = REQUEST_TIMEOUT_S,
        request_timeout_cap_s: float = REQUEST_TIMEOUT_CAP_S,
        deadline_growth: float = DEADLINE_GROWTH,
        deadline_retry_pause_s: float = DEADLINE_RETRY_PAUSE_S,
        fast_backoff_min_s: float = FAST_BACKOFF_MIN_S,
        fast_backoff_max_s: float = FAST_BACKOFF_MAX_S,
    ) -> None:
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
            raise ValueError("max_attempts must be an integer of at least 1")
        if not isinstance(model_id, str) or not model_id.startswith("gemma-"):
            raise ValueError("GemmaAIStudio requires a Gemma model; no paid fallback")
        self.model_id = model_id
        self._client: Any | None = None
        self._client_timeout_ms: int | None = None
        self._client_factory = client_factory
        self.free_tier_project = free_tier_project
        self._sleep = sleep
        self._monotonic = monotonic
        self._rng = rng
        self._wall_time = wall_time or (lambda: datetime.now(timezone.utc))
        self.max_attempts = max_attempts
        self.wall_clock_limit_s = _transport_number(
            "wall_clock_limit_s", wall_clock_limit_s, minimum=0, exclusive=True)
        # Accepted so an older config still loads. generate() does not read them.
        self.request_timeout_s = _transport_number("request_timeout_s", request_timeout_s, minimum=0, exclusive=True)
        self.request_timeout_cap_s = _transport_number(
            "request_timeout_cap_s", request_timeout_cap_s, minimum=self.request_timeout_s, exclusive=True)
        self.deadline_growth = _transport_number("deadline_growth", deadline_growth, minimum=1, exclusive=True)
        self.deadline_retry_pause_s = _transport_number("deadline_retry_pause_s", deadline_retry_pause_s, minimum=0)
        self.fast_backoff_min_s = _transport_number("fast_backoff_min_s", fast_backoff_min_s, minimum=0, exclusive=True)
        self.fast_backoff_max_s = _transport_number(
            "fast_backoff_max_s", fast_backoff_max_s, minimum=self.fast_backoff_min_s)

    def _timeout_ms(self, timeout_s: float) -> int:
        # Ceil to the next millisecond. Setup between the deadline sample and
        # this call is under _WALL_EPSILON_S, so the socket still ends with the case.
        return max(1, math.ceil(float(timeout_s) * 1000 - 1e-9))

    def _build_client(self, timeout_s: float) -> Any:
        if _vertex_requested():
            raise ProviderSetupError(
                "Vertex routing is disabled; unset GOOGLE_GENAI_USE_VERTEXAI and use the Gemini API key"
            )
        key = os.environ.get(API_KEY_ENV, "").strip()
        if not key:
            raise MissingAPIKey(f"{API_KEY_ENV} is not set")
        timeout_ms = self._timeout_ms(timeout_s)
        slack_ms = int(_WALL_EPSILON_S * 1000)
        # A cached client is reusable only while its socket timeout still fits
        # this attempt. A longer timeout would let the read run past the case.
        if (
            self._client is not None
            and self._client_timeout_ms is not None
            and self._client_timeout_ms <= timeout_ms + slack_ms
        ):
            return self._client
        factory = self._client_factory
        if factory is None:
            try:
                from google import genai
            except ImportError as exc:
                raise ProviderSetupError(
                    "google-genai is not installed; refusing to call any other API"
                ) from exc
            factory = genai.Client
        self._client = factory(
            vertexai=False, api_key=key,
            http_options={"base_url": AI_STUDIO_ENDPOINT, "timeout": timeout_ms},
        )
        self._client_timeout_ms = timeout_ms
        return self._client

    def _drop_client(self) -> None:
        self._client = None
        self._client_timeout_ms = None

    def _stamp(self) -> str:
        moment = self._wall_time()
        if not isinstance(moment, datetime):
            raise ValueError("wall_time must return a datetime")
        if moment.tzinfo is None or moment.tzinfo.utcoffset(moment) is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment.isoformat()

    def _fast_backoff_s(self, failed_attempt: int) -> float:
        """Exponential backoff with jitter. The first wait is fast_backoff_min_s; later waits reach the cap."""
        ceiling = min(self.fast_backoff_max_s, self.fast_backoff_min_s * (2 ** (failed_attempt - 1)))
        floor = min(self.fast_backoff_min_s, ceiling)
        unit = self._rng()
        if isinstance(unit, bool) or not isinstance(unit, (int, float)) or not math.isfinite(unit):
            unit = 0
        unit = min(1.0, max(0.0, float(unit)))
        return floor + unit * (ceiling - floor)

    def _provider_policy(self) -> dict[str, Any]:
        return {
            "free_tier_project": self.free_tier_project if isinstance(self.free_tier_project, bool) else None,
            "attestation_source": "caller_config" if isinstance(self.free_tier_project, bool) else "none",
            "vertexai": False,
            "base_url": AI_STUDIO_ENDPOINT,
            "wall_clock_limit_s": self.wall_clock_limit_s,
            "max_attempts": self.max_attempts,
            "fast_backoff_min_s": self.fast_backoff_min_s,
            "fast_backoff_max_s": self.fast_backoff_max_s,
        }

    def _attempt_row(
        self, *, elapsed_s: float, status: str, error_class: str,
        started_at: str, ended_at: str, finish_reason: str | None,
        input_tokens: int | None, output_tokens: int | None,
        thoughts_token_count: int | None, timeout_s: float,
    ) -> dict[str, Any]:
        return {
            "latency_ms": elapsed_s * 1000,
            "elapsed_s": elapsed_s,
            "started_at": started_at,
            "ended_at": ended_at,
            "status": status,
            "error_class": error_class,
            "finish_reason": finish_reason,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "prompt_token_count": input_tokens,
            "candidates_token_count": output_tokens,
            "thoughts_token_count": thoughts_token_count,
            "timeout_s": timeout_s,
        }

    def _remaining_s(self, deadline: float) -> float:
        return deadline - self._monotonic()

    def _read_stream(self, stream: Any, deadline: float) -> tuple[str, str | None, tuple[int | None, int | None, int | None], str, str]:
        parts: list[str] = []
        finish: str | None = None
        usage = (None, None, None)
        model = ""
        response_id = ""
        try:
            for chunk in stream:
                if self._monotonic() > deadline + _WALL_EPSILON_S:
                    raise TimeoutError("wall-clock limit")
                piece = _visible_text(chunk)
                if piece:
                    parts.append(piece)
                reason = _finish_reason(chunk)
                if reason:
                    finish = reason
                prompt, output, thoughts = _usage(chunk)
                usage = (
                    prompt if prompt is not None else usage[0],
                    output if output is not None else usage[1],
                    thoughts if thoughts is not None else usage[2],
                )
                found_model = _model_returned(chunk)
                if found_model:
                    model = found_model
                found_id = _response_id(chunk)
                if found_id:
                    response_id = found_id
            if self._monotonic() > deadline + _WALL_EPSILON_S:
                raise TimeoutError("wall-clock limit")
        except Exception as exc:
            raise _PartialStream(
                exc,
                visible="".join(parts),
                finish=finish,
                usage=usage,
                model=model,
                response_id=response_id,
            ) from exc
        return "".join(parts), finish, usage, model, response_id

    def _exchange(
        self, *, request_id: str, recorded: dict[str, Any], attempts: list[dict[str, Any]],
        total_started: float, status: str, raw_response: str, error: str,
        input_tokens: int | None, output_tokens: int | None, model_returned: str,
        provider_response_id: str, attempt: int,
    ) -> ProviderExchange:
        return ProviderExchange(
            request_id=request_id,
            model_requested=self.model_id,
            model_returned=model_returned,
            settings=recorded,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=(self._monotonic() - total_started) * 1000,
            raw_response=raw_response,
            status=status,
            error=error,
            provider_response_id=provider_response_id,
            attempt_count=attempt,
            attempts=list(attempts),
        )

    def generate(
        self,
        prompt: str,
        *,
        request_id: str,
        settings: dict[str, Any] | None = None,
    ) -> ProviderExchange:
        public = {key: value for key, value in dict(settings or {}).items() if key in _SETTINGS_KEYS}
        recorded_settings = redact(public)
        recorded_settings.setdefault("temperature", 0)
        recorded_settings.setdefault("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS)
        recorded_settings.setdefault("thinking_level", "minimal")
        api_config = _generation_config(recorded_settings)
        recorded = {**recorded_settings, "provider_policy": self._provider_policy()}
        attempts: list[dict[str, Any]] = []
        total_started = self._monotonic()
        case_deadline = total_started + float(self.wall_clock_limit_s)
        last_exchange: ProviderExchange | None = None
        for attempt in range(1, self.max_attempts + 1):
            remaining = self._remaining_s(case_deadline)
            if remaining <= _WALL_EPSILON_S:
                break
            # The socket timeout is the time still left, so a read that yields
            # nothing cannot use the original 900 s again.
            attempt_timeout_s = self._timeout_ms(remaining) / 1000
            client = self._build_client(attempt_timeout_s)
            started = self._monotonic()
            started_at = self._stamp()
            try:
                if self._remaining_s(case_deadline) <= _WALL_EPSILON_S:
                    raise TimeoutError("wall-clock limit")
                stream = client.models.generate_content_stream(
                    model=self.model_id,
                    contents=prompt,
                    config=api_config,
                )
                visible, finish, usage, model_returned, response_id = self._read_stream(stream, case_deadline)
            except Exception as exc:
                ended = self._monotonic()
                partial = exc if isinstance(exc, _PartialStream) else None
                cause = partial.cause if partial is not None else exc
                code = _status_code(cause)
                if partial is not None:
                    prompt, output, thoughts = partial.usage
                    fallback = _usage_from_exception(cause)
                    input_tokens = prompt if prompt is not None else fallback[0]
                    output_tokens = output if output is not None else fallback[1]
                    thoughts = thoughts if thoughts is not None else fallback[2]
                    finish_seen = partial.finish
                    raw = partial.visible or _raw_from_exception(cause)
                    model_seen = partial.model
                    response_seen = partial.response_id
                else:
                    input_tokens, output_tokens, thoughts = _usage_from_exception(cause)
                    finish_seen = None
                    raw = _raw_from_exception(cause)
                    model_seen = ""
                    response_seen = ""
                status = "quota" if _is_quota(cause, code) else "error"
                attempts.append(self._attempt_row(
                    elapsed_s=ended - started, status=status, error_class=type(cause).__name__,
                    started_at=started_at, ended_at=self._stamp(), finish_reason=finish_seen,
                    input_tokens=input_tokens, output_tokens=output_tokens,
                    thoughts_token_count=thoughts, timeout_s=attempt_timeout_s,
                ))
                exchange = self._exchange(
                    request_id=request_id, recorded=recorded, attempts=attempts,
                    total_started=total_started, status=status, raw_response=raw,
                    error=redact(str(cause)), input_tokens=input_tokens, output_tokens=output_tokens,
                    model_returned=model_seen, provider_response_id=response_seen, attempt=attempt,
                )
                if _is_quota(cause, code):
                    raise QuotaExceeded(exchange) from cause
                last_exchange = exchange
                # STOP was already received. Keep that finish and the usage on
                # this failure instead of calling the model again.
                if _is_stop(finish_seen):
                    return exchange
                kind = _retry_kind(cause, code)
                if kind is None or attempt >= self.max_attempts:
                    return exchange
                if kind == "disconnect":
                    self._drop_client()
                remaining = self._remaining_s(case_deadline)
                if remaining <= _WALL_EPSILON_S:
                    return exchange
                wait = min(self._fast_backoff_s(attempt), remaining)
                self._sleep(wait)
                if self._remaining_s(case_deadline) <= _WALL_EPSILON_S:
                    return exchange
                continue
            input_tokens, output_tokens, thoughts = usage
            if finish is None or not _is_stop(finish):
                # A missing finish_reason is an incomplete stream and can be
                # retried. Any other reason, including MAX_TOKENS and SAFETY,
                # is terminal. Valid JSON does not make it a graded result.
                incomplete = finish is None
                if incomplete:
                    error = "stream ended without finish_reason"
                    error_class = "StreamEnded"
                elif _is_max_tokens(finish):
                    error = "MAX_TOKENS with no parseable answer" if not _parseable_visible_answer(visible) else "finish_reason=MAX_TOKENS"
                    error_class = "MAX_TOKENS"
                else:
                    error = f"finish_reason={finish}"
                    error_class = finish or "StreamEnded"
                attempts.append(self._attempt_row(
                    elapsed_s=self._monotonic() - started, status="error", error_class=error_class,
                    started_at=started_at, ended_at=self._stamp(), finish_reason=finish,
                    input_tokens=input_tokens, output_tokens=output_tokens,
                    thoughts_token_count=thoughts, timeout_s=attempt_timeout_s,
                ))
                exchange = self._exchange(
                    request_id=request_id, recorded=recorded, attempts=attempts,
                    total_started=total_started, status="error", raw_response=visible,
                    error=error,
                    input_tokens=input_tokens, output_tokens=output_tokens,
                    model_returned=model_returned or self.model_id,
                    provider_response_id=response_id, attempt=attempt,
                )
                if not incomplete or attempt >= self.max_attempts:
                    return exchange
                self._drop_client()
                remaining = self._remaining_s(case_deadline)
                if remaining <= _WALL_EPSILON_S:
                    return exchange
                wait = min(self._fast_backoff_s(attempt), remaining)
                self._sleep(wait)
                if self._remaining_s(case_deadline) <= _WALL_EPSILON_S:
                    return exchange
                continue
            attempts.append(self._attempt_row(
                elapsed_s=self._monotonic() - started, status="ok", error_class="",
                started_at=started_at, ended_at=self._stamp(), finish_reason=finish,
                input_tokens=input_tokens, output_tokens=output_tokens,
                thoughts_token_count=thoughts, timeout_s=attempt_timeout_s,
            ))
            return self._exchange(
                request_id=request_id, recorded=recorded, attempts=attempts,
                total_started=total_started, status="ok", raw_response=visible, error="",
                input_tokens=input_tokens, output_tokens=output_tokens,
                model_returned=model_returned or self.model_id,
                provider_response_id=response_id, attempt=attempt,
            )
        if last_exchange is not None:
            return last_exchange
        raise ProviderSetupError("generate failed")
