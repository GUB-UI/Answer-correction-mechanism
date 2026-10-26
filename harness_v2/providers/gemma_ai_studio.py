"""Google AI Studio client for Gemma. One key, one model, no paid fallback."""

from __future__ import annotations

import math
import os
import random
import time
from typing import Any, Callable

from harness_v2.core.contracts import ProviderExchange
from harness_v2.storage.secrets import redact

MODEL_ID = "gemma-4-31b-it"
API_KEY_ENV = "GEMINI_API_KEY"
AI_STUDIO_ENDPOINT = "https://generativelanguage.googleapis.com"
MAX_ATTEMPTS = 6
# First-attempt HTTP timeout. google-genai sends this as the request deadline.
# 180s matched observed 504s on long OCR answers; 600s is the default.
REQUEST_TIMEOUT_S = 600
REQUEST_TIMEOUT_CAP_S = 1800
DEADLINE_GROWTH = 2
DEADLINE_RETRY_PAUSE_S = 1
FAST_BACKOFF_MIN_S = 30
FAST_BACKOFF_MAX_S = 120
REQUEST_TIMEOUT_MS = REQUEST_TIMEOUT_S * 1000
TRANSPORT_CONFIG_KEYS = (
    "request_timeout_s",
    "request_timeout_cap_s",
    "deadline_growth",
    "deadline_retry_pause_s",
    "max_attempts",
    "fast_backoff_min_s",
    "fast_backoff_max_s",
)
_FAST_HTTP = {500, 502, 503}
_DEADLINE_HTTP = {408, 504}
_VERTEX_ENV = ("GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_GENAI_USE_VERTEX_AI")
_SETTINGS_KEYS = ("temperature", "max_output_tokens", "top_p", "top_k", "seed")


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
    """504 and read timeouts need a longer deadline, not the same one again."""
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
    if isinstance(exc, ConnectionError):
        return True
    name = type(exc).__name__.lower()
    return "connection" in name or "connect" in name


def _raw_from_exception(exc: BaseException) -> str:
    response = getattr(exc, "response", None)
    body = getattr(response, "text", None)
    if isinstance(body, str) and body:
        return body
    details = getattr(exc, "details", None)
    if isinstance(details, str) and details:
        return details
    return str(exc)


def _response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text
    candidates = getattr(response, "candidates", None) or []
    parts: list[str] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str):
                parts.append(part_text)
    return "".join(parts)


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

    429 stops immediately. A 504 or read timeout is retried with a longer
    deadline. A fast 500/502/503 waits on exponential backoff with jitter.
    The model id is fixed. No second project, key, or API. client_factory
    is a test seam; it receives the same pinned SDK options. Timeout and
    retry settings are transport policy: they are not generation settings.
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
        max_attempts: int = MAX_ATTEMPTS,
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
        self.max_attempts = max_attempts
        self.request_timeout_s = _transport_number("request_timeout_s", request_timeout_s, minimum=0, exclusive=True)
        self.request_timeout_cap_s = _transport_number(
            "request_timeout_cap_s", request_timeout_cap_s, minimum=self.request_timeout_s, exclusive=True)
        self.deadline_growth = _transport_number("deadline_growth", deadline_growth, minimum=1, exclusive=True)
        self.deadline_retry_pause_s = _transport_number("deadline_retry_pause_s", deadline_retry_pause_s, minimum=0)
        self.fast_backoff_min_s = _transport_number("fast_backoff_min_s", fast_backoff_min_s, minimum=0, exclusive=True)
        self.fast_backoff_max_s = _transport_number(
            "fast_backoff_max_s", fast_backoff_max_s, minimum=self.fast_backoff_min_s)

    def _timeout_ms(self, timeout_s: float) -> int:
        return int(round(float(timeout_s) * 1000))

    def _build_client(self, timeout_s: float) -> Any:
        if _vertex_requested():
            raise ProviderSetupError(
                "Vertex routing is disabled; unset GOOGLE_GENAI_USE_VERTEXAI and use the Gemini API key"
            )
        key = os.environ.get(API_KEY_ENV, "").strip()
        if not key:
            raise MissingAPIKey(f"{API_KEY_ENV} is not set")
        timeout_ms = self._timeout_ms(timeout_s)
        if self._client is not None and self._client_timeout_ms == timeout_ms:
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

    def _fast_backoff_s(self, failed_attempt: int) -> float:
        """Exponential backoff with jitter. The first wait is fast_backoff_min_s; later waits reach the cap."""
        ceiling = min(self.fast_backoff_max_s, self.fast_backoff_min_s * (2 ** (failed_attempt - 1)))
        floor = min(self.fast_backoff_min_s, ceiling)
        unit = self._rng()
        if isinstance(unit, bool) or not isinstance(unit, (int, float)) or not math.isfinite(unit):
            unit = 0
        unit = min(1.0, max(0.0, float(unit)))
        return floor + unit * (ceiling - floor)

    def _next_timeout_s(self, timeout_s: float) -> float:
        grown = float(timeout_s) * float(self.deadline_growth)
        return min(float(self.request_timeout_cap_s), grown)

    def _provider_policy(self) -> dict[str, Any]:
        return {
            "free_tier_project": self.free_tier_project if isinstance(self.free_tier_project, bool) else None,
            "attestation_source": "caller_config" if isinstance(self.free_tier_project, bool) else "none",
            "vertexai": False,
            "base_url": AI_STUDIO_ENDPOINT,
            "request_timeout_s": self.request_timeout_s,
            "request_timeout_cap_s": self.request_timeout_cap_s,
            "deadline_growth": self.deadline_growth,
            "deadline_retry_pause_s": self.deadline_retry_pause_s,
            "max_attempts": self.max_attempts,
            "fast_backoff_min_s": self.fast_backoff_min_s,
            "fast_backoff_max_s": self.fast_backoff_max_s,
        }

    def _attempt_row(self, *, elapsed_s: float, status: str, error_class: str, timeout_s: float,
                     input_tokens: int | None, output_tokens: int | None,
                     thoughts_token_count: int | None) -> dict[str, Any]:
        return {
            "latency_ms": elapsed_s * 1000,
            "elapsed_s": elapsed_s,
            "status": status,
            "error_class": error_class,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "thoughts_token_count": thoughts_token_count,
            "timeout_s": timeout_s,
        }

    def generate(
        self,
        prompt: str,
        *,
        request_id: str,
        settings: dict[str, Any] | None = None,
    ) -> ProviderExchange:
        public = {key: value for key, value in dict(settings or {}).items() if key in _SETTINGS_KEYS}
        config = redact(public)
        config.setdefault("temperature", 0)
        recorded = {**config, "provider_policy": self._provider_policy()}
        last_error = ""
        attempts: list[dict[str, Any]] = []
        total_started = self._monotonic()
        timeout_s = float(self.request_timeout_s)
        for attempt in range(1, self.max_attempts + 1):
            client = self._build_client(timeout_s)
            started = self._monotonic()
            try:
                response = client.models.generate_content(
                    model=self.model_id,
                    contents=prompt,
                    config=config,
                )
            except Exception as exc:
                elapsed_s = self._monotonic() - started
                code = _status_code(exc)
                raw = _raw_from_exception(exc)
                input_tokens, output_tokens, thoughts = _usage_from_exception(exc)
                status = "quota" if _is_quota(exc, code) else "error"
                attempts.append(self._attempt_row(
                    elapsed_s=elapsed_s, status=status, error_class=type(exc).__name__,
                    timeout_s=timeout_s, input_tokens=input_tokens, output_tokens=output_tokens,
                    thoughts_token_count=thoughts,
                ))
                exchange = ProviderExchange(
                    request_id=request_id,
                    model_requested=self.model_id,
                    model_returned="",
                    settings=recorded,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=(self._monotonic() - total_started) * 1000,
                    raw_response=raw,
                    status=status,
                    error=redact(str(exc)),
                    attempt_count=attempt,
                    attempts=list(attempts),
                )
                if _is_quota(exc, code):
                    raise QuotaExceeded(exchange) from exc
                last_error = exchange.error
                if attempt >= self.max_attempts:
                    return exchange
                if code in _FAST_HTTP or _is_connection(exc):
                    self._sleep(self._fast_backoff_s(attempt))
                    continue
                if _is_deadline(exc, code):
                    timeout_s = self._next_timeout_s(timeout_s)
                    self._sleep(float(self.deadline_retry_pause_s))
                    continue
                return exchange
            elapsed_s = self._monotonic() - started
            input_tokens, output_tokens, thoughts = _usage(response)
            attempts.append(self._attempt_row(
                elapsed_s=elapsed_s, status="ok", error_class="", timeout_s=timeout_s,
                input_tokens=input_tokens, output_tokens=output_tokens,
                thoughts_token_count=thoughts,
            ))
            return ProviderExchange(
                request_id=request_id,
                model_requested=self.model_id,
                model_returned=_model_returned(response) or self.model_id,
                settings=recorded,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=(self._monotonic() - total_started) * 1000,
                raw_response=_response_text(response),
                status="ok",
                provider_response_id=_response_id(response),
                attempt_count=attempt,
                attempts=list(attempts),
            )
        raise ProviderSetupError(last_error or "generate failed")
