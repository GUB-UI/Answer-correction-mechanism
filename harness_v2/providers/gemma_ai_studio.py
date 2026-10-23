"""Google AI Studio client for Gemma. One key, one model, no paid fallback."""

from __future__ import annotations

import os
import time
from typing import Any, Callable

from harness_v2.core.contracts import ProviderExchange
from harness_v2.storage.secrets import redact

MODEL_ID = "gemma-4-31b-it"
API_KEY_ENV = "GEMINI_API_KEY"
MAX_ATTEMPTS = 3
_TRANSIENT_STATUS = {408, 500, 502, 503, 504}


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


def _is_transient(exc: BaseException, code: int | None) -> bool:
    if code in _TRANSIENT_STATUS:
        return True
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    name = type(exc).__name__.lower()
    return "timeout" in name or "connection" in name


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


def _usage(response: Any) -> tuple[int | None, int | None]:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return None, None
    prompt = getattr(usage, "prompt_token_count", None)
    output = getattr(usage, "candidates_token_count", None)
    return (
        prompt if isinstance(prompt, int) else None,
        output if isinstance(output, int) else None,
    )


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
    """Calls `google-genai` only after GEMINI_API_KEY is present.

    429 stops immediately. Other transient errors retry with backoff up to
    MAX_ATTEMPTS. The model id is fixed. No second project, key, or API.
    """

    def __init__(
        self,
        *,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.perf_counter,
        max_attempts: int = MAX_ATTEMPTS,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self._client = client
        self._sleep = sleep
        self._monotonic = monotonic
        self.max_attempts = max_attempts

    def _build_client(self) -> Any:
        if self._client is not None:
            return self._client
        key = os.environ.get(API_KEY_ENV, "").strip()
        if not key:
            raise MissingAPIKey(f"{API_KEY_ENV} is not set")
        try:
            from google import genai
        except ImportError as exc:
            raise ProviderSetupError(
                "google-genai is not installed; refusing to call any other API"
            ) from exc
        self._client = genai.Client(api_key=key)
        return self._client

    def generate(
        self,
        prompt: str,
        *,
        request_id: str,
        settings: dict[str, Any] | None = None,
    ) -> ProviderExchange:
        public = {
            key: value
            for key, value in dict(settings or {}).items()
            if key != "model"
            and not any(
                part in key.lower()
                for part in ("api_key", "apikey", "secret", "token", "password", "authorization")
            )
        }
        recorded = redact(public)
        recorded.setdefault("temperature", 0)
        client = self._build_client()
        last_error = ""
        for attempt in range(1, self.max_attempts + 1):
            started = self._monotonic()
            try:
                response = client.models.generate_content(
                    model=MODEL_ID,
                    contents=prompt,
                    config=recorded,
                )
            except Exception as exc:
                latency_ms = (self._monotonic() - started) * 1000
                code = _status_code(exc)
                raw = _raw_from_exception(exc)
                exchange = ProviderExchange(
                    request_id=request_id,
                    model_requested=MODEL_ID,
                    model_returned="",
                    settings=recorded,
                    input_tokens=None,
                    output_tokens=None,
                    latency_ms=latency_ms,
                    raw_response=raw,
                    status="quota" if _is_quota(exc, code) else "error",
                    error=redact(str(exc)),
                    attempt_count=attempt,
                )
                if _is_quota(exc, code):
                    raise QuotaExceeded(exchange) from exc
                last_error = exchange.error
                if _is_transient(exc, code) and attempt < self.max_attempts:
                    self._sleep(min(8.0, 0.5 * (2 ** (attempt - 1))))
                    continue
                exchange.error = last_error
                return exchange
            latency_ms = (self._monotonic() - started) * 1000
            input_tokens, output_tokens = _usage(response)
            return ProviderExchange(
                request_id=request_id,
                model_requested=MODEL_ID,
                model_returned=_model_returned(response) or MODEL_ID,
                settings=recorded,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                raw_response=_response_text(response),
                status="ok",
                provider_response_id=_response_id(response),
                attempt_count=attempt,
            )
        raise ProviderSetupError(last_error or "generate failed")
