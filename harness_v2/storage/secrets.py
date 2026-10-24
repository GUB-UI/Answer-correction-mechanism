"""Strip secrets before anything is written to disk."""

from __future__ import annotations

import json
import os
import re
from typing import Any

_SECRET = re.compile(r"AIza[0-9A-Za-z\-_]{8,}|sk-[A-Za-z0-9]{8,}")
_SENSITIVE_PARTS = ("api_key", "apikey", "secret", "password", "authorization", "token")


def _sensitive_key(key: str, value: Any = None) -> bool:
    lowered = str(key).lower().replace("-", "_")
    if (lowered.endswith(("_token_runs", "_token_attempts"))
            and type(value) in {int, float}):
        return False
    if lowered.endswith("_tokens") or lowered.endswith("_token_count"):
        return False
    return any(part in lowered for part in _SENSITIVE_PARTS)


def _env_secrets() -> list[str]:
    found: list[str] = []
    for name in ("GEMINI_API_KEY",):
        value = os.environ.get(name, "").strip()
        if len(value) >= 8:
            found.append(value)
    return found


def redact(value: Any) -> Any:
    secrets = _env_secrets()
    return _redact(value, secrets)


def _redact_text(text: str, secrets: list[str]) -> str:
    cleaned = _SECRET.sub("[redacted]", text)
    for secret in secrets:
        cleaned = cleaned.replace(secret, "[redacted]")
    return cleaned


def _redact_embedded(text: str, secrets: list[str]) -> str:
    """Redact secret keys inside JSON objects that sit inside a larger string."""
    decoder = json.JSONDecoder()
    pieces: list[str] = []
    index = 0
    while index < len(text):
        if text[index] in "{[":
            try:
                loaded, end = decoder.raw_decode(text, index)
            except json.JSONDecodeError:
                pieces.append(text[index])
                index += 1
                continue
            if isinstance(loaded, (dict, list)):
                cleaned = _redact(loaded, secrets)
                pieces.append(text[index:end] if cleaned == loaded else json.dumps(cleaned, ensure_ascii=False))
                index = end
                continue
        pieces.append(text[index])
        index += 1
    return _redact_text("".join(pieces), secrets)


def _redact(value: Any, secrets: list[str]) -> Any:
    if isinstance(value, str):
        return _redact_embedded(value, secrets)
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if _sensitive_key(str(key), item):
                cleaned[str(key)] = "[redacted]"
            else:
                cleaned[str(key)] = _redact(item, secrets)
        return cleaned
    if isinstance(value, list):
        return [_redact(item, secrets) for item in value]
    return value
