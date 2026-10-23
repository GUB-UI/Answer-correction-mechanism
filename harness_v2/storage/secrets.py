"""Strip secrets before anything is written to disk."""

from __future__ import annotations

import os
import re
from typing import Any

_SECRET = re.compile(r"AIza[0-9A-Za-z\-_]{8,}|sk-[A-Za-z0-9]{8,}")
_SENSITIVE_PARTS = ("api_key", "apikey", "secret", "password", "authorization", "token")


def _sensitive_key(key: str) -> bool:
    lowered = str(key).lower().replace("-", "_")
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


def _redact(value: Any, secrets: list[str]) -> Any:
    if isinstance(value, str):
        text = _SECRET.sub("[redacted]", value)
        for secret in secrets:
            text = text.replace(secret, "[redacted]")
        return text
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if _sensitive_key(str(key)):
                cleaned[str(key)] = "[redacted]"
            else:
                cleaned[str(key)] = _redact(item, secrets)
        return cleaned
    if isinstance(value, list):
        return [_redact(item, secrets) for item in value]
    return value
