"""File and GitHub handoff for Grokbot. No HTTP model API.

A request and its result share `request_id`. GitHub delivery writes the
issue, branch, and pull-request text as files. Opening them is a separate
step and is not done here.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from harness_v2.storage.secrets import redact
from harness_v2.storage.permissions import private_directory, private_write_text
from harness_v2.evaluation.independent import scrub_score_text

_REQUEST_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_DATA_URI = re.compile(r"data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=\s]+", re.IGNORECASE)
_IMAGE_CONTENT = re.compile(
    r"data\s*:\s*image/|!\[[^\]]*\]\(|<\s*(?:img|svg)\b|"
    r"https?://[^\s\"<>]+\.(?:png|jpe?g|gif|webp|bmp|tiff?|svg)(?:\b|\?)|"
    r"(?:iVBORw0KGgo|/9j/|R0lGOD|UklGR)[A-Za-z0-9+/=]{8,}", re.IGNORECASE,
)

_SCHEMAS = Path(__file__).resolve().parent / "schemas"


class SchemaError(ValueError):
    pass


def _load_schema(name: str) -> dict[str, Any]:
    return json.loads((_SCHEMAS / name).read_text(encoding="utf-8"))


def _check(value: Any, schema: dict[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    expected = schema.get("type")
    if expected == "object":
        if not isinstance(value, dict):
            return [f"{path or 'object'} must be an object"]
        required = schema.get("required") or []
        for key in required:
            if key not in value:
                errors.append(f"{path + '.' if path else ''}{key} is required")
        if schema.get("additionalProperties") is False:
            allowed = set((schema.get("properties") or {}).keys())
            for key in value:
                if key not in allowed:
                    errors.append(f"{path + '.' if path else ''}{key} is not allowed")
        for key, sub in (schema.get("properties") or {}).items():
            if key in value:
                child = f"{path}.{key}" if path else key
                errors.extend(_check(value[key], sub, child))
        return errors
    if expected == "array":
        if not isinstance(value, list):
            return [f"{path} must be an array"]
        item_schema = schema.get("items") or {}
        for index, item in enumerate(value):
            errors.extend(_check(item, item_schema, f"{path}[{index}]"))
        return errors
    if expected == "string":
        if not isinstance(value, str):
            return [f"{path} must be a string"]
        if schema.get("minLength") and len(value) < schema["minLength"]:
            return [f"{path} is empty"]
        return errors
    if "enum" in schema:
        if value not in schema["enum"]:
            return [f"{path} must be one of {schema['enum']}"]
        return errors
    if expected == "boolean" and not isinstance(value, bool):
        return [f"{path} must be a boolean"]
    if expected == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return [f"{path} must be a finite number"]
        if "minimum" in schema and value < schema["minimum"]:
            return [f"{path} is below minimum"]
    return errors


def safe_request_id(request_id: str) -> str:
    if not isinstance(request_id, str) or _REQUEST_ID.fullmatch(request_id) is None:
        raise SchemaError(f"request_id is not a safe token: {request_id!r}")
    return request_id


def strip_image_data(value: Any) -> Any:
    if isinstance(value, str):
        return _DATA_URI.sub("[stripped-image]", value)
    if isinstance(value, dict):
        return {str(key): strip_image_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [strip_image_data(item) for item in value]
    return value


def _inside(root: Path, path: Path) -> Path:
    resolved_root = root.resolve()
    resolved = path.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise SchemaError(f"path escapes the exchange directory: {path}")
    return resolved


def validate_request(payload: dict[str, Any]) -> None:
    errors = _check(payload, _load_schema("grokbot_request.schema.json"), "")
    if errors:
        raise SchemaError("; ".join(errors))
    _reject_images(payload)


def _reject_images(value: Any) -> None:
    if isinstance(value, str) and _IMAGE_CONTENT.search(value):
        raise SchemaError("image-bearing content cannot be published")
    if isinstance(value, dict):
        for item in value.values():
            _reject_images(item)
    elif isinstance(value, list):
        for item in value:
            _reject_images(item)


def _blind_text(value: Any) -> Any:
    if isinstance(value, str):
        return scrub_score_text(value)
    if isinstance(value, dict):
        return {key: _blind_text(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_blind_text(item) for item in value]
    return value


def _publishable_request(payload: dict[str, Any]) -> dict[str, Any]:
    # Validate before redaction: unsupported/image fields must not be hidden by
    # cleanup and then published under an unrestricted payload object.
    validate_request(payload)
    cleaned = redact(payload)
    if cleaned.get("independent_evaluation"):
        cleaned["payload"] = _blind_text(cleaned["payload"])
    validate_request(cleaned)
    return cleaned


def validate_result(payload: dict[str, Any]) -> None:
    errors = _check(payload, _load_schema("grokbot_result.schema.json"), "")
    if errors:
        raise SchemaError("; ".join(errors))


class FileExchange:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        private_directory(self.root)
        private_directory(self.root / "requests")
        private_directory(self.root / "results")

    def _path(self, folder: str, request_id: str) -> Path:
        token = safe_request_id(request_id)
        return _inside(self.root / folder, self.root / folder / f"{token}.json")

    def publish_request(self, payload: dict[str, Any]) -> Path:
        cleaned = _publishable_request(payload)
        private_directory(self.root, managed=True)
        private_directory(self.root / "requests", managed=True)
        path = self._path("requests", cleaned["request_id"])
        private_write_text(path, json.dumps(cleaned, ensure_ascii=False, indent=2))
        return path

    def write_result(self, payload: dict[str, Any]) -> Path:
        cleaned = redact(strip_image_data(payload))
        validate_result(cleaned)
        request_path = self._path("requests", cleaned["request_id"])
        if not request_path.exists():
            raise SchemaError(f"no request with request_id {cleaned['request_id']}")
        private_directory(self.root, managed=True)
        private_directory(self.root / "results", managed=True)
        path = self._path("results", cleaned["request_id"])
        private_write_text(path, json.dumps(cleaned, ensure_ascii=False, indent=2))
        return path

    def read_result(self, request_id: str) -> dict[str, Any]:
        path = self._path("results", request_id)
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise SchemaError("result is not an object")
        validate_result(payload)
        if payload.get("request_id") != request_id:
            raise SchemaError("result request_id does not match the file")
        return payload


class GitHubHandoff:
    """Write the Issue, branch name, and PR body. Do not call GitHub."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def write(self, payload: dict[str, Any]) -> Path:
        cleaned = _publishable_request(payload)
        request_id = safe_request_id(cleaned["request_id"])
        folder = _inside(self.root, self.root / "handoff" / request_id)
        private_directory(self.root, managed=True)
        private_directory(self.root / "handoff", managed=True)
        private_directory(folder, managed=True)
        branch = f"harness/{request_id}"
        private_write_text(folder / "request.json",
            json.dumps(cleaned, ensure_ascii=False, indent=2),
        )
        private_write_text(folder / "branch.txt", branch + "\n")
        private_write_text(folder / "issue.md",
            f"# Grokbot {cleaned['kind']} {request_id}\n\n"
            f"experiment: {cleaned['experiment_id']}\n\n"
            f"commit: {cleaned['commit']}\n\n"
            "Open this file as a GitHub issue. The harness does not post it.\n",
        )
        private_write_text(folder / "pr.md",
            f"# PR for {request_id}\n\n"
            f"Branch `{branch}` holds the handoff files for this request_id.\n"
            "Open the pull request separately. This file is the body.\n",
        )
        return folder
