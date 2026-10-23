"""File and GitHub handoff for Grokbot. No HTTP model API.

A request and its result share `request_id`. GitHub delivery writes the
issue, branch, and pull-request text as files. Opening them is a separate
step and is not done here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from harness_v2.storage.secrets import redact

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
    return errors


def validate_request(payload: dict[str, Any]) -> None:
    errors = _check(payload, _load_schema("grokbot_request.schema.json"), "")
    if errors:
        raise SchemaError("; ".join(errors))


def validate_result(payload: dict[str, Any]) -> None:
    errors = _check(payload, _load_schema("grokbot_result.schema.json"), "")
    if errors:
        raise SchemaError("; ".join(errors))


class FileExchange:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        (self.root / "requests").mkdir(parents=True, exist_ok=True)
        (self.root / "results").mkdir(parents=True, exist_ok=True)

    def publish_request(self, payload: dict[str, Any]) -> Path:
        cleaned = redact(payload)
        validate_request(cleaned)
        path = self.root / "requests" / f"{cleaned['request_id']}.json"
        path.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def write_result(self, payload: dict[str, Any]) -> Path:
        cleaned = redact(payload)
        validate_result(cleaned)
        request_path = self.root / "requests" / f"{cleaned['request_id']}.json"
        if not request_path.exists():
            raise SchemaError(f"no request with request_id {cleaned['request_id']}")
        path = self.root / "results" / f"{cleaned['request_id']}.json"
        path.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def read_result(self, request_id: str) -> dict[str, Any]:
        path = self.root / "results" / f"{request_id}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("request_id") != request_id:
            raise SchemaError("result request_id does not match the file")
        return payload


class GitHubHandoff:
    """Write the Issue, branch name, and PR body. Do not call GitHub."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def write(self, payload: dict[str, Any]) -> Path:
        cleaned = redact(payload)
        validate_request(cleaned)
        request_id = cleaned["request_id"]
        folder = self.root / "handoff" / request_id
        folder.mkdir(parents=True, exist_ok=True)
        branch = f"harness/{request_id}"
        (folder / "request.json").write_text(
            json.dumps(cleaned, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (folder / "branch.txt").write_text(branch + "\n", encoding="utf-8")
        (folder / "issue.md").write_text(
            f"# Grokbot {cleaned['kind']} {request_id}\n\n"
            f"experiment: {cleaned['experiment_id']}\n\n"
            f"commit: {cleaned['commit']}\n\n"
            "Open this file as a GitHub issue. The harness does not post it.\n",
            encoding="utf-8",
        )
        (folder / "pr.md").write_text(
            f"# PR for {request_id}\n\n"
            f"Branch `{branch}` holds the handoff files for this request_id.\n"
            "Open the pull request separately. This file is the body.\n",
            encoding="utf-8",
        )
        return folder
