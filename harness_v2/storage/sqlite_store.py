"""SQLite run records. The database stores redacted JSON, not images or keys."""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from harness_v2.core.contracts import RunRecord
from harness_v2.storage.artifacts import ArtifactStore
from harness_v2.storage.secrets import redact

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    commit_sha TEXT NOT NULL,
    provider TEXT NOT NULL,
    model_id TEXT NOT NULL,
    rubric_version TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    input_ids_json TEXT NOT NULL,
    outputs_json TEXT NOT NULL,
    checks_json TEXT NOT NULL,
    latency_ms REAL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    failure_reason TEXT NOT NULL,
    proposal_link TEXT NOT NULL,
    request_id TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""


class RunStore:
    def __init__(self, db_path: Path, artifacts: ArtifactStore) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifacts = artifacts
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def save_run(self, record: RunRecord) -> RunRecord:
        cleaned = redact(record.to_dict())
        run_id = str(cleaned.get("run_id") or "") or uuid.uuid4().hex
        cleaned["run_id"] = run_id
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs (
                    run_id, experiment_id, commit_sha, provider, model_id,
                    rubric_version, prompt_version, input_ids_json, outputs_json,
                    checks_json, latency_ms, input_tokens, output_tokens,
                    failure_reason, proposal_link, request_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    cleaned["experiment_id"],
                    cleaned["commit"],
                    cleaned["provider"],
                    cleaned["model_id"],
                    cleaned["rubric_version"],
                    cleaned["prompt_version"],
                    json.dumps(cleaned["input_ids"], ensure_ascii=False),
                    json.dumps(cleaned["outputs"], ensure_ascii=False),
                    json.dumps(cleaned["checks"], ensure_ascii=False),
                    cleaned["latency_ms"],
                    cleaned["input_tokens"],
                    cleaned["output_tokens"],
                    cleaned["failure_reason"],
                    cleaned["proposal_link"],
                    cleaned["request_id"],
                    cleaned["created_at"],
                ),
            )
        record.run_id = run_id
        return record

    def list_runs(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM runs ORDER BY created_at, run_id").fetchall()
        found: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["input_ids"] = json.loads(item.pop("input_ids_json"))
            item["outputs"] = json.loads(item.pop("outputs_json"))
            item["checks"] = json.loads(item.pop("checks_json"))
            item["commit"] = item.pop("commit_sha")
            found.append(item)
        return found
