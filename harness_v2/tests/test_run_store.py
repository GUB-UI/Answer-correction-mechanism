from __future__ import annotations

import sqlite3

import pytest

from harness_v2.core.contracts import RunRecord
from harness_v2.storage.artifacts import ArtifactRejected, ArtifactStore
from harness_v2.storage.sqlite_store import RunStore


def test_store_redacts_secrets_and_rejects_images(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "unique-test-key-value")
    artifacts = ArtifactStore(tmp_path / "artifacts")
    store = RunStore(tmp_path / "runs.sqlite", artifacts)
    with pytest.raises(ArtifactRejected):
        artifacts.write_text("sheet.png", "not-an-image")
    artifacts.write_text("note.txt", "unique-test-key-value")
    assert "unique-test-key-value" not in artifacts.read_text("note.txt")
    store.save_run(
        RunRecord(
            experiment_id="exp",
            commit="abc",
            provider="fixture",
            model_id="fixture",
            rubric_version="v0",
            prompt_version="v0",
            input_ids=["i1"],
            outputs={"note": "unique-test-key-value"},
            checks=[],
            latency_ms=1,
            input_tokens=3,
            output_tokens=4,
            failure_reason="",
            proposal_link="",
            request_id="r",
            created_at="2026-10-23T00:00:00Z",
        )
    )
    blob = (tmp_path / "runs.sqlite").read_bytes()
    assert b"unique-test-key-value" not in blob
    saved = store.list_runs()[0]
    assert saved["input_tokens"] == 3
    assert saved["output_tokens"] == 4
    reopened = RunStore(store.db_path, artifacts, readonly=True)
    assert reopened.list_runs() == [saved]
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        reopened.update_outputs(saved["run_id"], {"note": "replacement"})
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        reopened.save_run(RunRecord(**{**saved, "run_id": "another-run"}))
    assert reopened.list_runs() == [saved]


def test_readonly_store_does_not_create_missing_database_or_parents(tmp_path):
    artifacts = ArtifactStore(tmp_path / "artifacts")
    path = tmp_path / "missing" / "database.sqlite"
    store = RunStore(path, artifacts, readonly=True)
    with pytest.raises(sqlite3.OperationalError, match="unable to open database"):
        store.list_runs()
    assert not path.parent.exists()


def test_readonly_store_skips_schema_initialization(tmp_path):
    from harness_v2.storage.permissions import private_open

    path = tmp_path / "empty.sqlite"
    with private_open(path, "wb"):
        pass
    store = RunStore(path, ArtifactStore(tmp_path / "artifacts"), readonly=True)
    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        store.list_runs()
    assert path.read_bytes() == b""
    with store._connect() as conn:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("CREATE TABLE forbidden (value TEXT)")
