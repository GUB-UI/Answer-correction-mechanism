"""Private data permissions must not depend on the caller's umask."""
import contextlib
import json
import os
import stat
import sys
from types import SimpleNamespace

import pytest

from harness_v2.storage.artifacts import ArtifactStore, ImmutableArtifactError
from harness_v2.storage.sqlite_store import RunStore


def mode(path):
    return stat.S_IMODE(os.stat(path).st_mode)


@pytest.fixture(autouse=True)
def permissive_umask():
    previous = os.umask(0o000)
    try:
        yield
    finally:
        os.umask(previous)


@pytest.mark.parametrize("name", ["ocr_private/a.txt", "grading_private/a.txt", "private/nested/a.txt", "raw/a.txt"])
def test_private_artifact_creation_and_existing_opens_are_owner_only(tmp_path, name):
    root = tmp_path / "new" / "artifacts"
    store = ArtifactStore(root)
    path, _ = store.write_private_text(name, "exact\r\n")
    assert mode(path) == 0o600
    for folder in [root.parent, root, *path.relative_to(root).parents]:
        target = folder if folder.is_absolute() else root / folder
        assert mode(target) == 0o700
    path.chmod(0o666)
    path.parent.chmod(0o777)
    assert store.read_bytes(name) == b"exact\r\n"
    assert mode(path) == 0o600
    assert mode(path.parent) == 0o700
    path.chmod(0o666)
    assert store.read_text(name) == "exact\n"
    assert mode(path) == 0o600
    path.chmod(0o666)
    with pytest.raises(ImmutableArtifactError):
        store.write_private_text(name, "replacement")
    assert mode(path) == 0o600
    assert path.read_bytes() == b"exact\r\n"


def test_existing_artifact_directories_are_secured(tmp_path):
    root = tmp_path / "artifacts"
    folder = root / "ocr_private" / "nested"
    folder.mkdir(parents=True)
    root.chmod(0o777)
    store = ArtifactStore(root)
    store.write_text("ocr_private/nested/a.txt", "answer")
    assert mode(root) == mode(folder.parent) == mode(folder) == 0o700
    assert mode(folder / "a.txt") == 0o600


@pytest.mark.parametrize("journal_mode,suffix", [("PERSIST", "-journal"), ("WAL", "-wal")])
def test_database_and_live_sidecars_are_private(tmp_path, journal_mode, suffix):
    path = tmp_path / "new" / "database" / "runs.sqlite"
    store = RunStore(path, ArtifactStore(tmp_path / "artifacts"))
    assert mode(path) == 0o600
    assert mode(path.parent) == mode(path.parent.parent) == 0o700
    with store._connect() as conn:
        conn.execute(f"PRAGMA journal_mode={journal_mode}")
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("INSERT INTO runs SELECT * FROM runs")
        conn.execute("CREATE TABLE private_test (value TEXT)")
        sidecar = path.with_name(path.name + suffix)
        assert sidecar.exists()
        assert mode(sidecar) == 0o600
        if journal_mode == "WAL":
            assert mode(path.with_name(path.name + "-shm")) == 0o600
    path.chmod(0o666)
    sidecar.chmod(0o666) if sidecar.exists() else None
    path.parent.chmod(0o777)
    store.list_runs()
    assert mode(path) == 0o600
    assert mode(path.parent) == 0o700
    if sidecar.exists():
        assert mode(sidecar) == 0o600


def test_existing_database_and_sidecars_are_secured_before_connect(tmp_path):
    path = tmp_path / "runs.sqlite"
    store = RunStore(path, ArtifactStore(tmp_path / "artifacts"))
    # A zero-length rollback journal is safe for SQLite to inspect on opening.
    journal = path.with_name(path.name + "-journal")
    journal.touch(mode=0o666)
    path.chmod(0o666)
    store = RunStore(path, store.artifacts)
    assert mode(path) == 0o600
    assert mode(journal) == 0o600


def test_cpu_raw_ocr_outputs_are_private_on_create_and_rewrite(tmp_path, monkeypatch):
    from PIL import Image
    from harness_v2.recognition.run_ocr_cpu import _run_model

    class Tokens:
        shape = [1, 2]
        def __getitem__(self, index):
            return [0, 1]

    processor = SimpleNamespace(
        apply_chat_template=lambda *args, **kwargs: {"input_ids": SimpleNamespace(shape=[1, 1])},
        decode=lambda *args, **kwargs: "raw student OCR")
    model = SimpleNamespace(eval=lambda: SimpleNamespace(generate=lambda **kwargs: Tokens()))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        set_num_threads=lambda n: None, float32="float32", inference_mode=contextlib.nullcontext))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoProcessor=SimpleNamespace(from_pretrained=lambda *a, **kw: processor),
        AutoModelForImageTextToText=SimpleNamespace(from_pretrained=lambda *a, **kw: model)))
    image = tmp_path / "sheet.png"
    Image.new("RGB", (2, 2)).save(image)
    out = tmp_path / "raw" / "outputs"
    args = SimpleNamespace(threads=1, out=str(out), images=[str(image)], max_edge=0, max_new_tokens=2)
    assert _run_model(args) == 0
    assert mode(out) == mode(out.parent) == 0o700
    for path in out.iterdir():
        assert mode(path) == 0o600
        path.chmod(0o666)
    out.chmod(0o777)
    assert _run_model(args) == 0
    assert mode(out) == 0o700
    assert mode(out / "sheet.md") == mode(out / "stats.jsonl") == 0o600
    assert (out / "sheet.md").read_text() == "raw student OCR"


def test_cli_result_with_raw_grade_is_private(tmp_path):
    from harness_v2.__main__ import main
    source = tmp_path / "case.json"
    source.write_text(json.dumps({"request_id": "r", "answer_text": "x",
        "rubric": [{"item_id": "a", "max_points": 1}], "new_completion": "invalid raw answer"}))
    out = tmp_path / "out"
    assert main(["--input", str(source), "--out", str(out)]) == 0
    assert mode(out) == 0o700
    assert mode(out / "result.json") == 0o600
    (out / "result.json").chmod(0o666)
    source.write_text("{}")
    assert main(["--input", str(source), "--out", str(out)]) == 0
    assert mode(out / "result.json") == 0o600


def test_imported_raw_ocr_is_secured_when_opened(tmp_path):
    from harness_v2.recognition.ocr import XiaomiOCR
    source = tmp_path / "raw.txt"
    source.write_text("raw answer")
    source.chmod(0o666)
    artifacts = ArtifactStore(tmp_path / "artifacts")
    XiaomiOCR(artifacts).import_file(source, ocr_id="r")
    assert mode(source) == 0o600


def test_exchange_and_handoff_student_text_is_owner_only(tmp_path):
    from harness_v2.providers.grokbot_exchange import FileExchange, GitHubHandoff
    request = {"request_id": "r", "kind": "grade", "experiment_id": "e", "commit": "c",
               "rubric_version": "v0", "prompt_version": "v0", "input_ids": ["i"],
               "payload": {"answer_text": "private student work"},
               "created_at": "2026-10-23T00:00:00Z", "independent_evaluation": True}
    exchange = FileExchange(tmp_path / "exchange")
    exchange.publish_request(request)
    exchange.write_result({"request_id": "r", "status": "completed", "outputs": {}, "failure_reason": ""})
    folder = GitHubHandoff(tmp_path / "gh").write(request)
    for root in (exchange.root, folder.parents[1]):
        assert mode(root) == 0o700
        for path in root.rglob("*"):
            assert mode(path) == (0o700 if path.is_dir() else 0o600)
    for path in (exchange.root / "requests" / "r.json", exchange.root / "results" / "r.json", folder / "request.json"):
        path.chmod(0o666)
    exchange.publish_request(request)
    exchange.read_result("r")
    GitHubHandoff(tmp_path / "gh").write(request)
    assert mode(exchange.root / "requests" / "r.json") == 0o600
    assert mode(exchange.root / "results" / "r.json") == 0o600
    assert mode(folder / "request.json") == 0o600


def test_legacy_runner_outputs_are_private(tmp_path):
    from scripts.legacy_runner import run
    source = tmp_path / "case.json"
    source.write_text(json.dumps({"answer_text": "private answer", "legacy_completion": '{"score": 1}'}))
    out = tmp_path / "legacy"
    run(source, out)
    assert mode(out) == 0o700
    assert mode(out / "input_used.json") == mode(out / "result.json") == 0o600
    for path in out.iterdir():
        path.chmod(0o666)
    run(source, out)
    assert mode(out / "input_used.json") == mode(out / "result.json") == 0o600
