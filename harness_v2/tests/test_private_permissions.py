"""Private data permissions must not depend on the caller's umask."""
import contextlib
import hashlib
import json
import os
import stat
import sys
from types import SimpleNamespace

import pytest

from harness_v2.storage.artifacts import ArtifactStore, ImmutableArtifactError
from harness_v2.storage.permissions import private_directory, private_open
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
def test_private_artifact_writes_are_private_and_reads_preserve_modes(tmp_path, name):
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
    assert mode(path) == 0o666
    assert mode(path.parent) == 0o777
    path.chmod(0o666)
    assert store.read_text(name) == "exact\n"
    assert mode(path) == 0o666
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
    assert mode(path) == 0o666
    assert mode(path.parent) == 0o777
    if sidecar.exists():
        assert mode(sidecar) == 0o666
    with store._connect() as conn:
        conn.execute("INSERT INTO runs SELECT * FROM runs")
    assert mode(path) == 0o600
    assert mode(path.parent) == 0o777
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


@pytest.mark.parametrize("journal_mode", ["PERSIST", "WAL"])
def test_reopened_database_listing_preserves_modes(tmp_path, journal_mode):
    path = tmp_path / "database.sqlite"
    store = RunStore(path, ArtifactStore(tmp_path / "artifacts"))
    # Keep the writer open so WAL and SHM remain present during the read.
    with store._connect() as writer:
        writer.execute(f"PRAGMA journal_mode={journal_mode}")
        writer.execute("CREATE TABLE listing_probe (value TEXT)")
        writer.commit()
        sidecars = [path.with_name(path.name + suffix)
                    for suffix in ("-journal", "-wal", "-shm")
                    if path.with_name(path.name + suffix).exists()]
        assert sidecars
        for file in [path, *sidecars]:
            file.chmod(0o644)
        path.parent.chmod(0o755)
        watched = [path, *sidecars, path.parent]
        before = {file: mode(file) for file in watched}
        assert RunStore(path, store.artifacts, readonly=True).list_runs() == []
        assert {file: mode(file) for file in watched} == before


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


@pytest.mark.parametrize("explicit_metadata", [False, True])
def test_imported_raw_ocr_and_metadata_keep_input_permissions(tmp_path, explicit_metadata):
    from harness_v2.recognition.ocr import XiaomiOCR
    source = tmp_path / "raw.txt"
    source.write_text("raw answer")
    sidecar = tmp_path / ("metadata.json" if explicit_metadata else "raw.txt.meta.json")
    sidecar.write_text('{}')
    for path in (source, sidecar):
        path.chmod(0o644)
    tmp_path.chmod(0o755)
    artifacts = ArtifactStore(tmp_path / "artifacts")
    result = XiaomiOCR(artifacts).import_file(
        source, ocr_id="r", metadata_path=sidecar if explicit_metadata else None)
    assert mode(source) == mode(sidecar) == 0o644
    assert mode(tmp_path) == 0o755
    assert mode(artifacts.root / result.artifact_name) == 0o600


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
    exchange.root.chmod(0o755)
    (exchange.root / "results").chmod(0o755)
    exchange.read_result("r")
    assert mode(exchange.root / "results" / "r.json") == 0o666
    assert mode(exchange.root) == mode(exchange.root / "results") == 0o755
    exchange.write_result({"request_id": "r", "status": "completed", "outputs": {}, "failure_reason": ""})
    GitHubHandoff(tmp_path / "gh").write(request)
    assert mode(exchange.root / "requests" / "r.json") == 0o600
    assert mode(exchange.root / "results" / "r.json") == 0o600
    assert mode(folder / "request.json") == 0o600


@pytest.mark.parametrize("read_mode", ["r", "rt", "rb", "tr", "br"])
def test_private_read_preserves_external_file_and_directory_modes(tmp_path, read_mode):
    folder = tmp_path / "shared"
    folder.mkdir(mode=0o755)
    source = folder / "input.txt"
    source.write_text("answer")
    source.chmod(0o644)
    with private_open(source, read_mode) as handle:
        assert handle.read() == (b"answer" if "b" in read_mode else "answer")
    assert mode(source) == 0o644
    assert mode(folder) == 0o755


def test_missing_private_read_does_not_create_parents(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        private_open(missing / "input.txt")
    assert not missing.exists()


@pytest.mark.parametrize("write_mode", ["w", "wb", "a", "ab", "x", "xb", "r+"])
def test_private_output_writes_preserve_preexisting_parents(tmp_path, write_mode):
    tmp_path.chmod(0o755)
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o755)
    path = shared / "output.txt"
    if write_mode[0] != "x":
        path.write_text("old")
        path.chmod(0o644)
    with private_open(path, write_mode) as handle:
        handle.write(b"new" if "b" in write_mode else "new")
    assert mode(path) == 0o600
    assert mode(shared) == mode(tmp_path) == 0o755


@pytest.mark.parametrize("umask", [0o000, 0o777])
def test_new_private_outputs_are_owner_only_without_chmodding_ancestors(tmp_path, umask):
    tmp_path.chmod(0o755)
    previous = os.umask(umask)
    try:
        output = tmp_path / "outputs" / "nested" / "answer.txt"
        with private_open(output, "w") as handle:
            handle.write("answer")
        assert mode(output) == 0o600
        assert mode(output.parent) == mode(output.parent.parent) == 0o700
        assert mode(tmp_path) == 0o755
    finally:
        os.umask(previous)


def test_private_directory_tightens_only_explicit_managed_output(tmp_path):
    tmp_path.chmod(0o755)
    output = tmp_path / "output"
    output.mkdir(mode=0o755)
    private_directory(output)
    assert mode(output) == 0o755
    private_directory(output, managed=True)
    assert mode(output) == 0o700
    assert mode(tmp_path) == 0o755


def test_artifact_reads_preserve_modes_and_do_not_create_directories(tmp_path):
    root = tmp_path / "artifacts"
    store = ArtifactStore(root)
    path = store.write_text("nested/a.txt", "answer")
    for directory in (root, path.parent):
        directory.chmod(0o755)
    path.chmod(0o644)
    reopened = ArtifactStore(root)
    assert reopened.read_text("nested/a.txt") == "answer"
    assert reopened.read_bytes("nested/a.txt") == b"answer"
    with pytest.raises(FileNotFoundError):
        reopened.read_text("missing/a.txt")
    assert not (root / "missing").exists()
    assert mode(path) == 0o644
    assert mode(root) == mode(path.parent) == 0o755


def test_database_in_shared_directory_preserves_parent_modes(tmp_path):
    tmp_path.chmod(0o755)
    path = tmp_path / "database.sqlite"
    store = RunStore(path, ArtifactStore(tmp_path / "artifacts"))
    assert mode(path) == 0o600
    assert mode(tmp_path) == 0o755
    store.list_runs()
    assert mode(tmp_path) == 0o755


def test_runner_dataset_metrics_and_comparison_preserve_shared_inputs(tmp_path):
    from harness_v2.evaluation.comparison import compare_experiments
    from harness_v2.evaluation.dataset import load_dataset
    from harness_v2.evaluation.metrics import compute_metrics
    from harness_v2.run import run_experiment

    shared = tmp_path / "shared"
    shared.mkdir(mode=0o755)
    tmp_path.chmod(0o755)
    answer = shared / "answer.txt"
    answer.write_text("student work")
    rubric = shared / "rubric.json"
    rubric.write_text(json.dumps([{"item_id": "work", "max_points": 2}]))
    labels = shared / "labels.json"
    labels.write_text(json.dumps({"dataset_id": "shared", "version": "1", "reference_scores": {}}))
    def descriptor(path):
        return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = shared / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "dataset_id": "shared", "version": "1",
        "cases": [{"case_id": "case", "problem_id": "problem", "input_version": "manual",
                   "problem_text": "Explain your work", "rubric_version": "v1",
                   "grading_input": descriptor(answer), "rubric": descriptor(rubric),
                   "ocr_metadata": {"engine": "manual"}}],
        "references": descriptor(labels)}))
    config = shared / "config.json"
    config.write_text(json.dumps({"name": "permission-test", "provider": "fixture", "model": "fixture",
        "prompt_version": "v1", "rubric_version": "v1", "generation_settings": {},
        "fixture_completion": json.dumps({"items": [{"item_id": "work", "score": 1, "max_points": 2,
            "math_reason": "Work shown", "evidence_spans": [{"text": "student work"}]}],
            "total_score": 1})}))
    inputs = [answer, rubric, labels, manifest, config]
    for path in inputs:
        path.chmod(0o644)
    out = tmp_path / "private-output"
    summary = run_experiment(config, manifest, repeats=1, out=out, commit="permission-test")
    assert summary["completed"] == 1
    dataset = load_dataset(manifest)
    payload = json.loads((out / "results.json").read_text())
    compute_metrics(payload, dataset)
    comparison = compare_experiments(payload, payload, dataset=dataset)
    assert comparison["comparable"], comparison["reasons"]
    assert all(mode(path) == 0o644 for path in inputs)
    assert mode(shared) == mode(tmp_path) == 0o755
    assert mode(out) == 0o700
    assert mode(out / "results.json") == 0o600


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
