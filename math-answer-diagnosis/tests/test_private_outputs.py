"""Private output permissions, including creation time under a permissive umask."""
import os
import stat
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.config import load_config
from src.runtime import services
from src.storage import Storage


def mode(path):
    return stat.S_IMODE(path.stat().st_mode)


@pytest.mark.parametrize("mask", [0, 0o022, 0o077])
def test_storage_outputs_are_private_from_creation(tmp_path, monkeypatch, mask):
    config = load_config(tmp_path)
    config = replace(config, data_dir_override=tmp_path / "new-parent" / "data")
    created = []
    original_open = os.open
    original_replace = os.replace
    original_mkdir = os.mkdir
    created_directories = []

    def capture_mkdir(path, permissions=0o777, **kwargs):
        original_mkdir(path, permissions, **kwargs)
        created_directories.append(stat.S_IMODE(Path(path).stat().st_mode))

    def capture_open(path, flags, permissions=0o777, **kwargs):
        fd = original_open(path, flags, permissions, **kwargs)
        if flags & os.O_CREAT:
            created.append(stat.S_IMODE(os.fstat(fd).st_mode))
        return fd

    def capture_replace(source, target):
        assert mode(Path(source)) == 0o600
        return original_replace(source, target)

    monkeypatch.setattr(os, "open", capture_open)
    monkeypatch.setattr(os, "replace", capture_replace)
    monkeypatch.setattr(os, "mkdir", capture_mkdir)
    old_mask = os.umask(mask)
    try:
        storage = Storage(config)
        image = config.resolve_path(storage.save_image(b"image"))
        answer_image = config.resolve_path(storage.save_answer_image("answer", b"image"))
        path = config.records_dir / "diagnosis_results.jsonl"
        storage._append_jsonl(path, {"record": 1})
        storage._write_jsonl(path, [{"record": 2}])
    finally:
        os.umask(old_mask)
    assert mode(image) == mode(answer_image) == mode(path) == 0o600
    assert all(mode(p) == 0o600 for p in config.records_dir.iterdir())
    assert all(mode(p) == 0o700 for p in [config.data_dir.parent, config.data_dir, config.images_dir, config.records_dir])
    assert created and all(value == 0o600 for value in created)
    assert created_directories and all(value == 0o700 for value in created_directories)


def test_reading_existing_inputs_and_directories_does_not_change_permissions(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("services: {mode: external}\n")
    config_file.chmod(0o644)
    config = load_config(tmp_path)
    directories = [tmp_path, config.data_dir, config.images_dir, config.records_dir, config.run_dir]
    for path in directories:
        path.mkdir(exist_ok=True)
        path.chmod(0o755)
    source = config.images_dir / "input.png"
    source.write_bytes(b"input")
    source.chmod(0o644)
    records = config.records_dir / "problems.jsonl"
    records.write_text("")
    records.chmod(0o644)
    storage = Storage(config)
    assert storage.list_problems() == []
    assert config.resolve_path(str(source)).read_bytes() == b"input"
    manager = services.ServiceManager({}, config.run_dir)
    with manager._locked():
        pass
    assert all(mode(path) == 0o755 for path in directories)
    assert all(mode(path) == 0o644 for path in [source, records, config_file])


def test_service_outputs_are_private_from_creation(tmp_path, monkeypatch):
    run_dir = tmp_path / "new-parent" / "run"
    spec = services.ServiceSpec("llm", ["unused"], 8000)
    manager = services.ServiceManager({"llm": spec}, run_dir, mode="managed")
    process = MagicMock(pid=12345)
    process.poll.return_value = None
    monkeypatch.setattr(services.subprocess, "Popen", MagicMock(return_value=process))
    monkeypatch.setattr(manager, "read_pid", lambda name: None)
    monkeypatch.setattr(manager, "is_healthy", lambda name: True)
    created = []
    original_open = os.open

    def capture_open(path, flags, permissions=0o777, **kwargs):
        fd = original_open(path, flags, permissions, **kwargs)
        if flags & os.O_CREAT:
            created.append(stat.S_IMODE(os.fstat(fd).st_mode))
        return fd

    monkeypatch.setattr(os, "open", capture_open)
    old_mask = os.umask(0)
    try:
        with manager._locked():
            manager._start_unlocked("llm")
    finally:
        os.umask(old_mask)
    assert mode(run_dir.parent) == mode(run_dir) == 0o700
    assert {path.name for path in run_dir.iterdir()} == {"services.lock", "llm.pid", "llm.log", "llm.launch.json"}
    assert all(mode(path) == 0o600 for path in run_dir.iterdir())
    assert created and all(value == 0o600 for value in created)
