"""Managed configuration changes must not keep an old healthy process."""
import json
import os
import signal
import stat
import subprocess
import sys
from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from src.config import load_config
from src.private_io import write_private
from src.runtime import services


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Service configuration tests must not access the network")

    monkeypatch.setattr("socket.socket.connect", forbidden)


@pytest.mark.parametrize("change", ["model_id", "max_new_tokens", "quantization", "command", "port", "alias"])
def test_cached_manager_refuses_conflicting_managed_configuration(tmp_path, monkeypatch, change):
    config = load_config(tmp_path)
    config = replace(config, services=replace(config.services, mode="managed"))
    monkeypatch.setattr(services, "_MANAGERS", {})
    old, new = MagicMock(managed=True), MagicMock(managed=True)
    build = MagicMock(side_effect=[old, new])
    monkeypatch.setattr(services, "build_service_manager", build)
    assert services.get_service_manager(config) is old
    assert services.get_service_manager(config) is old
    assert not old.stop_all.called
    if change == "alias":
        changed = replace(config, diagnosis=replace(config.diagnosis, model_name="alias-B"))
    else:
        value = {"model_id": "weights-B", "max_new_tokens": 1024, "quantization": "none",
                 "command": "custom-server", "port": 9000}[change]
        changed = replace(config, services=replace(
            config.services, llm=replace(config.services.llm, **{change: value}),
        ))
    with pytest.raises(services.ServiceStartError, match="configuration"):
        services.get_service_manager(changed)
    old.stop_all.assert_not_called()
    assert services.get_service_manager(config) is old
    assert build.call_count == 1


@pytest.mark.parametrize("launch", ["matching", "old_config", "missing", "wrong_pid", "invalid"])
def test_new_manager_checks_persisted_launch_before_reusing_healthy_process(tmp_path, monkeypatch, launch):
    old_spec = services.ServiceSpec("llm", ["server", "--model", "A", "--cap", "512"], 8000)
    new_spec = replace(old_spec, command=["server", "--model", "B", "--cap", "1024"])
    old = services.ServiceManager({"llm": old_spec}, tmp_path, mode="managed")
    manager = services.ServiceManager({"llm": new_spec}, tmp_path, mode="managed")
    pid = 12345
    if launch != "missing":
        identity = old._launch_identity("llm", pid) if launch == "old_config" else manager._launch_identity("llm", pid)
        if launch == "wrong_pid":
            identity["pid"] = pid + 1
        manager.launch_file("llm").write_text("bad json" if launch == "invalid" else json.dumps(identity))
    running = {"pid": pid}
    monkeypatch.setattr(manager, "read_pid", lambda name: running["pid"])
    monkeypatch.setattr(manager, "is_healthy", lambda name: running["pid"] is not None)
    stop = MagicMock(side_effect=lambda name: running.update(pid=None))
    start = MagicMock()
    monkeypatch.setattr(manager, "_stop_unlocked", stop)
    monkeypatch.setattr(manager, "_start_unlocked", start)
    manager.ensure("llm")
    assert stop.call_count == start.call_count == (0 if launch == "matching" else 1)


def test_start_records_the_launched_spec_and_pid(tmp_path, monkeypatch):
    spec = services.ServiceSpec("llm", ["server", "--model", "B"], 8000)
    manager = services.ServiceManager({"llm": spec}, tmp_path, mode="managed")
    process = MagicMock(pid=12345)
    process.poll.return_value = None
    monkeypatch.setattr(services.subprocess, "Popen", MagicMock(return_value=process))
    monkeypatch.setattr(manager, "read_pid", lambda name: None)
    monkeypatch.setattr(manager, "is_healthy", lambda name: True)
    manager._start_unlocked("llm")
    assert manager.pid_file("llm").read_text() == "12345"
    assert manager._launch_matches("llm", process.pid)


def test_conflicting_session_cannot_stop_original_manager(tmp_path, monkeypatch):
    config = load_config(tmp_path)
    config = replace(config, services=replace(config.services, mode="managed"))
    changed = replace(config, services=replace(config.services,
                      llm=replace(config.services.llm, model_id="different-weights")))
    monkeypatch.setattr(services, "_MANAGERS", {})
    manager = MagicMock(managed=True)
    build = MagicMock(return_value=manager)
    monkeypatch.setattr(services, "build_service_manager", build)
    assert services.get_service_manager(config) is manager
    for _ in range(2):
        with pytest.raises(RuntimeError, match="configuration"):
            services.get_service_manager(changed)
        assert services.get_service_manager(config) is manager
    manager.stop_all.assert_not_called()
    assert build.call_count == 1


def _mode(path):
    return stat.S_IMODE(path.stat().st_mode)


def _sleep_process():
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )


def _release(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def test_independent_managers_refuse_conflicting_launch_before_stopping(tmp_path, monkeypatch):
    run = tmp_path / "new-parent" / "run"
    spec_a = services.ServiceSpec("llm", ["server", "--model", "A"], 8000)
    spec_b = services.ServiceSpec("llm", ["server", "--model", "B"], 8000)
    manager_a = services.ServiceManager({"llm": spec_a}, run, mode="managed")
    manager_b = services.ServiceManager({"llm": spec_b}, run, mode="managed")
    process = _sleep_process()
    try:
        write_private(manager_a.pid_file("llm"), str(process.pid))
        write_private(manager_a.launch_file("llm"), json.dumps(manager_a._launch_identity("llm", process.pid)))
        monkeypatch.setattr(manager_a, "is_healthy", lambda name: True)
        manager_a.ensure("llm")
        note = run / "input.txt"
        note.write_text("keep")
        note.chmod(0o644)
        assert process.poll() is None
        with pytest.raises(services.ServiceStartError, match="configuration"):
            manager_b.ensure("llm")
        with pytest.raises(services.ServiceStartError, match="configuration"):
            manager_b.stop("llm")
        with pytest.raises(services.ServiceStartError, match="configuration"):
            manager_b.stop_all()
        assert process.poll() is None
        assert not any(event.get("action") == "stop" for event in [*manager_a.events, *manager_b.events])
        assert _mode(run) == _mode(run.parent) == 0o700
        assert _mode(note) == 0o644
        assert _mode(manager_a.owner_file()) == _mode(run / "services.lock") == 0o600
        assert manager_a.stop("llm") is True
    finally:
        _release(process)


def test_same_identity_managers_share_a_run_directory(tmp_path, monkeypatch):
    spec = services.ServiceSpec("llm", ["server", "--model", "A"], 8000)
    manager_a = services.ServiceManager({"llm": spec}, tmp_path / "run", mode="managed")
    manager_b = services.ServiceManager({"llm": spec}, tmp_path / "run", mode="managed")
    process = _sleep_process()
    try:
        write_private(manager_a.pid_file("llm"), str(process.pid))
        write_private(manager_a.launch_file("llm"), json.dumps(manager_a._launch_identity("llm", process.pid)))
        monkeypatch.setattr(manager_a, "is_healthy", lambda name: True)
        monkeypatch.setattr(manager_b, "is_healthy", lambda name: True)
        manager_a.ensure("llm")
        manager_b.ensure("llm")
        assert process.poll() is None
        assert not any(event.get("action") == "stop" for event in [*manager_a.events, *manager_b.events])
    finally:
        _release(process)


def test_dead_owner_can_be_replaced_without_refusing(tmp_path, monkeypatch):
    spec_a = services.ServiceSpec("llm", ["server", "--model", "A"], 8000)
    spec_b = services.ServiceSpec("llm", ["server", "--model", "B"], 8000)
    manager_a = services.ServiceManager({"llm": spec_a}, tmp_path, mode="managed")
    manager_b = services.ServiceManager({"llm": spec_b}, tmp_path, mode="managed")
    write_private(manager_a.owner_file(), json.dumps({
        "pid": 999999, "identity": manager_a._session_identity(),
    }))
    monkeypatch.setattr(manager_b, "read_pid", lambda name: 4242)
    monkeypatch.setattr(manager_b, "is_healthy", lambda name: True)
    monkeypatch.setattr(manager_b, "_stop_unlocked", MagicMock())
    monkeypatch.setattr(manager_b, "_start_unlocked", MagicMock())
    manager_b.launch_file("llm").write_text(json.dumps(manager_a._launch_identity("llm", 4242)))
    manager_b.ensure("llm")
    manager_b._stop_unlocked.assert_called_once_with("llm")
    manager_b._start_unlocked.assert_called_once_with("llm")


def test_checkouts_sharing_run_dir_conflict_under_the_filesystem_lock(tmp_path, monkeypatch):
    real = tmp_path / "real-run"
    alias = tmp_path / "alias-run"
    real.mkdir()
    real.chmod(0o755)
    alias.symlink_to(real, target_is_directory=True)
    preserved = real / "input.txt"
    preserved.write_text("keep")
    preserved.chmod(0o644)
    root_a = tmp_path / "checkout-a"
    root_b = tmp_path / "checkout-b"
    root_a.mkdir()
    root_b.mkdir()
    (root_a / "config.yaml").write_text("runtime: {device: mac}\n", encoding="utf-8")
    (root_b / "config.yaml").write_text("runtime: {device: mac}\n", encoding="utf-8")
    monkeypatch.setenv("SERVICE_MODE", "managed")
    monkeypatch.setenv("SERVICE_RUN_DIR", str(real))
    config_a = load_config(root_a)
    monkeypatch.setenv("SERVICE_RUN_DIR", str(alias))
    config_b = load_config(root_b)
    spec_a = services.ServiceSpec("llm", ["server", "--model", "A"], 8000, cwd=root_a)
    spec_b = services.ServiceSpec("llm", ["server", "--model", "B"], 8000, cwd=root_b)
    manager_a = services.ServiceManager({"llm": spec_a}, config_a.run_dir, mode="managed")
    manager_b = services.ServiceManager({"llm": spec_b}, config_b.run_dir, mode="managed")
    monkeypatch.setattr(services, "_MANAGERS", {})
    build = MagicMock(side_effect=[manager_a, manager_b])
    monkeypatch.setattr(services, "build_service_manager", build)
    assert services.canonical_run_dir(config_a.run_dir) == services.canonical_run_dir(config_b.run_dir)
    assert services.get_service_manager(config_a) is manager_a
    with pytest.raises(services.ServiceStartError, match="configuration"):
        services.get_service_manager(config_b)
    assert build.call_count == 1
    manager_a.stop_all = MagicMock()
    services._MANAGERS.clear()
    with pytest.raises(services.ServiceStartError, match="configuration"):
        services.get_service_manager(config_b)
    assert build.call_count == 2
    manager_a.stop_all.assert_not_called()
    assert _mode(real) == 0o755
    assert _mode(preserved) == 0o644
    assert _mode(manager_a.owner_file()) == 0o600
