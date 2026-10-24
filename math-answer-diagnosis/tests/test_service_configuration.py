"""Managed configuration changes must not keep an old healthy process."""
import json
from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from src.config import load_config
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
