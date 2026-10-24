"""GPU を使うモデルサーバー（GLM-OCR / UniMERNet / 診断LLM）の起動・停止管理。

Colab では VRAM に全モデルを同時に載せられないため、各モデルを別プロセスの
HTTP サーバーとして起動し、``exclusive_gpu`` が有効なら1つずつだけ常駐させる。
プロセスを終了させれば CUDA メモリは確実に解放される（同一プロセス内で
``del model`` するより確実）。

Mac（oMLX / mlx-vlm を手動起動する構成）では ``mode=external`` とし、
ここでは何も起動・停止しない。

PID ファイルと排他ロックを run_dir に置くので、Streamlit とノートブック/CLI の
どちらから操作しても同じ状態を共有できる。
"""

from __future__ import annotations

try:
    import fcntl
except ImportError:  # Windows には無い。managed 起動は Colab / Mac 向け。
    fcntl = None  # type: ignore[assignment]
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from src.config import AppConfig
from src.private_io import ensure_private_directory, private_open, write_private

GLM_OCR = "glm_ocr"
XIAOMI_OCR = "xiaomi_ocr"
UNIMERNET = "unimernet"
LLM = "llm"
SERVICE_ORDER = (XIAOMI_OCR, GLM_OCR, UNIMERNET, LLM)


@dataclass
class ServiceSpec:
    name: str
    command: list[str]
    port: int
    host: str = "127.0.0.1"
    health_path: str = "/health"
    uses_gpu: bool = True
    startup_timeout_sec: float = 1800.0
    env: dict[str, str] = field(default_factory=dict)
    cwd: Path | None = None

    @property
    def health_url(self) -> str:
        return f"http://{self.host}:{self.port}{self.health_path}"


class ServiceStartError(RuntimeError):
    pass


def http_health(url: str, timeout: float = 3.0) -> bool:
    """200 を返せば健康。JSON に ok:false が含まれる場合（ロード中）は不健康とみなす。"""
    try:
        with urlopen(url, timeout=timeout) as response:
            if response.status != 200:
                return False
            body = response.read()
    except (URLError, HTTPError, TimeoutError, ConnectionError, OSError):
        return False
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return True
    if isinstance(payload, dict) and payload.get("ok") is False:
        return False
    return True


_SESSION_CONFLICT = (
    "Service configuration conflicts with an existing session in this run directory; "
    "use a separate run directory or explicitly stop and restart the application."
)


def canonical_run_dir(path: Path) -> str:
    """One key for every path that names the same run directory."""
    return str(Path(path).resolve())


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        # 終了済みでも回収されていない子プロセス（ゾンビ）は生存扱いしない。
        waited, _ = os.waitpid(pid, os.WNOHANG)
        if waited == pid:
            return False
    except ChildProcessError:
        pass
    stat = Path(f"/proc/{pid}/stat")
    if stat.is_file():
        try:
            if stat.read_text().split()[2] == "Z":
                return False
        except (OSError, IndexError):
            pass
    return True


class ServiceManager:
    def __init__(
        self,
        specs: dict[str, ServiceSpec],
        run_dir: Path,
        *,
        mode: str = "external",
        exclusive_gpu: bool = True,
        health_check: Callable[[str], bool] = http_health,
        poll_interval_sec: float = 2.0,
        stop_timeout_sec: float = 60.0,
    ) -> None:
        self.specs = specs
        self.run_dir = run_dir
        self.mode = mode
        self.exclusive_gpu = exclusive_gpu
        self.health_check = health_check
        self.poll_interval_sec = poll_interval_sec
        self.stop_timeout_sec = stop_timeout_sec
        self._thread_lock = threading.RLock()
        self.events: list[dict[str, Any]] = []

    @property
    def managed(self) -> bool:
        return self.mode == "managed"

    def pid_file(self, name: str) -> Path:
        return self.run_dir / f"{name}.pid"

    def log_file(self, name: str) -> Path:
        return self.run_dir / f"{name}.log"

    def launch_file(self, name: str) -> Path:
        return self.run_dir / f"{name}.launch.json"

    def owner_file(self) -> Path:
        return self.run_dir / "services.owner.json"

    def _session_identity(self) -> str:
        payload = {
            "mode": self.mode,
            "exclusive_gpu": self.exclusive_gpu,
            "specs": {name: asdict(self.specs[name]) for name in sorted(self.specs)},
        }
        encoded = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=True)
        return hashlib.sha256(encoded.encode()).hexdigest()

    def _owner_record(self) -> dict[str, Any] | None:
        try:
            value = json.loads(self.owner_file().read_text())
        except (OSError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    def _holder_pids(self, record: dict[str, Any] | None) -> list[int]:
        if not record:
            return []
        raw = record.get("holders")
        if not isinstance(raw, list):
            raw = [record.get("pid")]
        holders: list[int] = []
        for pid in raw:
            if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 or pid in holders:
                continue
            holders.append(pid)
        return holders

    def _write_owner(self, identity: str, holders: list[int]) -> None:
        write_private(
            self.owner_file(),
            json.dumps({"pid": holders[-1], "holders": holders, "identity": identity}),
        )

    def _claim_session(self) -> None:
        """Record this process as a holder of the run directory, or refuse.

        Must run under the shared filesystem lock, before any service is stopped.
        Every live holder is kept. A different launch identity is refused while
        any of them remains. Dead holders are dropped, and a directory with no
        live holder can be taken over. Matching identities share the directory.
        """
        identity = self._session_identity()
        record = self._owner_record()
        same = bool(record) and record.get("identity") == identity
        live = [pid for pid in self._holder_pids(record) if _pid_alive(pid)]
        if live and not same:
            raise ServiceStartError(_SESSION_CONFLICT)
        me = os.getpid()
        if same:
            holders = [pid for pid in live if pid != me]
            holders.append(me)
            if holders == self._holder_pids(record):
                return
            self._write_owner(identity, holders)
            return
        self._write_owner(identity, [me])

    def _launch_identity(self, name: str, pid: int) -> dict[str, Any]:
        spec = json.dumps(asdict(self.specs[name]), sort_keys=True, default=str)
        return {"pid": pid, "spec_sha256": hashlib.sha256(spec.encode()).hexdigest()}

    def _launch_matches(self, name: str, pid: int) -> bool:
        try:
            return json.loads(self.launch_file(name).read_text()) == self._launch_identity(name, pid)
        except (OSError, ValueError):
            return False

    def _record(self, action: str, name: str, **extra: Any) -> None:
        self.events.append({"time": time.time(), "action": action, "service": name, **extra})

    @contextmanager
    def _locked(self):
        ensure_private_directory(self.run_dir)
        with self._thread_lock:
            if fcntl is None:
                yield
                return
            with private_open(self.run_dir / "services.lock", "a") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def read_pid(self, name: str) -> int | None:
        path = self.pid_file(name)
        if not path.is_file():
            return None
        try:
            pid = int(path.read_text().strip())
        except ValueError:
            path.unlink(missing_ok=True)
            return None
        if not _pid_alive(pid):
            path.unlink(missing_ok=True)
            return None
        return pid

    def is_healthy(self, name: str) -> bool:
        return self.health_check(self.specs[name].health_url)

    def ensure(self, name: str) -> None:
        """name のサービスを使える状態にする。排他モードでは他の GPU サービスを先に止める。"""
        if not self.managed:
            return
        if name not in self.specs:
            raise KeyError(f"Unknown service: {name}")
        with self._locked():
            self._claim_session()
            spec = self.specs[name]
            if self.exclusive_gpu and spec.uses_gpu:
                for other_name, other in self.specs.items():
                    if other_name != name and other.uses_gpu:
                        self._stop_unlocked(other_name)
            pid = self.read_pid(name)
            if pid is not None:
                if self._launch_matches(name, pid) and self.is_healthy(name):
                    return
                self._stop_unlocked(name)
            if self.read_pid(name) is None and self.is_healthy(name):
                # 管理外で起動済み（手動起動など）。止められないがそのまま使う。
                self._record("reuse_external", name)
                return
            self._start_unlocked(name)

    def _start_unlocked(self, name: str) -> None:
        spec = self.specs[name]
        existing = self.read_pid(name)
        if existing is not None:
            self._stop_unlocked(name)
        ensure_private_directory(self.run_dir)
        log_path = self.log_file(name)
        env = {**os.environ, **spec.env}
        with private_open(log_path, "ab") as log:
            log.write(f"\n===== start {name}: {shlex.join(spec.command)} =====\n".encode())
            log.flush()
            process = subprocess.Popen(
                spec.command,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                cwd=str(spec.cwd) if spec.cwd else None,
                env=env,
                start_new_session=True,
            )
        write_private(self.pid_file(name), str(process.pid))
        write_private(self.launch_file(name), json.dumps(self._launch_identity(name, process.pid)))
        self._record("start", name, pid=process.pid)
        deadline = time.monotonic() + spec.startup_timeout_sec
        while time.monotonic() < deadline:
            if process.poll() is not None:
                self.pid_file(name).unlink(missing_ok=True)
                raise ServiceStartError(
                    f"{name} サーバーが起動途中で終了しました (exit={process.returncode})。\n"
                    f"ログ: {log_path}\n{tail_text(log_path)}"
                )
            if self.is_healthy(name):
                self._record("ready", name, pid=process.pid)
                return
            time.sleep(self.poll_interval_sec)
        self._stop_unlocked(name)
        raise ServiceStartError(
            f"{name} サーバーが {spec.startup_timeout_sec:.0f} 秒以内に起動しませんでした。\n"
            f"ログ: {log_path}\n{tail_text(log_path)}"
        )

    def stop(self, name: str) -> bool:
        with self._locked():
            if self.managed:
                self._claim_session()
            return self._stop_unlocked(name)

    def _stop_unlocked(self, name: str) -> bool:
        pid = self.read_pid(name)
        if pid is None:
            return False
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            self.pid_file(name).unlink(missing_ok=True)
            return False
        except PermissionError:
            os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + self.stop_timeout_sec
        while time.monotonic() < deadline and _pid_alive(pid):
            time.sleep(0.2)
        if _pid_alive(pid):
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            time.sleep(0.5)
        self.pid_file(name).unlink(missing_ok=True)
        self._record("stop", name, pid=pid)
        return True

    def stop_all(self) -> list[str]:
        stopped: list[str] = []
        with self._locked():
            if self.managed:
                self._claim_session()
            for name in self.specs:
                if self._stop_unlocked(name):
                    stopped.append(name)
        return stopped

    def status(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for name, spec in self.specs.items():
            pid = self.read_pid(name)
            rows.append(
                {
                    "service": name,
                    "pid": pid,
                    "healthy": self.is_healthy(name),
                    "port": spec.port,
                    "uses_gpu": spec.uses_gpu,
                    "log": str(self.log_file(name)),
                    "command": shlex.join(spec.command),
                }
            )
        return rows


def tail_text(path: Path, lines: int = 30) -> str:
    if not path.is_file():
        return ""
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def _port_from_url(url: str, default: int) -> int:
    from urllib.parse import urlparse

    parsed = urlparse(url)
    return parsed.port or default


def build_service_specs(
    config: AppConfig, *, bind_free_port: bool = False
) -> dict[str, ServiceSpec]:
    root = config.project_root
    scripts = root / "scripts"
    services = config.services
    timeout = services.startup_timeout_sec
    python = sys.executable

    glm_port = config.ocr_api.port
    if bind_free_port:
        from src.runtime.ports import assign_listen_port

        glm_port = assign_listen_port(GLM_OCR, glm_port, config.ocr_api.host)
    if services.glm_ocr.command:
        glm_command = shlex.split(services.glm_ocr.command)
    else:
        glm_command = [
            python,
            str(scripts / "glmocr_server.py"),
            "--host",
            "127.0.0.1",
            "--port",
            str(glm_port),
            "--model",
            services.glm_ocr.model_id,
        ]
    glm_env = {
        "GLMOCR_MIN_EDGE": str(config.ocr_api.min_edge),
        "GLMOCR_MAX_EDGE": str(config.ocr_api.max_edge),
        "GLMOCR_PROMPT_SUFFIX": config.ocr_api.prompt_suffix,
    }

    xiaomi_port = config.ocr_api.port
    if bind_free_port:
        xiaomi_port = assign_listen_port(XIAOMI_OCR, xiaomi_port, config.ocr_api.host)
    if services.xiaomi_ocr.command:
        xiaomi_command = shlex.split(services.xiaomi_ocr.command)
    else:
        xiaomi_command = [
            python,
            str(scripts / "xiaomi_ocr_server.py"),
            "--host",
            "127.0.0.1",
            "--port",
            str(xiaomi_port),
            "--model",
            services.xiaomi_ocr.model_id,
        ]

    if config.ocr.revision:
        if not services.glm_ocr.command:
            glm_command.extend(["--revision", config.ocr.revision])
        if not services.xiaomi_ocr.command:
            xiaomi_command.extend(["--revision", config.ocr.revision])

    unimernet_port = _port_from_url(config.formula.base_url, 8091)
    unimernet_env = {"UNIMERNET_PORT": str(unimernet_port)}
    if services.unimernet.model_dir:
        unimernet_env["UNIMERNET_MODEL_DIR"] = services.unimernet.model_dir
    if services.unimernet.python:
        unimernet_env["UNIMERNET_PYTHON"] = services.unimernet.python
    if services.unimernet.command:
        unimernet_command = shlex.split(services.unimernet.command)
    else:
        unimernet_command = ["bash", str(scripts / "start_unimernet.sh")]

    llm = services.llm
    if llm.command:
        llm_command = shlex.split(llm.command)
    else:
        llm_command = [
            python,
            str(scripts / "llm_server.py"),
            "--host",
            "127.0.0.1",
            "--port",
            str(llm.port),
            "--model",
            llm.model_id,
            "--served-model-name",
            config.diagnosis.model_name,
            "--quantization",
            llm.quantization,
            "--max-new-tokens-cap",
            str(llm.max_new_tokens),
        ]

    return {
        XIAOMI_OCR: ServiceSpec(
            name=XIAOMI_OCR,
            command=xiaomi_command,
            port=xiaomi_port,
            startup_timeout_sec=timeout,
            cwd=root,
        ),
        GLM_OCR: ServiceSpec(
            name=GLM_OCR,
            command=glm_command,
            port=glm_port,
            startup_timeout_sec=timeout,
            env=glm_env,
            cwd=root,
        ),
        UNIMERNET: ServiceSpec(
            name=UNIMERNET,
            command=unimernet_command,
            port=unimernet_port,
            startup_timeout_sec=timeout,
            env=unimernet_env,
            cwd=root,
        ),
        LLM: ServiceSpec(
            name=LLM,
            command=llm_command,
            port=llm.port,
            startup_timeout_sec=timeout,
            cwd=root,
        ),
    }


def build_service_manager(config: AppConfig) -> ServiceManager:
    return ServiceManager(
        build_service_specs(config, bind_free_port=config.services.mode == "managed"),
        config.run_dir,
        mode=config.services.mode,
        exclusive_gpu=config.services.exclusive_gpu,
    )


_MANAGERS: dict[str, tuple[AppConfig, ServiceManager]] = {}
_MANAGERS_LOCK = threading.RLock()


def get_service_manager(config: AppConfig) -> ServiceManager:
    key = canonical_run_dir(config.run_dir)
    with _MANAGERS_LOCK:
        entry = _MANAGERS.get(key)
        if entry is None or entry[0] != config:
            if entry is not None and (entry[1].managed or config.services.mode == "managed"):
                raise ServiceStartError(_SESSION_CONFLICT)
            manager = build_service_manager(config)
            if manager.managed is True:
                with manager._locked():
                    manager._claim_session()
            _MANAGERS[key] = (config, manager)
        return _MANAGERS[key][1]
