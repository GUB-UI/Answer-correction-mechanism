"""Keep agent and debug logs out of the tracked repository tree."""

from fnmatch import fnmatchcase
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
LOG_PATTERNS = ("codex_*.log", "*_stderr.log", "*_stdout.log")


def test_no_tracked_agent_or_debug_logs():
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True, text=True
    )
    logs = [
        name
        for name in result.stdout.split("\0")
        if name and any(fnmatchcase(Path(name).name, pattern) for pattern in LOG_PATTERNS)
    ]
    assert not logs, f"Agent/debug logs must not be tracked: {logs}"


def test_agent_and_debug_logs_are_ignored_at_any_depth():
    paths = [
        prefix + name
        for prefix in ("", "nested/logs/")
        for name in ("codex_session.log", "agent_stderr.log", "agent_stdout.log")
    ]
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin"],
        cwd=ROOT,
        input="\n".join(paths) + "\n",
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == paths
