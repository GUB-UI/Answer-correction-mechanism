"""Keep agent and debug logs out of the tracked repository tree."""

from fnmatch import fnmatchcase
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
LOG_PATTERNS = (
    "stderr.log", "stdout.log", "debug.log", "agent*.log",
    "codex_*.log", "*_stderr.log", "*_stdout.log", "handshake_log*.txt",
    "codex_fix_*.md", "codex_review_*.md", "codex_rereview_*.md",
)


def require_git_work_tree():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=ROOT, capture_output=True, text=True,
        )
    except OSError:
        pytest.skip("Git is unavailable for repository hygiene checks")
    if result.returncode != 0 or result.stdout.strip() != "true":
        pytest.skip("Repository hygiene checks require a Git work tree")


def test_no_tracked_agent_or_debug_logs():
    require_git_work_tree()
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True, text=True
    )
    removed_artifacts = {"codex_fix_gpt-6.1-sol.md", "handshake_log_after_fix.txt"}
    assert removed_artifacts.isdisjoint(result.stdout.split("\0")), (
        "Removed agent artifacts must not be tracked"
    )
    logs = [
        name
        for name in result.stdout.split("\0")
        if name and any(fnmatchcase(Path(name).name, pattern) for pattern in LOG_PATTERNS)
    ]
    assert not logs, f"Agent/debug logs must not be tracked: {logs}"


def test_agent_and_debug_logs_are_ignored_at_any_depth():
    require_git_work_tree()
    paths = [
        prefix + name
        for prefix in ("", "nested/logs/")
        for name in (
            "stderr.log", "stdout.log", "debug.log",
            "agent.log", "agent-session.log", "agent_session.log",
            "codex_session.log", "agent_stderr.log", "agent_stdout.log",
            "worker_stderr.log", "worker_stdout.log",
            "handshake_log.txt", "handshake_log_after_fix.txt",
            "codex_fix_gpt-6.1-sol.md", "codex_review_run.md", "codex_rereview_run.md",
        )
    ]
    assert all(
        any(fnmatchcase(Path(name).name, pattern) for pattern in LOG_PATTERNS)
        for name in paths
    )
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin"],
        cwd=ROOT,
        input="\n".join(paths) + "\n",
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == paths


def test_legitimate_documents_are_not_ignored():
    require_git_work_tree()
    tracked = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.md", "*.txt", "*.log"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    paths = [name for name in tracked.stdout.split("\0") if name]
    paths += [
        prefix + name
        for prefix in ("", "nested/docs/")
        for name in ("codex_notes.md", "codex_fix.md", "codex_review.md", "review.md")
    ]
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin"],
        cwd=ROOT, input="\n".join(paths) + "\n", capture_output=True, text=True,
    )
    assert result.returncode == 1, result.stderr
    assert not result.stdout, f"Legitimate documents must not be ignored: {result.stdout}"


@pytest.mark.parametrize("check", [
    test_no_tracked_agent_or_debug_logs,
    test_agent_and_debug_logs_are_ignored_at_any_depth,
    test_legitimate_documents_are_not_ignored,
])
def test_hygiene_checks_skip_without_git(check, monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(pytest.skip.Exception, match="Git"):
        check()


@pytest.mark.parametrize("check", [
    test_no_tracked_agent_or_debug_logs,
    test_agent_and_debug_logs_are_ignored_at_any_depth,
    test_legitimate_documents_are_not_ignored,
])
def test_hygiene_checks_skip_outside_work_tree(check, monkeypatch, tmp_path):
    monkeypatch.setitem(globals(), "ROOT", tmp_path)
    with pytest.raises(pytest.skip.Exception, match="Git"):
        check()
