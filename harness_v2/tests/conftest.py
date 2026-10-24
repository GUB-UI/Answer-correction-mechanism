from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest


@pytest.fixture(autouse=True)
def isolated_commit_resolver(monkeypatch):
    """Keep experiments independent of the checkout; expose Git resolution for unit tests."""
    import harness_v2.run as runner
    resolver = getattr(runner, 'resolve_implementation_commit', None)
    monkeypatch.setattr(runner, 'resolve_implementation_commit', lambda: 'test-revision', raising=False)
    return resolver
