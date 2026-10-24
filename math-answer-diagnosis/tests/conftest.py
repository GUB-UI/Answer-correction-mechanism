from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import pytest


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    """Keep the suite offline, including tests using the real SDK with mock transports."""
    def forbidden(*args, **kwargs):
        pytest.fail("Tests must not make network connections")

    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr("socket.socket.connect_ex", forbidden)
