"""起動するサーバーのポート。使用中なら次の空きを選ぶ。

Mac の external モードでは呼ばない。既に動いている mlx-vlm の 8080 を
空き扱いで飛ばすと、SDK が別ポートを見てしまう。
"""

from __future__ import annotations

import socket

_assigned: dict[str, int] = {}


def clear_assigned_ports() -> None:
    _assigned.clear()


def port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex((host, port)) == 0


def assign_listen_port(name: str, preferred: int, host: str = "127.0.0.1") -> int:
    """preferred が LISTEN 中なら、その次から最大 30 個を探す。一度決めた値は固定する。"""
    cached = _assigned.get(name)
    if cached is not None:
        return cached
    chosen = preferred
    if port_in_use(host, preferred):
        for candidate in range(preferred + 1, preferred + 31):
            if not port_in_use(host, candidate):
                chosen = candidate
                break
        print(
            f"[ports] {name}: {host}:{preferred} は使用中なので {chosen} を使います",
            flush=True,
        )
    _assigned[name] = chosen
    return chosen


def assigned_port(name: str, fallback: int) -> int:
    return _assigned.get(name, fallback)
