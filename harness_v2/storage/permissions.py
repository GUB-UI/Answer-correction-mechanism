"""Owner-only storage without changing the process-wide umask."""
from __future__ import annotations

import os
from pathlib import Path
from typing import IO


def private_directory(path: Path, *, managed: bool = False) -> Path:
    """Create private directories, preserving existing ancestors.

    Only explicitly managed harness output directories may have their existing
    permissions tightened. Implicit parents can be shared/user directories.
    """
    path = Path(path)
    if not path.parent.exists():
        private_directory(path.parent)
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        if not path.is_dir():
            raise
        if not managed:
            return path
    path.chmod(0o700)
    return path


def private_open(path: Path, mode: str = "r", *, encoding: str = "utf-8") -> IO:
    """Open reads unchanged; create/tighten only harness-managed output writes.

    Callers must use write modes only for private outputs, never external inputs.
    Existing parent directories are preserved; owners manage output roots via
    ``private_directory(..., managed=True)`` when writing.
    """
    path = Path(path)
    if not any(flag in mode for flag in "wax+"):
        return path.open(mode, **({} if "b" in mode else {"encoding": encoding}))
    private_directory(path.parent)
    flags = os.O_RDWR if "+" in mode else os.O_WRONLY if mode[0] in "wax" else os.O_RDONLY
    if mode[0] in "wax":
        flags |= os.O_CREAT
    if mode[0] == "x":
        flags |= os.O_EXCL
    if mode[0] == "a":
        flags |= os.O_APPEND
    fd = os.open(path, flags, 0o600)
    try:
        os.fchmod(fd, 0o600)
        if mode[0] == "w":
            os.ftruncate(fd, 0)
        handle = os.fdopen(fd, mode, **({} if "b" in mode else {"encoding": encoding}))
    except BaseException:
        os.close(fd)
        raise
    return handle


def private_write_text(path: Path, text: str) -> None:
    with private_open(path, "w") as handle:
        handle.write(text)
