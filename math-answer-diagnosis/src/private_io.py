"""Owner-only output creation; existing input directories are left untouched."""
from __future__ import annotations

import os
import uuid
from pathlib import Path


def ensure_private_directory(path: Path) -> None:
    try:
        path.mkdir(mode=0o700)
    except FileNotFoundError:
        ensure_private_directory(path.parent)
        ensure_private_directory(path)
    except FileExistsError:
        if not path.is_dir():
            raise
    else:
        # Only a directory just created here is adjusted, including restrictive umasks.
        path.chmod(0o700)


def private_open(path: Path, mode: str, *, encoding: str | None = None):
    """Open an output with no group/other access from its first instant."""
    flags = os.O_WRONLY | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    if "x" in mode:
        flags |= os.O_EXCL
    elif "a" in mode:
        flags |= os.O_APPEND
    elif "w" in mode:
        flags |= os.O_TRUNC
    else:
        raise ValueError("private_open only supports output modes")
    fd = os.open(path, flags, 0o600)
    try:
        os.fchmod(fd, 0o600)
        return os.fdopen(fd, mode, encoding=encoding)
    except BaseException:
        os.close(fd)
        raise


def write_private(path: Path, data: str | bytes) -> None:
    """Atomically replace an output using an owner-only temporary file."""
    ensure_private_directory(path.parent)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        binary = isinstance(data, bytes)
        with private_open(temporary, "xb" if binary else "x", encoding=None if binary else "utf-8") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
