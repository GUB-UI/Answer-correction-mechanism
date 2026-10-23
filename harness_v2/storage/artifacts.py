"""Immutable text artifacts. Images and student files are not copied in."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from harness_v2.storage.secrets import redact

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}


class ImmutableArtifactError(FileExistsError):
    pass


class ArtifactRejected(ValueError):
    pass


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        if not name or name.startswith(("/", "\\")) or ".." in Path(name).parts:
            raise ArtifactRejected(f"artifact name is not relative: {name}")
        path = (self.root / name).resolve()
        if self.root.resolve() not in path.parents and path != self.root.resolve():
            raise ArtifactRejected(f"artifact name escapes the store: {name}")
        if path.suffix.lower() in _IMAGE_SUFFIXES:
            raise ArtifactRejected("student images are not stored in the harness")
        return path

    def _exclusive_bytes(self, name: str, data: bytes) -> Path:
        path = self._path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError as exc:
            raise ImmutableArtifactError(f"artifact already exists: {name}") from exc
        try:
            os.write(fd, data)
        finally:
            os.close(fd)
        return path

    def write_text(self, name: str, text: str) -> Path:
        return self._exclusive_bytes(name, redact(text).encode("utf-8"))

    def write_private_text(self, name: str, text: str) -> tuple[Path, str]:
        """Exact bytes. Callers keep a redacted sibling for anything committed."""
        data = text.encode("utf-8")
        path = self._exclusive_bytes(name, data)
        return path, hashlib.sha256(data).hexdigest()

    def write_json(self, name: str, payload: Any) -> Path:
        return self.write_text(name, json.dumps(redact(payload), ensure_ascii=False, indent=2))

    def read_text(self, name: str) -> str:
        return self._path(name).read_text(encoding="utf-8")

    def read_bytes(self, name: str) -> bytes:
        return self._path(name).read_bytes()
