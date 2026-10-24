"""Immutable text artifacts. Images and student files are not copied in."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from harness_v2.storage.secrets import redact
from harness_v2.storage.permissions import private_directory, private_open

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}


class ImmutableArtifactError(FileExistsError):
    pass


class ArtifactRejected(ValueError):
    pass


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        private_directory(self.root)

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
        private_directory(self.root, managed=True)
        folder = self.root
        for part in path.parent.relative_to(self.root.resolve()).parts:
            folder = folder / part
            private_directory(folder, managed=True)
        try:
            with private_open(path, "xb") as handle:
                handle.write(data)
        except FileExistsError as exc:
            path.chmod(0o600)
            raise ImmutableArtifactError(f"artifact already exists: {name}") from exc
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
        with self._path(name).open(encoding="utf-8") as handle:
            return handle.read()

    def read_bytes(self, name: str) -> bytes:
        with self._path(name).open("rb") as handle:
            return handle.read()
