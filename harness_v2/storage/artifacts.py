"""Immutable text artifacts. Images and student files are not copied in."""

from __future__ import annotations

import json
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

    def write_text(self, name: str, text: str) -> Path:
        path = self._path(name)
        if path.exists():
            raise ImmutableArtifactError(f"artifact already exists: {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(redact(text), encoding="utf-8")
        return path

    def write_json(self, name: str, payload: Any) -> Path:
        return self.write_text(name, json.dumps(redact(payload), ensure_ascii=False, indent=2))

    def read_text(self, name: str) -> str:
        return self._path(name).read_text(encoding="utf-8")
