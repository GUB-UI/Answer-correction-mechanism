"""Xiaomi-OCR-0 as its own step.

Imported text and CPU output are stored twice: an exclusive exact copy under
``ocr_private/``, and a redacted sibling under ``ocr/`` for anything that
can be committed. The grading input is the exact copy and its sha256.

CPU inference runs ``harness_v2.recognition.run_ocr_cpu`` in a subprocess.
bfloat16 is refused. The model is not downloaded by unit tests.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from harness_v2.recognition.run_ocr_cpu import (
    MODEL_ID,
    REVISION,
    CPUOCRRefused,
    assert_runnable,
    available_bytes,
    cpu_command,
)
from harness_v2.storage.artifacts import ArtifactStore

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class CPURequirement:
    reasonable: bool
    model_id: str
    message: str


@dataclass(frozen=True)
class OCRArtifact:
    ocr_id: str
    source_name: str
    engine: str
    raw_text: str
    artifact_name: str
    sha256: str
    redacted_name: str
    revision: str = ""
    dtype: str = ""
    elapsed_s: float | None = None


class OCRNotRunnable(RuntimeError):
    def __init__(self, requirement: CPURequirement) -> None:
        super().__init__(requirement.message)
        self.requirement = requirement


class OCRImportError(ValueError):
    pass


def assess_cpu_xiaomi(*, dtype: str = "float32", free_bytes: int | None = None) -> CPURequirement:
    free = available_bytes() if free_bytes is None else free_bytes
    try:
        assert_runnable(dtype, free)
    except CPUOCRRefused as exc:
        return CPURequirement(False, MODEL_ID, str(exc))
    return CPURequirement(
        True,
        MODEL_ID,
        f"{MODEL_ID} can be attempted on CPU in float32 at revision {REVISION}.",
    )


class XiaomiOCR:
    def __init__(
        self,
        artifacts: ArtifactStore,
        *,
        runner: Callable[[Path], tuple[str, dict[str, Any]]] | None = None,
        free_bytes: Callable[[], int] | None = None,
        dtype: str = "float32",
    ) -> None:
        self.artifacts = artifacts
        self._runner = runner
        self._free_bytes = free_bytes or available_bytes
        self.dtype = dtype

    def import_file(self, path: Path, *, ocr_id: str) -> OCRArtifact:
        source = Path(path)
        if source.suffix.lower() in _IMAGE_SUFFIXES:
            raise OCRImportError("import OCR text, not a student image")
        if not source.is_file():
            raise OCRImportError(f"OCR text file not found: {source}")
        raw = source.read_text(encoding="utf-8")
        return self._store(ocr_id, source.name, raw)

    def store_text(self, raw: str, *, ocr_id: str, source_name: str) -> OCRArtifact:
        return self._store(ocr_id, source_name, raw)

    def run_cpu(self, image_path: Path, *, ocr_id: str) -> OCRArtifact:
        requirement = assess_cpu_xiaomi(dtype=self.dtype, free_bytes=self._free_bytes())
        if not requirement.reasonable:
            raise OCRNotRunnable(requirement)
        if self._runner is not None:
            text, meta = self._runner(Path(image_path))
        else:
            text, meta = _subprocess_ocr(Path(image_path), dtype=self.dtype)
        return self._store(
            ocr_id,
            Path(image_path).name,
            text,
            revision=str(meta.get("revision") or REVISION),
            dtype=str(meta.get("dtype") or "float32"),
            elapsed_s=_optional_float(meta.get("wall_s", meta.get("elapsed_s"))),
        )

    def _store(
        self,
        ocr_id: str,
        source_name: str,
        raw: str,
        *,
        revision: str = "",
        dtype: str = "",
        elapsed_s: float | None = None,
    ) -> OCRArtifact:
        private_name = f"ocr_private/{ocr_id}.txt"
        public_name = f"ocr/{ocr_id}.txt"
        _path, digest = self.artifacts.write_private_text(private_name, raw)
        self.artifacts.write_text(public_name, raw)
        return OCRArtifact(
            ocr_id=ocr_id,
            source_name=source_name,
            engine=MODEL_ID,
            raw_text=raw,
            artifact_name=private_name,
            sha256=digest,
            redacted_name=public_name,
            revision=revision,
            dtype=dtype,
            elapsed_s=elapsed_s,
        )


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _subprocess_ocr(image_path: Path, *, dtype: str) -> tuple[str, dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="xiaomi-ocr-") as directory:
        out_dir = Path(directory)
        command = cpu_command(image_path, out_dir, dtype=dtype)
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            raise OCRNotRunnable(
                CPURequirement(
                    False,
                    MODEL_ID,
                    completed.stderr.strip() or f"CPU OCR exited {completed.returncode}",
                )
            )
        text_path = out_dir / f"{image_path.stem}.md"
        stats_path = out_dir / "stats.jsonl"
        if not text_path.is_file():
            raise OCRNotRunnable(CPURequirement(False, MODEL_ID, "CPU OCR wrote no transcript"))
        meta: dict[str, Any] = {"revision": REVISION, "dtype": dtype}
        if stats_path.is_file():
            line = stats_path.read_text(encoding="utf-8").strip().splitlines()[-1]
            loaded = json.loads(line)
            if isinstance(loaded, dict):
                meta.update(loaded)
        return text_path.read_text(encoding="utf-8"), meta
