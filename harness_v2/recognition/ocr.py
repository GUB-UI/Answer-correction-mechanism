"""Xiaomi-OCR-0 as its own step. Raw text is stored once and not rewritten.

This machine is a CPU-only Linux box (8 vCPU, about 15 GB RAM, no GPU).
SeerRay-Lab/Xiaomi-OCR-0 is a 0.8B model whose fp16 weights are about 1.6 GB,
so the weights can fit. A full transformers runtime plus a page of handwriting
was not run here, and 8 CPU cores are not a reasonable interactive OCR path
(expect minutes per page, unmeasured). Phase 1 therefore does not launch the
model. Import a text file produced by a separate OCR process instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from harness_v2.storage.artifacts import ArtifactStore

MODEL_ID = "SeerRay-Lab/Xiaomi-OCR-0"
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


class OCRNotRunnable(RuntimeError):
    def __init__(self, requirement: CPURequirement) -> None:
        super().__init__(requirement.message)
        self.requirement = requirement


class OCRImportError(ValueError):
    pass


def assess_cpu_xiaomi() -> CPURequirement:
    return CPURequirement(
        reasonable=False,
        model_id=MODEL_ID,
        message=(
            f"{MODEL_ID} is not launched on this CPU-only box "
            "(8 vCPU, about 15 GB RAM, no GPU). Weights can fit in RAM, "
            "but page OCR on CPU is not a reasonable step in this harness. "
            "Run OCR in a separate process and import the text file."
        ),
    )


class XiaomiOCR:
    def __init__(self, artifacts: ArtifactStore) -> None:
        self.artifacts = artifacts

    def import_file(self, path: Path, *, ocr_id: str) -> OCRArtifact:
        source = Path(path)
        if source.suffix.lower() in _IMAGE_SUFFIXES:
            raise OCRImportError("import OCR text, not a student image")
        if not source.is_file():
            raise OCRImportError(f"OCR text file not found: {source}")
        raw = source.read_text(encoding="utf-8")
        return self._store(ocr_id, source.name, raw)

    def run_cpu(self, image_path: Path, *, ocr_id: str) -> OCRArtifact:
        del image_path, ocr_id
        raise OCRNotRunnable(assess_cpu_xiaomi())

    def _store(self, ocr_id: str, source_name: str, raw: str) -> OCRArtifact:
        name = f"ocr/{ocr_id}.txt"
        self.artifacts.write_text(name, raw)
        return OCRArtifact(
            ocr_id=ocr_id,
            source_name=source_name,
            engine=MODEL_ID,
            raw_text=raw,
            artifact_name=name,
        )
