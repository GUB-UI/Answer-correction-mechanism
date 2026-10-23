from __future__ import annotations

import hashlib

import pytest

from harness_v2.recognition.candidates import readings_from_ocr
from harness_v2.recognition.ocr import OCRImportError, OCRNotRunnable, XiaomiOCR
from harness_v2.recognition.run_ocr_cpu import (
    MAX_NEW_TOKENS,
    MIN_FREE_BYTES,
    REVISION,
    assert_runnable,
    cpu_command,
    main,
)
from harness_v2.recognition.structure import structure_text
from harness_v2.storage.artifacts import ArtifactStore, ImmutableArtifactError
from pathlib import Path


def test_ocr_import_is_immutable_and_images_are_rejected(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    ocr = XiaomiOCR(store)
    source = tmp_path / "page.txt"
    source.write_text("x+1", encoding="utf-8")
    first = ocr.import_file(source, ocr_id="ocr1")
    assert first.raw_text == "x+1"
    stored = store.read_bytes(first.artifact_name)
    assert stored == b"x+1"
    assert hashlib.sha256(stored).hexdigest() == first.sha256
    assert store.read_text(first.redacted_name) == "x+1"
    with pytest.raises(ImmutableArtifactError):
        ocr.import_file(source, ocr_id="ocr1")
    image = tmp_path / "sheet.png"
    image.write_bytes(b"png")
    with pytest.raises(OCRImportError):
        ocr.import_file(image, ocr_id="ocr2")
    assert not list((tmp_path / "artifacts").rglob("*.png"))


def test_private_ocr_is_exact_and_redacted_copy_drops_secrets(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    ocr = XiaomiOCR(store)
    source = tmp_path / "page.txt"
    source.write_text("work sk-FAKEKEY123456", encoding="utf-8")
    artifact = ocr.import_file(source, ocr_id="secret")
    assert store.read_bytes(artifact.artifact_name) == b"work sk-FAKEKEY123456"
    assert "sk-FAKEKEY123456" not in store.read_text(artifact.redacted_name)
    assert artifact.sha256 == hashlib.sha256(b"work sk-FAKEKEY123456").hexdigest()


def test_cpu_path_refuses_bf16_and_low_memory_and_records_a_run(tmp_path) -> None:
    with pytest.raises(Exception, match="bfloat16"):
        assert_runnable("bfloat16", 10**12)
    with pytest.raises(Exception, match="4.5"):
        assert_runnable("float32", 1024)
    assert_runnable("float32", MIN_FREE_BYTES)
    command = cpu_command(Path("page.png"), Path("out"))
    assert command[command.index("--dtype") + 1] == "float32"
    assert command[command.index("--max-new-tokens") + 1] == str(MAX_NEW_TOKENS)
    assert command[command.index("--max-edge") + 1] == "0"
    assert REVISION == "ff990be318c4c68f7e621dd690231200f47e48c4"
    assert main(["page.png", "--dtype", "bfloat16"]) == 2

    store = ArtifactStore(tmp_path / "artifacts")

    def runner(_path: Path) -> tuple[str, dict]:
        return "transcribed", {"revision": REVISION, "dtype": "float32", "wall_s": 1.25}

    ocr = XiaomiOCR(store, runner=runner, free_bytes=lambda: MIN_FREE_BYTES)
    image = tmp_path / "page.png"
    image.write_bytes(b"png")
    artifact = ocr.run_cpu(image, ocr_id="cpu1")
    assert artifact.revision == REVISION
    assert artifact.dtype == "float32"
    assert artifact.elapsed_s == 1.25
    assert artifact.raw_text == "transcribed"
    assert store.read_bytes(artifact.artifact_name) == b"transcribed"
    refused = XiaomiOCR(store, free_bytes=lambda: 1024)
    with pytest.raises(OCRNotRunnable):
        refused.run_cpu(image, ocr_id="cpu2")


def test_structure_and_candidates_keep_raw_text() -> None:
    regions = structure_text("line\n$x+1$")
    assert any(region.kind == "math" and region.text == "$x+1$" for region in regions)
    readings = readings_from_ocr("raw", ["other"])
    assert readings[0].is_raw is True
    assert readings[1].is_raw is False
    assert readings[0].text == "raw"
