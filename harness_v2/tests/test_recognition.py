from __future__ import annotations

import pytest

from harness_v2.recognition.candidates import readings_from_ocr
from harness_v2.recognition.ocr import OCRImportError, OCRNotRunnable, XiaomiOCR
from harness_v2.recognition.structure import structure_text
from harness_v2.storage.artifacts import ArtifactStore, ImmutableArtifactError


def test_ocr_import_is_immutable_and_cpu_is_refused(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    ocr = XiaomiOCR(store)
    source = tmp_path / "page.txt"
    source.write_text("x+1", encoding="utf-8")
    first = ocr.import_file(source, ocr_id="ocr1")
    assert first.raw_text == "x+1"
    with pytest.raises(ImmutableArtifactError):
        ocr.import_file(source, ocr_id="ocr1")
    image = tmp_path / "sheet.png"
    image.write_bytes(b"png")
    with pytest.raises(OCRImportError):
        ocr.import_file(image, ocr_id="ocr2")
    with pytest.raises(OCRNotRunnable) as caught:
        ocr.run_cpu(image, ocr_id="ocr3")
    assert caught.value.requirement.reasonable is False
    assert "8 vCPU" in caught.value.requirement.message
    assert not list((tmp_path / "artifacts").rglob("*.png"))


def test_structure_and_candidates_keep_raw_text() -> None:
    regions = structure_text("line\n$x+1$")
    assert any(region.kind == "math" and region.text == "$x+1$" for region in regions)
    readings = readings_from_ocr("raw", ["other"])
    assert readings[0].is_raw is True
    assert readings[1].is_raw is False
    assert readings[0].text == "raw"
