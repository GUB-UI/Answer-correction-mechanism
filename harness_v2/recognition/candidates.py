"""Hold OCR readings. Alternatives are not promoted over the raw text."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReadingCandidate:
    candidate_id: str
    text: str
    source: str
    is_raw: bool


def readings_from_ocr(raw_text: str, alternatives: list[str] | None = None) -> list[ReadingCandidate]:
    readings = [ReadingCandidate("c0", raw_text, "ocr_raw", True)]
    for index, text in enumerate(alternatives or [], start=1):
        readings.append(ReadingCandidate(f"c{index}", text, "alternative", False))
    return readings
