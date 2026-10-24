"""Split OCR text into lines and math spans. This does not correct the text."""

from __future__ import annotations

import re
from dataclasses import dataclass

_MATH = re.compile(r"\$\$[\s\S]+?\$\$|\$[^$\n]+\$")


@dataclass(frozen=True)
class Region:
    region_id: str
    kind: str
    text: str
    start: int
    end: int


def structure_text(text: str) -> list[Region]:
    regions: list[Region] = []
    index = 0
    for match in _MATH.finditer(text):
        regions.append(
            Region(
                region_id=f"m{index}",
                kind="math",
                text=match.group(0),
                start=match.start(),
                end=match.end(),
            )
        )
        index += 1
    for line_index, line in enumerate(text.splitlines()):
        regions.append(
            Region(
                region_id=f"l{line_index}",
                kind="line",
                text=line,
                start=0,
                end=len(line),
            )
        )
    return regions
