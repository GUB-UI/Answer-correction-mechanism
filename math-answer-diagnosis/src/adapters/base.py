from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.models import DiagnosisResult, OCRResult, ProblemRecord


@dataclass
class OCRRunContext:
    answer_id: str
    image_path: Path
    ocr_engine: str


@dataclass
class DiagnosisRunContext:
    answer_id: str
    problem_id: str
    ocr_id: str
    problem: ProblemRecord
    used_text: str
    model_name: str
    prompt_type: str


class OCRAdapter(ABC):
    @abstractmethod
    def run(self, context: OCRRunContext) -> OCRResult:
        raise NotImplementedError


class DiagnosisAdapter(ABC):
    @abstractmethod
    def run(self, context: DiagnosisRunContext) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def to_diagnosis_result(
        self,
        context: DiagnosisRunContext,
        raw_output: dict[str, Any],
        diagnosis_id: str,
        created_at: str,
    ) -> DiagnosisResult:
        raise NotImplementedError
