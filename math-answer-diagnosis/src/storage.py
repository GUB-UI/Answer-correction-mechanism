from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from src.config import AppConfig, load_config
from src.models import (
    AnswerRecord,
    DiagnosisResult,
    OCRResult,
    ProblemRecord,
)

T = TypeVar("T", bound=BaseModel)

RECORD_FILES = {
    "problems": "problems.jsonl",
    "answers": "answers.jsonl",
    "ocr_results": "ocr_results.jsonl",
    "diagnosis_results": "diagnosis_results.jsonl",
}


class Storage:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        self.config.images_dir.mkdir(parents=True, exist_ok=True)
        self.config.records_dir.mkdir(parents=True, exist_ok=True)
        for filename in RECORD_FILES.values():
            path = self.config.records_dir / filename
            if not path.exists():
                path.touch()

    def _record_path(self, record_type: str) -> Path:
        filename = RECORD_FILES[record_type]
        return self.config.records_dir / filename

    def generate_id(self, prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    def save_image(self, image_bytes: bytes, extension: str = ".png") -> str:
        ext = extension if extension.startswith(".") else f".{extension}"
        image_id = self.generate_id("img")
        path = self.config.images_dir / f"{image_id}{ext}"
        path.write_bytes(image_bytes)
        return str(path.relative_to(self.config.project_root))

    def save_answer_image(
        self,
        answer_id: str,
        image_bytes: bytes,
        extension: str = ".png",
    ) -> str:
        ext = extension if extension.startswith(".") else f".{extension}"
        path = self.config.images_dir / f"{answer_id}{ext}"
        path.write_bytes(image_bytes)
        return str(path.relative_to(self.config.project_root))

    @staticmethod
    def _append_jsonl(path: Path, payload: dict) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        if not path.exists():
            return []
        rows: list[dict] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _load_models(self, record_type: str, model_cls: type[T]) -> list[T]:
        rows = self._read_jsonl(self._record_path(record_type))
        return [model_cls.model_validate(row) for row in rows]

    def save_problem(self, problem: ProblemRecord) -> ProblemRecord:
        self._append_jsonl(self._record_path("problems"), problem.model_dump())
        return problem

    def list_problems(self) -> list[ProblemRecord]:
        return self._load_models("problems", ProblemRecord)

    def get_problem(self, problem_id: str) -> ProblemRecord | None:
        for problem in self.list_problems():
            if problem.problem_id == problem_id:
                return problem
        return None

    def save_answer(self, answer: AnswerRecord) -> AnswerRecord:
        self._append_jsonl(self._record_path("answers"), answer.model_dump())
        return answer

    def list_answers(self) -> list[AnswerRecord]:
        return self._load_models("answers", AnswerRecord)

    def get_answer(self, answer_id: str) -> AnswerRecord | None:
        for answer in self.list_answers():
            if answer.answer_id == answer_id:
                return answer
        return None

    def save_ocr_result(self, ocr_result: OCRResult) -> OCRResult:
        self._append_jsonl(
            self._record_path("ocr_results"),
            ocr_result.model_dump(),
        )
        return ocr_result

    def list_ocr_results(self) -> list[OCRResult]:
        return self._load_models("ocr_results", OCRResult)

    def get_ocr_result(self, ocr_id: str) -> OCRResult | None:
        for result in self.list_ocr_results():
            if result.ocr_id == ocr_id:
                return result
        return None

    def get_ocr_results_for_answer(self, answer_id: str) -> list[OCRResult]:
        return [r for r in self.list_ocr_results() if r.answer_id == answer_id]

    def update_ocr_result(self, ocr_result: OCRResult) -> OCRResult:
        rows = self._read_jsonl(self._record_path("ocr_results"))
        updated = False
        for index, row in enumerate(rows):
            if row.get("ocr_id") == ocr_result.ocr_id:
                rows[index] = ocr_result.model_dump()
                updated = True
                break
        if not updated:
            raise KeyError(f"OCR result not found: {ocr_result.ocr_id}")
        self._write_jsonl(self._record_path("ocr_results"), rows)
        return ocr_result

    def save_diagnosis_result(self, diagnosis: DiagnosisResult) -> DiagnosisResult:
        self._append_jsonl(
            self._record_path("diagnosis_results"),
            diagnosis.model_dump(),
        )
        return diagnosis

    def list_diagnosis_results(self) -> list[DiagnosisResult]:
        return self._load_models("diagnosis_results", DiagnosisResult)

    def get_diagnosis_result(self, diagnosis_id: str) -> DiagnosisResult | None:
        for result in self.list_diagnosis_results():
            if result.diagnosis_id == diagnosis_id:
                return result
        return None

    def get_diagnosis_results_for_answer(
        self,
        answer_id: str,
    ) -> list[DiagnosisResult]:
        return [
            r for r in self.list_diagnosis_results() if r.answer_id == answer_id
        ]


def validate_anonymized_id(student_id: str) -> str:
    value = student_id.strip()
    if not value:
        raise ValueError("匿名化生徒IDは必須です")
    if re.search(r"[\s/\\]", value):
        raise ValueError("匿名化生徒IDに空白やパス区切り文字は使えません")
    return value
