from __future__ import annotations

from pathlib import Path
from typing import Any

from src.adapters.base import DiagnosisAdapter, DiagnosisRunContext, OCRAdapter, OCRRunContext
from src.adapters.diagnosis_helpers import build_diagnosis_result, parse_diagnosis_json
from src.config import AppConfig, load_config
from src.models import DiagnosisResult, OCRResult, utc_now_iso
from src.prompts import DEFAULT_DIAGNOSIS_PROMPT_TYPE, DIAGNOSIS_PROMPTS, OCR_PROMPT
from src.storage import Storage
from src.utils import detect_uncertain_parts, is_ocr_suspect


class MockOCRAdapter(OCRAdapter):
    def run(self, context: OCRRunContext) -> OCRResult:
        raw_text = (
            "x^2 + 5x + 6 = 0\n"
            "(x + 2)(x + 3) = 0\n"
            "x = -2, -3"
        )
        uncertain_parts = detect_uncertain_parts(raw_text)
        return OCRResult(
            ocr_id="",
            answer_id=context.answer_id,
            ocr_engine=context.ocr_engine,
            raw_text=raw_text,
            used_text=raw_text,
            uncertain_parts=uncertain_parts,
            ocr_suspect=is_ocr_suspect(raw_text, uncertain_parts),
            human_corrected=False,
            correction_note=None,
            raw_output={"provider": "mock", "image_path": str(context.image_path)},
            created_at=utc_now_iso(),
        )


class MockDiagnosisAdapter(DiagnosisAdapter):
    def run(self, context: DiagnosisRunContext) -> dict[str, Any]:
        return {
            "score": 3.0,
            "max_score": 5.0,
            "error_categories": ["計算ミス"],
            "error_locations": [
                {
                    "line": "1行目",
                    "description": "符号の読み取りが不自然な可能性",
                }
            ],
            "reasoning_summary": "因数分解の方針は妥当ですが、途中式に不自然な点があります。",
            "student_feedback": "因数分解の流れはよいので、符号を一行ずつ確認しましょう。",
            "teacher_review_notes": "OCR結果の符号が不自然なため、原文確認を推奨します。",
            "possible_ocr_issue": True,
            "confidence": 0.68,
            "mock_input_excerpt": context.used_text[:120],
        }

    def to_diagnosis_result(
        self,
        context: DiagnosisRunContext,
        raw_output: dict[str, Any],
        diagnosis_id: str,
        created_at: str,
    ) -> DiagnosisResult:
        return build_diagnosis_result(
            context,
            raw_output,
            diagnosis_id,
            created_at=created_at,
        )
