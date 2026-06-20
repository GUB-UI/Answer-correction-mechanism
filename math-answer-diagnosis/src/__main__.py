from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

from src.adapters import (
    ManualCorrectionOCRAdapter,
    create_diagnosis_adapter,
    create_ocr_adapter,
)
from src.adapters.base import DiagnosisRunContext, OCRRunContext
from src.config import load_config
from src.models import AnswerRecord, ProblemRecord, utc_now_iso
from src.storage import Storage, validate_anonymized_id


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def cmd_smoke(_: argparse.Namespace) -> int:
    root = _project_root()
    config = load_config(root)
    storage = Storage(config)

    print("=== config ===")
    print(f"LM Studio URL: {config.lmstudio.base_url}")
    print(f"OCR provider: {config.ocr.provider} ({config.ocr.model_name})")
    print(
        f"Diagnosis provider: {config.diagnosis.provider} "
        f"({config.diagnosis.model_name})"
    )

    problem = ProblemRecord(
        problem_id=storage.generate_id("prob"),
        title="二次方程式",
        problem_text="x^2 + 5x + 6 = 0 を解きなさい。",
        correct_answer="x = -2, -3",
        rubric="因数分解または解の公式で正解を導ければ満点。",
        unit="二次関数",
        difficulty="basic",
        created_at=utc_now_iso(),
    )
    storage.save_problem(problem)
    print(f"saved problem: {problem.problem_id}")

    image = Image.new("RGB", (200, 120), color="white")
    image_path = config.images_dir / f"{storage.generate_id('img')}.png"
    image.save(image_path)
    answer = AnswerRecord(
        answer_id=storage.generate_id("ans"),
        problem_id=problem.problem_id,
        image_path=str(image_path.relative_to(root)),
        student_anonymized_id=validate_anonymized_id("student_001"),
        created_at=utc_now_iso(),
    )
    storage.save_answer(answer)
    print(f"saved answer: {answer.answer_id}")

    ocr_adapter = create_ocr_adapter("mock", config)
    ocr_result = ocr_adapter.run(
        OCRRunContext(
            answer_id=answer.answer_id,
            image_path=root / answer.image_path,
            ocr_engine=config.ocr.model_name,
        )
    )
    ocr_result.ocr_id = storage.generate_id("ocr")
    storage.save_ocr_result(ocr_result)
    print(f"saved OCR: {ocr_result.ocr_id}")
    print(f"  raw_text: {ocr_result.raw_text!r}")
    print(f"  used_text: {ocr_result.used_text!r}")

    manual = ManualCorrectionOCRAdapter(storage)
    corrected = manual.apply_correction(
        ocr_result.ocr_id,
        "x^2 - 5x + 6 = 0\n(x - 2)(x - 3) = 0\nx = 2, 3",
        ocr_suspect=True,
        correction_note="OCRが -5x を +5x と誤読していた",
    )
    print(f"corrected OCR used_text: {corrected.used_text!r}")

    diagnosis_adapter = create_diagnosis_adapter("mock", config)
    diagnosis_context = DiagnosisRunContext(
        answer_id=answer.answer_id,
        problem_id=problem.problem_id,
        ocr_id=corrected.ocr_id,
        problem=problem,
        used_text=corrected.used_text,
        model_name=config.diagnosis.model_name,
        prompt_type="basic_scoring",
    )
    diagnosis_raw = diagnosis_adapter.run(diagnosis_context)
    diagnosis = diagnosis_adapter.to_diagnosis_result(
        diagnosis_context,
        diagnosis_raw,
        diagnosis_id=storage.generate_id("diag"),
        created_at=utc_now_iso(),
    )
    storage.save_diagnosis_result(diagnosis)
    print(f"saved diagnosis: {diagnosis.diagnosis_id}")
    print(f"  possible_ocr_issue: {diagnosis.possible_ocr_issue}")

    reloaded_problems = storage.list_problems()
    reloaded_answers = storage.list_answers()
    reloaded_ocr = storage.list_ocr_results()
    reloaded_diagnosis = storage.list_diagnosis_results()
    print("=== reload counts ===")
    print(
        json.dumps(
            {
                "problems": len(reloaded_problems),
                "answers": len(reloaded_answers),
                "ocr_results": len(reloaded_ocr),
                "diagnosis_results": len(reloaded_diagnosis),
            },
            ensure_ascii=False,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Math answer diagnosis CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("smoke", help="Run end-to-end mock smoke test")
    args = parser.parse_args(argv)

    if args.command == "smoke":
        return cmd_smoke(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
