from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.config import load_config
from src.adapters import ManualCorrectionOCRAdapter, create_ocr_adapter
from src.adapters.base import OCRRunContext
from src.models import AnswerRecord, OCRResult, ProblemRecord, utc_now_iso
from src.storage import Storage, validate_anonymized_id


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    (root / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "runtime": {"device": "test", "use_lmstudio": True},
                "lmstudio": {
                    "base_url": "http://localhost:1234/v1",
                    "api_key": "lm-studio",
                },
                "models": {
                    "ocr": {"provider": "mock", "model_name": "zai-small-ocr"},
                    "diagnosis": {
                        "provider": "mock",
                        "model_name": "local-diagnosis-model",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return root


@pytest.fixture
def storage(project_root: Path) -> Storage:
    return Storage(load_config(project_root))


def test_data_directories_are_created(storage: Storage) -> None:
    assert storage.config.images_dir.is_dir()
    assert storage.config.records_dir.is_dir()
    assert (storage.config.records_dir / "problems.jsonl").is_file()


def test_generate_id_has_prefix(storage: Storage) -> None:
    value = storage.generate_id("prob")
    assert value.startswith("prob_")


def test_validate_anonymized_id_rejects_blank() -> None:
    with pytest.raises(ValueError):
        validate_anonymized_id("   ")


def test_problem_roundtrip(storage: Storage) -> None:
    problem = ProblemRecord(
        problem_id=storage.generate_id("prob"),
        title="title",
        problem_text="text",
        correct_answer="answer",
        rubric="rubric",
        unit="unit",
        difficulty="basic",
        created_at=utc_now_iso(),
    )
    storage.save_problem(problem)
    assert storage.list_problems()[-1].problem_id == problem.problem_id


def test_answer_roundtrip(storage: Storage) -> None:
    answer = AnswerRecord(
        answer_id=storage.generate_id("ans"),
        problem_id="prob_001",
        image_path=storage.save_image(b"data", extension=".png"),
        student_anonymized_id=validate_anonymized_id("student_001"),
        created_at=utc_now_iso(),
    )
    storage.save_answer(answer)
    loaded = storage.get_answer(answer.answer_id)
    assert loaded is not None
    assert loaded.student_anonymized_id == "student_001"


def test_jsonl_append_only(storage: Storage) -> None:
    for index in range(2):
        storage.save_problem(
            ProblemRecord(
                problem_id=f"prob_{index}",
                title=f"title{index}",
                problem_text="text",
                correct_answer="answer",
                rubric="rubric",
                unit="unit",
                difficulty="basic",
                created_at=utc_now_iso(),
            )
        )
    path = storage.config.records_dir / "problems.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 2


def test_save_answer_with_internal_image_id(storage: Storage, project_root: Path) -> None:
    image_rel = storage.save_image(b"fake-image", extension=".png")
    answer = AnswerRecord(
        answer_id=storage.generate_id("ans"),
        problem_id="prob_001",
        image_path=image_rel,
        student_anonymized_id=validate_anonymized_id("student_001"),
        created_at=utc_now_iso(),
    )
    storage.save_answer(answer)
    loaded = storage.get_answer(answer.answer_id)
    assert loaded is not None
    assert "student_001" not in loaded.image_path
    assert (project_root / loaded.image_path).exists()


def test_jsonl_reload(storage: Storage) -> None:
    problem = ProblemRecord(
        problem_id="prob_jsonl",
        title="title",
        problem_text="text",
        correct_answer="answer",
        rubric="rubric",
        unit="unit",
        difficulty="basic",
        created_at=utc_now_iso(),
    )
    storage.save_problem(problem)
    reloaded = Storage(storage.config)
    assert reloaded.get_problem("prob_jsonl") is not None


def test_mock_ocr_saves_raw_and_used_text(storage: Storage, project_root: Path) -> None:
    image_rel = storage.save_image(b"png", extension=".png")
    answer = AnswerRecord(
        answer_id=storage.generate_id("ans"),
        problem_id="prob_001",
        image_path=image_rel,
        student_anonymized_id="student_001",
        created_at=utc_now_iso(),
    )
    adapter = create_ocr_adapter("mock", storage.config)
    result = adapter.run(
        OCRRunContext(
            answer_id=answer.answer_id,
            image_path=project_root / image_rel,
            ocr_engine="mock-ocr",
        )
    )
    result.ocr_id = storage.generate_id("ocr")
    storage.save_ocr_result(result)
    loaded = storage.get_ocr_result(result.ocr_id)
    assert loaded is not None
    assert loaded.raw_text == loaded.used_text


def test_manual_correction_updates_used_text(storage: Storage) -> None:
    ocr = OCRResult(
        ocr_id=storage.generate_id("ocr"),
        answer_id="ans_001",
        ocr_engine="mock",
        raw_text="x^2 + 5x + 6 = 0",
        used_text="x^2 + 5x + 6 = 0",
        created_at=utc_now_iso(),
    )
    storage.save_ocr_result(ocr)
    manual = ManualCorrectionOCRAdapter(storage)
    updated = manual.apply_correction(
        ocr.ocr_id,
        "x^2 - 5x + 6 = 0",
        ocr_suspect=True,
        correction_note="符号修正",
    )
    assert updated.used_text == "x^2 - 5x + 6 = 0"
    assert updated.human_corrected is True
