from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from src.adapters.base import DiagnosisRunContext, OCRRunContext
from src.adapters.diagnosis_adapter import LMStudioDiagnosisAdapter, create_diagnosis_adapter
from src.adapters.lmstudio_client import LMStudioClient
from src.adapters.ocr_adapter import GLMOCRSdkAdapter, create_ocr_adapter
from src.config import load_config
from src.models import ProblemRecord, utc_now_iso
from src.storage import Storage


def _write_test_config(root: Path) -> None:
    (root / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "runtime": {"device": "test"},
                "ocr": {"provider": "mock", "model_name": "glm-ocr"},
                "lmstudio": {
                    "base_url": "http://localhost:1234/v1",
                    "api_key": "lm-studio",
                },
                "models": {
                    "diagnosis": {
                        "provider": "mock",
                        "model_name": "local-diagnosis-model",
                    },
                },
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    _write_test_config(root)
    return root


def test_load_config_reads_lmstudio_settings(project_root: Path) -> None:
    os.environ["LMSTUDIO_BASE_URL"] = "http://localhost:4321/v1"
    os.environ["LMSTUDIO_API_KEY"] = "override-key"
    try:
        config = load_config(project_root)
        assert config.lmstudio.base_url == "http://localhost:4321/v1"
        assert config.lmstudio.api_key == "override-key"
    finally:
        os.environ.pop("LMSTUDIO_BASE_URL", None)
        os.environ.pop("LMSTUDIO_API_KEY", None)


def test_load_config_reads_ocr_from_top_level(project_root: Path) -> None:
    config = load_config(project_root)
    assert config.ocr.provider == "mock"
    assert config.ocr.model_name == "glm-ocr"


def test_create_ocr_adapter_mock(project_root: Path) -> None:
    config = load_config(project_root)
    adapter = create_ocr_adapter("mock", config)
    image_path = project_root / "sample.png"
    image_path.write_bytes(b"png")
    result = adapter.run(
        OCRRunContext(
            answer_id="ans_001",
            image_path=image_path,
            ocr_engine="mock",
        )
    )
    assert result.raw_text
    assert result.used_text == result.raw_text


def test_glmocr_sdk_adapter_not_configured(project_root: Path) -> None:
    config = load_config(project_root)
    adapter = GLMOCRSdkAdapter(config=config)
    image_path = project_root / "sample.png"
    image_path.write_bytes(b"png")
    with pytest.raises(RuntimeError, match="GLM-OCR SDK OCR failed"):
        adapter.run(
            OCRRunContext(
                answer_id="ans_001",
                image_path=image_path,
                ocr_engine="glm-ocr",
            )
        )


def test_create_ocr_adapter_rejects_unknown_provider(project_root: Path) -> None:
    config = load_config(project_root)
    with pytest.raises(ValueError, match="Unsupported OCR provider"):
        create_ocr_adapter("zai", config)


def test_mock_diagnosis_adapter(project_root: Path) -> None:
    config = load_config(project_root)
    adapter = create_diagnosis_adapter("mock", config)
    problem = ProblemRecord(
        problem_id="prob_001",
        title="title",
        problem_text="text",
        correct_answer="answer",
        rubric="rubric",
        unit="unit",
        difficulty="basic",
        created_at=utc_now_iso(),
    )
    context = DiagnosisRunContext(
        answer_id="ans_001",
        problem_id=problem.problem_id,
        ocr_id="ocr_001",
        problem=problem,
        used_text="x^2 - 5x + 6 = 0",
        model_name="mock",
        prompt_type="basic_scoring",
    )
    raw = adapter.run(context)
    diagnosis = adapter.to_diagnosis_result(
        context,
        raw,
        diagnosis_id="diag_001",
        created_at=utc_now_iso(),
    )
    assert diagnosis.possible_ocr_issue is True
    assert diagnosis.error_categories


def test_lmstudio_diagnosis_adapter_parses_json(project_root: Path) -> None:
    config = load_config(project_root)
    client = MagicMock(spec=LMStudioClient)
    client.chat_completion.return_value = (
        '{"score": 4, "max_score": 5, "error_categories": ["計算ミス"], '
        '"error_locations": [], "reasoning_summary": "ok", '
        '"student_feedback": "good", "teacher_review_notes": "check", '
        '"possible_ocr_issue": false, "confidence": 0.9}'
    )
    adapter = LMStudioDiagnosisAdapter(config=config, lmstudio_client=client)
    problem = ProblemRecord(
        problem_id="prob_001",
        title="title",
        problem_text="text",
        correct_answer="answer",
        rubric="rubric",
        unit="unit",
        difficulty="basic",
        created_at=utc_now_iso(),
    )
    context = DiagnosisRunContext(
        answer_id="ans_001",
        problem_id=problem.problem_id,
        ocr_id="ocr_001",
        problem=problem,
        used_text="x = 1",
        model_name="local-diagnosis-model",
        prompt_type="basic_scoring",
    )
    raw = adapter.run(context)
    diagnosis = adapter.to_diagnosis_result(
        context,
        raw,
        diagnosis_id="diag_001",
        created_at=utc_now_iso(),
    )
    assert diagnosis.score == 4.0
    assert diagnosis.max_score == 5.0


def test_mock_diagnosis_possible_ocr_issue_saved(project_root: Path) -> None:
    storage = Storage(load_config(project_root))
    config = storage.config
    adapter = create_diagnosis_adapter("mock", config)
    problem = ProblemRecord(
        problem_id="prob_001",
        title="title",
        problem_text="text",
        correct_answer="answer",
        rubric="rubric",
        unit="unit",
        difficulty="basic",
        created_at=utc_now_iso(),
    )
    context = DiagnosisRunContext(
        answer_id="ans_001",
        problem_id=problem.problem_id,
        ocr_id="ocr_001",
        problem=problem,
        used_text="x^2 - 5x + 6 = 0",
        model_name="mock",
        prompt_type="basic_scoring",
    )
    raw = adapter.run(context)
    diagnosis = adapter.to_diagnosis_result(
        context,
        raw,
        diagnosis_id=storage.generate_id("diag"),
        created_at=utc_now_iso(),
    )
    storage.save_diagnosis_result(diagnosis)
    loaded = storage.get_diagnosis_result(diagnosis.diagnosis_id)
    assert loaded is not None
    assert loaded.possible_ocr_issue is True
