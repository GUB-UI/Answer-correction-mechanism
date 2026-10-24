from __future__ import annotations

import json
import io
import sys
from pathlib import Path

import pytest
import yaml

from src.batch_pipeline import BatchOptions, run_batch
from src.config import load_config
from src.models import AnswerRecord, ProblemRecord, utc_now_iso
from src.runtime.gpu import GPUInfo, parse_nvidia_smi, select_profile
from src.runtime.services import ServiceManager, ServiceSpec, ServiceStartError, http_health
from src.storage import Storage

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
from _openai_compat import (  # noqa: E402
    chat_completion_response,
    content_text,
    split_assistant_prefill,
    strip_think,
)


def test_parse_nvidia_smi_and_tiers() -> None:
    gpus = parse_nvidia_smi("NVIDIA A100-SXM4-40GB, 40960, 12\nNVIDIA L4, 23034, 0\n")
    assert gpus[0].name.startswith("NVIDIA A100")
    assert select_profile(gpus[0]).tier == "a100"
    assert select_profile(gpus[0]).comparable_to_27b is True
    assert select_profile(gpus[1]).tier == "l4"
    assert select_profile(gpus[1]).llm_model_id == "Qwen/Qwen3.8-27B"
    t4 = select_profile(GPUInfo(name="Tesla T4", total_mib=15360))
    assert t4.tier == "t4"
    assert t4.comparable_to_27b is False
    assert "Qwen3.5-9B" in t4.llm_model_id
    cpu = select_profile(None)
    assert cpu.tier == "cpu"
    assert cpu.layout_device == "cpu"
    assert parse_nvidia_smi("garbage\n") == []


def test_chat_template_helpers() -> None:
    messages = [
        {"role": "system", "content": "規則"},
        {"role": "user", "content": [{"type": "text", "text": "採点して"}, {"type": "image_url", "image_url": {"url": "data:x"}}]},
        {"role": "assistant", "content": "<think>\n\n</think>\n\n"},
    ]
    history, prefill, is_empty = split_assistant_prefill(messages)
    assert is_empty is True
    assert len(history) == 2
    assert content_text(messages[1]["content"]) == "採点して"
    assert strip_think("<think>考え中</think>\n{\"score\": 1}") == '{"score": 1}'
    payload = chat_completion_response("m", "本文", {"prompt_tokens": 3, "completion_tokens": 2})
    assert payload["choices"][0]["message"]["content"] == "本文"
    assert payload["usage"]["total_tokens"] == 5


def test_openai_handler_reports_loading_then_serves() -> None:
    from _openai_compat import ServerState, make_handler

    class Backend:
        model_name = "tiny"

        def load(self) -> None:
            return None

        def generate(self, request: dict) -> tuple[str, dict[str, int]]:
            if request.get("fail"):
                raise RuntimeError("boom")
            return "ok", {"prompt_tokens": 1, "completion_tokens": 1}

    state = ServerState(Backend())
    handler = object.__new__(make_handler(state))
    payloads = []
    handler._send_json = lambda payload, status=200: payloads.append((payload, status))
    handler.path = "/health"
    handler.do_GET()
    assert payloads[-1][0]["ok"] is False
    state.ready = True
    handler.path = "/v1/health"
    handler.do_GET()
    assert payloads[-1][0]["ok"] is True
    body = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()
    handler.path = "/v1/chat/completions"
    handler.headers = {"Content-Length": str(len(body))}
    handler.rfile = io.BytesIO(body)
    handler.do_POST()
    payload, status = payloads[-1]
    assert status == 200
    assert payload["choices"][0]["message"]["content"] == "ok"


def test_service_manager_starts_stops_and_is_exclusive(tmp_path: Path) -> None:
    glm_port, llm_port = 18080, 18081
    command = [sys.executable, "-c", "import time; time.sleep(60)"]
    specs = {
        "glm_ocr": ServiceSpec("glm_ocr", command, glm_port, startup_timeout_sec=10),
        "llm": ServiceSpec("llm", command, llm_port, startup_timeout_sec=10),
    }
    manager = ServiceManager(
        specs, tmp_path / "run", mode="managed", exclusive_gpu=True, poll_interval_sec=0.05,
        health_check=lambda url: manager.read_pid("llm" if ":18081/" in url else "glm_ocr") is not None,
    )
    manager.ensure("glm_ocr")
    assert manager.is_healthy("glm_ocr")
    glm_pid = manager.read_pid("glm_ocr")
    assert glm_pid is not None
    manager.ensure("llm")
    assert manager.is_healthy("llm")
    assert manager.read_pid("glm_ocr") is None
    assert manager.stop("llm") is True
    assert manager.is_healthy("llm") is False

    external = ServiceManager(specs, tmp_path / "run2", mode="external")
    external.ensure("glm_ocr")
    assert external.events == []


def test_service_manager_reports_crash(tmp_path: Path) -> None:
    spec = ServiceSpec(
        "llm",
        [sys.executable, "-c", "import sys; sys.exit(3)"],
        port=1,
        startup_timeout_sec=3,
    )
    manager = ServiceManager({"llm": spec}, tmp_path / "run", mode="managed", poll_interval_sec=0.05,
                             health_check=lambda url: False)
    with pytest.raises(ServiceStartError, match="終了"):
        manager.ensure("llm")


def test_config_data_dir_and_service_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "config.yaml").write_text("runtime: {device: mac}\n", encoding="utf-8")
    data = tmp_path / "drive" / "experiment"
    monkeypatch.setenv("APP_DATA_DIR", str(data))
    monkeypatch.setenv("SERVICE_MODE", "managed")
    monkeypatch.setenv("LLM_MODEL_ID", "Qwen/Qwen3.8-27B")
    monkeypatch.setenv("DIAGNOSIS_PROVIDER", "colab_cuda")
    config = load_config(root)
    storage = Storage(config)
    stored = storage.save_image(b"png", extension=".png")
    assert stored.startswith("data/images/")
    assert (data / "images").is_dir()
    assert config.resolve_path(stored).is_file()
    assert config.services.mode == "managed"
    assert config.services.llm.model_id == "Qwen/Qwen3.8-27B"
    assert config.diagnosis.provider == "colab_cuda"
    mac_path = "/Users/ssh56/Desktop/F2/Answer-correction-mechanism/math-answer-diagnosis/data/images/ans_old.png"
    (data / "images" / "ans_old.png").write_bytes(b"old")
    assert config.resolve_path(mac_path).name == "ans_old.png"


def test_batch_runs_phases_and_reuses_results(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "runtime": {"device": "test"},
                "services": {"mode": "external"},
                "ocr": {"provider": "mock"},
                "models": {"diagnosis": {"provider": "mock", "model_name": "mock"}},
            }
        ),
        encoding="utf-8",
    )
    config = load_config(root)
    storage = Storage(config)
    storage.save_problem(
        ProblemRecord(
            problem_id="prob_1",
            title="t",
            problem_text="問題",
            correct_answer="答",
            rubric="基準",
            created_at=utc_now_iso(),
            score_allocation=[{"question": "(1)", "max_score": 5}],
        )
    )
    image = storage.save_image(b"fake", extension=".png")
    storage.save_answer(
        AnswerRecord(
            answer_id="ans_1",
            problem_id="prob_1",
            image_path=image,
            student_anonymized_id="student_001",
            created_at=utc_now_iso(),
        )
    )
    phases: list[str] = []
    options = BatchOptions(ocr_provider="mock", diagnosis_provider="mock")
    report = run_batch(
        storage,
        config,
        storage.list_answers(),
        options,
        progress=lambda phase, index, total, answer_id: phases.append(phase),
    )
    assert report.items[0].status == "review_required"
    assert report.items[0].score == 3.0
    assert phases == ["ocr", "diagnosis"]
    again = run_batch(storage, config, storage.list_answers(), options)
    assert again.items[0].status == "review_required"
    assert len(storage.list_diagnosis_results()) == 1
