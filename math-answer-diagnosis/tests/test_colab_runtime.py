from __future__ import annotations

from pathlib import Path

from src.adapters.diagnosis_helpers import normalize_question_scores
from src.runtime.colab_env import COLAB_ENV
from src.runtime.envfile import update_env_file
from src.runtime.ocr_image import resize_for_ocr
from src.runtime.ports import assign_listen_port, clear_assigned_ports
from src.runtime.quant_dtype import checkpoint_compute_dtype, resolve_load_dtype

from PIL import Image


def test_colab_env_replaces_mac_dotenv_values(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text(
        "OCR_MODEL_NAME=mlx-community/GLM-OCR-bf16\nDIAGNOSIS_PROVIDER=omlx\nOCR_API_PORT=8080\n",
        encoding="utf-8",
    )
    update_env_file(path, COLAB_ENV)
    text = path.read_text(encoding="utf-8")
    assert "OCR_PROVIDER=xiaomi_ocr" in text
    assert "OCR_MODEL_NAME=SeerRay-Lab/Xiaomi-OCR-0" in text
    assert "UNIMERNET_ENABLED=false" in text
    assert "DIAGNOSIS_PROVIDER=colab_cuda" in text
    assert "OCR_API_PORT=8081" in text
    assert "mlx-community" not in text
    assert "DIAGNOSIS_PROVIDER=omlx" not in text


def test_busy_glm_port_moves_to_the_next_free_port(monkeypatch) -> None:
    clear_assigned_ports()
    busy = 18080
    monkeypatch.setattr("src.runtime.ports.port_in_use", lambda host, port: port == busy)
    try:
        chosen = assign_listen_port("glm_ocr", busy)
        assert chosen == busy + 1
        assert assign_listen_port("glm_ocr", busy) == chosen
    finally:
        clear_assigned_ports()


def test_prequantized_checkpoint_keeps_float16_on_bf16_gpu() -> None:
    quant = {
        "quant_method": "bitsandbytes",
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": "float16",
    }
    assert checkpoint_compute_dtype(quant) == "float16"
    assert resolve_load_dtype("auto", "float16", cuda_bf16=True) == "float16"
    assert resolve_load_dtype("bfloat16", "float16", cuda_bf16=True) == "bfloat16"
    assert checkpoint_compute_dtype(None) is None
    assert resolve_load_dtype("auto", None, cuda_bf16=True) == "bfloat16"
    assert checkpoint_compute_dtype({"load_in_4bit": True}) == "float16"


def test_question_scores_drop_total_and_unknown_ids() -> None:
    raw = [
        {"question": "全体", "score": 5, "max_score": 10},
        {"question": "問1", "score": 2, "max_score": 4},
        {"question": "(9)", "score": 1, "max_score": 1},
    ]
    allocation = [{"question": "(1)", "max_score": 4}, {"question": "(2)", "max_score": 6}]
    scores = normalize_question_scores(raw, allocation)
    assert scores == [
        {"question": "(1)", "score": 2.0, "max_score": 4.0, "reason": "", "status": "validated"},
    ]
    assert normalize_question_scores([{"question": "全体", "score": 3}]) == []
    kept = normalize_question_scores([{"question": "(2)", "score": 1, "max_score": 3}])
    assert kept[0]["question"] == "(2)"


def test_ocr_suspect_flags_garbage() -> None:
    from src.adapters.ocr_quality import assess_ocr_text

    assert "chinese_in_japanese" in assess_ocr_text("よって 最小值 は $x=1$ である。当且仅当。")
    repeated = "x<5であるから次へ\n" * 3
    assert "repeated_lines" in assess_ocr_text(repeated)
    reasons = assess_ocr_text(
        "x<5",
        replacements=[{"glm": "x<5", "skipped": "unimernet_diverged", "unimernet_raw": r"\beta"}],
    )
    assert "unimernet_disagreement" in reasons
    assert assess_ocr_text("よって $x<5$ である。") == []


def test_empty_question_scores_follow_allocation() -> None:
    from src.adapters.base import DiagnosisRunContext
    from src.adapters.diagnosis_helpers import build_diagnosis_result
    from src.models import ProblemRecord, utc_now_iso

    problem = ProblemRecord(
        problem_id="prob",
        title="t",
        created_at=utc_now_iso(),
        score_allocation=[
            {"question": "(1)", "max_score": 4},
            {"question": "(2)", "max_score": 6},
        ],
    )
    context = DiagnosisRunContext(
        answer_id="ans",
        problem_id="prob",
        ocr_id="ocr",
        problem=problem,
        used_text="x=1",
        model_name="mock",
        prompt_type="basic_scoring",
    )
    full = build_diagnosis_result(
        context,
        {"score": 10, "max_score": 10, "question_scores": [], "possible_ocr_issue": False},
        "diag",
    )
    assert [item["score"] for item in full.question_scores] == [None, None]
    held = build_diagnosis_result(
        context,
        {
            "score": 0,
            "max_score": 10,
            "question_scores": [],
            "possible_ocr_issue": True,
            "teacher_review_notes": "",
        },
        "diag2",
    )
    assert held.review_status == "review_required"
    assert all(item["score"] is None for item in held.question_scores)
    assert "要教師確認" in held.teacher_review_notes
    partial = build_diagnosis_result(
        context,
        {
            "score": 0,
            "max_score": 10,
            "possible_ocr_issue": True,
            "question_scores": [
                {"question": "(1)", "score": 2, "max_score": 4, "reason": "途中まで正しい"},
                {"question": "(2)", "score": 3, "max_score": 6, "reason": "x>=2 は正しい"},
            ],
        },
        "diag3",
    )
    assert partial.score == 0
    assert partial.review_status == "review_required"


def test_xiaomi_adapter_sends_japanese_prompt(tmp_path: Path) -> None:
    from src.adapters.ocr_adapter import XiaomiOCRAdapter, create_ocr_adapter
    from src.adapters.base import OCRRunContext
    from src.config import load_config
    from tests.test_adapters import _write_test_config

    root = tmp_path / "project"
    root.mkdir()
    _write_test_config(root)
    image = root / "sample.png"
    Image.new("RGB", (20, 10), "white").save(image)
    seen: dict[str, object] = {}

    def complete(messages: list) -> str:
        seen["messages"] = messages
        return "よって $x<5$"

    config = load_config(root)
    adapter = XiaomiOCRAdapter(config, complete=complete)
    result = adapter.run(
        OCRRunContext(answer_id="ans", image_path=image, ocr_engine="SeerRay-Lab/Xiaomi-OCR-0")
    )
    assert result.raw_text == "よって $x<5$"
    assert result.ocr_suspect is False
    prompt = seen["messages"][0]["content"][1]["text"]
    assert "Markdown" in prompt and "LaTeX" in prompt and "中国語に翻訳しない" in prompt
    assert create_ocr_adapter("xiaomi_ocr", config).__class__.__name__ == "XiaomiOCRAdapter"


def test_resize_for_ocr_is_opt_in() -> None:
    image = Image.new("RGB", (20, 10), "white")
    assert resize_for_ocr(image).size == (20, 10)
    enlarged = resize_for_ocr(image, min_edge=40)
    assert min(enlarged.size) == 40
    capped = resize_for_ocr(Image.new("RGB", (1000, 500), "white"), max_edge=100)
    assert max(capped.size) == 100
