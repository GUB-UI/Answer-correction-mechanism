from __future__ import annotations

import socket
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
    assert "OCR_MODEL_NAME=zai-org/GLM-OCR" in text
    assert "DIAGNOSIS_PROVIDER=colab_cuda" in text
    assert "OCR_API_PORT=8081" in text
    assert "mlx-community" not in text
    assert "DIAGNOSIS_PROVIDER=omlx" not in text


def test_busy_glm_port_moves_to_the_next_free_port() -> None:
    clear_assigned_ports()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    busy = sock.getsockname()[1]
    try:
        chosen = assign_listen_port("glm_ocr", busy)
        assert chosen != busy
        assert assign_listen_port("glm_ocr", busy) == chosen
    finally:
        sock.close()
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
        {"question": "(1)", "score": 2.0, "max_score": 4.0, "reason": ""},
    ]
    assert normalize_question_scores([{"question": "全体", "score": 3}]) == []
    kept = normalize_question_scores([{"question": "(2)", "score": 1, "max_score": 3}])
    assert kept[0]["question"] == "(2)"


def test_resize_for_ocr_is_opt_in() -> None:
    image = Image.new("RGB", (20, 10), "white")
    assert resize_for_ocr(image).size == (20, 10)
    enlarged = resize_for_ocr(image, min_edge=40)
    assert min(enlarged.size) == 40
    capped = resize_for_ocr(Image.new("RGB", (1000, 500), "white"), max_edge=100)
    assert max(capped.size) == 100
