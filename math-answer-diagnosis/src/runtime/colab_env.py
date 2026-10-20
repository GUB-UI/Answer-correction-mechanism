"""Colab の .env で Mac の値を上書きするキー。

`.env.example` の OCR_MODEL_NAME と DIAGNOSIS_PROVIDER は Mac 用で、
環境変数は config.colab.yaml より優先される。ここを書き戻さないと
Colab が mlx のモデル名と omlx のままになる。
"""

from __future__ import annotations

# Hugging Face の公式 ID。呼称「xiaomi ocr 0」は SeerRay-Lab/Xiaomi-OCR-0（0.8B）。
XIAOMI_OCR_MODEL_ID = "SeerRay-Lab/Xiaomi-OCR-0"

COLAB_ENV: dict[str, str] = {
    "APP_CONFIG_FILE": "config.colab.yaml",
    "OCR_PROVIDER": "xiaomi_ocr",
    "OCR_MODEL_NAME": XIAOMI_OCR_MODEL_ID,
    "XIAOMI_OCR_MODEL_ID": XIAOMI_OCR_MODEL_ID,
    "DIAGNOSIS_PROVIDER": "colab_cuda",
    "OCR_API_HOST": "127.0.0.1",
    # Colab の VM は 8080 を自前の node が使っている。
    "OCR_API_PORT": "8081",
    "SERVICE_MODE": "managed",
    # A100 の掃引では UniMERNet をオンにすると複数行が壊れた。既定はオフ。
    "UNIMERNET_ENABLED": "false",
}


def colab_env_updates(profile_env: dict[str, str] | None = None) -> dict[str, str]:
    updates = dict(COLAB_ENV)
    if profile_env:
        updates.update(profile_env)
    return updates
