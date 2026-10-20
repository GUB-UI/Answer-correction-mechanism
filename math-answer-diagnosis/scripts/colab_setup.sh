#!/usr/bin/env bash
# Colab でアプリ本体と、隔離した UniMERNet 環境を用意する。
# GLM-OCR と診断 LLM は Colab の GPU 版 PyTorch を使うので、ここでは torch を入れ直さない。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 -m pip install -q -U pip
python3 -m pip install -q --upgrade-strategy only-if-needed -r requirements-colab.txt

if [[ -n "${APP_DATA_DIR:-}" ]]; then
  mkdir -p "${APP_DATA_DIR}/images" "${APP_DATA_DIR}/records" "${APP_DATA_DIR}/models"
fi
if [[ -n "${HF_HOME:-}" ]]; then
  mkdir -p "${HF_HOME}"
fi
mkdir -p "${ROOT}/.run"

# UniMERNet は transformers==4.42.4 固定なので、GLM-OCR / Qwen（transformers 5 系）と混ぜない。
UNIMERNET_VENV="${ROOT}/.venv-unimernet"
if [[ ! -x "${UNIMERNET_VENV}/bin/python" ]]; then
  if ! python3 -m venv "${UNIMERNET_VENV}"; then
    python3 -m pip install -q uv
    uv venv --python 3.12 "${UNIMERNET_VENV}"
  fi
fi
"${UNIMERNET_VENV}/bin/python" -m pip install -q -U pip
if [[ "${UNIMERNET_TORCH:-cpu}" == "cuda" ]]; then
  "${UNIMERNET_VENV}/bin/python" -m pip install -q torch torchvision \
    --index-url https://download.pytorch.org/whl/cu124
else
  # 数式領域は小さいので既定は CPU。GPU 版 torch の版ずれを避ける。
  "${UNIMERNET_VENV}/bin/python" -m pip install -q torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu
fi
"${UNIMERNET_VENV}/bin/python" -m pip install -q unimernet huggingface_hub pillow "pyarrow<17"

MODEL_DIR="${UNIMERNET_MODEL_DIR:-${ROOT}/data/models/unimernet_small}"
if [[ -n "${APP_DATA_DIR:-}" && -z "${UNIMERNET_MODEL_DIR:-}" ]]; then
  MODEL_DIR="${APP_DATA_DIR}/models/unimernet_small"
fi
if [[ ! -f "${MODEL_DIR}/unimernet_small.pth" ]]; then
  echo "Downloading wanderkid/unimernet_small to ${MODEL_DIR}"
  mkdir -p "${MODEL_DIR}"
  "${UNIMERNET_VENV}/bin/python" - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="wanderkid/unimernet_small", local_dir="${MODEL_DIR}")
PY
fi

if [[ ! -f "${ROOT}/.env" ]]; then
  cp "${ROOT}/.env.example" "${ROOT}/.env"
fi

echo
echo "Colab の準備ができました。"
echo "次: APP_CONFIG_FILE=config.colab.yaml python3 -m src gpu-profile --write-env .env"
