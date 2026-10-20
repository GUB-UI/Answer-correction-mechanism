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
# Colab の Python には ensurepip が無く、python3 -m venv は途中で失敗する。
# 失敗した環境には pip も入らない。uv は仮想環境の pip を使わずにパッケージを入れる。
if ! python3 -m uv --version >/dev/null 2>&1; then
  python3 -m pip install -q uv
fi
UNIMERNET_VENV="${ROOT}/.venv-unimernet"
# ディレクトリが既にあると uv venv は置き換え確認で入力待ちになる。Colab ではそこで止まる。
if [[ -x "${UNIMERNET_VENV}/bin/python" ]]; then
  python3 -m uv venv --allow-existing --python 3.12 "${UNIMERNET_VENV}"
else
  python3 -m uv venv --clear --force --python 3.12 "${UNIMERNET_VENV}"
fi
UV_PIP=(python3 -m uv pip install --python "${UNIMERNET_VENV}/bin/python")
if [[ "${UNIMERNET_TORCH:-cpu}" == "cuda" ]]; then
  "${UV_PIP[@]}" torch torchvision --index-url https://download.pytorch.org/whl/cu124
else
  # 数式領域は小さいので既定は CPU。GPU 版 torch の版ずれを避ける。
  "${UV_PIP[@]}" torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi
"${UV_PIP[@]}" unimernet huggingface_hub pillow "pyarrow<17"

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
