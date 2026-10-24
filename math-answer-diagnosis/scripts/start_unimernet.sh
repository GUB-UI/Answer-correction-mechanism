#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${ROOT}/.venv-unimernet"
PYTHON="${UNIMERNET_PYTHON:-${VENV}/bin/python}"
MODEL_DIR="${UNIMERNET_MODEL_DIR:-${ROOT}/data/models/unimernet_small}"
PORT="${UNIMERNET_PORT:-8091}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "UniMERNet python is missing: ${PYTHON}" >&2
  echo "Mac: scripts/setup_production.sh / Colab: scripts/colab_setup.sh を先に実行してください。" >&2
  exit 1
fi

if [[ ! -f "${MODEL_DIR}/unimernet_small.pth" ]]; then
  echo "Downloading wanderkid/unimernet_small to ${MODEL_DIR} ..."
  mkdir -p "${MODEL_DIR}"
  "${PYTHON}" - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="wanderkid/unimernet_small",
    local_dir="${MODEL_DIR}",
)
PY
fi

exec "${PYTHON}" "${ROOT}/scripts/unimernet_server.py" \
  --port "${PORT}" \
  --model-dir "${MODEL_DIR}"
