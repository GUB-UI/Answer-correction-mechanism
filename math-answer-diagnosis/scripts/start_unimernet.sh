#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${ROOT}/.venv-unimernet"
MODEL_DIR="${UNIMERNET_MODEL_DIR:-${ROOT}/data/models/unimernet_small}"
PORT="${UNIMERNET_PORT:-8091}"

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "UniMERNet venv is missing. Run scripts/setup_production.sh first." >&2
  exit 1
fi

if [[ ! -f "${MODEL_DIR}/unimernet_small.pth" ]]; then
  echo "Downloading wanderkid/unimernet_small to ${MODEL_DIR} ..."
  mkdir -p "${MODEL_DIR}"
  "${VENV}/bin/python" - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="wanderkid/unimernet_small",
    local_dir="${MODEL_DIR}",
    local_dir_use_symlinks=False,
)
PY
fi

exec "${VENV}/bin/python" "${ROOT}/scripts/unimernet_server.py" \
  --port "${PORT}" \
  --model-dir "${MODEL_DIR}"
