#!/usr/bin/env bash
set -euo pipefail

F2_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OMLX_BIN="${F2_ROOT}/omlx-runtime/bin/omlx"
MODEL_DIR="${F2_ROOT}/omlx-models"
BASE_PATH="${F2_ROOT}/omlx-state"
PORT="${OMLX_PORT:-8000}"

if [[ ! -x "${OMLX_BIN}" ]]; then
  echo "oMLX is not installed at ${OMLX_BIN}" >&2
  exit 1
fi

if lsof -iTCP:"${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "oMLX already listening on :${PORT}"
  exit 0
fi

mkdir -p "${BASE_PATH}"
echo "Starting oMLX on :${PORT} with Qwen3.8-27B-oQ4e-mtp + Lightning MTP..."
exec "${OMLX_BIN}" serve \
  --model-dir "${MODEL_DIR}" \
  --base-path "${BASE_PATH}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --max-concurrent-requests 1 \
  --memory-guard balanced
