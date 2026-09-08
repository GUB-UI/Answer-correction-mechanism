#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

APP_VENV="${ROOT}/.venv"
MLX_VENV="${ROOT}/.venv-mlx"
OMLX_URL="${OMLX_BASE_URL:-http://localhost:8000/v1}"
OMLX_PORT="${OMLX_PORT:-8000}"
MLX_PORT="${MLX_PORT:-8080}"
UNIMERNET_PORT="${UNIMERNET_PORT:-8091}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"

port_open() {
  python3 - "$1" <<'PY'
import socket, sys
port = int(sys.argv[1])
s = socket.socket()
s.settimeout(1)
try:
    s.connect(("127.0.0.1", port))
except OSError:
    sys.exit(1)
finally:
    s.close()
PY
}

if [[ ! -x "${APP_VENV}/bin/streamlit" ]]; then
  echo "app venv is missing. Run scripts/setup_production.sh first." >&2
  exit 1
fi
if [[ ! -x "${MLX_VENV}/bin/python" ]]; then
  echo "mlx-vlm venv is missing. Run scripts/setup_production.sh first." >&2
  exit 1
fi

if [[ ! -x "${ROOT}/.venv-unimernet/bin/python" ]]; then
  echo "UniMERNet venv is missing. Run scripts/setup_production.sh first." >&2
  exit 1
fi

if ! port_open "${OMLX_PORT}"; then
  echo "Starting oMLX diagnosis server..."
  nohup bash "${ROOT}/scripts/start_omlx.sh" > "${ROOT}/.omlx.log" 2>&1 &
  echo $! > "${ROOT}/.omlx.pid"
fi

echo "Waiting for oMLX on ${OMLX_URL}..."
ready=0
for _ in $(seq 1 90); do
  if curl -sf "${OMLX_URL}/models" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 2
done
if [[ "${ready}" -ne 1 ]]; then
  echo "oMLX did not become ready. See ${ROOT}/.omlx.log" >&2
  exit 1
fi

if ! port_open "${MLX_PORT}"; then
  echo "Starting mlx-vlm server on port ${MLX_PORT}..."
  nohup "${MLX_VENV}/bin/mlx_vlm.server" --trust-remote-code --port "${MLX_PORT}" \
    --model mlx-community/GLM-OCR-bf16 --log-level INFO \
    > "${ROOT}/.mlx-vlm.log" 2>&1 &
  echo $! > "${ROOT}/.mlx-vlm.pid"
fi

echo "Waiting for mlx-vlm on :${MLX_PORT}..."
ready=0
for _ in $(seq 1 90); do
  if curl -sf --max-time 3 -X POST "http://127.0.0.1:${MLX_PORT}/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"mlx-community/GLM-OCR-bf16","messages":[{"role":"user","content":[{"type":"text","text":"ping"}]}],"max_tokens":1}' \
    >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 2
done
if [[ "${ready}" -ne 1 ]]; then
  echo "mlx-vlm did not become ready. See ${ROOT}/.mlx-vlm.log" >&2
  exit 1
fi

if ! port_open "${UNIMERNET_PORT}"; then
  echo "Starting UniMERNet Small on port ${UNIMERNET_PORT}..."
  nohup bash "${ROOT}/scripts/start_unimernet.sh" > "${ROOT}/.unimernet.log" 2>&1 &
  echo $! > "${ROOT}/.unimernet.pid"
fi

echo "Waiting for UniMERNet on :${UNIMERNET_PORT}..."
ready=0
for _ in $(seq 1 90); do
  if curl -sf "http://127.0.0.1:${UNIMERNET_PORT}/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 2
done
if [[ "${ready}" -ne 1 ]]; then
  echo "UniMERNet did not become ready. See ${ROOT}/.unimernet.log" >&2
  exit 1
fi

echo "Starting Streamlit on :${STREAMLIT_PORT}..."
exec "${APP_VENV}/bin/streamlit" run app.py --server.port "${STREAMLIT_PORT}"
