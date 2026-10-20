#!/usr/bin/env bash
# Streamlit を起動し、cloudflared の一時 URL で Colab の外から開く。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PORT="${STREAMLIT_PORT:-8501}"
mkdir -p "${ROOT}/.run"

port_open() {
  python3 - "$1" <<'PY'
import socket, sys
s = socket.socket()
s.settimeout(1)
try:
    s.connect(("127.0.0.1", int(sys.argv[1])))
except OSError:
    sys.exit(1)
finally:
    s.close()
PY
}

if ! port_open "${PORT}"; then
  echo "Starting Streamlit on :${PORT}"
  nohup python3 -m streamlit run app.py \
    --server.port "${PORT}" \
    --server.address 127.0.0.1 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    > "${ROOT}/.run/streamlit.log" 2>&1 &
  echo $! > "${ROOT}/.run/streamlit.pid"
  for _ in $(seq 1 40); do
    if curl -sf "http://127.0.0.1:${PORT}/_stcore/health" >/dev/null; then
      break
    fi
    sleep 1
  done
fi

BIN="${CLOUDFLARED:-}"
if [[ -z "${BIN}" ]]; then
  if command -v cloudflared >/dev/null 2>&1; then
    BIN="cloudflared"
  else
    BIN="${ROOT}/.run/cloudflared"
    if [[ ! -x "${BIN}" ]]; then
      echo "Downloading cloudflared..."
      curl -fL --retry 3 -o "${BIN}" \
        https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
      chmod +x "${BIN}"
    fi
  fi
fi

echo "公開URLは次の行の https://....trycloudflare.com です。"
echo "トンネルは公開されるので、.env の APP_PASSWORD を設定してください。"
exec "${BIN}" tunnel --url "http://127.0.0.1:${PORT}" --no-autoupdate
