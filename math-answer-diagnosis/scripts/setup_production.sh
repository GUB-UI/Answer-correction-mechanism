#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PATH="${HOME}/.local/bin:${PATH}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it from https://docs.astral.sh/uv/" >&2
  exit 1
fi

echo "Installing Python 3.12..."
uv python install 3.12

echo "Creating app venv (Streamlit + GLM-OCR SDK)..."
uv venv --python 3.12 "${ROOT}/.venv"
uv pip install --python "${ROOT}/.venv/bin/python" -r requirements.txt
uv pip install --python "${ROOT}/.venv/bin/python" "glmocr[selfhosted]"

echo "Creating UniMERNet Small venv..."
uv venv --python 3.12 "${ROOT}/.venv-unimernet"
uv pip install --python "${ROOT}/.venv-unimernet/bin/python" unimernet huggingface_hub pillow "pyarrow<17"
mkdir -p "${ROOT}/data/models/unimernet_small"
if [[ ! -f "${ROOT}/data/models/unimernet_small/unimernet_small.pth" ]]; then
  echo "Downloading wanderkid/unimernet_small ..."
  "${ROOT}/.venv-unimernet/bin/python" - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="wanderkid/unimernet_small",
    local_dir="${ROOT}/data/models/unimernet_small",
)
PY
fi
echo "Creating mlx-vlm venv..."
uv venv --python 3.12 "${ROOT}/.venv-mlx"
uv pip install --python "${ROOT}/.venv-mlx/bin/python" "git+https://github.com/Blaizzy/mlx-vlm.git" jinja2

if [[ ! -f "${ROOT}/.env" ]]; then
  cp "${ROOT}/.env.example" "${ROOT}/.env"
fi

echo
echo "Production environments are ready."
echo "Start the stack with: bash scripts/start_production.sh"
