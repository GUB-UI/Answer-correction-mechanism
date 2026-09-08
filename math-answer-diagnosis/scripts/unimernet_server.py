#!/usr/bin/env python3
"""UniMERNet Small inference server. Isolated from the GLM-OCR app venv."""

from __future__ import annotations

import argparse
import io
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from PIL import Image


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def write_config(model_dir: Path) -> Path:
    cfg_path = model_dir / "inference.yaml"
    pretrained = model_dir / "unimernet_small.pth"
    cfg_path.write_text(
        f"""
model:
  arch: unimernet
  model_type: unimernet
  model_name: unimernet
  load_pretrained: True
  pretrained: "{pretrained}"
  tokenizer_name: nougat
  tokenizer_config:
    path: "{model_dir}"
  model_config:
    model_name: "{model_dir}"
    max_seq_len: 1536

datasets:
  formula_rec_eval:
    vis_processor:
      eval:
        name: "formula_image_eval"
        image_size:
          - 192
          - 672

run:
  task: unimernet_train
  device: cpu
  evaluate: False
  generate_cfg:
    temperature: 0.0
    do_sample: False
    top_p: 0.95
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return cfg_path


def load_processor(model_dir: Path):
    import argparse as ap

    from unimernet.common.config import Config
    import unimernet.tasks as tasks
    from unimernet.processors import load_processor

    cfg_path = write_config(model_dir)
    args = ap.Namespace(cfg_path=str(cfg_path), options=None)
    cfg = Config(args)
    task = tasks.setup_task(cfg)
    device = _device()
    model = task.build_model(cfg).to(device)
    model.eval()
    vis_processor = load_processor(
        "formula_image_eval",
        cfg.config.datasets.formula_rec_eval.vis_processor.eval,
    )
    return model, vis_processor, device


class FormulaProcessor:
    def __init__(self, model_dir: Path) -> None:
        self.model, self.vis_processor, self.device = load_processor(model_dir)

    def predict(self, image: Image.Image) -> str:
        tensor = self.vis_processor(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model.generate({"image": tensor}, temperature=0.0, do_sample=False)
        pred = output["pred_str"][0]
        return str(pred).strip()


PROCESSOR: FormulaProcessor | None = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:  # noqa: A003
        print("[%s] %s" % (self.log_date_time_string(), format % args))

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] == "/health":
            self._send_json({"ok": PROCESSOR is not None, "model": "unimernet_small"})
            return
        self._send_json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        if PROCESSOR is None:
            self._send_json({"error": "model not loaded"}, 503)
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b""
        path = self.path.split("?", 1)[0]
        try:
            if path == "/predict":
                image = Image.open(io.BytesIO(raw)).convert("RGB")
            elif path == "/predict_path":
                payload = json.loads(raw.decode("utf-8"))
                image = Image.open(payload["image_path"]).convert("RGB")
            else:
                self._send_json({"error": "not found"}, 404)
                return
            latex = PROCESSOR.predict(image)
            self._send_json({"latex": latex})
        except Exception as exc:
            self._send_json({"error": str(exc)}, 500)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()
    model_dir = Path(args.model_dir).resolve()
    if not (model_dir / "unimernet_small.pth").is_file():
        raise SystemExit(f"UniMERNet Small weights not found: {model_dir}")

    global PROCESSOR
    print(f"Loading UniMERNet Small from {model_dir} ...", flush=True)
    PROCESSOR = FormulaProcessor(model_dir)
    print(f"UniMERNet Small ready on http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
