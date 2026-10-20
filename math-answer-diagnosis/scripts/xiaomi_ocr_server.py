#!/usr/bin/env python3
"""Xiaomi-OCR-0 を transformers で動かす OpenAI 互換サーバー（Colab / CUDA 用）。

手書き答案はページ全体を1回で読む。レイアウトで切らない。
GLM-OCR に戻すときはこのサーバーを起動せず、OCR_PROVIDER=glmocr_sdk にする。

    python scripts/xiaomi_ocr_server.py --port 8081 --model SeerRay-Lab/Xiaomi-OCR-0

モデルは Qwen3.5-0.8B 系。transformers は Qwen3_5ForConditionalGeneration を
含む版（5.17 以降）が要る。Colab の CUDA 版 torch は入れ直さない。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _openai_compat import (  # noqa: E402
    content_text,
    decode_image_url,
    image_urls,
    pick_device,
    pick_dtype,
    serve,
    strip_think,
)
from src.prompts import XIAOMI_OCR_PROMPT  # noqa: E402
from src.runtime.ocr_image import resize_for_ocr  # noqa: E402


class XiaomiOcrBackend:
    def __init__(
        self,
        model_id: str,
        *,
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens_cap: int = 4096,
    ) -> None:
        self.model_id = model_id
        self.model_name = model_id
        self.requested_device = device
        self.requested_dtype = dtype
        self.max_new_tokens_cap = max_new_tokens_cap
        self.model = None
        self.processor = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        device = pick_device(self.requested_device)
        dtype = pick_dtype(device, self.requested_dtype)
        print(f"[xiaomi-ocr] loading {self.model_id} on {device} ({dtype})", flush=True)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            dtype=dtype,
            device_map=device,
        )
        self.model.eval()
        if device.startswith("cuda"):
            used = torch.cuda.memory_allocated() / 1024**3
            print(f"[xiaomi-ocr] VRAM allocated: {used:.2f} GiB", flush=True)

    def _prepare_image(self, image: Any) -> Any:
        # 拡大は既定オフ。GLM の掃引では拡大だけで繰り返しが起きた。
        try:
            min_edge = int(os.environ.get("XIAOMI_OCR_MIN_EDGE") or 0)
            max_edge = int(os.environ.get("XIAOMI_OCR_MAX_EDGE") or 0)
        except ValueError:
            return image
        if min_edge <= 0 and max_edge <= 0:
            return image
        return resize_for_ocr(image, min_edge=min_edge, max_edge=max_edge)

    def _build_messages(self, request: dict[str, Any]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for message in request["messages"]:
            role = message.get("role", "user")
            if role == "assistant":
                continue
            content = message.get("content")
            parts: list[dict[str, Any]] = [
                {"type": "image", "image": self._prepare_image(decode_image_url(url))}
                for url in image_urls(content)
            ]
            text = content_text(content).strip()
            if not text and parts:
                text = os.environ.get("XIAOMI_OCR_PROMPT", "").strip() or XIAOMI_OCR_PROMPT
            if text:
                parts.append({"type": "text", "text": text})
            if parts:
                messages.append({"role": role, "content": parts})
        return messages

    def generate(self, request: dict[str, Any]) -> tuple[str, dict[str, int]]:
        import torch

        assert self.model is not None and self.processor is not None
        messages = self._build_messages(request)
        template_kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        try:
            inputs = self.processor.apply_chat_template(
                messages,
                enable_thinking=False,
                **template_kwargs,
            )
        except TypeError:
            inputs = self.processor.apply_chat_template(messages, **template_kwargs)
        inputs = inputs.to(self.model.device)
        max_new_tokens = min(
            int(request.get("max_tokens") or self.max_new_tokens_cap),
            self.max_new_tokens_cap,
        )
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = output[0][prompt_len:]
        text = strip_think(self.processor.decode(new_tokens, skip_special_tokens=True))
        usage = {
            "prompt_tokens": int(prompt_len),
            "completion_tokens": int(new_tokens.shape[0]),
            "finish_reason": "length" if new_tokens.shape[0] >= max_new_tokens else "stop",
        }
        return text, usage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--model", default="SeerRay-Lab/Xiaomi-OCR-0")
    parser.add_argument("--device", default="auto", help="auto / cuda / cpu")
    parser.add_argument("--dtype", default="auto", help="auto / bfloat16 / float16 / float32")
    parser.add_argument("--max-new-tokens-cap", type=int, default=4096)
    args = parser.parse_args()
    backend = XiaomiOcrBackend(
        args.model,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens_cap=args.max_new_tokens_cap,
    )
    serve(backend, args.host, args.port)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
