#!/usr/bin/env python3
"""GLM-OCR を transformers で動かす OpenAI 互換サーバー（Colab / CUDA 用）。

Mac では mlx-vlm（:8080）が担っていた役割を置き換える。GLM-OCR 公式 SDK
（selfhosted モード）は PP-DocLayoutV3 でレイアウト解析した領域ごとに
``POST /chat/completions`` を送ってくるので、それに応答する。

    python scripts/glmocr_server.py --port 8080 --model zai-org/GLM-OCR
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _openai_compat import (  # noqa: E402
    content_text,
    decode_image_url,
    image_urls,
    pick_device,
    pick_dtype,
    serve,
)


class GlmOcrBackend:
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
        print(f"[glmocr] loading {self.model_id} on {device} ({dtype})", flush=True)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            dtype=dtype,
            device_map=device,
        )
        self.model.eval()
        if device.startswith("cuda"):
            used = torch.cuda.memory_allocated() / 1024**3
            print(f"[glmocr] VRAM allocated: {used:.2f} GiB", flush=True)

    def _build_messages(self, request: dict[str, Any]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for message in request["messages"]:
            role = message.get("role", "user")
            content = message.get("content")
            parts: list[dict[str, Any]] = [
                {"type": "image", "image": decode_image_url(url)} for url in image_urls(content)
            ]
            text = content_text(content)
            if text:
                parts.append({"type": "text", "text": text})
            if parts:
                messages.append({"role": role, "content": parts})
        return messages

    def generate(self, request: dict[str, Any]) -> tuple[str, dict[str, int]]:
        import torch

        assert self.model is not None and self.processor is not None
        messages = self._build_messages(request)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        max_new_tokens = min(
            int(request.get("max_tokens") or self.max_new_tokens_cap),
            self.max_new_tokens_cap,
        )
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            # OCR は再現性を優先し常に greedy。SDK の temperature=0.01 相当。
            "do_sample": False,
        }
        repetition_penalty = request.get("repetition_penalty")
        if repetition_penalty:
            generate_kwargs["repetition_penalty"] = float(repetition_penalty)
        with torch.inference_mode():
            output = self.model.generate(**inputs, **generate_kwargs)
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = output[0][prompt_len:]
        text = self.processor.decode(new_tokens, skip_special_tokens=True).strip()
        usage = {
            "prompt_tokens": int(prompt_len),
            "completion_tokens": int(new_tokens.shape[0]),
            "finish_reason": "length" if new_tokens.shape[0] >= max_new_tokens else "stop",
        }
        return text, usage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--model", default="zai-org/GLM-OCR")
    parser.add_argument("--device", default="auto", help="auto / cuda / cpu")
    parser.add_argument("--dtype", default="auto", help="auto / bfloat16 / float16 / float32")
    parser.add_argument("--max-new-tokens-cap", type=int, default=4096)
    args = parser.parse_args()
    backend = GlmOcrBackend(
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
