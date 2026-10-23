#!/usr/bin/env python3
"""診断LLM（Qwen3.8-27B 4bit など）を transformers で動かす OpenAI 互換サーバー。

Colab / CUDA 用。Mac の oMLX と同じ ``/v1/chat/completions`` を提供するので、
アプリ側の LMStudioClient はそのまま使える。

    # A100: 事前量子化済み（Unsloth Dynamic 4bit）
    python scripts/llm_server.py --model unsloth/Qwen3.8-27B-unsloth-bnb-4bit
    # L4: 公式重みをロード時に NF4 量子化
    python scripts/llm_server.py --model Qwen/Qwen3.8-27B --quantization bnb4

GGUF（llama.cpp）や vLLM を使いたい場合は、このスクリプトの代わりに
``LLM_SERVER_COMMAND`` で任意の OpenAI 互換サーバーの起動コマンドを指定できる。
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
    pick_device,
    serve,
    split_assistant_prefill,
    strip_think,
)
from src.runtime.quant_dtype import (  # noqa: E402
    checkpoint_compute_dtype,
    resolve_load_dtype,
)


def build_prompt(tokenizer: Any, messages: list[dict[str, Any]], enable_thinking: bool) -> str:
    """チャットテンプレートでプロンプト文字列を作る。

    クライアントの空 ``<think></think>`` プレフィルは enable_thinking=False に置き換え、
    それ以外の assistant プレフィルは continue_final_message で続きを生成させる。
    """
    history, prefill, is_empty_think = split_assistant_prefill(messages)
    if is_empty_think:
        enable_thinking = False
        prefill = None
    plain = [
        {"role": m.get("role", "user"), "content": content_text(m.get("content"))}
        for m in history
    ]
    template_kwargs: dict[str, Any] = {"tokenize": False, "enable_thinking": enable_thinking}
    if prefill:
        plain.append({"role": "assistant", "content": prefill})
        return tokenizer.apply_chat_template(
            plain, continue_final_message=True, **template_kwargs
        )
    return tokenizer.apply_chat_template(plain, add_generation_prompt=True, **template_kwargs)


class TransformersChatBackend:
    def __init__(
        self,
        model_id: str,
        *,
        served_model_name: str | None = None,
        quantization: str = "bnb4",
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens_cap: int = 2048,
    ) -> None:
        self.model_id = model_id
        self.model_name = served_model_name or model_id
        self.quantization = quantization
        self.requested_device = device
        self.requested_dtype = dtype
        self.max_new_tokens_cap = max_new_tokens_cap
        self.model = None
        self.tokenizer = None

    def backend_fingerprint(self) -> dict[str, Any] | None:
        if self.model is None or self.tokenizer is None:
            return None
        quantization = getattr(self.model.config, "quantization_config", None)
        if quantization is not None and hasattr(quantization, "to_dict"):
            quantization = quantization.to_dict()
        return {
            "model_id": self.model_id,
            "settings": {
                "model_revision": getattr(self.model.config, "_commit_hash", None),
                "generation_config": self.model.generation_config.to_dict(),
                "max_new_tokens_cap": self.max_new_tokens_cap,
                "quantization_config": quantization,
                "dtype": str(self.model.dtype),
                "device": str(self.model.device),
                "chat_template": self.tokenizer.chat_template,
            },
        }

    def _model_config(self) -> Any:
        from transformers import AutoConfig

        return AutoConfig.from_pretrained(self.model_id)

    def _quantization_config(self, device: str, dtype: Any, config: Any) -> Any:
        from transformers import BitsAndBytesConfig

        if self.quantization in {"none", "", "bf16", "fp16"}:
            return None
        if self.quantization != "bnb4":
            raise ValueError(f"Unsupported quantization: {self.quantization}")
        if getattr(config, "quantization_config", None):
            print("[llm] checkpoint is pre-quantized; using its quantization_config", flush=True)
            return None
        if not device.startswith("cuda"):
            print("[llm] bitsandbytes 4bit requires CUDA; loading unquantized", flush=True)
            return None
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    def _load_dtype(self, device: str, config: Any) -> Any:
        import torch

        quant = getattr(config, "quantization_config", None)
        if quant is not None and not isinstance(quant, dict) and hasattr(quant, "to_dict"):
            quant = quant.to_dict()
        checkpoint = checkpoint_compute_dtype(quant if isinstance(quant, dict) else None)
        cuda_bf16 = device.startswith("cuda") and torch.cuda.is_bf16_supported()
        name = resolve_load_dtype(
            self.requested_dtype,
            checkpoint,
            cuda_bf16=cuda_bf16,
        )
        if checkpoint and self.requested_dtype == "auto":
            print(f"[llm] pre-quantized checkpoint compute dtype={name}", flush=True)
        return getattr(torch, name)

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = pick_device(self.requested_device)
        config = self._model_config()
        dtype = self._load_dtype(device, config)
        quantization_config = self._quantization_config(device, dtype, config)
        print(
            f"[llm] loading {self.model_id} on {device} dtype={dtype} "
            f"quantization={'nf4(on-load)' if quantization_config else self.quantization}",
            flush=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        kwargs: dict[str, Any] = {
            "dtype": dtype,
            "device_map": "auto" if device.startswith("cuda") else device,
        }
        if quantization_config is not None:
            kwargs["quantization_config"] = quantization_config
        # Qwen3.5 / 3.8 は画像入力付きの構成だが、採点はテキストのみなので
        # 言語モデル部分だけを読み込み VRAM を節約する。
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        self.model.eval()
        if device.startswith("cuda"):
            used = torch.cuda.memory_allocated() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"[llm] VRAM allocated: {used:.2f} / {total:.1f} GiB", flush=True)

    def generate(self, request: dict[str, Any]) -> tuple[str, dict[str, int]]:
        import torch

        assert self.model is not None and self.tokenizer is not None
        template_kwargs = request.get("chat_template_kwargs") or {}
        enable_thinking = bool(template_kwargs.get("enable_thinking", False))
        prompt = build_prompt(self.tokenizer, request["messages"], enable_thinking)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        max_new_tokens = min(
            int(request.get("max_tokens") or self.max_new_tokens_cap),
            self.max_new_tokens_cap,
        )
        temperature = float(request.get("temperature") or 0.0)
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        if temperature > 0.01:
            generate_kwargs.update(
                do_sample=True,
                temperature=temperature,
                top_p=float(request.get("top_p") or 0.95),
            )
        else:
            generate_kwargs["do_sample"] = False
        with torch.inference_mode():
            output = self.model.generate(**inputs, **generate_kwargs)
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = output[0][prompt_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        usage = {
            "prompt_tokens": int(prompt_len),
            "completion_tokens": int(new_tokens.shape[0]),
            "finish_reason": "length" if new_tokens.shape[0] >= max_new_tokens else "stop",
        }
        return strip_think(text), usage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="unsloth/Qwen3.8-27B-unsloth-bnb-4bit")
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--quantization", default="bnb4", choices=["bnb4", "none"])
    parser.add_argument("--device", default="auto", help="auto / cuda / cpu")
    parser.add_argument("--dtype", default="auto", help="auto / bfloat16 / float16 / float32")
    parser.add_argument("--max-new-tokens-cap", type=int, default=2048)
    args = parser.parse_args()
    backend = TransformersChatBackend(
        args.model,
        served_model_name=args.served_model_name,
        quantization=args.quantization,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens_cap=args.max_new_tokens_cap,
    )
    serve(backend, args.host, args.port)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    raise SystemExit(main())
