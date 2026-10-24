"""事前量子化チェックポイントの計算 dtype。

Unsloth の bnb-4bit は計算が float16 で、lm_head は量子化されない。
A100 だからといって重み全体を bfloat16 にすると、
``mat1 and mat2 ... Half != BFloat16`` で生成が 500 になる。
"""

from __future__ import annotations

from typing import Any

_ALIASES = {
    "fp16": "float16",
    "float16": "float16",
    "half": "float16",
    "torch.float16": "float16",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "torch.bfloat16": "bfloat16",
    "fp32": "float32",
    "float32": "float32",
    "torch.float32": "float32",
}


def normalize_dtype_name(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text.startswith("torch."):
        text = text
    return _ALIASES.get(text)


def checkpoint_compute_dtype(quantization_config: dict[str, Any] | None) -> str | None:
    """4bit 済みチェックポイントなら、その compute dtype。無ければ float16。"""
    if not isinstance(quantization_config, dict) or not quantization_config:
        return None
    method = str(quantization_config.get("quant_method") or "")
    prequantized = bool(quantization_config.get("load_in_4bit")) or method in {
        "bitsandbytes",
        "bnb",
    }
    if not prequantized:
        return None
    named = normalize_dtype_name(
        quantization_config.get("bnb_4bit_compute_dtype")
        or quantization_config.get("compute_dtype")
    )
    return named or "float16"


def resolve_load_dtype(
    requested: str,
    checkpoint_dtype: str | None,
    *,
    cuda_bf16: bool,
) -> str:
    """--dtype が auto で事前量子化ならチェックポイント側を使う。"""
    if requested and requested != "auto":
        normalized = normalize_dtype_name(requested)
        if normalized is None:
            raise ValueError(f"Unsupported dtype: {requested}")
        return normalized
    if checkpoint_dtype:
        return checkpoint_dtype
    if cuda_bf16:
        return "bfloat16"
    return "float16"
