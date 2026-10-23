"""CPU subprocess for SeerRay-Lab/Xiaomi-OCR-0.

Equivalent to the checked CPU script: pinned revision, thinking off, greedy
decoding, 4096 new tokens, and no resize. bfloat16 is refused because it
breaks the CPU transcription. Torch is imported only after those checks, and
only inside ``main``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

MODEL_ID = "SeerRay-Lab/Xiaomi-OCR-0"
REVISION = "ff990be318c4c68f7e621dd690231200f47e48c4"
MAX_NEW_TOKENS = 4096
MIN_FREE_BYTES = int(4.5 * 1024**3)
PROMPT = """この画像は高校数学の手書き答案です。ページ全体を、読み順に Markdown で書き起こしてください。
数式は LaTeX にしてください。行の中の式は $...$、独立した式は $$...$$ です。
表は Markdown の表にしてください。
日本語はそのまま書き、中国語に翻訳しないでください。
ヘッダーとフッターは無視してください。
採点、解説、誤りの指摘はしないでください。
読めない箇所は [不明] と書いてください。"""


class CPUOCRRefused(RuntimeError):
    pass


def available_bytes() -> int:
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return 0


def assert_runnable(dtype: str, free_bytes: int) -> None:
    if dtype != "float32":
        raise CPUOCRRefused(
            f"refusing dtype {dtype} on CPU; bfloat16 breaks Xiaomi-OCR-0 output. Use float32."
        )
    if free_bytes < MIN_FREE_BYTES:
        raise CPUOCRRefused(
            f"refusing to load {MODEL_ID}: need about 4.5 GiB free, saw {free_bytes} bytes"
        )


def cpu_command(image: Path, out_dir: Path, *, dtype: str = "float32") -> list[str]:
    return [
        sys.executable,
        "-m",
        "harness_v2.recognition.run_ocr_cpu",
        str(image),
        "--out",
        str(out_dir),
        "--dtype",
        dtype,
        "--max-new-tokens",
        str(MAX_NEW_TOKENS),
        "--max-edge",
        "0",
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Xiaomi-OCR-0 on CPU in float32.")
    parser.add_argument("images", nargs="+")
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--max-edge", type=int, default=0)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        assert_runnable(args.dtype, available_bytes())
    except CPUOCRRefused as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return _run_model(args)


def _run_model(args: argparse.Namespace) -> int:
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    torch.set_num_threads(args.threads)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    ).eval()
    load_s = time.perf_counter() - started
    for image_path in args.images:
        image = Image.open(image_path).convert("RGB")
        if args.max_edge and max(image.size) > args.max_edge:
            scale = args.max_edge / max(image.size)
            image = image.resize(
                (round(image.width * scale), round(image.height * scale)),
                Image.Resampling.LANCZOS,
            )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        generated_at = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        elapsed = time.perf_counter() - generated_at
        prompt_len = int(inputs["input_ids"].shape[1])
        new_tokens = int(output.shape[1] - prompt_len)
        text = processor.decode(output[0][prompt_len:], skip_special_tokens=True)
        name = Path(image_path).stem
        (out_dir / f"{name}.md").write_text(text, encoding="utf-8")
        stats = {
            "image": name,
            "revision": REVISION,
            "dtype": "float32",
            "max_new_tokens": args.max_new_tokens,
            "max_edge": args.max_edge,
            "enable_thinking": False,
            "do_sample": False,
            "new_tokens": new_tokens,
            "wall_s": round(elapsed, 3),
            "load_s": round(load_s, 3),
            "model": MODEL_ID,
        }
        with (out_dir / "stats.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(stats) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
