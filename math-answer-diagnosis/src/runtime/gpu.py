"""GPU 検出と、VRAM に応じた実行プロファイル（A100 → L4 → T4 の順に縮退）。"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class GPUInfo:
    name: str
    total_mib: int
    used_mib: int = 0

    @property
    def total_gib(self) -> float:
        return self.total_mib / 1024


@dataclass(frozen=True)
class RuntimeProfile:
    tier: str
    llm_model_id: str
    llm_quantization: str
    diagnosis_model_name: str
    llm_max_new_tokens: int
    layout_device: str
    comparable_to_27b: bool
    notes: list[str] = field(default_factory=list)

    def env(self) -> dict[str, str]:
        return {
            "RUNTIME_DEVICE": f"colab_{self.tier}",
            "LLM_MODEL_ID": self.llm_model_id,
            "LLM_QUANTIZATION": self.llm_quantization,
            "DIAGNOSIS_MODEL_NAME": self.diagnosis_model_name,
            "LLM_MAX_NEW_TOKENS": str(self.llm_max_new_tokens),
            "OCR_LAYOUT_DEVICE": self.layout_device,
        }

    def as_dict(self) -> dict:
        return asdict(self)


def parse_nvidia_smi(output: str) -> list[GPUInfo]:
    gpus: list[GPUInfo] = []
    for line in output.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            total = int(float(parts[1]))
            used = int(float(parts[2])) if len(parts) > 2 else 0
        except ValueError:
            continue
        gpus.append(GPUInfo(name=parts[0], total_mib=total, used_mib=used))
    return gpus


def detect_gpus() -> list[GPUInfo]:
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return parse_nvidia_smi(output)


# 27B の 4bit 版。A100 は Unsloth Dynamic 4bit（一部層を16bitで保持、約22GB）。
# L4 は公式重みをロード時に NF4 量子化（約17GB）。T4(16GB) には 27B 4bit が載らない。
MODEL_27B_DYNAMIC_4BIT = "unsloth/Qwen3.8-27B-unsloth-bnb-4bit"
MODEL_27B_OFFICIAL = "Qwen/Qwen3.8-27B"
MODEL_T4_FALLBACK = "Qwen/Qwen3.5-9B"


def select_profile(gpu: GPUInfo | None) -> RuntimeProfile:
    if gpu is None:
        return RuntimeProfile(
            tier="cpu",
            llm_model_id=MODEL_T4_FALLBACK,
            llm_quantization="none",
            diagnosis_model_name="Qwen3.5-9B-cpu",
            llm_max_new_tokens=1024,
            layout_device="cpu",
            comparable_to_27b=False,
            notes=[
                "GPU が見つかりません。Colab の「ランタイムのタイプを変更」で GPU を選んでください。",
                "CPU では 27B は実用的に動きません。mock で画面確認のみ行ってください。",
            ],
        )
    if gpu.total_mib >= 38_000:
        return RuntimeProfile(
            tier="a100",
            llm_model_id=MODEL_27B_DYNAMIC_4BIT,
            llm_quantization="bnb4",
            diagnosis_model_name="Qwen3.8-27B-bnb4",
            llm_max_new_tokens=2048,
            layout_device="cuda",
            comparable_to_27b=True,
            notes=[f"{gpu.name} ({gpu.total_gib:.0f}GiB): 推奨構成。27B Dynamic 4bit を使います。"],
        )
    if gpu.total_mib >= 20_000:
        return RuntimeProfile(
            tier="l4",
            llm_model_id=MODEL_27B_OFFICIAL,
            llm_quantization="bnb4",
            diagnosis_model_name="Qwen3.8-27B-bnb4",
            llm_max_new_tokens=1536,
            layout_device="cuda",
            comparable_to_27b=True,
            notes=[
                f"{gpu.name} ({gpu.total_gib:.0f}GiB): 公式 27B をロード時に NF4 量子化します。",
                "初回は約56GBのダウンロードが必要です（Drive キャッシュ推奨）。",
                "A100 の Dynamic 4bit とは量子化方式が違うため、厳密な比較では model_name で区別してください。",
            ],
        )
    return RuntimeProfile(
        tier="t4",
        llm_model_id=MODEL_T4_FALLBACK,
        llm_quantization="bnb4",
        diagnosis_model_name="Qwen3.5-9B-bnb4",
        llm_max_new_tokens=1536,
        layout_device="cpu",
        comparable_to_27b=False,
        notes=[
            f"{gpu.name} ({gpu.total_gib:.0f}GiB): 27B の 4bit 版（約17GB以上）は載りません。",
            "代替として Qwen3.5-9B を NF4 で使います。27B の結果とは混ぜずに扱ってください。",
            "27B を使いたい場合は A100 か L4 のランタイムを選び直してください。",
        ],
    )


def detect_profile() -> tuple[GPUInfo | None, RuntimeProfile]:
    gpus = detect_gpus()
    gpu = gpus[0] if gpus else None
    return gpu, select_profile(gpu)
