"""Identity of an OCR request, including ordered image content hashes."""
from __future__ import annotations

import hashlib
import os
from dataclasses import asdict
from typing import Any

from src.config import AppConfig
from src.models import OCRProvenance


def image_hashes(config: AppConfig, image_paths: list[str]) -> list[str]:
    hashes = []
    for path in image_paths:
        with config.resolve_path(path).open("rb") as stream:
            hashes.append(hashlib.file_digest(stream, "sha256").hexdigest())
    return hashes


def canonical_engine(provider: str) -> str:
    return {"glm-ocr": "glmocr_sdk", "xiaomi-ocr": "xiaomi_ocr"}.get(provider, provider)


def ocr_provenance(
    config: AppConfig,
    image_paths: list[str],
    *,
    provider: str,
    adapter: Any,
    apply_formula: bool = False,
) -> OCRProvenance:
    from src.prompts import XIAOMI_OCR_PROMPT

    engine = canonical_engine(provider)
    settings: dict[str, Any] = {
        "version": 1,
        "adapter": f"{type(adapter).__module__}.{type(adapter).__qualname__}",
        "api": asdict(config.ocr_api),
        "formula_enabled": apply_formula,
    }
    if engine == "glmocr_sdk":
        sdk_config = config.project_root / "glmocr.config.yaml"
        settings["sdk_config_sha256"] = (
            hashlib.sha256(sdk_config.read_bytes()).hexdigest() if sdk_config.is_file() else None
        )
    if engine == "xiaomi_ocr":
        settings.update(prompt=XIAOMI_OCR_PROMPT, temperature=0, max_tokens=4096)
        settings["server_settings"] = {
            key: os.environ.get(key, "")
            for key in ("XIAOMI_OCR_MIN_EDGE", "XIAOMI_OCR_MAX_EDGE", "XIAOMI_OCR_PROMPT")
        }
    if config.services.mode == "managed" and engine in {"glmocr_sdk", "xiaomi_ocr"}:
        service = config.services.glm_ocr if engine == "glmocr_sdk" else config.services.xiaomi_ocr
        settings["server_model_id"] = service.model_id
        settings["server_command_sha256"] = hashlib.sha256(service.command.encode()).hexdigest()
    settings["service_mode"] = config.services.mode
    if apply_formula:
        settings["formula"] = asdict(config.formula)
        settings["formula_model_dir"] = config.services.unimernet.model_dir
        settings["formula_command_sha256"] = hashlib.sha256(
            config.services.unimernet.command.encode()
        ).hexdigest()
    return OCRProvenance(
        engine=engine,
        model_id=config.ocr.model_name if engine != "mock" else "mock",
        model_revision=config.ocr.revision,
        settings=settings,
        image_sha256=image_hashes(config, image_paths),
    )
