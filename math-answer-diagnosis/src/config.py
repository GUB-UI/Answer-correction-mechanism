from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class LMStudioSettings:
    base_url: str
    api_key: str


@dataclass(frozen=True)
class ModelSettings:
    provider: str
    model_name: str


@dataclass(frozen=True)
class FormulaOCRSettings:
    enabled: bool
    model_name: str
    base_url: str


@dataclass(frozen=True)
class AppConfig:
    runtime_device: str
    use_lmstudio: bool
    lmstudio: LMStudioSettings
    ocr: ModelSettings
    formula: FormulaOCRSettings
    diagnosis: ModelSettings
    require_anonymization: bool
    store_raw_images_locally: bool
    project_root: Path

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "images"

    @property
    def records_dir(self) -> Path:
        return self.data_dir / "records"


def _env_override(key: str, default: str) -> str:
    value = os.environ.get(key)
    if value is not None and value.strip():
        return value.strip()
    return default


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _resolve_ocr_section(raw: dict[str, Any]) -> dict[str, Any]:
    ocr = raw.get("ocr")
    if isinstance(ocr, dict) and ocr:
        return ocr
    models = raw.get("models", {})
    legacy = models.get("ocr") if isinstance(models, dict) else None
    return legacy if isinstance(legacy, dict) else {}


def load_config(project_root: Path | None = None) -> AppConfig:
    root = (project_root or _default_project_root()).resolve()
    load_dotenv(root / ".env", override=False)

    config_path = root / "config.yaml"
    raw: dict[str, Any] = {}
    if config_path.is_file():
        with config_path.open(encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            raw = loaded if isinstance(loaded, dict) else {}

    runtime = raw.get("runtime", {})
    llm_api = raw.get("omlx") if isinstance(raw.get("omlx"), dict) else {}
    if not llm_api:
        llm_api = raw.get("lmstudio", {}) if isinstance(raw.get("lmstudio"), dict) else {}
    models = raw.get("models", {})
    ocr = _resolve_ocr_section(raw)
    formula = ocr.get("formula", {}) if isinstance(ocr.get("formula"), dict) else {}
    diagnosis = models.get("diagnosis", {}) if isinstance(models, dict) else {}
    safety = raw.get("safety", {})

    return AppConfig(
        runtime_device=str(runtime.get("device", "apple_silicon_mac")),
        use_lmstudio=_as_bool(runtime.get("use_omlx"), True)
        if "use_omlx" in runtime
        else _as_bool(runtime.get("use_lmstudio"), True),
        lmstudio=LMStudioSettings(
            base_url=_env_override(
                "OMLX_BASE_URL",
                _env_override(
                    "LMSTUDIO_BASE_URL",
                    str(llm_api.get("base_url", "http://localhost:8000/v1")),
                ),
            ),
            api_key=_env_override(
                "OMLX_API_KEY",
                _env_override(
                    "LMSTUDIO_API_KEY",
                    str(llm_api.get("api_key", "omlx")),
                ),
            ),
        ),
        ocr=ModelSettings(
            provider=_env_override("OCR_PROVIDER", str(ocr.get("provider", "mock"))),
            model_name=_env_override(
                "OCR_MODEL_NAME",
                str(ocr.get("model_name", "glm-ocr")),
            ),
        ),
        formula=FormulaOCRSettings(
            enabled=_as_bool(
                _env_override(
                    "UNIMERNET_ENABLED",
                    str(formula.get("enabled", True)),
                ),
                True,
            ),
            model_name=_env_override(
                "UNIMERNET_MODEL_NAME",
                str(formula.get("model_name", "wanderkid/unimernet_small")),
            ),
            base_url=_env_override(
                "UNIMERNET_BASE_URL",
                str(formula.get("base_url", "http://127.0.0.1:8091")),
            ),
        ),
        diagnosis=ModelSettings(
            provider=_env_override(
                "DIAGNOSIS_PROVIDER",
                str(diagnosis.get("provider", "mock")),
            ),
            model_name=_env_override(
                "DIAGNOSIS_MODEL_NAME",
                str(diagnosis.get("model_name", "local-diagnosis-model")),
            ),
        ),
        require_anonymization=_as_bool(
            safety.get("require_anonymization"),
            True,
        ),
        store_raw_images_locally=_as_bool(
            safety.get("store_raw_images_locally"),
            True,
        ),
        project_root=root,
    )


def _default_project_root() -> Path:
    return Path(__file__).resolve().parent.parent
