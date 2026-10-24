from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

SERVICE_MODES = ("external", "managed")


@dataclass(frozen=True)
class LMStudioSettings:
    base_url: str
    api_key: str


@dataclass(frozen=True)
class ModelSettings:
    provider: str
    model_name: str
    revision: str = ""


@dataclass(frozen=True)
class FormulaOCRSettings:
    enabled: bool
    model_name: str
    base_url: str


@dataclass(frozen=True)
class OCRApiSettings:
    host: str = "127.0.0.1"
    port: int = 8080
    layout_device: str = "cpu"
    # 0 はリサイズしない。手書き写真向けの実験用で、品質を保証しない。
    min_edge: int = 0
    max_edge: int = 0
    prompt_suffix: str = ""


@dataclass(frozen=True)
class GLMOCRServiceSettings:
    model_id: str = "zai-org/GLM-OCR"
    command: str = ""


@dataclass(frozen=True)
class XiaomiOCRServiceSettings:
    model_id: str = "SeerRay-Lab/Xiaomi-OCR-0"
    command: str = ""


@dataclass(frozen=True)
class UniMERNetServiceSettings:
    python: str = ""
    model_dir: str = ""
    command: str = ""


@dataclass(frozen=True)
class LLMServiceSettings:
    port: int = 8000
    model_id: str = "unsloth/Qwen3.8-27B-unsloth-bnb-4bit"
    quantization: str = "bnb4"
    max_new_tokens: int = 2048
    command: str = ""


@dataclass(frozen=True)
class ServiceSettings:
    mode: str = "external"
    exclusive_gpu: bool = True
    run_dir: str = ""
    startup_timeout_sec: float = 1800.0
    glm_ocr: GLMOCRServiceSettings = field(default_factory=GLMOCRServiceSettings)
    xiaomi_ocr: XiaomiOCRServiceSettings = field(default_factory=XiaomiOCRServiceSettings)
    unimernet: UniMERNetServiceSettings = field(default_factory=UniMERNetServiceSettings)
    llm: LLMServiceSettings = field(default_factory=LLMServiceSettings)


@dataclass(frozen=True)
class ComparisonSettings:
    ratio_threshold: float = 0.2
    abs_threshold: float | None = None
    recheck_mode: str = "blind"


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
    data_dir_override: Path | None = None
    ocr_api: OCRApiSettings = field(default_factory=OCRApiSettings)
    services: ServiceSettings = field(default_factory=ServiceSettings)
    comparison: ComparisonSettings = field(default_factory=ComparisonSettings)
    app_password: str = ""
    tagging_model_name: str = ""

    @property
    def data_dir(self) -> Path:
        return self.data_dir_override or (self.project_root / "data")

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "images"

    @property
    def records_dir(self) -> Path:
        return self.data_dir / "records"

    @property
    def run_dir(self) -> Path:
        if self.services.run_dir:
            return Path(self.services.run_dir)
        return self.project_root / ".run"

    def to_stored_path(self, path: Path) -> str:
        """保存用の相対パス。データ置き場を Drive に移しても `data/...` 形式を保つ。"""
        return str(Path("data") / path.resolve().relative_to(self.data_dir.resolve()))

    def resolve_path(self, stored: str) -> Path:
        from src.sources import resolve_stored_path

        return resolve_stored_path(self.project_root, stored, data_dir=self.data_dir)


def _env_override(key: str, default: str) -> str:
    value = os.environ.get(key)
    if value is not None and value.strip():
        return value.strip()
    return default


def _env_first(keys: list[str], default: str) -> str:
    for key in keys:
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


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return float(value)


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    return value if isinstance(value, dict) else {}


def _resolve_ocr_section(raw: dict[str, Any]) -> dict[str, Any]:
    ocr = raw.get("ocr")
    if isinstance(ocr, dict) and ocr:
        return ocr
    models = raw.get("models", {})
    legacy = models.get("ocr") if isinstance(models, dict) else None
    return legacy if isinstance(legacy, dict) else {}


def _config_path(root: Path) -> Path:
    name = os.environ.get("APP_CONFIG_FILE", "").strip()
    if not name:
        return root / "config.yaml"
    path = Path(name)
    return path if path.is_absolute() else root / path


def _load_services(raw: dict[str, Any]) -> ServiceSettings:
    services = _section(raw, "services")
    glm = _section(services, "glm_ocr")
    xiaomi = _section(services, "xiaomi_ocr")
    unimernet = _section(services, "unimernet")
    llm = _section(services, "llm")
    mode = _env_override("SERVICE_MODE", str(services.get("mode", "external"))).lower()
    if mode not in SERVICE_MODES:
        raise ValueError(f"Unsupported SERVICE_MODE: {mode} (choose from {SERVICE_MODES})")
    return ServiceSettings(
        mode=mode,
        exclusive_gpu=_as_bool(
            _env_override("SERVICE_EXCLUSIVE_GPU", str(services.get("exclusive_gpu", True))),
            True,
        ),
        run_dir=_env_override("SERVICE_RUN_DIR", str(services.get("run_dir", "") or "")),
        startup_timeout_sec=float(
            _env_override(
                "SERVICE_STARTUP_TIMEOUT_SEC",
                str(services.get("startup_timeout_sec", 1800)),
            )
        ),
        glm_ocr=GLMOCRServiceSettings(
            model_id=_env_override(
                "GLMOCR_SERVER_MODEL_ID", str(glm.get("model_id", "zai-org/GLM-OCR"))
            ),
            command=_env_override("GLMOCR_SERVER_COMMAND", str(glm.get("command", "") or "")),
        ),
        xiaomi_ocr=XiaomiOCRServiceSettings(
            model_id=_env_override(
                "XIAOMI_OCR_MODEL_ID",
                str(xiaomi.get("model_id", "SeerRay-Lab/Xiaomi-OCR-0")),
            ),
            command=_env_override(
                "XIAOMI_OCR_SERVER_COMMAND", str(xiaomi.get("command", "") or "")
            ),
        ),
        unimernet=UniMERNetServiceSettings(
            python=_env_override("UNIMERNET_PYTHON", str(unimernet.get("python", "") or "")),
            model_dir=_env_override(
                "UNIMERNET_MODEL_DIR", str(unimernet.get("model_dir", "") or "")
            ),
            command=_env_override(
                "UNIMERNET_SERVER_COMMAND", str(unimernet.get("command", "") or "")
            ),
        ),
        llm=LLMServiceSettings(
            port=int(_env_override("LLM_SERVER_PORT", str(llm.get("port", 8000)))),
            model_id=_env_override(
                "LLM_MODEL_ID",
                str(llm.get("model_id", "unsloth/Qwen3.8-27B-unsloth-bnb-4bit")),
            ),
            quantization=_env_override(
                "LLM_QUANTIZATION", str(llm.get("quantization", "bnb4"))
            ).lower(),
            max_new_tokens=int(
                _env_override("LLM_MAX_NEW_TOKENS", str(llm.get("max_new_tokens", 2048)))
            ),
            command=_env_override("LLM_SERVER_COMMAND", str(llm.get("command", "") or "")),
        ),
    )


def load_config(project_root: Path | None = None) -> AppConfig:
    root = (project_root or _default_project_root()).resolve()
    load_dotenv(root / ".env", override=False)

    config_path = _config_path(root)
    raw: dict[str, Any] = {}
    if config_path.is_file():
        with config_path.open(encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            raw = loaded if isinstance(loaded, dict) else {}

    runtime = _section(raw, "runtime")
    llm_api = _section(raw, "omlx") or _section(raw, "lmstudio")
    models = _section(raw, "models")
    ocr = _resolve_ocr_section(raw)
    formula = ocr.get("formula", {}) if isinstance(ocr.get("formula"), dict) else {}
    ocr_api = ocr.get("api", {}) if isinstance(ocr.get("api"), dict) else {}
    diagnosis = _section(models, "diagnosis")
    safety = _section(raw, "safety")
    storage = _section(raw, "storage")
    comparison = _section(raw, "comparison")

    data_dir_value = _env_override("APP_DATA_DIR", str(storage.get("data_dir", "") or ""))
    data_dir_override: Path | None = None
    if data_dir_value:
        candidate = Path(data_dir_value).expanduser()
        data_dir_override = candidate if candidate.is_absolute() else root / candidate

    return AppConfig(
        runtime_device=_env_override(
            "RUNTIME_DEVICE", str(runtime.get("device", "apple_silicon_mac"))
        ),
        use_lmstudio=_as_bool(runtime.get("use_omlx"), True)
        if "use_omlx" in runtime
        else _as_bool(runtime.get("use_lmstudio"), True),
        lmstudio=LMStudioSettings(
            base_url=_env_first(
                ["LLM_BASE_URL", "OMLX_BASE_URL", "LMSTUDIO_BASE_URL"],
                str(llm_api.get("base_url", "http://localhost:8000/v1")),
            ),
            api_key=_env_first(
                ["LLM_API_KEY", "OMLX_API_KEY", "LMSTUDIO_API_KEY"],
                str(llm_api.get("api_key", "omlx")),
            ),
        ),
        ocr=ModelSettings(
            revision=_env_override("OCR_MODEL_REVISION", str(ocr.get("revision", "") or "")),
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
        data_dir_override=data_dir_override,
        ocr_api=OCRApiSettings(
            host=_env_override("OCR_API_HOST", str(ocr_api.get("host", "127.0.0.1"))),
            port=int(_env_override("OCR_API_PORT", str(ocr_api.get("port", 8080)))),
            layout_device=_env_override(
                "OCR_LAYOUT_DEVICE", str(ocr_api.get("layout_device", "cpu"))
            ),
            min_edge=int(_env_override("GLMOCR_MIN_EDGE", str(ocr_api.get("min_edge", 0) or 0))),
            max_edge=int(_env_override("GLMOCR_MAX_EDGE", str(ocr_api.get("max_edge", 0) or 0))),
            prompt_suffix=_env_override(
                "GLMOCR_PROMPT_SUFFIX", str(ocr_api.get("prompt_suffix", "") or "")
            ),
        ),
        services=_load_services(raw),
        comparison=ComparisonSettings(
            ratio_threshold=float(
                _env_override(
                    "COMPARE_RATIO_THRESHOLD", str(comparison.get("ratio_threshold", 0.2))
                )
            ),
            abs_threshold=_as_optional_float(
                _env_override(
                    "COMPARE_ABS_THRESHOLD", str(comparison.get("abs_threshold", "") or "")
                )
            ),
            recheck_mode=_env_override(
                "COMPARE_RECHECK_MODE", str(comparison.get("recheck_mode", "blind"))
            ),
        ),
        app_password=_env_override("APP_PASSWORD", ""),
        tagging_model_name=_env_override(
            "TAGGING_MODEL_NAME",
            _env_override("DIAGNOSIS_MODEL_NAME", str(diagnosis.get("model_name", "local-diagnosis-model"))),
        ),
    )


def _default_project_root() -> Path:
    return Path(__file__).resolve().parent.parent
