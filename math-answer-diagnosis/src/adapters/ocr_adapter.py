from __future__ import annotations

from pathlib import Path

from src.adapters.base import OCRAdapter, OCRRunContext
from src.adapters.mock_adapters import MockOCRAdapter
from src.config import AppConfig, load_config
from src.models import OCRResult, utc_now_iso
from src.prompts import OCR_PROMPT
from src.storage import Storage


class GLMOCRSdkAdapter(OCRAdapter):
    """GLM-OCR 公式 SDK を Python から呼び出す Adapter。"""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()

    def run(self, context: OCRRunContext) -> OCRResult:
        raw_text, raw_output = self._run_via_sdk(context)
        return OCRResult(
            ocr_id="",
            answer_id=context.answer_id,
            ocr_engine=context.ocr_engine,
            raw_text=raw_text,
            used_text=raw_text,
            uncertain_parts=[],
            ocr_suspect=False,
            human_corrected=False,
            correction_note=None,
            raw_output=raw_output,
            created_at=utc_now_iso(),
        )

    def _run_via_sdk(self, context: OCRRunContext) -> tuple[str, dict]:
        try:
            raw_text = self._invoke_sdk(context.image_path)
            return raw_text, {
                "provider": "glmocr_sdk",
                "model_name": self.config.ocr.model_name,
                "image_path": str(context.image_path),
            }
        except Exception as exc:
            raise RuntimeError(f"GLM-OCR SDK OCR failed: {exc}") from exc

    def _invoke_sdk(self, image_path: Path) -> str:
        try:
            # TODO: 実際の GLM-OCR SDK の呼び出しに合わせて実装する
            # from glmocr import parse
            # result = parse(str(image_path), prompt=OCR_PROMPT)
            # return str(result).strip()
            _ = OCR_PROMPT  # SDK がプロンプト対応時に使用
            _ = image_path
            raise NotImplementedError(
                "GLM-OCR SDK integration is not configured yet. "
                "Please install the official GLM-OCR SDK and update GLMOCRSdkAdapter."
            )
        except NotImplementedError:
            raise
        except Exception as exc:
            raise RuntimeError(f"GLM-OCR SDK OCR failed: {exc}") from exc


class ManualCorrectionOCRAdapter:
    """既存 OCR 結果の used_text を人間が修正するための Adapter。"""

    def __init__(self, storage: Storage | None = None) -> None:
        self.storage = storage or Storage()

    def apply_correction(
        self,
        ocr_id: str,
        used_text: str,
        *,
        ocr_suspect: bool | None = None,
        correction_note: str | None = None,
    ) -> OCRResult:
        existing = self.storage.get_ocr_result(ocr_id)
        if existing is None:
            raise KeyError(f"OCR result not found: {ocr_id}")

        updated = existing.model_copy(
            update={
                "used_text": used_text,
                "human_corrected": used_text != existing.raw_text,
                "ocr_suspect": (
                    existing.ocr_suspect if ocr_suspect is None else ocr_suspect
                ),
                "correction_note": correction_note,
            }
        )
        return self.storage.update_ocr_result(updated)


def create_ocr_adapter(
    provider: str | None = None,
    config: AppConfig | None = None,
) -> OCRAdapter:
    cfg = config or load_config()
    selected = (provider or cfg.ocr.provider).lower()
    if selected == "mock":
        return MockOCRAdapter()
    if selected in {"glmocr_sdk", "glm-ocr"}:
        return GLMOCRSdkAdapter(config=cfg)
    raise ValueError(f"Unsupported OCR provider: {selected}")
