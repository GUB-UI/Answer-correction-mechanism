from __future__ import annotations

import mimetypes
from enum import Enum
from pathlib import Path
from typing import Literal

from src.adapters.base import OCRAdapter, OCRRunContext
from src.adapters.lmstudio_client import LMStudioClient
from src.adapters.mock_adapters import MockOCRAdapter
from src.config import AppConfig, load_config
from src.models import OCRResult, utc_now_iso
from src.prompts import OCR_PROMPT
from src.storage import Storage
from src.utils import detect_uncertain_parts, image_to_base64, is_ocr_suspect


class ZaiOCRMode(str, Enum):
    LM_STUDIO = "lm_studio"
    API = "api"


class ZaiOCRAdapter(OCRAdapter):
    """Z.ai 系小型 OCR モデル用 Adapter（LM Studio / API 両対応の骨組み）。"""

    def __init__(
        self,
        config: AppConfig | None = None,
        mode: ZaiOCRMode | Literal["lm_studio", "api"] = ZaiOCRMode.LM_STUDIO,
        lmstudio_client: LMStudioClient | None = None,
    ) -> None:
        self.config = config or load_config()
        self.mode = ZaiOCRMode(mode)
        self.lmstudio_client = lmstudio_client or LMStudioClient(self.config)

    def run(self, context: OCRRunContext) -> OCRResult:
        if self.mode == ZaiOCRMode.LM_STUDIO:
            raw_text, raw_output = self._run_via_lm_studio(context)
        else:
            raw_text, raw_output = self._run_via_api(context)

        uncertain_parts = detect_uncertain_parts(raw_text)
        return OCRResult(
            ocr_id="",
            answer_id=context.answer_id,
            ocr_engine=context.ocr_engine,
            raw_text=raw_text,
            used_text=raw_text,
            uncertain_parts=uncertain_parts,
            ocr_suspect=is_ocr_suspect(raw_text, uncertain_parts),
            human_corrected=False,
            correction_note=None,
            raw_output=raw_output,
            created_at=utc_now_iso(),
        )

    def _run_via_lm_studio(self, context: OCRRunContext) -> tuple[str, dict]:
        mime = self._guess_mime(context.image_path)
        image_data_url = f"data:{mime};base64,{image_to_base64(context.image_path)}"
        text = self.lmstudio_client.vision_chat_completion(
            system_prompt=OCR_PROMPT,
            user_prompt="画像内の答案を読み取ってください。",
            image_data_url=image_data_url,
            model=self.config.ocr.model_name,
        )
        return text.strip(), {
            "provider": "zai",
            "mode": self.mode.value,
            "transport": "lm_studio",
            "model_name": self.config.ocr.model_name,
        }

    def _run_via_api(self, context: OCRRunContext) -> tuple[str, dict]:
        raise NotImplementedError(
            "Z.ai API 経由の OCR は将来実装予定です。"
            "現時点では mode='lm_studio' を使用してください。"
        )

    @staticmethod
    def _guess_mime(image_path: Path) -> str:
        mime, _ = mimetypes.guess_type(image_path.name)
        return mime or "image/png"


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
    if selected in {"zai", "zai-small-ocr"}:
        return ZaiOCRAdapter(config=cfg)
    raise ValueError(f"Unsupported OCR provider: {selected}")
