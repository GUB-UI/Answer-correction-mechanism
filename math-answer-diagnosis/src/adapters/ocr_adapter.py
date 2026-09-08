from __future__ import annotations

from pathlib import Path
from typing import Any

from src.adapters.base import OCRAdapter, OCRRunContext
from src.adapters.formula_merge import (
    block_bbox,
    crop_region,
    latex_is_plausible,
    merge_region_content,
    rebuild_markdown,
    region_prefers_unimernet,
    _as_pages,
)
from src.adapters.mock_adapters import MockOCRAdapter
from src.adapters.unimernet_client import UniMERNetClient
from src.config import AppConfig, load_config
from src.models import OCRResult, utc_now_iso
from src.storage import Storage


def _import_glmocr() -> Any:
    try:
        from glmocr import GlmOcr
    except ImportError as exc:
        raise RuntimeError(
            "GLM-OCR SDK is not installed. "
            "Install the official SDK and use a self-hosted mlx-vlm server."
        ) from exc
    return GlmOcr


def _extract_markdown(result: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(result, list):
        if not result:
            raise RuntimeError("GLM-OCR SDK returned an empty result list")
        return _extract_markdown(result[0])

    raw: dict[str, Any] = {}
    if hasattr(result, "to_dict"):
        try:
            maybe_raw = result.to_dict()
        except Exception:
            maybe_raw = None
        if isinstance(maybe_raw, dict):
            raw = maybe_raw

    if isinstance(raw, dict) and raw.get("error"):
        raise RuntimeError(str(raw["error"]))

    markdown = getattr(result, "markdown_result", None)
    if not (isinstance(markdown, str) and markdown.strip()):
        markdown = raw.get("markdown_result") if isinstance(raw, dict) else None
    if not (isinstance(markdown, str) and markdown.strip()):
        raise RuntimeError("GLM-OCR SDK returned empty text")
    return markdown.strip(), raw


class GLMOCRSdkAdapter(OCRAdapter):
    """GLM-OCR + UniMERNet Small。日本語・文書は GLM、数字・数式は UniMERNet を優先する。"""

    def __init__(
        self,
        config: AppConfig | None = None,
        formula_client: UniMERNetClient | None = None,
    ) -> None:
        self.config = config or load_config()
        self.formula_client = formula_client

    def run(self, context: OCRRunContext) -> OCRResult:
        raw_text, raw_output = self._run_via_sdk(context)
        engine = context.ocr_engine
        replacements = raw_output.get("formula_replacements") or []
        if any("unimernet" in item for item in replacements if isinstance(item, dict)):
            engine = f"{engine}+unimernet_small"
        return OCRResult(
            ocr_id="",
            answer_id=context.answer_id,
            ocr_engine=engine,
            raw_text=raw_text,
            used_text=raw_text,
            uncertain_parts=[],
            ocr_suspect=False,
            human_corrected=False,
            correction_note=None,
            raw_output=raw_output,
            created_at=utc_now_iso(),
            source_kind=context.source_kind,
            source_id=context.source_id or context.answer_id,
            image_paths=[context.image_rel] if context.image_rel else [],
        )

    def _run_via_sdk(self, context: OCRRunContext) -> tuple[str, dict[str, Any]]:
        try:
            raw_text, parsed = self._invoke_sdk(context.image_path)
            replacements: list[dict[str, Any]] = []
            if self.config.formula.enabled:
                raw_text, replacements = self._prefer_unimernet_math(
                    context.image_path,
                    raw_text,
                    parsed,
                )
            raw_output = {
                "provider": "glmocr_sdk",
                "model_name": self.config.ocr.model_name,
                "formula_model": self.config.formula.model_name,
                "image_path": str(context.image_path),
                "result": parsed,
                "formula_replacements": replacements,
            }
            return raw_text, raw_output
        except Exception as exc:
            raise RuntimeError(f"GLM-OCR SDK OCR failed: {exc}") from exc

    def _invoke_sdk(self, image_path: Path) -> tuple[str, dict[str, Any]]:
        GlmOcr = _import_glmocr()
        config_path = self.config.project_root / "glmocr.config.yaml"
        kwargs: dict[str, Any] = {
            "mode": "selfhosted",
            "model": self.config.ocr.model_name,
            "layout_device": "cpu",
        }
        if config_path.is_file():
            kwargs["config_path"] = str(config_path)
        with GlmOcr(**kwargs) as parser:
            result = parser.parse(
                str(image_path),
                save_layout_visualization=False,
            )
        return _extract_markdown(result)

    def _prefer_unimernet_math(
        self,
        image_path: Path,
        markdown: str,
        parsed: dict[str, Any],
    ) -> tuple[str, list[dict[str, Any]]]:
        client = self.formula_client or UniMERNetClient(self.config.formula.base_url)
        if not client.health():
            return markdown, [{"skipped": "unimernet_unavailable"}]

        from copy import deepcopy

        from PIL import Image

        pages = _as_pages(parsed.get("json_result"))
        if not pages:
            return markdown, []

        image = Image.open(image_path).convert("RGB")
        replacements: list[dict[str, Any]] = []
        updated_pages: list[list[dict[str, Any]]] = []
        for page in pages:
            updated_page: list[dict[str, Any]] = []
            for block in page:
                block = deepcopy(block)
                if region_prefers_unimernet(block):
                    crop = crop_region(image, block_bbox(block))
                    if crop is not None:
                        try:
                            latex = client.predict_image(crop)
                            glm_content = str(block.get("content") or "")
                            if not latex_is_plausible(latex):
                                replacements.append(
                                    {
                                        "label": block.get("label"),
                                        "native_label": block.get("native_label"),
                                        "glm": glm_content,
                                        "skipped": "unimernet_implausible",
                                        "unimernet_raw": latex,
                                    }
                                )
                            else:
                                block["content"] = merge_region_content(block, latex)
                                block["glm_content"] = glm_content
                                replacements.append(
                                    {
                                        "label": block.get("label"),
                                        "native_label": block.get("native_label"),
                                        "glm": glm_content,
                                        "unimernet": latex,
                                    }
                                )
                        except Exception as exc:
                            replacements.append(
                                {
                                    "label": block.get("label"),
                                    "error": str(exc),
                                }
                            )
                updated_page.append(block)
            updated_pages.append(updated_page)
        if any("unimernet" in item for item in replacements):
            parsed["json_result"] = updated_pages
            parsed["glm_markdown_result"] = markdown
            markdown = rebuild_markdown(updated_pages)
        return markdown, replacements


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
