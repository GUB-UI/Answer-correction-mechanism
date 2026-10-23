from __future__ import annotations

from pathlib import Path
from typing import Any

from src.adapters.base import OCRAdapter, OCRRunContext
from src.adapters.formula_merge import (
    block_bbox,
    comparison_base,
    crop_region,
    latex_diverges_from_base,
    merge_region_content,
    rebuild_markdown,
    region_math_span,
    region_prefers_unimernet,
    replace_math_spans,
    _as_pages,
)
from src.adapters.ocr_quality import assess_ocr_text
from src.adapters.mock_adapters import MockOCRAdapter
from src.adapters.unimernet_client import UniMERNetClient
from src.config import AppConfig, load_config
from src.models import OCRResult, utc_now_iso
from src.storage import Storage


def _formula_summary(replacements: list[dict[str, Any]]) -> dict[str, Any]:
    replaced = sum(1 for item in replacements if "unimernet" in item and "skipped" not in item)
    skipped = [str(item.get("skipped")) for item in replacements if item.get("skipped")]
    return {"replaced": replaced, "skipped": skipped}


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


def refine_markdown_with_unimernet(
    image_path: Path,
    markdown: str,
    parsed: dict[str, Any],
    client: UniMERNetClient,
) -> tuple[str, list[dict[str, Any]]]:
    """GLM-OCR のレイアウト結果のうち数式・数字領域を UniMERNet で読み直す。"""
    if not client.health():
        return markdown, [{"skipped": "unimernet_unavailable"}]

    from copy import deepcopy

    from PIL import Image

    parsed.setdefault("glm_markdown_result", markdown)
    source = parsed.get("json_result")
    pages = _as_pages(source)
    if not pages:
        # レイアウトが無いと切り出しできない。raw_text と used_text は同じ GLM 文のまま。
        return markdown, [{"skipped": "no_layout_regions"}]

    image = Image.open(image_path).convert("RGB")
    replacements: list[dict[str, Any]] = []
    updated_pages: list[list[dict[str, Any]]] = []
    considered = False
    for page in pages:
        updated_page: list[dict[str, Any]] = []
        for block in page:
            block = deepcopy(block)
            whole = region_prefers_unimernet(block)
            span_only = region_math_span(block)
            if whole or span_only:
                considered = True
                crop = crop_region(image, block_bbox(block))
                if crop is None:
                    replacements.append(
                        {
                            "label": block.get("label"),
                            "native_label": block.get("native_label"),
                            "skipped": "no_bbox",
                        }
                    )
                else:
                    # 再実行時は GLM の元テキストを基準にする。
                    glm_content = str(block.get("glm_content") or block.get("content") or "")
                    block["content"] = glm_content
                    try:
                        latex = client.predict_image(crop)
                        base_text = comparison_base(glm_content, span_only=span_only)
                        if latex_diverges_from_base(latex, base_text):
                            replacements.append(
                                {
                                    "label": block.get("label"),
                                    "native_label": block.get("native_label"),
                                    "glm": glm_content,
                                    "skipped": "unimernet_diverged",
                                    "unimernet_raw": latex,
                                }
                            )
                        else:
                            if span_only:
                                block["content"] = replace_math_spans(glm_content, latex)
                            else:
                                block["content"] = merge_region_content(block, latex)
                            block["glm_content"] = glm_content
                            replacements.append(
                                {
                                    "label": block.get("label"),
                                    "native_label": block.get("native_label"),
                                    "glm": glm_content,
                                    "unimernet": latex,
                                    "span_only": span_only,
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
    if not considered:
        labels = sorted(
            {
                str(block.get("native_label") or block.get("label") or "")
                for page in pages
                for block in page
            }
        )
        return markdown, [{"skipped": "no_formula_regions", "labels": labels}]
    if any("unimernet" in item for item in replacements):
        parsed["json_result"] = updated_pages
        markdown = rebuild_markdown(updated_pages)
    return markdown, replacements


def _ensure_service(config: AppConfig, name: str) -> None:
    from src.runtime.services import get_service_manager

    get_service_manager(config).ensure(name)


class GLMOCRSdkAdapter(OCRAdapter):
    """GLM-OCR + UniMERNet Small。日本語・文書は GLM、数字・数式は UniMERNet を優先する。

    apply_formula=False にすると GLM-OCR だけを実行する（一括実行で GPU を
    GLM-OCR → UniMERNet の順に切り替えるため）。数式の置換は
    ``refine_ocr_result_with_unimernet`` で後から行える。
    """

    def __init__(
        self,
        config: AppConfig | None = None,
        formula_client: UniMERNetClient | None = None,
        *,
        apply_formula: bool | None = None,
    ) -> None:
        self.config = config or load_config()
        self.formula_client = formula_client
        self.apply_formula = (
            self.config.formula.enabled if apply_formula is None else apply_formula
        )

    def run(self, context: OCRRunContext) -> OCRResult:
        raw_text, raw_output = self._run_via_sdk(context)
        engine = context.ocr_engine
        replacements = raw_output.get("formula_replacements") or []
        if any("unimernet" in item for item in replacements if isinstance(item, dict)):
            engine = f"{engine}+unimernet_small"
        reasons = assess_ocr_text(raw_text, replacements=replacements)
        raw_output = {**raw_output, "ocr_suspect_reasons": reasons}
        return OCRResult(
            ocr_id="",
            answer_id=context.answer_id,
            ocr_engine=engine,
            raw_text=raw_text,
            used_text=raw_text,
            uncertain_parts=[],
            ocr_suspect=bool(reasons),
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
            _ensure_service(self.config, "glm_ocr")
            raw_text, parsed = self._invoke_sdk(context.image_path)
            replacements: list[dict[str, Any]] = []
            if self.apply_formula:
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
                "glm_markdown": parsed.get("glm_markdown_result"),
                "formula_replacements": replacements,
                "formula_summary": _formula_summary(replacements),
                "formula_applied": bool(self.apply_formula),
            }
            return raw_text, raw_output
        except Exception as exc:
            raise RuntimeError(f"GLM-OCR SDK OCR failed: {exc}") from exc

    def _invoke_sdk(self, image_path: Path) -> tuple[str, dict[str, Any]]:
        GlmOcr = _import_glmocr()
        config_path = self.config.project_root / "glmocr.config.yaml"
        from src.runtime.ports import assigned_port

        kwargs: dict[str, Any] = {
            "mode": "selfhosted",
            "model": self.config.ocr.model_name,
            "layout_device": self.config.ocr_api.layout_device,
            "ocr_api_host": self.config.ocr_api.host,
            "ocr_api_port": assigned_port("glm_ocr", self.config.ocr_api.port),
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
        _ensure_service(self.config, "unimernet")
        client = self.formula_client or UniMERNetClient(self.config.formula.base_url)
        return refine_markdown_with_unimernet(image_path, markdown, parsed, client)


def refine_ocr_result_with_unimernet(
    ocr_result: OCRResult,
    config: AppConfig,
    client: UniMERNetClient | None = None,
) -> OCRResult:
    """保存済み GLM-OCR 結果に UniMERNet の数式置換を後から適用した新しい OCRResult を返す。

    人が修正済み（human_corrected）の used_text は上書きしない。
    """
    _ensure_service(config, "unimernet")
    formula_client = client or UniMERNetClient(config.formula.base_url)
    raw_output = dict(ocr_result.raw_output or {})
    pages = raw_output.get("pages")
    page_outputs: list[dict[str, Any]] = (
        [dict(page) for page in pages if isinstance(page, dict)]
        if isinstance(pages, list)
        else [raw_output]
    )
    texts: list[str] = []
    any_replaced = False
    collected: list[dict[str, Any]] = []
    for page in page_outputs:
        parsed = page.get("result") if isinstance(page.get("result"), dict) else {}
        image_value = page.get("image_path")
        if not parsed or not image_value:
            continue
        image_path = config.resolve_path(str(image_value))
        base_markdown = str(
            parsed.get("glm_markdown_result") or parsed.get("markdown_result") or ""
        )
        markdown, replacements = refine_markdown_with_unimernet(
            image_path, base_markdown, parsed, formula_client
        )
        summary = _formula_summary(replacements)
        page["result"] = parsed
        page["formula_replacements"] = replacements
        page["formula_summary"] = summary
        page["formula_applied"] = True
        collected.extend(replacements)
        any_replaced = any_replaced or summary["replaced"] > 0
        texts.append(markdown)

    if not texts:
        return ocr_result
    if len(texts) > 1:
        combined = "\n\n".join(
            f"## 画像{index}\n{text}" for index, text in enumerate(texts, start=1)
        ).strip()
    else:
        combined = texts[0].strip()
    summary = _formula_summary(collected)
    if isinstance(pages, list):
        raw_output["pages"] = page_outputs
        raw_output["formula_summary"] = summary
        raw_output["formula_replacements"] = collected
    else:
        raw_output = page_outputs[0]
        raw_output["formula_summary"] = summary
    reasons = assess_ocr_text(combined, replacements=collected)
    raw_output["ocr_suspect_reasons"] = reasons
    engine = ocr_result.ocr_engine
    if any_replaced and "unimernet" not in engine:
        engine = f"{engine}+unimernet_small"
    return ocr_result.model_copy(
        update={
            "raw_text": combined,
            "used_text": ocr_result.used_text if ocr_result.human_corrected else combined,
            "raw_output": raw_output,
            "ocr_engine": engine,
            "ocr_suspect": ocr_result.ocr_suspect if ocr_result.human_corrected else bool(reasons),
        }
    )


class ManualCorrectionOCRAdapter:
    """元の OCR を残し、手動転記を別の入力として保存する。"""

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
                "ocr_id": self.storage.generate_id("transcription"),
                "created_at": utc_now_iso(),
                "parent_ocr_id": existing.ocr_id,
                "input_type": "manual_transcription",
                "provenance": None,
                "used_text": used_text,
                "human_corrected": True,
                "ocr_suspect": (
                    existing.ocr_suspect if ocr_suspect is None else ocr_suspect
                ),
                "correction_note": correction_note,
            }
        )
        self.storage.save_ocr_result(updated)
        return updated


def _post_chat_completion(url: str, payload: dict[str, Any], timeout: float = 600.0) -> str:
    import json
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen

    body = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            parsed = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Xiaomi OCR HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Xiaomi OCR request failed: {exc}") from exc
    choices = parsed.get("choices") if isinstance(parsed, dict) else None
    if not choices:
        raise RuntimeError(f"Xiaomi OCR returned no choices: {parsed}")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Xiaomi OCR returned empty text")
    return content.strip()


class XiaomiOCRAdapter(OCRAdapter):
    """ページ全体を Xiaomi-OCR-0 で Markdown / LaTeX にする。

    手書きは領域切り出しをしない（公式の推奨はページ全体）。
    UniMERNet はここでは呼ばない。Colab の既定は UNIMERNET_ENABLED=false。
    GLM-OCR は provider を glmocr_sdk に戻すと使える。
    """

    def __init__(
        self,
        config: AppConfig | None = None,
        complete: Any | None = None,
    ) -> None:
        self.config = config or load_config()
        # complete(messages) -> str。テストはサーバーを起動しない。
        self.complete = complete

    def run(self, context: OCRRunContext) -> OCRResult:
        from src.prompts import XIAOMI_OCR_PROMPT
        from src.utils import image_to_base64

        suffix = self.config.ocr_api.prompt_suffix.strip()
        prompt = XIAOMI_OCR_PROMPT if not suffix else f"{XIAOMI_OCR_PROMPT}\n{suffix}"
        encoded = image_to_base64(context.image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        try:
            raw_text = self._complete(messages)
        except Exception as exc:
            raise RuntimeError(f"Xiaomi OCR failed: {exc}") from exc
        reasons = assess_ocr_text(raw_text)
        raw_output = {
            "provider": "xiaomi_ocr",
            "model_name": self.config.ocr.model_name,
            "image_path": str(context.image_path),
            "prompt": prompt,
            "ocr_suspect_reasons": reasons,
            "formula_applied": False,
            "formula_summary": {"replaced": 0, "skipped": ["unimernet_disabled"]},
        }
        return OCRResult(
            ocr_id="",
            answer_id=context.answer_id,
            ocr_engine=context.ocr_engine,
            raw_text=raw_text,
            used_text=raw_text,
            uncertain_parts=[],
            ocr_suspect=bool(reasons),
            human_corrected=False,
            correction_note=None,
            raw_output=raw_output,
            created_at=utc_now_iso(),
            source_kind=context.source_kind,
            source_id=context.source_id or context.answer_id,
            image_paths=[context.image_rel] if context.image_rel else [],
        )

    def _complete(self, messages: list[dict[str, Any]]) -> str:
        if self.complete is not None:
            return str(self.complete(messages)).strip()
        from src.runtime.ports import assigned_port
        from src.runtime.services import XIAOMI_OCR

        _ensure_service(self.config, XIAOMI_OCR)
        port = assigned_port(XIAOMI_OCR, self.config.ocr_api.port)
        url = f"http://{self.config.ocr_api.host}:{port}/v1/chat/completions"
        payload = {
            "model": self.config.ocr.model_name,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 4096,
        }
        return _post_chat_completion(url, payload)


def create_ocr_adapter(
    provider: str | None = None,
    config: AppConfig | None = None,
) -> OCRAdapter:
    cfg = config or load_config()
    selected = (provider or cfg.ocr.provider).lower()
    if selected == "mock":
        return MockOCRAdapter()
    if selected in {"xiaomi_ocr", "xiaomi-ocr"}:
        return XiaomiOCRAdapter(config=cfg)
    if selected in {"glmocr_sdk", "glm-ocr"}:
        return GLMOCRSdkAdapter(config=cfg)
    raise ValueError(f"Unsupported OCR provider: {selected}")
