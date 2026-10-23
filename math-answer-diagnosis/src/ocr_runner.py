from __future__ import annotations

from typing import Any

from src.adapters.base import OCRAdapter, OCRRunContext
from src.models import OCRProvenance, OCRResult, utc_now_iso
from src.ocr_provenance import image_hashes, ocr_provenance
from src.storage import Storage

GLMOCR_SDK_ERROR_MESSAGE = """GLM-OCR SDKの実行に失敗しました。
OCRサーバー（既定: http://127.0.0.1:8080。Mac は mlx-vlm、Colab は scripts/glmocr_server.py）が
起動していることと、公式SDKのインストールを確認してください。
Colab（SERVICE_MODE=managed）ではサーバーは自動起動されます。ログは .run/ 以下を確認してください。"""


def run_ocr_for_images(
    adapter: OCRAdapter,
    storage: Storage,
    *,
    source_kind: str,
    source_id: str,
    image_paths: list[str],
    engine: str,
    provenance: OCRProvenance | None = None,
) -> OCRResult:
    if not image_paths:
        raise ValueError("OCRする画像がありません。")

    if provenance is None:
        provider = {
            "MockOCRAdapter": "mock",
            "GLMOCRSdkAdapter": "glmocr_sdk",
            "XiaomiOCRAdapter": "xiaomi_ocr",
        }.get(type(adapter).__name__, storage.config.ocr.provider)
        provenance = ocr_provenance(
            storage.config, image_paths, provider=provider, adapter=adapter,
            apply_formula=bool(getattr(adapter, "apply_formula", False)),
        )
    page_texts: list[str] = []
    page_raw: list[dict[str, Any]] = []
    engine_names: list[str] = []
    suspect_reasons: list[str] = []
    answer_id = source_id if source_kind == "student_answer" else ""

    for index, rel in enumerate(image_paths, start=1):
        image_path = storage.config.resolve_path(rel)
        if not image_path.is_file():
            raise FileNotFoundError(f"画像が見つかりません: {image_path}")
        part = adapter.run(
            OCRRunContext(
                answer_id=answer_id,
                image_path=image_path,
                ocr_engine=engine,
                source_kind=source_kind,
                source_id=source_id,
                image_rel=rel,
            )
        )
        label = f"## 画像{index}\n{part.raw_text}" if len(image_paths) > 1 else part.raw_text
        page_texts.append(label)
        page_raw.append(part.raw_output)
        engine_names.append(part.ocr_engine)
        if part.ocr_suspect:
            suspect_reasons.extend(part.raw_output.get("ocr_suspect_reasons") or ["ocr_suspect"])

    combined = "\n\n".join(page_texts).strip()
    result = OCRResult(
        provenance=(provenance if image_hashes(storage.config, image_paths) == provenance.image_sha256 else None),
        ocr_id=storage.generate_id("ocr"),
        answer_id=answer_id,
        ocr_engine="+".join(dict.fromkeys(engine_names)) or engine,
        raw_text=combined,
        used_text=combined,
        uncertain_parts=[],
        ocr_suspect=bool(suspect_reasons),
        human_corrected=False,
        correction_note=None,
        raw_output={
            "source_kind": source_kind,
            "source_id": source_id,
            "pages": page_raw,
            "ocr_suspect_reasons": suspect_reasons,
        },
        created_at=utc_now_iso(),
        source_kind=source_kind,
        source_id=source_id,
        image_paths=list(image_paths),
    )
    storage.save_ocr_result(result)
    return result


def typed_text_as_ocr(
    storage: Storage,
    *,
    answer_id: str,
    typed_text: str,
) -> OCRResult:
    text = typed_text.strip()
    if not text:
        raise ValueError("テキスト答案が空です。")
    result = OCRResult(
        ocr_id=storage.generate_id("ocr"),
        answer_id=answer_id,
        ocr_engine="typed_text",
        raw_text=text,
        used_text=text,
        uncertain_parts=[],
        ocr_suspect=False,
        human_corrected=False,
        correction_note="画像ではなくテキストとして入力された答案",
        raw_output={"provider": "typed_text"},
        created_at=utc_now_iso(),
        source_kind="student_answer",
        source_id=answer_id,
        image_paths=[],
    )
    storage.save_ocr_result(result)
    return result
