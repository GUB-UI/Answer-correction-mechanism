from __future__ import annotations

from typing import Any

from src.adapters.base import OCRAdapter, OCRRunContext
from src.models import OCRResult, utc_now_iso
from src.sources import resolve_stored_path
from src.storage import Storage

GLMOCR_SDK_ERROR_MESSAGE = """GLM-OCR SDKの実行に失敗しました。
mlx-vlm サーバー（既定: http://localhost:8080）が起動していることと、
公式SDKのインストールを確認してください。"""


def run_ocr_for_images(
    adapter: OCRAdapter,
    storage: Storage,
    *,
    source_kind: str,
    source_id: str,
    image_paths: list[str],
    engine: str,
) -> OCRResult:
    if not image_paths:
        raise ValueError("OCRする画像がありません。")

    page_texts: list[str] = []
    page_raw: list[dict[str, Any]] = []
    engine_names: list[str] = []
    answer_id = source_id if source_kind == "student_answer" else ""

    for index, rel in enumerate(image_paths, start=1):
        image_path = resolve_stored_path(storage.config.project_root, rel)
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

    combined = "\n\n".join(page_texts).strip()
    result = OCRResult(
        ocr_id=storage.generate_id("ocr"),
        answer_id=answer_id,
        ocr_engine="+".join(dict.fromkeys(engine_names)) or engine,
        raw_text=combined,
        used_text=combined,
        uncertain_parts=[],
        ocr_suspect=False,
        human_corrected=False,
        correction_note=None,
        raw_output={
            "source_kind": source_kind,
            "source_id": source_id,
            "pages": page_raw,
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
