from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx


def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v.strip() if v and v.strip() else default


ImageUrlFormat = Literal["raw_base64", "data_uri"]


def _guess_mime(image_path: Path) -> str:
    mime, _ = mimetypes.guess_type(image_path.name)
    if mime in ("image/jpeg", "image/png", "image/webp"):
        return mime
    ext = image_path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    return "image/png"


def _maybe_resize_image(data: bytes, max_edge: int) -> bytes:
    if max_edge <= 0:
        return data
    try:
        from io import BytesIO

        from PIL import Image

        im = Image.open(BytesIO(data))
        im = im.convert("RGB") if im.mode not in ("RGB", "L") else im
        w, h = im.size
        edge = max(w, h)
        if edge <= max_edge:
            return data
        scale = max_edge / float(edge)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        im = im.resize((nw, nh), Image.Resampling.LANCZOS)
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=88)
        return buf.getvalue()
    except Exception:
        return data


def _encode_image_uri(b64: str, mime: str, fmt: ImageUrlFormat) -> str:
    if fmt == "data_uri":
        return f"data:{mime};base64,{b64}"
    return b64


def _extract_json(text: str) -> dict[str, Any]:
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", t)
    if fence:
        t = fence.group(1).strip()
    return json.loads(t)


@dataclass
class PipelineConfig:
    """環境変数で上書き可能。コンストラクタ引数優先。"""

    base_url: str | None = None
    api_key: str | None = None
    timeout_sec: float = 600.0
    ocr_model: str | None = None
    llm_model: str | None = None
    image_url_format: ImageUrlFormat | None = None
    max_image_edge: int | None = None

    def resolved_base_url(self) -> str:
        if self.base_url:
            return self.base_url.rstrip("/")
        return _env("LM_STUDIO_BASE_URL", "http://localhost:1234/v1").rstrip("/")

    def resolved_api_key(self) -> str:
        if self.api_key is not None:
            return self.api_key
        return _env("LM_STUDIO_API_KEY", "lm-studio")

    def resolved_ocr_model(self) -> str | None:
        if self.ocr_model is not None:
            return self.ocr_model or None
        m = os.environ.get("LM_STUDIO_OCR_MODEL")
        return m.strip() if m and m.strip() else None

    def resolved_llm_model(self) -> str | None:
        if self.llm_model is not None:
            return self.llm_model or None
        m = os.environ.get("LM_STUDIO_LLM_MODEL")
        return m.strip() if m and m.strip() else None

    def resolved_image_url_format(self) -> ImageUrlFormat:
        if self.image_url_format:
            return self.image_url_format
        v = (_env("LM_STUDIO_IMAGE_URL_FORMAT", "raw_base64")).lower()
        return "data_uri" if v == "data_uri" else "raw_base64"

    def resolved_max_image_edge(self) -> int:
        if self.max_image_edge is not None:
            return self.max_image_edge
        try:
            return int(_env("LM_STUDIO_MAX_IMAGE_EDGE", "0"))
        except ValueError:
            return 0


def _chat_raw(
    *,
    cfg: PipelineConfig,
    body: dict[str, Any],
) -> httpx.Response:
    url = f"{cfg.resolved_base_url()}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cfg.resolved_api_key()}",
    }
    return httpx.post(
        url,
        json=body,
        headers=headers,
        timeout=cfg.timeout_sec,
    )


def _chat(
    *,
    cfg: PipelineConfig,
    messages: list[dict[str, Any]],
    model: str | None,
    response_format_json: bool = False,
) -> str:
    body: dict[str, Any] = {
        "messages": messages,
        "temperature": 0.2,
        "stream": False,
    }
    if model:
        body["model"] = model
    if response_format_json:
        body["response_format"] = {"type": "json_object"}

    r = _chat_raw(cfg=cfg, body=body)
    if r.status_code >= 400 and response_format_json:
        rb = dict(body)
        rb.pop("response_format", None)
        r = _chat_raw(cfg=cfg, body=rb)
    r.raise_for_status()
    data = r.json()
    choice = data.get("choices") or [{}]
    msg = choice[0].get("message") or {}
    content = msg.get("content")
    if not isinstance(content, str):
        raise RuntimeError(f"応答に content がありません: {data}")
    return content


OCR_SYSTEM = """あなたは教科書や答案用紙のテキストを精密に転記する OCR です。
レイアウトを保ちつつ、画像内に書かれた文字・数式記号・箇条書きを漏れなく転写してください。
推測で補完しないでください。読めない場合は （判読不明） と記してください。
出力は転記テキストのみ（前置きや説明文は書かない）。"""

ANALYSIS_SYSTEM = """あなたは採点補助手順を実行します。入力は OCR で得られた生徒の解答テキストです。
次のキーを持つ JSON オブジェクトだけを返してください（他の説明・コードフェンス・日本語前置き禁止）。
キー仕様:
- "items": 問題単位の配列。各要素は {"question_id": string|null, "student_answer": string, "judgment": "correct"|"incorrect"|"partial"|"unknown", "feedback": string}
- "summary": string（全体コメント）

JSON 以外は一切出力しない。"""


def _ocr_via_lm_studio(
    *,
    cfg: PipelineConfig,
    image_path: Path,
    image_bytes: bytes,
    mime: str,
) -> str:
    b64 = base64.b64encode(image_bytes).decode("ascii")
    url_val = _encode_image_uri(b64, mime, cfg.resolved_image_url_format())
    user_content: list[dict[str, Any]] = [
        {
            "type": "image_url",
            "image_url": {"url": url_val},
        },
        {
            "type": "text",
            "text": "画像内の解答をすべて転記してください。",
        },
    ]
    messages = [
        {"role": "system", "content": OCR_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    return _chat(cfg=cfg, messages=messages, model=cfg.resolved_ocr_model())


def _analyze_via_lm_studio(*, cfg: PipelineConfig, ocr_text: str) -> dict[str, Any]:
    user = f"以下は OCR で得られたテキストです。指定スキーマの JSON のみ出力してください。\n\n---\n{ocr_text}\n---"
    messages = [
        {"role": "system", "content": ANALYSIS_SYSTEM},
        {"role": "user", "content": user},
    ]
    raw = _chat(
        cfg=cfg,
        messages=messages,
        model=cfg.resolved_llm_model(),
        response_format_json=True,
    )
    return _extract_json(raw)


def process_answer_image(
    image_path: str | Path,
    *,
    config: PipelineConfig | None = None,
) -> dict[str, Any]:
    """
    解答画像を LM Studio のチャット completions API で OCR し、続けて LLM で JSON 構造化結果を生成する。

    戻り値例::
        {
          "ok": true,
          "ocr_raw_text": "...",
          "llm_json": { "structured": {...}, ... },
          "meta": { "image": "...", "ocr_model_env": ..., ... }
        }
    """
    cfg = config or PipelineConfig()
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    data = path.read_bytes()
    mime = _guess_mime(path)
    edge = cfg.resolved_max_image_edge()
    data = _maybe_resize_image(data, edge)

    ocr_text = _ocr_via_lm_studio(cfg=cfg, image_path=path, image_bytes=data, mime=mime)
    llm_obj = _analyze_via_lm_studio(cfg=cfg, ocr_text=ocr_text)

    return {
        "ok": True,
        "ocr_raw_text": ocr_text.strip(),
        "llm_json": llm_obj,
        "meta": {
            "image": str(path.resolve()),
            "mime": mime,
            "lm_studio_base_url": cfg.resolved_base_url(),
            "ocr_model": cfg.resolved_ocr_model(),
            "llm_model": cfg.resolved_llm_model(),
            "image_url_format": cfg.resolved_image_url_format(),
        },
    }
