from __future__ import annotations

import re
from typing import Any

from PIL import Image

FORMULA_LABELS = {
    "formula",
    "isolate_formula",
    "inline_formula",
    "display_formula",
    "equation",
    "formula_number",
    "number",
}

CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
MATH_SPAN_RE = re.compile(r"\$\$[\s\S]+?\$\$|\$[^$\n]+\$")


def _as_pages(json_result: Any) -> list[list[dict[str, Any]]]:
    if isinstance(json_result, str):
        import json

        try:
            json_result = json.loads(json_result)
        except json.JSONDecodeError:
            return []
    if not isinstance(json_result, list) or not json_result:
        return []
    if isinstance(json_result[0], dict):
        return [json_result]
    pages: list[list[dict[str, Any]]] = []
    for page in json_result:
        if isinstance(page, list):
            pages.append([block for block in page if isinstance(block, dict)])
    return pages


def looks_math_dominant(content: str) -> bool:
    text = (content or "").strip()
    if not text:
        return False
    cjk = len(CJK_RE.findall(text))
    if cjk >= 2:
        return False
    math_chars = len(re.findall(r"[0-9=<>+\-*/\\^{}_$]", text))
    if math_chars >= 6:
        return True
    return cjk == 0 and math_chars >= 4


# 1行の目安。0-1000 正規化では約 80、その 1.5 倍を超える高さは複数行とみなす。
# ピクセル座標では 160px。幅/高さが 3 以上の横長も 1 行。
_SINGLE_LINE_NORM_HEIGHT = 120.0
_SINGLE_LINE_PIXEL_HEIGHT = 160.0
_SINGLE_LINE_ASPECT = 3.0
_GREEK_CMD_RE = re.compile(
    r"\\(?:alpha|beta|gamma|delta|epsilon|varepsilon|zeta|eta|theta|vartheta|"
    r"iota|kappa|lambda|mu|nu|xi|pi|varpi|rho|sigma|tau|upsilon|phi|varphi|"
    r"chi|psi|omega)\b"
)


def region_is_single_line(block: dict[str, Any]) -> bool:
    """改行が無く、領域が1行分の高さ（または横長）のときだけ True。"""
    content = str(block.get("content") or "")
    if "\n" in content:
        return False
    bbox = block_bbox(block)
    if not bbox or len(bbox) < 4:
        return True
    try:
        x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    except (TypeError, ValueError):
        return True
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    if width <= 0 or height <= 0:
        return False
    if width / height >= _SINGLE_LINE_ASPECT:
        return True
    normalized = max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1000
    limit = _SINGLE_LINE_NORM_HEIGHT if normalized else _SINGLE_LINE_PIXEL_HEIGHT
    return height < limit


def region_prefers_unimernet(block: dict[str, Any]) -> bool:
    """1行の数式・数字領域だけブロックごと UniMERNet に渡す。

    複数行のブロックをそのまま渡すと、UniMERNet が1式用のモデルとして
    array やギリシャ文字のごみを返す。その場合は元の OCR を残す。
    """
    if not region_is_single_line(block):
        return False
    label = str(block.get("native_label") or block.get("label") or "").lower()
    content = str(block.get("content") or "")
    if label in FORMULA_LABELS or "formula" in label:
        return True
    return looks_math_dominant(content)


def region_math_span(block: dict[str, Any]) -> bool:
    """日本語交じりの1行で $...$ があるとき、その数式だけ読み直す。"""
    if not region_is_single_line(block):
        return False
    if region_prefers_unimernet(block):
        return False
    return bool(MATH_SPAN_RE.search(str(block.get("content") or ""))) and bool(block_bbox(block))


def block_bbox(block: dict[str, Any]) -> Any:
    return block.get("bbox_2d") or block.get("bbox")


def _to_pixel_bbox(
    bbox: Any, image_size: tuple[int, int]
) -> tuple[float, float, float, float] | None:
    if not bbox or len(bbox) < 4:
        return None
    try:
        x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    except (TypeError, ValueError):
        return None
    width, height = image_size
    # GLM-OCR の bbox_2d は 0-1000 正規化。1000 超はピクセル座標とみなす。
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1000:
        x1 = x1 / 1000.0 * width
        y1 = y1 / 1000.0 * height
        x2 = x2 / 1000.0 * width
        y2 = y2 / 1000.0 * height
    return x1, y1, x2, y2


def crop_region(image: Image.Image, bbox: Any, pad: int = 10) -> Image.Image | None:
    converted = _to_pixel_bbox(bbox, image.size)
    if converted is None:
        return None
    x1, y1, x2, y2 = converted
    width, height = image.size
    left = max(0, int(min(x1, x2)) - pad)
    top = max(0, int(min(y1, y2)) - pad)
    right = min(width, int(max(x1, x2)) + pad)
    bottom = min(height, int(max(y1, y2)) + pad)
    if right - left < 8 or bottom - top < 8:
        return None
    return image.crop((left, top, right, bottom))


def latex_is_plausible(latex: str) -> bool:
    text = (latex or "").strip()
    if len(text) < 1 or len(text) > 1500:
        return False
    if CJK_RE.search(text):
        return False
    return bool(re.search(r"[0-9=<>+\-*/\\^{}_()a-zA-Z]", text))


def _compact_math(text: str) -> str:
    cleaned = (text or "").replace("$$", "").replace("$", "")
    cleaned = re.sub(r"\\(?:begin|end)\{[^}]+\}", "", cleaned)
    return re.sub(r"\s+", "", cleaned)


def latex_diverges_from_base(latex: str, base: str) -> bool:
    """UniMERNet の出力が元 OCR から大きく外れていれば True（採用しない）。"""
    if not latex_is_plausible(latex):
        return True
    base_compact = _compact_math(base)
    latex_compact = _compact_math(latex)
    if not base_compact:
        return False
    if len(latex_compact) > max(40, int(len(base_compact) * 2.5)):
        return True
    if len(base_compact) >= 8 and len(latex_compact) < max(2, int(len(base_compact) * 0.35)):
        return True
    extra_greek = set(_GREEK_CMD_RE.findall(latex)) - set(_GREEK_CMD_RE.findall(base))
    if extra_greek:
        return True
    if "\\begin{" in latex and "\\begin{" not in base and latex.count("\\\\") >= 2:
        return True
    return False


def comparison_base(content: str, *, span_only: bool) -> str:
    if not span_only:
        return content
    spans = list(MATH_SPAN_RE.finditer(content))
    if not spans:
        return content
    return spans[0].group(0)


def wrap_latex(latex: str, *, display: bool) -> str:
    cleaned = latex.strip()
    cleaned = cleaned.removeprefix("$$").removesuffix("$$").strip()
    cleaned = cleaned.removeprefix("$").removesuffix("$").strip()
    if display:
        return f"$$\n{cleaned}\n$$"
    return f"${cleaned}$"


def replace_math_spans(content: str, latex: str) -> str:
    spans = list(MATH_SPAN_RE.finditer(content))
    if not spans:
        return wrap_latex(latex, display="$$" in content)
    if len(spans) == 1:
        span = spans[0]
        display = span.group(0).startswith("$$")
        return content[: span.start()] + wrap_latex(latex, display=display) + content[span.end() :]
    # 複数数式がある行は、数字のスペース割れが目立つ span だけ置き換える。
    spaced = re.compile(r"\d\s+\d")
    replaced = content
    offset = 0
    applied = False
    for span in spans:
        chunk = span.group(0)
        if not spaced.search(chunk):
            continue
        display = chunk.startswith("$$")
        start = span.start() + offset
        end = span.end() + offset
        wrapped = wrap_latex(latex, display=display)
        replaced = replaced[:start] + wrapped + replaced[end:]
        offset += len(wrapped) - (end - start)
        applied = True
        break
    if applied:
        return replaced
    span = spans[0]
    display = span.group(0).startswith("$$")
    return content[: span.start()] + wrap_latex(latex, display=display) + content[span.end() :]


def merge_region_content(block: dict[str, Any], latex: str) -> str:
    content = str(block.get("content") or "")
    label = str(block.get("native_label") or block.get("label") or "").lower()
    display = "formula" in label and "inline" not in label
    if looks_math_dominant(content) or label in FORMULA_LABELS or "formula" in label:
        return wrap_latex(latex, display=display or content.strip().startswith("$$"))
    if MATH_SPAN_RE.search(content):
        return replace_math_spans(content, latex)
    return wrap_latex(latex, display=False)


def rebuild_markdown(pages: list[list[dict[str, Any]]]) -> str:
    parts: list[str] = []
    for page in pages:
        for block in sorted(page, key=lambda item: item.get("index", 0)):
            text = str(block.get("content") or "").strip()
            if text:
                parts.append(text)
    return "\n\n".join(parts)
