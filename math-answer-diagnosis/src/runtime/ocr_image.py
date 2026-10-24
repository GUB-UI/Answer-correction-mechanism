"""GLM-OCR に渡す前の画像サイズ。既定では何もしない。

手書きノートの写真では小さい文字が潰れることがある。効くかどうかは資料次第なので、
拡大・縮小は GLMOCR_MIN_EDGE / GLMOCR_MAX_EDGE を付けたときだけ行う。
"""

from __future__ import annotations

from PIL import Image


def resize_for_ocr(image: Image.Image, *, min_edge: int = 0, max_edge: int = 0) -> Image.Image:
    width, height = image.size
    if width < 1 or height < 1:
        return image
    long_edge = max(width, height)
    short_edge = min(width, height)
    scale = 1.0
    if min_edge > 0 and short_edge < min_edge:
        scale = min_edge / short_edge
    if max_edge > 0 and long_edge * scale > max_edge:
        scale = max_edge / long_edge
    if abs(scale - 1.0) < 0.01:
        return image
    resized = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return image.resize(resized, Image.Resampling.LANCZOS)
