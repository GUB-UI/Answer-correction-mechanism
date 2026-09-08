from __future__ import annotations

import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image


class UniMERNetClient:
    """UniMERNet Small のローカル推論サーバークライアント。"""

    def __init__(self, base_url: str = "http://127.0.0.1:8091") -> None:
        self.base_url = base_url.rstrip("/")

    def health(self) -> bool:
        try:
            with urlopen(f"{self.base_url}/health", timeout=2.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return bool(payload.get("ok"))
        except (URLError, HTTPError, TimeoutError, json.JSONDecodeError, OSError):
            return False

    def predict_image(self, image: Image.Image, timeout: int = 60) -> str:
        import io

        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        return self.predict_png_bytes(buffer.getvalue(), timeout=timeout)

    def predict_png_bytes(self, png_bytes: bytes, timeout: int = 60) -> str:
        request = Request(
            f"{self.base_url}/predict",
            data=png_bytes,
            headers={"Content-Type": "application/octet-stream"},
            method="POST",
        )
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        latex = str(payload.get("latex") or "").strip()
        if not latex:
            raise RuntimeError("UniMERNet returned empty latex")
        return latex

    def predict_path(self, image_path: Path, timeout: int = 60) -> str:
        payload = json.dumps({"image_path": str(image_path)}).encode("utf-8")
        request = Request(
            f"{self.base_url}/predict_path",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        latex = str(body.get("latex") or "").strip()
        if not latex:
            raise RuntimeError("UniMERNet returned empty latex")
        return latex
