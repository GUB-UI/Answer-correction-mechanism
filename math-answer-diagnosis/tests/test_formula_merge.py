from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.adapters.formula_merge import (
    crop_region,
    latex_is_plausible,
    looks_math_dominant,
    merge_region_content,
    rebuild_markdown,
    region_prefers_unimernet,
    replace_math_spans,
    wrap_latex,
)
from src.adapters.ocr_adapter import GLMOCRSdkAdapter
from src.adapters.base import OCRRunContext
from src.config import load_config
from tests.test_adapters import _write_test_config


def test_formula_labels_prefer_unimernet() -> None:
    assert region_prefers_unimernet({"label": "formula", "content": "E=mc^2"})
    assert region_prefers_unimernet({"native_label": "inline_formula", "content": "a+b"})
    assert region_prefers_unimernet({"label": "text", "content": "$5x-29 < -3x+11$"})
    assert region_prefers_unimernet({"label": "text", "content": "8x<40"})
    assert not region_prefers_unimernet({"label": "text", "content": "次の不等式を解け。"})
    assert not region_prefers_unimernet(
        {"label": "text", "content": "4. (1) $5 x-2 9 < -3 x+11$ を解け。"}
    )


def test_looks_math_dominant() -> None:
    assert looks_math_dominant("$8x < 40$, $x < 5$")
    assert looks_math_dominant("8x<40")
    assert not looks_math_dominant("自然数であるから次の値をすべて求めよ。")


def test_replace_spaced_digits_span() -> None:
    content = "よって $5 x - 2 9 < -3 x + 1 1$ である。"
    updated = replace_math_spans(content, "5x-29 < -3x+11")
    assert "5x-29 < -3x+11" in updated
    assert "よって" in updated


def test_wrap_latex_strips_existing_dollars() -> None:
    assert wrap_latex("$$a+b$$", display=False) == "$a+b$"


def test_merge_formula_region_replaces_whole_block() -> None:
    block = {"label": "formula", "content": "$$5 x - 2 9$$"}
    assert "5x-29" in merge_region_content(block, "5x-29")


def test_crop_region_clamps_pixel_coords() -> None:
    image = Image.new("RGB", (100, 40), "white")
    crop = crop_region(image, [10, 5, 80, 2000], pad=2)
    assert crop is not None
    assert crop.size[1] <= 40


def test_crop_region_normalizes_glm_0_1000() -> None:
    image = Image.new("RGB", (200, 100), "white")
    crop = crop_region(image, [0, 0, 500, 500], pad=0)
    assert crop is not None
    assert crop.size == (100, 50)


def test_latex_is_plausible() -> None:
    assert latex_is_plausible("5x-29 < -3x+11")
    assert not latex_is_plausible("次の不等式")


def test_rebuild_markdown_keeps_order() -> None:
    pages = [
        [
            {"index": 1, "content": "second"},
            {"index": 0, "content": "first"},
        ]
    ]
    assert rebuild_markdown(pages) == "first\n\nsecond"


def test_glmocr_prefers_unimernet_on_math_region(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _write_test_config(root)
    config = load_config(root)

    class FakeResult:
        markdown_result = "自然数を求めよ。\n\n$$5 x-2 9 < -3 x+11$$"

        def to_dict(self) -> dict:
            return {
                "markdown_result": self.markdown_result,
                "json_result": [
                    [
                        {
                            "index": 0,
                            "label": "text",
                            "native_label": "text",
                            "content": "自然数を求めよ。",
                            "bbox_2d": [0, 0, 1000, 200],
                        },
                        {
                            "index": 1,
                            "label": "formula",
                            "native_label": "display_formula",
                            "content": "$$5 x-2 9 < -3 x+11$$",
                            "bbox_2d": [0, 200, 1000, 1000],
                        },
                    ]
                ],
            }

    class FakeFormula:
        def health(self) -> bool:
            return True

        def predict_image(self, image: Image.Image) -> str:
            return "5x-29 < -3x+11"

    image_path = root / "sample.png"
    Image.new("RGB", (80, 40), "white").save(image_path)

    adapter = GLMOCRSdkAdapter(config=config, formula_client=FakeFormula())
    adapter._invoke_sdk = lambda path: (  # type: ignore[method-assign]
        FakeResult().markdown_result,
        FakeResult().to_dict(),
    )
    result = adapter.run(
        OCRRunContext(
            answer_id="ans_001",
            image_path=image_path,
            ocr_engine="glm-ocr",
        )
    )
    assert "5x-29 < -3x+11" in result.used_text
    assert "自然数を求めよ" in result.used_text
    assert result.ocr_engine.endswith("+unimernet_small")
    assert result.raw_output["formula_replacements"][0]["unimernet"] == "5x-29 < -3x+11"
