import json

import pytest
from PIL import Image

from harness_v2.recognition import run_ocr_cpu as ocr
from harness_v2.recognition.skeleton import cer, gate_a1, skeleton


def _img():
    img = Image.new("RGB", (40, 20), (200, 200, 200))
    for x in range(10, 20):
        img.putpixel((x, 5), (90, 90, 90))
    return img


def test_none_is_identity():
    img = _img()
    assert ocr.preprocess_image(img, "none").tobytes() == img.tobytes()


def test_bundle_is_deterministic_and_scaled():
    a = ocr.preprocess_image(_img(), "gray_autocontrast_x1.5")
    b = ocr.preprocess_image(_img(), "gray_autocontrast_x1.5")
    assert a.size == (60, 30) and a.mode == "RGB"
    assert ocr.image_sha256(a) == ocr.image_sha256(b)


def test_bundle_caps_long_edge():
    big = Image.new("RGB", (3000, 100), (255, 255, 255))
    assert max(ocr.preprocess_image(big, "gray_autocontrast_x1.5").size) == 4096


def test_unknown_preprocess_rejected():
    with pytest.raises(ValueError):
        ocr.preprocess_image(_img(), "sharpen")
    with pytest.raises(SystemExit):
        ocr._parser().parse_args(["x.jpg", "--preprocess", "sharpen"])


def test_default_preprocess_is_none():
    assert ocr._parser().parse_args(["x.jpg"]).preprocess == "none"
    assert ocr.PREPROCESS_PARAMS["none"] == {}
    json.dumps(ocr.PREPROCESS_PARAMS)


@pytest.mark.parametrize("a,b", [
    (r"$\frac{1}{2}a$", "1/2a"),
    (r"\dfrac{a+1}{2}", "(a+1)/2"),
    (r"\frac{\frac{1}{2}}{3}", "(1/2)/3"),
    (r"$x \leq 1$", "x≤1"),
    (r"x \le 1", "x≦1"),
    (r"x\geq 2", "x ≧ 2"),
    ("a−1", "a-1"),
    (r"x^2", "x²"),
    (r"\sqrt{2}", "√2"),
    (r"2 \times 3", "2×3"),
    ("```markdown\n答 $x=1$\n```", "答 x=1"),
])
def test_equivalent_notation_matches(a, b):
    assert skeleton(a) == skeleton(b)


@pytest.mark.parametrize("a,b", [
    (r"\frac{1}{2}", "1/3"), (r"x\leq1", "x<1"), ("-17", "17"), ("x=1", "x=2"), ("最小値", "最大値"),
])
def test_different_math_stays_different(a, b):
    assert skeleton(a) != skeleton(b)


def test_cer_and_gate():
    assert cer("abc", "abc") == 0
    assert cer("abd", "abc") == pytest.approx(1 / 3)
    base = {"a": 0.10, "b": 0.05, "c": 0.20}
    assert gate_a1(base, {"a": 0.10, "b": 0.05, "c": 0.15}, target="c")["passed"]
    assert not gate_a1(base, {"a": 0.12, "b": 0.05, "c": 0.15}, target="c")["passed"]
    assert not gate_a1(base, {"a": 0.10, "b": 0.05, "c": 0.19}, target="c")["passed"]
    with pytest.raises(ValueError):
        cer("x", "")


@pytest.mark.parametrize("a,b", [
    (r"\frac{x^{2}}{3}", "x²/3"),
    (r"\frac{\sqrt{2}}{2}", "√2/2"),
    (r"\frac{a_{1}}{b}", "a1/b"),
    (r"\frac{{1}}{2}", "1/2"),
    (r"\frac{1}{\frac{2}{3}}", "1/(2/3)"),
    (r"\frac12", "1/2"),
    ("\\frac｛1｝｛2｝", "1/2"),
])
def test_grouped_fraction_arguments(a, b):
    assert skeleton(a) == skeleton(b)


def test_gate_boundaries_inclusive():
    assert gate_a1({"t": 0.30, "o": 0.06}, {"t": 0.28, "o": 0.07}, target="t")["passed"]
    assert not gate_a1({"t": 0.30, "o": 0.06}, {"t": 0.2801, "o": 0.06}, target="t")["passed"]
    assert not gate_a1({"t": 0.30, "o": 0.06}, {"t": 0.20, "o": 0.0701}, target="t")["passed"]


@pytest.mark.parametrize("name", ["none", "gray_autocontrast_x1.5"])
def test_runner_records_provenance(tmp_path, monkeypatch, name):
    import contextlib
    import hashlib
    import sys
    from types import SimpleNamespace

    seen = {}

    class Tokens:
        shape = [1, 3]
        def __getitem__(self, index):
            return [0, 1, 2]

    def template(messages, **kwargs):
        seen["image"] = messages[0]["content"][0]["image"]
        seen["kwargs"] = kwargs
        return {"input_ids": SimpleNamespace(shape=[1, 1])}

    def generate(**kwargs):
        seen["generate"] = kwargs
        return Tokens()

    processor = SimpleNamespace(apply_chat_template=template, decode=lambda *a, **k: "text")
    model = SimpleNamespace(eval=lambda: SimpleNamespace(generate=generate))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        set_num_threads=lambda n: None, float32="float32", inference_mode=contextlib.nullcontext))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoProcessor=SimpleNamespace(from_pretrained=lambda *a, **kw: processor),
        AutoModelForImageTextToText=SimpleNamespace(from_pretrained=lambda *a, **kw: model)))
    path = tmp_path / "sheet.png"
    _img().save(path)
    args = SimpleNamespace(threads=1, out=str(tmp_path / "o"), images=[str(path)], max_edge=0,
                           max_new_tokens=7, preprocess=name)
    assert ocr._run_model(args) == 0
    stats = json.loads((tmp_path / "o" / "stats.jsonl").read_text())
    assert stats["original_image_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert stats["model_image_sha256"] == ocr.image_sha256(seen["image"])
    assert stats["model_image_size"] == list(seen["image"].size)
    assert stats["model_image_size"] == ([40, 20] if name == "none" else [60, 30])
    assert stats["preprocess"] == {"name": name, "params": ocr.PREPROCESS_PARAMS[name]}
    assert stats["prompt_sha256"] == hashlib.sha256(ocr.PROMPT.encode()).hexdigest()
    assert stats["revision"] == ocr.REVISION and stats["dtype"] == "float32"
    assert stats["do_sample"] is False and seen["generate"]["do_sample"] is False
    assert seen["generate"]["max_new_tokens"] == 7 and stats["enable_thinking"] is False
    assert seen["kwargs"]["enable_thinking"] is False
