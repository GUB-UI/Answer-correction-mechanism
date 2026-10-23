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
