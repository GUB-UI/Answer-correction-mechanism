"""Synthetic transcript checks. They do not read a dataset or a reference score."""
import json

from harness_v2.core.pipeline import Pipeline
from harness_v2.grading.transcript_quality import transcript_reasons
from harness_v2.providers.fixture import FixtureProvider
from harness_v2.tests.test_p3_setup import store_at
from harness_v2.tests.test_pipeline import _case


def _raw(answer: str, uncertainty: float = 0.2, confidence: float | None = None) -> str:
    payload = {
        "items": [{
            "item_id": "R1",
            "max_points": 4,
            "score": 4,
            "evidence_spans": [{"text": answer}],
            "math_reason": "matches",
            "uncertainty": uncertainty,
            "review_reason": "",
        }],
        "total_score": 4,
    }
    if confidence is not None:
        payload["confidence"] = confidence
    return json.dumps(payload, ensure_ascii=False)


def _run(tmp_path, answer: str, *, gate: str, uncertainty: float = 0.2, confidence: float | None = None):
    raw = _raw(answer, uncertainty, confidence)
    case = _case("t", raw)
    case["answer_text"] = answer
    case["transcript_quality_gate"] = gate
    case["rubric"] = [{"item_id": "R1", "max_points": 4, "description": "work"}]
    record = Pipeline(store_at(tmp_path), FixtureProvider(raw)).run(
        case, experiment_id="e", commit="c")
    return record.outputs["grading"], record.outputs["transcript_quality_gate"]


def test_clear_text_stays_validated_and_the_score_is_unchanged(tmp_path):
    grade, flag = _run(tmp_path, "x + 1 = 2\n2·y = 8\ny = 4", gate="on")
    assert flag == "on"
    assert grade["status"] == "validated"
    assert grade["total_score"] == 4
    assert grade["items"][0]["score"] == 4
    assert grade["failure_reason"] == ""


def test_garbling_routes_to_review_without_changing_the_score(tmp_path):
    samples = ["x + # = 2", "(x+1", r"\frac{1}{2", "x\u044b = 2", "x + 1 = 2\ufffd"]
    for answer in samples:
        reasons = transcript_reasons(answer)
        assert reasons == ["transcript_garbled"], answer
    grade, _flag = _run(tmp_path, "x + # = 2", gate="on")
    assert grade["status"] == "review_required"
    assert grade["failure_reason"] == "transcript_garbled"
    assert grade["total_score"] == 4
    assert grade["items"][0]["score"] == 4
    assert any(check["name"] == "transcript_garbled" and check["ok"] is False for check in grade["checks"])


def test_gate_off_does_not_route_a_garbled_transcript(tmp_path):
    grade, flag = _run(tmp_path, "x + # = 2", gate="off")
    assert flag == "off"
    assert grade["status"] == "validated"
    assert grade["total_score"] == 4
    assert "transcript_garbled" not in grade["failure_reason"]


def test_consecutive_equations_that_disagree_are_inconsistent():
    assert transcript_reasons("2·y = 6\ny = 4") == ["transcript_inconsistent"]
    assert transcript_reasons("k·y = c\ny = d") == []
    assert transcript_reasons("2·y = 8\ny = 4") == []
    assert transcript_reasons("xは2") == []


def test_inconsistent_transcript_keeps_the_model_score(tmp_path):
    answer = "2·y = 6\ny = 4"
    grade, _flag = _run(tmp_path, answer, gate="on")
    assert grade["status"] == "review_required"
    assert grade["failure_reason"] == "transcript_inconsistent"
    assert grade["total_score"] == 4
    assert grade["items"][0]["score"] == 4


def test_uncertainty_with_a_soft_flag_routes_to_review(tmp_path):
    soft = "x = 2?"
    assert transcript_reasons(soft) == []
    assert transcript_reasons(soft, low_confidence=True) == ["transcript_garbled"]
    plain, _flag = _run(tmp_path, soft, gate="on", uncertainty=0.2)
    assert plain["status"] == "validated"
    assert plain["total_score"] == 4
    uncertain = tmp_path / "uncertain"
    clean_dir = tmp_path / "clean"
    uncertain.mkdir()
    clean_dir.mkdir()
    flagged, _flag = _run(uncertain, soft, gate="on", uncertainty=0.6)
    assert flagged["status"] == "review_required"
    assert "transcript_garbled" in flagged["failure_reason"]
    assert flagged["items"][0]["score"] == 4
    assert flagged["total_score"] == 4
    clean, _flag = _run(clean_dir, "x = 2", gate="on", uncertainty=0.6)
    assert "transcript_garbled" not in clean["failure_reason"]
    assert clean["items"][0]["score"] == 4


def test_low_reported_confidence_with_a_fullwidth_mix(tmp_path):
    mixed = "A\uff21 = 1"
    assert transcript_reasons(mixed) == []
    grade, _flag = _run(tmp_path, mixed, gate="on", uncertainty=0.0, confidence=0.2)
    assert grade["status"] == "review_required"
    assert grade["failure_reason"] == "transcript_garbled"
    assert grade["total_score"] == 4


def test_superscripts_are_exponents_and_do_not_fail_the_grade(tmp_path):
    assert transcript_reasons("x\u00b2 = 4\nx = 2") == []
    assert transcript_reasons("2\u00b3 = 8\n2\u00b2 = 4") == []
    assert transcript_reasons("10\u207b\u00b2 = 0.01\ny = 1") == []
    assert transcript_reasons("x\u00b2 = 4\nx = 3") == ["transcript_inconsistent"]
    grade, _flag = _run(tmp_path, "x\u00b2 = 4\nx = 2", gate="on")
    assert grade["status"] == "validated"
    assert grade["failure_reason"] == ""
    assert grade["total_score"] == 4
    assert grade["items"][0]["score"] == 4
    disagree = tmp_path / "disagree"
    disagree.mkdir()
    flagged, _flag = _run(disagree, "x\u00b2 = 4\nx = 3", gate="on")
    assert flagged["status"] == "review_required"
    assert flagged["failure_reason"] == "transcript_inconsistent"
    assert flagged["total_score"] == 4
    assert flagged["items"][0]["score"] == 4


def test_a_gate_error_records_the_error_and_keeps_the_grade(tmp_path, monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("gate blew up")

    monkeypatch.setattr("harness_v2.grading.transcript_quality.transcript_reasons", boom)
    grade, flag = _run(tmp_path, "x + 1 = 2", gate="on")
    assert flag == "on"
    assert grade["status"] == "validated"
    assert grade["total_score"] == 4
    assert grade["items"][0]["score"] == 4
    assert grade["failure_reason"] == ""
    assert "transcript_garbled" not in grade["failure_reason"]
    errors = [check for check in grade["checks"] if check["name"] == "gate_error"]
    assert errors == [{"name": "gate_error", "ok": False, "detail": "RuntimeError"}]
