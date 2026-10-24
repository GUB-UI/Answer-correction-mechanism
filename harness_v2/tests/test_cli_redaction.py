from __future__ import annotations

import json
from pathlib import Path

from harness_v2.__main__ import main
from harness_v2.storage.secrets import redact


def test_redact_parses_json_embedded_in_a_string() -> None:
    cleaned = redact('prefix {"api_key":"embedded-secret-value","note":"sk-FAKEKEY123456"}')
    assert "embedded-secret-value" not in cleaned
    assert "sk-FAKEKEY123456" not in cleaned


def test_result_json_comes_from_the_redacted_record(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "sk-FAKEKEY123456")
    case = {
        "input_id": "c1",
        "request_id": "req1",
        "problem_text": "solve sk-FAKEKEY123456",
        "answer_text": "x+1",
        "rubric_version": "v0",
        "prompt_version": "v0",
        "rubric": [{"item_id": "R1", "max_points": 4, "description": "work", "acceptable_expression": "1+x"}],
        "new_completion": json.dumps(
            {
                "items": [
                    {
                        "item_id": "R1",
                        "max_points": 4,
                        "score": 4,
                        "evidence_spans": [{"text": "x+1"}],
                        "math_reason": "matches",
                    }
                ],
                "total_score": 4,
                "leak": '{"api_key":"embedded-secret-value","token":"sk-FAKEKEY123456"}',
            }
        ),
    }
    source = tmp_path / "case.json"
    source.write_text(json.dumps(case), encoding="utf-8")
    out = tmp_path / "out"
    assert main(["--input", str(source), "--out", str(out), "--commit", "abc"]) == 0
    secret_needles = (b"sk-FAKEKEY123456", b"embedded-secret-value")
    written = [path for path in out.rglob("*") if path.is_file()]
    assert written
    for path in written:
        blob = path.read_bytes()
        for needle in secret_needles:
            assert needle not in blob, path
