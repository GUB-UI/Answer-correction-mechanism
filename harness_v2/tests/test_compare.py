from __future__ import annotations

import json
from pathlib import Path

from scripts.compare_old_vs_new import compare


def test_old_and_new_run_in_separate_directories(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    case = {
        "input_id": "c1",
        "request_id": "req1",
        "problem_text": "solve",
        "answer_text": "x+1",
        "rubric_version": "v0",
        "prompt_version": "v0",
        "reference_score": 99,
        "prior_ai_score": 1,
        "rubric": [{"item_id": "R1", "max_points": 4, "description": "work", "acceptable_expression": "1+x"}],
        "legacy_completion": json.dumps({"marker": "OLD_ONLY", "score": 4}),
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
            }
        ),
    }
    (inputs / "c1.json").write_text(json.dumps(case), encoding="utf-8")
    summary = compare(inputs, tmp_path / "out")
    assert summary["rows"][0]["old_exit"] == 0
    assert summary["rows"][0]["new_exit"] == 0
    old = json.loads((tmp_path / "out" / "c1" / "old" / "result.json").read_text(encoding="utf-8"))
    new = (tmp_path / "out" / "c1" / "new" / "result.json").read_text(encoding="utf-8")
    assert old["data"]["marker"] == "OLD_ONLY"
    assert "OLD_ONLY" not in new
    used = json.loads((tmp_path / "out" / "c1" / "old" / "input_used.json").read_text(encoding="utf-8"))
    assert "reference_score" not in used
    assert "prior_ai_score" not in used
    prompt = (tmp_path / "out" / "c1" / "new" / "artifacts" / "prompts" / "req1.txt").read_text(encoding="utf-8")
    assert "99" not in prompt
    assert "reference_score" not in prompt
