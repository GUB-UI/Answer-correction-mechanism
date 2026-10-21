from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_study():
    path = Path(__file__).resolve().parents[1] / "scripts" / "review_flag_study.py"
    spec = importlib.util.spec_from_file_location("review_flag_study", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_study_counts_a_sum_mismatch_and_skips_ocr_without_answer_text(tmp_path: Path) -> None:
    study = _load_study()
    payload = {
        "question_id": "1-A",
        "rubric_scores": [
            {"item_id": "R1", "score": 4, "max_score": 4, "evidence": "$x=1$", "reason": "過程がある"},
            {"item_id": "R2", "score": 4, "max_score": 4, "evidence": "$y=2$", "reason": "形がある"},
            {"item_id": "R3", "score": 0, "max_score": 2, "evidence": "$z=3$", "reason": "最終答が違う"},
        ],
        "score": 99,
        "max_score": 10,
        "needs_review": False,
        "grading_uncertain": False,
        "ocr_uncertain": False,
    }
    output = tmp_path / "outputs"
    output.mkdir()
    (output / "run_a.json").write_text(
        json.dumps({"text": "```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```"}),
        encoding="utf-8",
    )
    rows = [
        {
            "run_id": "run_a",
            "question": "1-A",
            "result_from": "run_a",
            "input_sha256": "abc",
            "fence": "True",
            "needs_review": "True",
            "review_reasons": "json_fence(コード囲み)",
            "ocr_reasons": "OCR疑い:```markdown",
            "evidence_missing": "",
        }
    ]
    summary = study.summarize(rows, study.load_outputs(output), {})
    assert summary["rows"] == 1
    assert summary["fence_stripped"] == 1
    assert summary["triggers"]["score_sum_mismatch"] == 1
    assert "key_formula_unreadable" not in summary["triggers"]
    assert summary["answer_text_skipped"] == 1
    assert summary["needs_review"] == 1
    assert summary["model_needs_review"] == 0
    assert summary["examples"]["score_sum_mismatch"][0]["run_id"] == "run_a"
    recorded = summary["recorded_prototype"]
    assert recorded["needs_review"] == 1
    assert recorded["review_reason_codes"]["json_fence(コード囲み)"] == 1
    assert recorded["ocr_reason_tokens"]["```markdown"] == 1
