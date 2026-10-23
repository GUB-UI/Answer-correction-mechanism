from __future__ import annotations

import json

from harness_v2.evaluation.independent import (
    grading_allowlist,
    prepare_independent_input,
    proposal_review_allowlist,
)


def test_allowlist_drops_scores_including_nested_json_strings() -> None:
    payload = {
        "problem_text": 'solve {"reference_score":8,"predicted_score":1}',
        "answer_text": "x+1",
        "predicted_score": 4,
        "reference_score": 99,
        "prior_ai_score": 1,
        "note": "reference_score: 99",
        "rubric": [
            {
                "item_id": "R1",
                "max_points": 4,
                "description": "work",
                "score": 4,
                "acceptable_expression": "1+x",
            }
        ],
    }
    view = prepare_independent_input(payload)
    assert set(view) == {"problem_text", "answer_text", "rubric"}
    blob = json.dumps(view)
    assert "reference_score" not in blob
    assert "predicted_score" not in blob
    assert "prior_ai_score" not in blob
    assert view["rubric"][0]["max_points"] == 4
    assert "score" not in view["rubric"][0]
    assert grading_allowlist(payload) == view


def test_proposal_review_keeps_the_failure_without_scores() -> None:
    payload = {
        "failure_case": {
            "case_id": "c1",
            "problem_id": "p1",
            "input_version": "v1",
            "model_version": "m",
            "harness_version": "0.1.0",
            "rubric_version": "v0",
            "predicted_score": 3,
            "reference_score": 9,
            "rubric_results": [
                {"item_id": "R1", "score": 3, "max_points": 4, "note": '{"predicted_score":3}'}
            ],
            "evidence": [],
            "errors": ["score_range"],
            "uncertainty": 0.2,
            "execution_log": [],
        },
        "diff": "reference_score: 9\n+ prompt line",
    }
    view = proposal_review_allowlist(payload)
    blob = json.dumps(view)
    assert "predicted_score" not in blob
    assert "reference_score" not in blob
    assert view["rubric_results"][0]["max_points"] == 4
    assert "score" not in view["rubric_results"][0]
    assert "prompt line" in view["diff"]
    assert view["case_id"] == "c1"
