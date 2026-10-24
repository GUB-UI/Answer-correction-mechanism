"""Finding 12: only declared text/rubric fields may be published."""
import json

import pytest

from harness_v2.providers.grokbot_exchange import FileExchange, GitHubHandoff, SchemaError


def request(payload):
    return {"request_id": "r", "kind": "grade", "experiment_id": "e", "commit": "c",
            "rubric_version": "v", "prompt_version": "v", "input_ids": ["i"],
            "payload": payload, "created_at": "now", "independent_evaluation": True}


@pytest.mark.parametrize("payload", [
    {"image_base64": "AAAA"}, {"answer_text": "data:image/png,raw-pixels"},
    {"answer_text": "DATA:IMAGE/svg+xml,%3Csvg%3E"},
    {"answer_text": "data:image/png;base64,AAAA"},
    {"notes": {"nested": {"image_base64": "AAAA"}}},
    {"answer_text": {"image_base64": "AAAA"}},
    {"rubric": [{"item_id": "a", "max_points": 3, "image_base64": "AAAA"}]},
    {"anything": "unrestricted metadata"},
    {"answer_text": "![sheet](https://example.test/sheet.png)"},
])
@pytest.mark.parametrize("delivery", ["file", "github"])
def test_publish_rejects_unallowed_or_image_bearing_fields(tmp_path, payload, delivery):
    target = FileExchange(tmp_path) if delivery == "file" else GitHubHandoff(tmp_path)
    publish = target.publish_request if delivery == "file" else target.write
    with pytest.raises(SchemaError):
        publish(request(payload))
    assert not list(tmp_path.rglob("*.json"))


def test_valid_text_is_blinded_for_independent_publication(tmp_path):
    path = FileExchange(tmp_path).publish_request(request({
        "answer_text": "[-1,1]; score: 4; 8/10点", "problem_text": "total_score: 8",
        "rubric": [{"item_id": "a", "max_points": 10, "description": "work"}],
    }))
    data = json.loads(path.read_text())
    assert "score: 4" not in data["payload"]["answer_text"]
    assert "8/10点" not in data["payload"]["answer_text"]
    assert "[-1,1]" in data["payload"]["answer_text"]
    assert data["payload"]["rubric"][0]["max_points"] == 10
