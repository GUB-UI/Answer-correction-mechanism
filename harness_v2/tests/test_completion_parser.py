"""LaTeX backslashes inside model JSON stay backslashes."""
import json

from harness_v2.core.contracts import RubricItem
from harness_v2.grading.scorer import COMPLETION_PARSER_VERSION, items_from_completion


def _raw(reason: str) -> str:
    return (
        '{"items":[{"item_id":"R1","score":1,"evidence_spans":[{"text":"x"}],'
        '"math_reason":"' + reason + '","uncertainty":0,"review_reason":""}],'
        '"total_score":1}'
    )


def _parsed(reason: str):
    items, _total, fence, repaired = items_from_completion(
        _raw(reason), [RubricItem("R1", 1, "work")])
    return items[0].math_reason, fence, repaired


def test_invalid_and_control_char_latex_escapes_stay_literal():
    strict = json.loads(_raw(r"\frac{1}{2}"))
    assert strict["items"][0]["math_reason"].startswith("\f")

    samples = {
        r"\le": r"\le",
        r"\frac{1}{2}": r"\frac{1}{2}",
        r"\times": r"\times",
        r"\neq": r"\neq",
        r"\beta": r"\beta",
        r"\right)": r"\right)",
    }
    for source, expected in samples.items():
        reason, fence, repaired = _parsed(source)
        assert reason == expected
        assert fence is False
        assert repaired is True
        assert "\f" not in reason and "\b" not in reason
        assert "\t" not in reason and "\r" not in reason


def test_genuine_newlines_escaped_backslash_and_unicode_are_not_rewritten():
    reason, _fence, repaired = _parsed("line one.\\n next")
    assert reason == "line one.\n next"
    assert repaired is False

    ended, _fence, repaired = _parsed("ends\\n")
    assert ended == "ends\n"
    assert repaired is False

    already, _fence, repaired = _parsed(r"\\frac{1}{2}")
    assert already == r"\frac{1}{2}"
    assert repaired is False

    letter, _fence, repaired = _parsed(r"\u03b2")
    assert letter == "β"
    assert repaired is False

    mixed, _fence, repaired = _parsed("\\\\neq 1.\\n Next")
    assert mixed == "\\neq 1.\n Next"
    assert repaired is False


def test_fenced_latex_is_repaired_after_the_fence_is_removed():
    raw = "```json\n" + _raw(r"\leq 2") + "\n```"
    items, _total, fence, repaired = items_from_completion(raw, [RubricItem("R1", 1, "work")])
    assert fence is True
    assert repaired is True
    assert items[0].math_reason == r"\leq 2"


def test_pipeline_records_parse_repaired(tmp_path):
    from harness_v2.core.pipeline import Pipeline
    from harness_v2.providers.fixture import FixtureProvider
    from harness_v2.tests.test_p3_setup import store_at
    from harness_v2.tests.test_pipeline import _case

    raw = (
        '{"items":[{"item_id":"R1","score":4,"evidence_spans":[{"text":"x+1"}],'
        '"math_reason":"\\frac{1}{2}","uncertainty":0.2,"review_reason":""}],'
        '"total_score":4}'
    )
    # The source above is a Python string, so the JSON text contains \frac.
    assert r"\frac" in raw and r"\\frac" not in raw
    store = store_at(tmp_path)
    record = Pipeline(store, FixtureProvider(raw)).run(
        _case("r", raw), experiment_id="e", commit="c")
    assert record.outputs["grading"]["parse_repaired"] is True
    assert record.outputs["grading"]["items"][0]["math_reason"] == r"\frac{1}{2}"
    assert "\f" not in record.outputs["grading"]["items"][0]["math_reason"]


def test_parser_version_changes_reuse_identity_and_is_not_caller_selectable(tmp_path):
    from harness_v2.evaluation.dataset import load_dataset
    from harness_v2.run import identity, load_config, request_id_for, reusable_config
    from harness_v2.tests.test_p3_dataset import make_dataset
    from harness_v2.tests.test_p3_runner import config_at

    path = config_at(tmp_path)
    loaded = load_config(path)
    assert loaded["completion_parser_version"] == COMPLETION_PARSER_VERSION
    dataset = load_dataset(make_dataset(tmp_path))
    case = dataset.cases[0]
    current = request_id_for(dataset, case, loaded, 1)
    stale = {**loaded, "completion_parser_version": "json_plain_v0"}
    assert request_id_for(dataset, case, stale, 1) != current
    assert identity({"config": reusable_config(loaded), "dataset": dataset.manifest, "commit": "c"}) != (
        identity({"config": reusable_config(stale), "dataset": dataset.manifest, "commit": "c"}))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["completion_parser_version"] = "json_plain_v0"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_config(path)["completion_parser_version"] == COMPLETION_PARSER_VERSION
