"""LaTeX backslashes inside model JSON stay backslashes.

A JSON newline stays a newline. A newline that also spells a known command
is not rewritten; the grade is marked ambiguous.
"""
import json

from harness_v2.core.contracts import RubricItem
from harness_v2.grading.scorer import COMPLETION_PARSER_VERSION, items_from_completion


def _raw(reason: str, evidence: str = "x") -> str:
    return (
        '{"items":[{"item_id":"R1","score":1,"evidence_spans":[{"text":"' + evidence + '"}],'
        '"math_reason":"' + reason + '","uncertainty":0,"review_reason":""}],'
        '"total_score":1}'
    )


def _parsed(reason: str, evidence: str = "x"):
    items, _total, fence, repaired, ambiguous = items_from_completion(
        _raw(reason, evidence), [RubricItem("R1", 1, "work")])
    return items[0], fence, repaired, ambiguous


def test_known_commands_stay_literal_and_invalid_escapes_are_doubled():
    strict = json.loads(_raw(r"\frac{1}{2}"))
    assert strict["items"][0]["math_reason"].startswith("\f")

    samples = {
        r"\le": r"\le",
        r"\frac{1}{2}": r"\frac{1}{2}",
        r"\times": r"\times",
        r"\theta": r"\theta",
        r"\tan ": r"\tan ",
        r"\to ": r"\to ",
        r"\triangle": r"\triangle",
        r"\text{m}": r"\text{m}",
        r"\tfrac{1}{2}": r"\tfrac{1}{2}",
        r"\beta": r"\beta",
        r"\bar{A}": r"\bar{A}",
        r"\boxed{x=1}": r"\boxed{x=1}",
        r"\begin{aligned}a\end{aligned}": r"\begin{aligned}a\end{aligned}",
        r"\right)": r"\right)",
        r"\rangle": r"\rangle",
        r"\rho": r"\rho",
        r"\left(x\right)": r"\left(x\right)",
    }
    for source, expected in samples.items():
        item, fence, repaired, ambiguous = _parsed(source)
        assert item.math_reason == expected
        assert fence is False
        assert repaired is True
        assert ambiguous is False
        assert "\f" not in item.math_reason and "\b" not in item.math_reason
        assert "\t" not in item.math_reason and "\r" not in item.math_reason
        assert "\n" not in item.math_reason


def test_newline_command_collisions_keep_the_newline_and_are_ambiguous():
    neq, _fence, repaired, ambiguous = _parsed(r"\neq")
    assert neq.math_reason == "\neq"
    assert "\\" not in neq.math_reason
    assert repaired is False
    assert ambiguous is True

    nu, _fence, repaired, ambiguous = _parsed(r"\nu")
    assert nu.math_reason == "\nu"
    assert repaired is False
    assert ambiguous is True

    # The same bytes as a newline plus the rest of the name. Not rewritten.
    for source, expected in ((r"\nabla", "\nabla"), (r"\not", "\not"), (r"\notin", "\notin")):
        item, _fence, repaired, ambiguous = _parsed(source)
        assert item.math_reason == expected
        assert "\\" not in item.math_reason
        assert repaired is False
        assert ambiguous is True

    # The command is not bounded, so this is a newline plus the word.
    prose, _fence, repaired, ambiguous = _parsed("x=1\\nequation")
    assert prose.math_reason == "x=1\nequation"
    assert repaired is False
    assert ambiguous is False


def test_genuine_newlines_tabs_escaped_backslash_and_unicode_are_not_rewritten():
    item, _fence, repaired, ambiguous = _parsed("line one.\\n next")
    assert item.math_reason == "line one.\n next"
    assert repaired is False
    assert ambiguous is False

    ended, _fence, repaired, ambiguous = _parsed("ends\\n")
    assert ended.math_reason == "ends\n"
    assert repaired is False
    assert ambiguous is False

    lines, _fence, repaired, ambiguous = _parsed("x=1\\nx+1=2", evidence="x=1\\nx+1=2")
    assert lines.math_reason == "x=1\nx+1=2"
    assert lines.evidence_spans[0].text == "x=1\nx+1=2"
    assert "\\" not in lines.evidence_spans[0].text
    assert repaired is False
    assert ambiguous is False

    tabbed, _fence, repaired, ambiguous = _parsed("\\tanswer")
    assert tabbed.math_reason == "\tanswer"
    assert repaired is False
    assert ambiguous is False

    spaced, _fence, repaired, ambiguous = _parsed("\\t next")
    assert spaced.math_reason == "\t next"
    assert repaired is False
    assert ambiguous is False

    # Backspace and form feed stay controls only when no letter follows.
    bare_b, _fence, repaired, ambiguous = _parsed("\\b.")
    assert bare_b.math_reason == "\b."
    assert repaired is False
    assert ambiguous is False
    bare_f, _fence, repaired, ambiguous = _parsed("\\f ")
    assert bare_f.math_reason == "\f "
    assert repaired is False
    assert ambiguous is False

    already, _fence, repaired, ambiguous = _parsed(r"\\frac{1}{2}")
    assert already.math_reason == r"\frac{1}{2}"
    assert repaired is False
    assert ambiguous is False

    letter, _fence, repaired, ambiguous = _parsed(r"\u03b2")
    assert letter.math_reason == "β"
    assert repaired is False
    assert ambiguous is False

    mixed, _fence, repaired, ambiguous = _parsed("\\\\neq 1.\\n Next")
    assert mixed.math_reason == "\\neq 1.\n Next"
    assert repaired is False
    assert ambiguous is False


def test_fenced_latex_is_repaired_after_the_fence_is_removed():
    raw = "```json\n" + _raw(r"\leq 2") + "\n```"
    items, _total, fence, repaired, ambiguous = items_from_completion(
        raw, [RubricItem("R1", 1, "work")])
    assert fence is True
    assert repaired is True
    assert ambiguous is False
    assert items[0].math_reason == r"\leq 2"


def test_pipeline_keeps_a_multiline_quote_validated(tmp_path):
    from harness_v2.core.pipeline import Pipeline
    from harness_v2.providers.fixture import FixtureProvider
    from harness_v2.tests.test_p3_setup import store_at
    from harness_v2.tests.test_pipeline import _case

    answer = "x=1\nx+1=2"
    raw = json.dumps({
        "items": [{
            "item_id": "R1",
            "score": 1,
            "evidence_spans": [{"text": answer}],
            "math_reason": "two lines",
            "uncertainty": 0,
            "review_reason": "",
        }],
        "total_score": 1,
    })
    assert "\\n" in raw and "\n" not in raw
    case = _case("lines", raw)
    case["answer_text"] = answer
    case["rubric"] = [{"item_id": "R1", "max_points": 1, "description": "work"}]
    record = Pipeline(store_at(tmp_path), FixtureProvider(raw)).run(
        case, experiment_id="e", commit="c")
    grade = record.outputs["grading"]
    assert grade["status"] == "validated"
    assert grade["failure_reason"] == ""
    assert grade["parse_repaired"] is False
    assert grade["parse_ambiguous"] is False
    quote = grade["items"][0]["evidence_spans"][0]["text"]
    assert quote == answer
    assert "\n" in quote and "\\" not in quote
    assert any(check["name"] == "evidence_traceable" and check["ok"] for check in grade["checks"])


def test_backslash_and_formfeed_commands_stay_in_reason_and_evidence(tmp_path):
    from harness_v2.core.pipeline import Pipeline
    from harness_v2.providers.fixture import FixtureProvider
    from harness_v2.tests.test_p3_setup import store_at
    from harness_v2.tests.test_pipeline import _case

    quote = (
        r"\boxed{x=1}"
        r"\begin{aligned}a\end{aligned}"
        r"\bar{A}\beta\frac{1}{2}"
        r"\left(x\right)\text{m}"
    )
    raw = _raw(quote, evidence=quote)
    strict = json.loads(_raw(r"\boxed{x=1}\frac{1}{2}"))
    strict_reason = strict["items"][0]["math_reason"]
    assert strict_reason.startswith("\b") and "\f" in strict_reason
    assert r"\boxed" not in strict_reason and r"\frac" not in strict_reason
    item, _fence, repaired, ambiguous = _parsed(quote, evidence=quote)
    assert item.math_reason == quote
    assert item.evidence_spans[0].text == quote
    assert "\b" not in item.math_reason and "\f" not in item.math_reason
    assert repaired is True and ambiguous is False

    case = _case("latex", raw)
    case["answer_text"] = quote
    case["rubric"] = [{"item_id": "R1", "max_points": 1, "description": "work"}]
    record = Pipeline(store_at(tmp_path), FixtureProvider(raw)).run(
        case, experiment_id="e", commit="c")
    grade = record.outputs["grading"]
    assert grade["status"] == "validated"
    assert grade["failure_reason"] == ""
    assert grade["parse_repaired"] is True
    assert grade["parse_ambiguous"] is False
    assert grade["items"][0]["math_reason"] == quote
    assert grade["items"][0]["evidence_spans"][0]["text"] == quote
    assert "\b" not in grade["items"][0]["math_reason"]
    assert "\f" not in grade["items"][0]["math_reason"]
    assert any(check["name"] == "evidence_traceable" and check["ok"] for check in grade["checks"])


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
    assert record.outputs["grading"]["parse_ambiguous"] is False
    assert record.outputs["grading"]["items"][0]["math_reason"] == r"\frac{1}{2}"
    assert "\f" not in record.outputs["grading"]["items"][0]["math_reason"]


def test_ambiguous_newline_command_is_review_required(tmp_path):
    from harness_v2.core.pipeline import Pipeline
    from harness_v2.providers.fixture import FixtureProvider
    from harness_v2.tests.test_p3_setup import store_at
    from harness_v2.tests.test_pipeline import _case

    raw = json.dumps({
        "items": [{
            "item_id": "R1",
            "score": 1,
            "evidence_spans": [{"text": "x"}],
            "math_reason": "\neq",
            "uncertainty": 0,
            "review_reason": "",
        }],
        "total_score": 1,
    })
    assert "\\neq" in raw
    case = _case("amb", raw)
    case["answer_text"] = "x"
    case["rubric"] = [{"item_id": "R1", "max_points": 1, "description": "work"}]
    record = Pipeline(store_at(tmp_path), FixtureProvider(raw)).run(
        case, experiment_id="e", commit="c")
    grade = record.outputs["grading"]
    assert grade["status"] == "review_required"
    assert grade["failure_reason"] == "parse_ambiguous"
    assert grade["parse_ambiguous"] is True
    assert grade["parse_repaired"] is False
    assert grade["items"][0]["math_reason"] == "\neq"
    assert "\\" not in grade["items"][0]["math_reason"]
    assert grade["items"][0]["evidence_spans"][0]["text"] == "x"


def test_parser_version_changes_reuse_identity_and_is_not_caller_selectable(tmp_path):
    from harness_v2.evaluation.dataset import load_dataset
    from harness_v2.run import identity, load_config, request_id_for, reusable_config
    from harness_v2.tests.test_p3_dataset import make_dataset
    from harness_v2.tests.test_p3_runner import config_at

    assert COMPLETION_PARSER_VERSION == "json_latex_escapes_v3"
    path = config_at(tmp_path)
    loaded = load_config(path)
    assert loaded["completion_parser_version"] == COMPLETION_PARSER_VERSION
    dataset = load_dataset(make_dataset(tmp_path))
    case = dataset.cases[0]
    current = request_id_for(dataset, case, loaded, 1)
    stale = {**loaded, "completion_parser_version": "json_latex_escapes_v2"}
    assert request_id_for(dataset, case, stale, 1) != current
    assert identity({"config": reusable_config(loaded), "dataset": dataset.manifest, "commit": "c"}) != (
        identity({"config": reusable_config(stale), "dataset": dataset.manifest, "commit": "c"}))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["completion_parser_version"] = "json_latex_escapes_v2"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_config(path)["completion_parser_version"] == COMPLETION_PARSER_VERSION
