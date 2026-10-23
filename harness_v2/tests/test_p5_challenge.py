import importlib.util
import json
from pathlib import Path

from harness_v2.evaluation.challenge import challenge_report
from harness_v2.evaluation.dataset import load_dataset

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("bcs", ROOT / "scripts" / "build_challenge_set.py")
bcs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bcs)


def test_transforms():
    assert bcs.full_width_digits("x=12") == "x=１２"
    assert bcs.space_operators("2x−4<−x+5") == "2x−4 < −x + 5"
    assert bcs.space_operators("a = b") == "a = b"


def test_variants_deterministic_and_zero_eligible():
    a, inv = bcs.variants("p:manual", "√9−√2")
    assert [label for label, _ in a] == ["O", "F"]
    assert inv["N_dropped"] == "zero eligible" and inv["S_dropped"] == "no-op"
    b1, _ = bcs.variants("p:x", "x<2 のとき 最小値 a")
    b2, _ = bcs.variants("p:x", "x<2 のとき 最小値 a")
    assert b1 == b2 and [label for label, _ in b1] == ["O", "F", "S", "N"]
    noisy = dict(b1)["N"]
    assert len(noisy) == len("x<2 のとき 最小値 a") - 1


def _kit(tmp_path):
    kit = tmp_path / "kit"
    kit.mkdir()
    (kit / "rubric.json").write_text(json.dumps({"P": [{"id": "R1", "max_score": 2, "label": "L", "rule": "r"}]}))
    (kit / "questions.json").write_text(json.dumps({"P": {"question": "q"}}))
    (kit / "manual_transcriptions.json").write_text(json.dumps([{"question_id": "P", "answer_text": "x=1 のとき"}]))
    c1 = tmp_path / "c1.json"
    c1.write_text("{}")
    return kit, c1


def test_builder_writes_label_free_private_dataset(tmp_path):
    kit, c1 = _kit(tmp_path)
    manifest = bcs.build_challenge_set(tmp_path / "out", kit=kit, c1_path=c1)
    ds = load_dataset(manifest)
    assert ds.references["reference_scores"] == {} and ds.references["reference_items"] == {}
    assert [c["input_version"] for c in ds.cases] == ["manual:O", "manual:F", "manual:S", "manual:N"]
    assert (manifest.parent.stat().st_mode & 0o777) == 0o700
    assert (manifest.stat().st_mode & 0o777) == 0o600
    assert ((tmp_path / "out" / "inventory.json").stat().st_mode & 0o777) == 0o600


def _run(rid, status, scores):
    items = [{"item_id": k, "score": v} for k, v in scores.items()] if scores is not None else []
    return {"request_id": rid, "outputs": {"grading": {"status": status, "items": items}}}


def test_report_pairs_by_repetition_and_counts_incomplete():
    expected = [{"request_id": f"{v}{r}", "case_id": f"P:manual:{v}", "repetition": r}
                for v in ("O", "F", "N") for r in (1, 2)]
    runs = [_run("O1", "validated", {"R1": 2}), _run("O2", "validated", {"R1": 1}),
            _run("F1", "validated", {"R1": 2}), _run("F2", "review_required", {"R1": 2}),
            _run("N1", "failed", None), _run("N2", "review_required", {"R1": 1})]
    report = challenge_report({"expected_runs": expected, "runs": runs})
    f, n = report["variants"]["F"], report["variants"]["N"]
    assert f["pairs"] == 2 and f["items_equal"] == 1 and f["item_agreement"] == 0.5
    assert f["newly_flagged_relative_to_O"] == 1
    assert n["incomplete_pairs"] == 1 and n["pairs"] == 1 and n["flagged"] == 1
    assert "reference" not in json.dumps(report)
