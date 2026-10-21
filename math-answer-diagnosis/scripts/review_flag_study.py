#!/usr/bin/env python3
"""v1.2 の既存採点へ規則フラグをかけ、引き金ごとの件数と例を出す。

答案の本文が添付に無い行では、OCR と根拠の照合はしない。
参照得点は使わない。配点は問題の満点だけを使う。

    python3 scripts/review_flag_study.py --csv results_v1.2.csv --outputs outputs/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.grading.parse import (  # noqa: E402
    ExpectedItem,
    GradingOutputError,
    parse_grading_text,
    validate_grading,
)
from src.grading.review_flags import evaluate_review_flags  # noqa: E402

# 問題の配点。教師の参照得点ではない。
RUBRIC: dict[str, list[ExpectedItem]] = {
    "1-A": [ExpectedItem("R1", 4), ExpectedItem("R2", 4), ExpectedItem("R3", 2)],
    "2-B": [
        ExpectedItem("R1", 1),
        ExpectedItem("R2", 3),
        ExpectedItem("R3", 2),
        ExpectedItem("R4", 2),
        ExpectedItem("R5", 2),
    ],
    "3-A": [ExpectedItem(f"R{index}", 2) for index in range(1, 6)],
}


def _trigger(reason: str) -> str:
    if reason.startswith("json_parse_failed"):
        return "json_parse_failed"
    if reason.startswith("schema:"):
        return "schema_invalid"
    if reason.startswith("items_mismatch"):
        return "items_mismatch"
    if reason.startswith("score_out_of_range") or reason.startswith("item_max_mismatch") or reason == "score_not_number":
        return "score_range"
    if reason.startswith("score_sum_mismatch"):
        return "score_sum_mismatch"
    if reason == "score_reason_mismatch" or "理由は" in reason:
        return "score_reason_mismatch"
    if reason == "key_formula_unreadable" or reason.startswith("key_formula_unreadable"):
        return "key_formula_unreadable"
    if reason.startswith("evidence_not_in_input"):
        return "evidence_not_in_input"
    return reason


def _split_labeled(field: str) -> dict[str, str]:
    found: dict[str, str] = {}
    if not field:
        return found
    for part in field.split(" | "):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        found[key.strip()] = value.strip()
    return found


def reconstruct_from_csv(row: dict[str, str]) -> dict[str, Any]:
    """生の出力が無い行を、CSV に残った項目・根拠・理由から組み立てる。"""
    evidence = _split_labeled(row.get("evidence") or "")
    reasons = _split_labeled(row.get("reasons") or "")
    items: list[dict[str, Any]] = []
    for part in (row.get("items") or "").split(";"):
        piece = part.strip()
        if "=" not in piece or "/" not in piece:
            continue
        item_id, rest = piece.split("=", 1)
        score_text, max_text = rest.split("/", 1)
        key = item_id.strip()
        items.append(
            {
                "item_id": key,
                "score": int(score_text),
                "max_score": int(max_text),
                "evidence": evidence.get(key, ""),
                "reason": reasons.get(key, ""),
            }
        )
    score_text = (row.get("score_after_fence") or row.get("score_strict") or "").strip()
    score = int(score_text) if score_text else sum(item["score"] for item in items)
    model: dict[str, Any] = {}
    raw_flags = row.get("model_flags") or ""
    if raw_flags:
        try:
            loaded = json.loads(raw_flags)
        except json.JSONDecodeError:
            loaded = None
        if isinstance(loaded, dict):
            model = loaded
    return {
        "question_id": row.get("question") or "",
        "rubric_scores": items,
        "score": score,
        "max_score": sum(item["max_score"] for item in items),
        "needs_review": bool(model.get("needs_review")),
        "grading_uncertain": bool(model.get("grading_uncertain")),
        "ocr_uncertain": bool(model.get("ocr_uncertain")),
        "review_reason": str(model.get("review_reason") or ""),
    }


def load_outputs(directory: Path | None) -> dict[str, dict[str, Any]]:
    if directory is None or not directory.is_dir():
        return {}
    found: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.json")):
        found[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return found


def load_answers(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("answers file must be a JSON object of id -> text")
    return {str(key): str(value) for key, value in loaded.items()}


def _answer_for(row: dict[str, str], answers: dict[str, str]) -> str | None:
    for key in (row.get("run_id"), row.get("input_sha256"), row.get("result_from")):
        if key and key in answers:
            return answers[key]
    return None


def _bump_example(examples: dict[str, list[dict[str, str]]], code: str, run_id: str, detail: str) -> None:
    bucket = examples[code]
    if len(bucket) < 3:
        bucket.append({"run_id": run_id, "detail": detail[:180]})


def recorded_prototype(rows: list[dict[str, str]]) -> dict[str, Any]:
    """添付 CSV に残っている試作フラグの内訳。新しい規則の再計算ではない。"""
    review_codes: dict[str, int] = defaultdict(int)
    ocr_tokens: dict[str, int] = defaultdict(int)
    examples: dict[str, list[dict[str, str]]] = defaultdict(list)
    needs_review = 0
    evidence_rows = 0
    for row in rows:
        run_id = row.get("run_id") or ""
        if (row.get("needs_review") or "").strip() == "True":
            needs_review += 1
        for part in (row.get("review_reasons") or "").split(";"):
            code = part.strip()
            if not code:
                continue
            label = code.split(":", 1)[0]
            review_codes[label] += 1
            _bump_example(examples, label, run_id, code)
        for part in (row.get("ocr_reasons") or "").split(";"):
            token = part.strip()
            if token.startswith("OCR疑い:"):
                token = token[len("OCR疑い:") :]
            if not token:
                continue
            ocr_tokens[token] += 1
            _bump_example(examples, f"ocr:{token}", run_id, part.strip())
        missing = (row.get("evidence_missing") or "").strip()
        if missing:
            evidence_rows += 1
            _bump_example(examples, "evidence_missing", run_id, missing)
    return {
        "needs_review": needs_review,
        "review_reason_codes": dict(sorted(review_codes.items(), key=lambda item: (-item[1], item[0]))),
        "ocr_reason_tokens": dict(sorted(ocr_tokens.items(), key=lambda item: (-item[1], item[0]))),
        "evidence_missing_rows": evidence_rows,
        "examples": examples,
    }


def summarize(
    rows: list[dict[str, str]],
    outputs: dict[str, dict[str, Any]],
    answers: dict[str, str] | None = None,
) -> dict[str, Any]:
    answers = answers or {}
    counts: dict[str, int] = defaultdict(int)
    examples: dict[str, list[dict[str, str]]] = defaultdict(list)
    needs_review = 0
    fence_stripped = 0
    fence_and_clean = 0
    text_missing = 0
    checked_against_answer = 0
    model_needs_review = 0
    model_grading_uncertain = 0
    model_ocr_uncertain = 0
    by_source = defaultdict(int)

    for row in rows:
        run_id = row.get("run_id") or ""
        question = row.get("question") or ""
        expected = RUBRIC.get(question)
        record = outputs.get(run_id) or outputs.get(row.get("result_from") or "")
        parse_error = None
        fence = False
        obj: dict[str, Any] | None
        if record and isinstance(record.get("text"), str):
            by_source["raw_text"] += 1
            try:
                parsed = parse_grading_text(record["text"])
            except GradingOutputError as exc:
                parse_error = str(exc)
                fence = exc.fence_stripped
                obj = None
            else:
                obj = parsed.data
                fence = parsed.fence_stripped
                question = str(obj.get("question_id") or question)
                expected = RUBRIC.get(question, expected)
        else:
            by_source["csv_only"] += 1
            text_missing += 1
            obj = reconstruct_from_csv(row)
            if row.get("fence") == "True":
                by_source["csv_fence_not_reparsed"] += 1
        answer = _answer_for(row, answers)
        if answer is not None:
            checked_against_answer += 1
        validation: list[str] = []
        if obj is not None and expected is not None:
            validation = validate_grading(obj, expected).errors
        elif obj is not None:
            validation = validate_grading(obj).errors
        flags = evaluate_review_flags(
            obj,
            answer_text=answer,
            validation_errors=validation,
            fence_stripped=fence,
            parse_error=parse_error,
        )
        if fence:
            fence_stripped += 1
            if not flags.needs_review:
                fence_and_clean += 1
        if flags.needs_review:
            needs_review += 1
        if obj is not None:
            model_needs_review += int(bool(obj.get("needs_review")))
            model_grading_uncertain += int(bool(obj.get("grading_uncertain")))
            model_ocr_uncertain += int(bool(obj.get("ocr_uncertain")))
        seen: set[str] = set()
        for reason in flags.review_reasons:
            code = _trigger(reason)
            if code in seen:
                continue
            seen.add(code)
            counts[code] += 1
            if len(examples[code]) < 3:
                examples[code].append({"run_id": run_id, "detail": reason[:180]})

    ordered = dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
    return {
        "rows": len(rows),
        "needs_review": needs_review,
        "fence_stripped": fence_stripped,
        "fence_stripped_without_review": fence_and_clean,
        "text_missing": text_missing,
        "answer_text_checked": checked_against_answer,
        "answer_text_skipped": len(rows) - checked_against_answer,
        "model_needs_review": model_needs_review,
        "model_grading_uncertain": model_grading_uncertain,
        "model_ocr_uncertain": model_ocr_uncertain,
        "sources": dict(by_source),
        "triggers": ordered,
        "examples": {key: examples[key] for key in ordered},
        "recorded_prototype": recorded_prototype(rows),
        "note": (
            "OCR と根拠の照合は答案本文がある行だけ行う。"
            "フェンスを外しただけでは needs_review にしない。"
            "参照得点は使っていない。"
        ),
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, default=None)
    parser.add_argument("--answers", type=Path, default=None, help="run_id または sha256 から答案本文への JSON")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    summary = summarize(_read_csv(args.csv), load_outputs(args.outputs), load_answers(args.answers))
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
