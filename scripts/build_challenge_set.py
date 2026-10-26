"""Build the reference-free Phase 5 robustness challenge set (Cycle B).

Reads only the manual transcriptions, problems and rubrics; never a reference
file. Variants per manual case (seed frozen):
  O  original (contemporaneous control)
  F  every ASCII digit -> full-width digit (formatting transform, math included)
  S  one space on both sides of each ASCII = + - < > not already space-adjacent
  N  delete k = max(1, round(0.03 * n)) CJK characters (n = eligible count);
     produced only when n >= 1
No-op variants are dropped. Variants carry a manipulation label only; they are
not claimed to be ambiguous or to need review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness_v2.evaluation.dataset import write_dataset  # noqa: E402
from harness_v2.storage.permissions import private_write_text  # noqa: E402

SEED = 20261024
NOISE_RATE = 0.03
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]")
_OPS = "=+-<>"


def full_width_digits(text: str) -> str:
    return "".join(chr(ord(ch) - 0x30 + 0xFF10) if "0" <= ch <= "9" else ch for ch in text)


def space_operators(text: str) -> str:
    out = []
    for i, ch in enumerate(text):
        if ch in _OPS:
            if out and not out[-1].isspace():
                out.append(" ")
            out.append(ch)
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if nxt and not nxt.isspace():
                out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


def delete_cjk(text: str, rng: random.Random) -> tuple[str, int, int]:
    positions = [m.start() for m in _CJK.finditer(text)]
    if not positions:
        return text, 0, 0
    k = max(1, round(NOISE_RATE * len(positions)))
    drop = set(rng.sample(positions, k))
    return "".join(ch for i, ch in enumerate(text) if i not in drop), len(positions), k


def variants(case_id: str, text: str) -> tuple[list[tuple[str, str]], dict]:
    rng = random.Random(f"{SEED}:{case_id}")
    noisy, n_cjk, k = delete_cjk(text, rng)
    inventory = {"cjk": n_cjk, "digits": sum(ch.isascii() and ch.isdigit() for ch in text),
                 "ascii_operators": sum(ch in _OPS for ch in text), "noise_deleted": k}
    out, seen = [("O", text)], {text}
    for label, value in (("F", full_width_digits(text)), ("S", space_operators(text)),
                         ("N", noisy if n_cjk else text)):
        if value in seen:
            inventory[f"{label}_dropped"] = "zero eligible" if label == "N" and not n_cjk else "no-op"
            continue
        seen.add(value)
        out.append((label, value))
    inventory["produced"] = [label for label, _ in out]
    return out, inventory


def manual_cases(kit: Path, c1_path: Path) -> list[dict]:
    def read(path):
        return json.loads(Path(path).read_bytes().decode("utf-8"))
    rubrics, questions = read(kit / "rubric.json"), read(kit / "questions.json")
    manual = {row["question_id"]: row["answer_text"] for row in read(kit / "manual_transcriptions.json")}
    c1 = read(c1_path)
    cases = []
    for problem_id, rows in rubrics.items():
        rubric = [{"item_id": r["id"], "max_points": r["max_score"],
                   "description": f"{r['label']}。{r['rule']}"} for r in rows]
        common = {"problem_id": problem_id, "problem_text": questions[problem_id]["question"],
                  "rubric": rubric, "rubric_version": "v1.1"}
        has_c1 = isinstance(c1.get(problem_id), dict) and isinstance(c1[problem_id].get("answer_text"), str)
        cases.append({**common, "source": "manual_C0" if has_c1 else "manual", "answer_text": manual[problem_id]})
        if has_c1:
            cases.append({**common, "source": "manual_C1", "answer_text": c1[problem_id]["answer_text"]})
    return cases


def build_challenge_set(out: Path, *, kit: Path, c1_path: Path) -> Path:
    rows, inventory = [], {}
    for case in manual_cases(kit, c1_path):
        source_id = f"{case['problem_id']}:{case['source']}"
        produced, inv = variants(source_id, case["answer_text"])
        inventory[source_id] = inv
        for label, text in produced:
            rows.append({k: case[k] for k in ("problem_id", "problem_text", "rubric", "rubric_version")}
                        | {"input_version": f"{case['source']}:{label}", "answer_text": text,
                           "ocr_metadata": {"engine": "manual", "challenge_source": source_id,
                                            "challenge_variant": label, "challenge_seed": SEED,
                                            "source_sha256": hashlib.sha256(case["answer_text"].encode()).hexdigest()}})
    manifest = write_dataset(out, dataset_id="harness-v2-challenge", version="1", cases=rows,
                             reference_scores={}, reference_items={})
    private_write_text(Path(out) / "inventory.json",
                       json.dumps(inventory, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kit", type=Path, default=Path("/workspace/experiment_v1_1/kit/data"))
    parser.add_argument("--c1", type=Path, default=Path("/workspace/experiment_v1_2/transcription_C1_v1.2.json"))
    args = parser.parse_args(argv)
    print(build_challenge_set(args.out, kit=args.kit, c1_path=args.c1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
