"""Build a private dataset_v2 split into harness inputs.

The dataset directory stays on the caller's machine. Conversion reads
``problem``, ``problem_summary``, and ``rubric`` from ``sol_draft`` and the
OCR transcripts for one split. It does not read ``answers`` or
``reference/scores_v1.json``.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

from harness_v2.evaluation.dataset import write_dataset

DATASET_ID = "dataset_v2"
RUBRIC_VERSION = "dataset_v2"
SPLITS = {"dev": "dev", "holdout": "holdout_sealed"}
TRANSCRIPT_MAP = "transcript_map.json"
TRANSCRIPT_MAP_VERSION = "transcript_map_v1"
_SOURCE_KEYS = ("problem", "problem_summary", "rubric")


class DatasetV2Error(ValueError):
    """A dataset_v2 input is missing, duplicated, or not the split it claims."""


def rubric_source(payload: dict) -> dict[str, Any]:
    """Copy only the rubric fields. ``answers`` and every other key are dropped."""
    if not isinstance(payload, dict):
        raise DatasetV2Error("sol_draft must be an object")
    missing = [key for key in _SOURCE_KEYS if key not in payload]
    if missing:
        raise DatasetV2Error("sol_draft missing " + ", ".join(missing))
    problem = _text(payload["problem"], "problem")
    summary = _text(payload["problem_summary"], "problem_summary")
    rubric = payload["rubric"]
    if not isinstance(rubric, dict):
        raise DatasetV2Error("rubric must be an object")
    subquestions = rubric.get("subquestions")
    if not isinstance(subquestions, list) or not subquestions:
        raise DatasetV2Error(f"rubric subquestions must be a nonempty list: {problem}")
    full_marks = _points(rubric.get("full_marks"), f"{problem} full_marks")
    copied = []
    seen: set[str] = set()
    for sub in subquestions:
        if not isinstance(sub, dict):
            raise DatasetV2Error(f"subquestion must be an object: {problem}")
        item_id = _text(sub.get("id"), f"{problem} subquestion id")
        if item_id in seen:
            raise DatasetV2Error(f"duplicate subquestion: {problem} {item_id}")
        seen.add(item_id)
        points = _points(sub.get("points"), f"{problem} {item_id}")
        model_answer = sub.get("model_answer", "")
        if not isinstance(model_answer, str):
            raise DatasetV2Error(f"model_answer must be a string: {problem} {item_id}")
        criteria = _criteria(sub.get("criteria", []), problem, item_id, points)
        copied.append({
            "id": item_id,
            "points": points,
            "model_answer": model_answer,
            "criteria": criteria,
        })
    allocated = sum(item["points"] for item in copied)
    if not _close(allocated, full_marks):
        raise DatasetV2Error(f"rubric points do not sum to full_marks: {problem}")
    return copy.deepcopy({
        "problem": problem,
        "problem_summary": summary,
        "rubric": {"full_marks": full_marks, "subquestions": copied},
    })


def harness_items(source: dict) -> list[dict]:
    """One harness item per subquestion. Allocations stay on the item, not a score."""
    items = []
    for sub in source["rubric"]["subquestions"]:
        lines = []
        if sub["model_answer"].strip():
            lines.append(sub["model_answer"].strip())
        for criterion in sub["criteria"]:
            lines.append(f"{criterion['desc']} ({_plain(criterion['points'])})")
        items.append({
            "item_id": sub["id"],
            "max_points": sub["points"],
            "description": "\n".join(lines),
            "acceptable_expression": "",
        })
    return items


def convert(dataset_dir: Path, split: str, out: Path, *,
            rubric_version: str = RUBRIC_VERSION) -> Path:
    """Write one split's harness dataset. Reference scores are not an input."""
    if split not in SPLITS:
        raise DatasetV2Error(f"unknown split: {split}")
    if not isinstance(rubric_version, str) or not rubric_version.strip():
        raise DatasetV2Error("rubric_version must be a nonempty string")
    root = Path(dataset_dir)
    loaded = _load_split(root)
    side = loaded[split]
    known_ids = set(loaded["dev"]["ids"]) | set(loaded["holdout"]["ids"])
    stems = _problem_stems(root / "sol_draft")
    transcripts = _transcripts(root, split, side["ids"], known_ids)
    sources: dict[str, dict] = {}
    cases = []
    for answer_id in side["ids"]:
        problem = _problem_for(answer_id, stems)
        if side["problems"] is not None and problem not in side["problems"]:
            raise DatasetV2Error(f"problem is not in the split: {answer_id}")
        writer, _image = _writer_and_image(answer_id, problem)
        if writer not in side["writers"]:
            raise DatasetV2Error(f"writer is not in the split: {answer_id}")
        if problem not in sources:
            sources[problem] = _load_problem(root, problem)
        source = sources[problem]
        raw, filename = transcripts[answer_id]
        cases.append({
            "case_id": answer_id,
            "problem_id": problem,
            "input_version": answer_id,
            "problem_text": f"{source['problem']}\n{source['problem_summary']}",
            "rubric_version": rubric_version,
            "rubric": harness_items(source),
            "grading_input_bytes": raw,
            "ocr_metadata": {
                "source": DATASET_ID,
                "split": split,
                "answer_id": answer_id,
                "transcript": f"ocr_gpu_v1/{SPLITS[split]}/{filename}",
            },
        })
    return write_dataset(
        Path(out),
        dataset_id=DATASET_ID,
        version=f"{loaded['version']}:{split}",
        cases=cases,
        reference_scores={},
        reference_items={},
    )


def _load_problem(root: Path, problem: str) -> dict:
    path = root / "sol_draft" / f"{problem}.json"
    payload = _load_json(path)
    source = rubric_source(payload)
    if source["problem"] != problem:
        raise DatasetV2Error(f"problem field does not match filename: {problem}")
    return source


def _load_split(root: Path) -> dict[str, Any]:
    payload = _load_json(root / "split.json")
    if not isinstance(payload, dict):
        raise DatasetV2Error("split.json must be an object")
    version = payload.get("version")
    if not isinstance(version, str) or not version.strip():
        raise DatasetV2Error("split version must be a nonempty string")
    sides = {name: _split_side(payload.get(name), name) for name in SPLITS}
    overlap = set(sides["dev"]["ids"]) & set(sides["holdout"]["ids"])
    if overlap:
        raise DatasetV2Error("answer id is in both splits: " + sorted(overlap)[0])
    return {"version": version, **sides}


def _split_side(payload: Any, name: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise DatasetV2Error(f"split {name} must be an object")
    ids = _answer_ids(payload.get("ids"), name)
    if "count" in payload and not _count_matches(payload["count"], len(ids)):
        raise DatasetV2Error(f"split count does not match ids: {name}")
    writers = payload.get("writers")
    if (not isinstance(writers, list) or not writers
            or any(not isinstance(item, str) or not item.strip() for item in writers)
            or len(writers) != len(set(writers))):
        raise DatasetV2Error(f"invalid writers: {name}")
    problems = None
    if "problems" in payload:
        raw = payload["problems"]
        if (not isinstance(raw, list)
                or any(not isinstance(item, str) or not item.strip() for item in raw)
                or len(raw) != len(set(raw))):
            raise DatasetV2Error(f"invalid problems: {name}")
        problems = set(raw)
    return {"ids": ids, "writers": set(writers), "problems": problems}


def _answer_ids(values: Any, label: str) -> list[str]:
    if not isinstance(values, list) or not values:
        raise DatasetV2Error(f"{label} ids must be a nonempty list")
    seen: set[str] = set()
    ids = []
    for value in values:
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise DatasetV2Error(f"invalid answer id in {label}")
        if "/" in value or "\\" in value:
            raise DatasetV2Error(f"answer id escapes directory: {value}")
        if value in seen:
            raise DatasetV2Error(f"duplicate answer id: {value}")
        seen.add(value)
        ids.append(value)
    return ids


def _count_matches(count: Any, size: int) -> bool:
    return not isinstance(count, bool) and isinstance(count, int) and count == size


def _problem_stems(directory: Path) -> list[str]:
    if not directory.is_dir():
        raise DatasetV2Error("missing sol_draft")
    stems = []
    for path in directory.glob("*.json"):
        if not path.is_file():
            continue
        stem = path.stem
        if not stem.strip() or stem != stem.strip() or stem in {".", ".."}:
            raise DatasetV2Error(f"invalid problem filename: {path.name}")
        stems.append(stem)
    if not stems:
        raise DatasetV2Error("missing sol_draft")
    return stems


def _problem_for(answer_id: str, stems: list[str]) -> str:
    matches = [stem for stem in stems if answer_id.startswith(stem + "_")]
    if not matches:
        raise DatasetV2Error(f"answer id does not match a problem: {answer_id}")
    matches.sort(key=len, reverse=True)
    return matches[0]


def _writer_and_image(answer_id: str, problem: str) -> tuple[str, str]:
    writer, separator, image = answer_id[len(problem) + 1:].partition("_")
    if not writer or not separator or not image:
        raise DatasetV2Error(f"answer id is not problem_writer_image: {answer_id}")
    return writer, image


def _transcripts(root: Path, split: str, ids: list[str],
                 known_ids: set[str]) -> dict[str, tuple[bytes, str]]:
    directory = root / "ocr_gpu_v1" / SPLITS[split]
    if not directory.is_dir():
        raise DatasetV2Error(f"missing transcript directory: {SPLITS[split]}")
    mapping = _transcript_map(root)
    unexpected_keys = sorted(set(mapping) - known_ids)
    if unexpected_keys:
        raise DatasetV2Error(f"unexpected transcript map id: {unexpected_keys[0]}")
    resolved_root = directory.resolve()
    chosen: dict[str, tuple[bytes, str, Path]] = {}
    for answer_id in ids:
        filename = mapping.get(answer_id) or f"ans_{answer_id}.md"
        path = (directory / filename).resolve()
        if path.parent != resolved_root:
            raise DatasetV2Error(f"transcript map escapes directory: {answer_id}")
        if not path.is_file():
            raise DatasetV2Error(f"missing transcript: {answer_id}")
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise DatasetV2Error(f"missing transcript: {answer_id}") from exc
        if not raw:
            raise DatasetV2Error(f"empty transcript: {answer_id}")
        try:
            raw.decode("utf-8")
        except UnicodeError as exc:
            raise DatasetV2Error(f"transcript is not UTF-8: {answer_id}") from exc
        chosen[answer_id] = (raw, filename, path)
    used = {item[2] for item in chosen.values()}
    for path in directory.glob("ans_*.md"):
        if path.resolve() not in used:
            raise DatasetV2Error(f"unexpected transcript: {path.name}")
    return {answer_id: (raw, filename) for answer_id, (raw, filename, _path) in chosen.items()}


def _transcript_map(root: Path) -> dict[str, str]:
    path = root / "ocr_gpu_v1" / TRANSCRIPT_MAP
    if not path.exists():
        return {}
    payload = _load_json(path)
    if not isinstance(payload, dict) or payload.get("version") != TRANSCRIPT_MAP_VERSION:
        raise DatasetV2Error("invalid transcript map")
    pairs = payload.get("by_answer_id")
    if not isinstance(pairs, dict):
        raise DatasetV2Error("invalid transcript map")
    mapping = {}
    for answer_id, value in pairs.items():
        if not isinstance(answer_id, str) or not answer_id.strip():
            raise DatasetV2Error("invalid transcript map")
        mapping[answer_id] = _mapped_filename(value)
    return mapping


def _mapped_filename(value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise DatasetV2Error("invalid transcript map")
    if "/" in value or "\\" in value or value.startswith("."):
        raise DatasetV2Error("transcript map escapes directory")
    if value.endswith(".md"):
        return value
    return f"ans_{value}.md"


def _criteria(value: Any, problem: str, item_id: str, points: int | float) -> list[dict]:
    if not isinstance(value, list):
        raise DatasetV2Error(f"criteria must be a list: {problem} {item_id}")
    copied = []
    for criterion in value:
        if not isinstance(criterion, dict):
            raise DatasetV2Error(f"criterion must be an object: {problem} {item_id}")
        desc = _text(criterion.get("desc"), f"{problem} {item_id} criterion")
        allocated = _points(criterion.get("points"), f"{problem} {item_id} criterion")
        copied.append({"desc": desc, "points": allocated})
    if copied and not _close(sum(item["points"] for item in copied), points):
        raise DatasetV2Error(f"criteria points do not sum: {problem} {item_id}")
    return copied


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetV2Error(f"{label} must be a nonempty string")
    return value


def _points(value: Any, label: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
        raise DatasetV2Error(f"invalid points: {label}")
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _plain(value: int | float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-6)


def _load_json(path: Path) -> Any:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise DatasetV2Error(f"unreadable JSON: {path.name}") from exc
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetV2Error(f"invalid JSON: {path.name}") from exc
