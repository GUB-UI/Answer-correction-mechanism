"""Build and score a private dataset_v2 split.

The dataset directory stays on the caller's machine. Conversion reads
``problem``, ``problem_summary``, and ``rubric`` from ``sol_draft`` and the
OCR transcripts for one split. It does not read ``answers`` or
``reference/scores_v1.json``. Evaluation reads that score file only while
comparing grader output, and it does not fill in a missing score.

Grade one split with the existing Gemma or fixture provider::

    python -m harness_v2.evaluation.dataset_v2 baseline \\
        --dataset-dir "$DATASET_DIR" --split dev --provider gemma --out "$OUT"
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Callable

from harness_v2.evaluation.dataset import write_dataset
from harness_v2.evaluation.reference import optional_score
from harness_v2.providers.gemma_ai_studio import MODEL_ID
from harness_v2.run import run_experiment
from harness_v2.storage.permissions import private_directory, private_write_text

DATASET_ID = "dataset_v2"
RUBRIC_VERSION = "dataset_v2"
DEFAULT_PROMPT = "harness_v2_ambiguity_aware_v1"
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
    stems = _problem_stems(root / "sol_draft")
    transcripts = _transcripts(
        root, split, side["ids"], loaded["dev"]["ids"], loaded["holdout"]["ids"],
    )
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


def baseline(dataset_dir: Path, split: str, out: Path, *, provider: str,
             model: str | None = None, prompt_version: str = DEFAULT_PROMPT,
             generation_settings: dict | None = None, repeats: int = 1,
             provider_factory: Callable | None = None, commit: str | None = None,
             fixture_completion: Any = None, fixture_completions: dict | None = None,
             rubric_version: str = RUBRIC_VERSION, delay_s: float = 0) -> dict[str, Any]:
    """Grade one split and write metrics when every case finishes.

    A saved grade is reused only when the model id, prompt version, rubric
    hash, and generation settings still match that run.
    """
    if model is None:
        model = MODEL_ID if provider == "gemma" else "fixture"
    settings = {"temperature": 0}
    if generation_settings:
        settings.update(generation_settings)
    config: dict[str, Any] = {
        "name": f"{DATASET_ID}-{split}",
        "provider": provider,
        "model": model,
        "prompt_version": prompt_version,
        "rubric_version": rubric_version,
        "generation_settings": settings,
    }
    if fixture_completion is not None:
        config["fixture_completion"] = (fixture_completion if isinstance(fixture_completion, str)
                                         else json.dumps(fixture_completion, ensure_ascii=False))
    if fixture_completions is not None:
        config["fixture_completions"] = fixture_completions
    out = private_directory(Path(out), managed=True)
    manifest = convert(dataset_dir, split, out / "dataset", rubric_version=rubric_version)
    config_path = out / "config.json"
    private_write_text(config_path, json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    summary = run_experiment(config_path, manifest, repeats=repeats, out=out / "run",
                             delay_s=delay_s, provider_factory=provider_factory, commit=commit)
    result = {
        "completed": summary["completed"],
        "failed": summary["failed"],
        "pending": summary["pending"],
        "skipped": summary["skipped"],
        "stopped_reason": summary["stopped_reason"],
        "metrics_path": None,
    }
    if summary["stopped_reason"] or summary["failed"] or summary["pending"]:
        return result
    metrics = evaluate(dataset_dir, split, out / "run" / "results.json")
    metrics_path = out / "metrics.json"
    private_write_text(metrics_path, json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    result["metrics_path"] = str(metrics_path)
    return result


def evaluate(dataset_dir: Path, split: str, results_path: Path) -> dict[str, Any]:
    """Compare saved grader output to reference scores. This does not grade."""
    if split not in SPLITS:
        raise DatasetV2Error(f"unknown split: {split}")
    root = Path(dataset_dir)
    loaded = _load_split(root)
    reference = _load_reference(root / "reference" / "scores_v1.json", loaded)
    observations = _observations(Path(results_path), loaded[split], reference)
    return _metrics(split, observations, reference["excluded"])


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
    shared_writers = sides["dev"]["writers"] & sides["holdout"]["writers"]
    if shared_writers:
        raise DatasetV2Error("writer is in both splits: " + sorted(shared_writers)[0])
    _writers_match_ids(root, sides)
    return {"version": version, **sides}


def _writers_match_ids(root: Path, sides: dict[str, Any]) -> None:
    """Use the converter's own id parser. A writer letter later in the id does not count."""
    stems = _problem_stems(root / "sol_draft")
    parsed: dict[str, set[str]] = {}
    for name in SPLITS:
        found: set[str] = set()
        for answer_id in sides[name]["ids"]:
            writer = _parsed_writer(answer_id, stems)
            if writer not in sides[name]["writers"]:
                raise DatasetV2Error(f"unknown writer: {answer_id}")
            found.add(writer)
        parsed[name] = found
    shared = parsed["dev"] & parsed["holdout"]
    if shared:
        raise DatasetV2Error("writer is in both splits: " + sorted(shared)[0])


def _parsed_writer(answer_id: str, stems: list[str]) -> str:
    try:
        problem = _problem_for(answer_id, stems)
        writer, _image = _writer_and_image(answer_id, problem)
    except DatasetV2Error as exc:
        raise DatasetV2Error(f"unparsable answer id: {answer_id}") from exc
    return writer


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


def _transcripts(root: Path, split: str, ids: list[str], dev_ids: list[str],
                 holdout_ids: list[str]) -> dict[str, tuple[bytes, str]]:
    roots = _split_transcript_roots(root)
    directory = root / "ocr_gpu_v1" / SPLITS[split]
    if roots[split] is None:
        raise DatasetV2Error(f"missing transcript directory: {SPLITS[split]}")
    mapping = _transcript_map(root, set(dev_ids), set(holdout_ids))
    # Resolve both splits before any transcript bytes are read.
    located = _resolve_transcript_targets(
        root, roots, mapping, {"dev": dev_ids, "holdout": holdout_ids},
    )
    other = roots["holdout" if split == "dev" else "dev"]
    used = {path for _filename, path in located[split].values()}
    for path in directory.glob("ans_*.md"):
        try:
            real = path.resolve(strict=True)
        except OSError as exc:
            raise DatasetV2Error(f"unexpected transcript: {path.name}") from exc
        if other is not None and _within(other, real):
            raise DatasetV2Error(f"transcript path crosses splits: {path.name}")
        if real not in used:
            raise DatasetV2Error(f"unexpected transcript: {path.name}")
    chosen = {}
    for answer_id in ids:
        filename, path = located[split][answer_id]
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
        chosen[answer_id] = (raw, filename)
    return chosen


def _resolve_transcript_targets(root: Path, roots: dict[str, Path | None],
                                mapping: dict[str, str],
                                id_lists: dict[str, list[str]]) -> dict[str, dict[str, tuple[str, Path]]]:
    """Fail closed when any split target leaves its folder or aliases another target."""
    seen: dict[Path, tuple[str, str]] = {}
    located: dict[str, dict[str, tuple[str, Path]]] = {name: {} for name in SPLITS}
    for name in SPLITS:
        split_root = roots[name]
        if split_root is None:
            continue
        directory = root / "ocr_gpu_v1" / SPLITS[name]
        other = roots["holdout" if name == "dev" else "dev"]
        for answer_id in id_lists[name]:
            filename = f"ans_{mapping.get(answer_id, answer_id)}.md"
            real = _real_transcript(directory / filename, answer_id, split_root, other)
            label = (name, filename)
            previous = seen.get(real)
            if previous is not None and previous != label:
                raise DatasetV2Error(f"transcript paths resolve to the same file: {answer_id}")
            seen[real] = label
            located[name][answer_id] = (filename, real)
    return located


def _split_transcript_roots(root: Path) -> dict[str, Path | None]:
    roots = {name: _real_dir(root / "ocr_gpu_v1" / folder) for name, folder in SPLITS.items()}
    dev, hold = roots["dev"], roots["holdout"]
    if dev is not None and hold is not None and (_within(dev, hold) or _within(hold, dev)):
        raise DatasetV2Error("transcript directory crosses splits")
    return roots


def _real_dir(path: Path) -> Path | None:
    if not path.exists() and not path.is_symlink():
        return None
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise DatasetV2Error(f"missing transcript directory: {path.name}") from exc
    if not resolved.is_dir():
        raise DatasetV2Error(f"missing transcript directory: {path.name}")
    return resolved


def _real_transcript(path: Path, answer_id: str, split_root: Path, other_root: Path | None) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise DatasetV2Error(f"missing transcript: {answer_id}") from exc
    if other_root is not None and _within(other_root, resolved):
        raise DatasetV2Error(f"transcript path crosses splits: {answer_id}")
    if not resolved.is_file() or resolved.parent != split_root:
        raise DatasetV2Error(f"transcript map escapes directory: {answer_id}")
    return resolved


def _within(parent: Path, child: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
    except (OSError, ValueError):
        return False
    return True


def _transcript_map(root: Path, dev_ids: set[str], holdout_ids: set[str]) -> dict[str, str]:
    """Map each answer id to a transcript id in the same split. No cross-split targets."""
    path = root / "ocr_gpu_v1" / TRANSCRIPT_MAP
    if not path.exists():
        return {}
    payload = _load_json(path)
    if not isinstance(payload, dict) or payload.get("version") != TRANSCRIPT_MAP_VERSION:
        raise DatasetV2Error("invalid transcript map")
    pairs = payload.get("by_answer_id")
    if not isinstance(pairs, dict):
        raise DatasetV2Error("invalid transcript map")
    homes = (("dev", dev_ids), ("holdout", holdout_ids))
    mapping = {}
    for answer_id, value in pairs.items():
        if not isinstance(answer_id, str) or not answer_id.strip():
            raise DatasetV2Error("invalid transcript map")
        home = next((ids for _name, ids in homes if answer_id in ids), None)
        if home is None:
            raise DatasetV2Error(f"unexpected transcript map id: {answer_id}")
        target = _target_answer_id(value)
        if target in home:
            mapping[answer_id] = target
            continue
        other = dev_ids if home is holdout_ids else holdout_ids
        if target in other:
            raise DatasetV2Error(f"transcript map crosses splits: {answer_id}")
        raise DatasetV2Error(f"transcript map target is not in the split: {answer_id}")
    return mapping


def _target_answer_id(value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise DatasetV2Error("invalid transcript map")
    if "/" in value or "\\" in value or value.startswith("."):
        raise DatasetV2Error("transcript map escapes directory")
    if value.endswith(".md"):
        if not value.startswith("ans_"):
            raise DatasetV2Error("transcript map target is not in the split")
        answer_id = value[len("ans_"):-len(".md")]
        if not answer_id or answer_id.startswith(".") or "/" in answer_id or "\\" in answer_id:
            raise DatasetV2Error("transcript map target is not in the split")
        return answer_id
    return value


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


def _load_reference(path: Path, loaded: dict) -> dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise DatasetV2Error("reference scores must be an object")
    excluded = _string_set(payload.get("excluded_low_confidence", []), "excluded_low_confidence")
    answers = payload.get("answers")
    if not isinstance(answers, list):
        raise DatasetV2Error("reference answers must be a list")
    if "count" in payload and not _count_matches(payload.get("count"), len(answers)):
        raise DatasetV2Error("reference count does not match answers")
    rows: dict[str, dict] = {}
    for row in answers:
        if not isinstance(row, dict):
            raise DatasetV2Error("reference answer must be an object")
        answer_id = row.get("answer_id")
        if not isinstance(answer_id, str) or not answer_id.strip():
            raise DatasetV2Error("invalid reference answer id")
        if answer_id in rows:
            raise DatasetV2Error(f"duplicate reference: {answer_id}")
        problem = row.get("problem")
        if not isinstance(problem, str) or not answer_id.startswith(problem + "_"):
            raise DatasetV2Error(f"reference problem does not match answer id: {answer_id}")
        writer, _image = _writer_and_image(answer_id, problem)
        if row.get("writer") != writer:
            raise DatasetV2Error(f"reference writer does not match answer id: {answer_id}")
        rows[answer_id] = {
            "problem": problem,
            "writer": writer,
            "total": _optional_number(row, "total", answer_id),
            "subs": _reference_subs(row.get("scores", []), answer_id),
        }
    known = set(loaded["dev"]["ids"]) | set(loaded["holdout"]["ids"]) | excluded
    unexpected = sorted(set(rows) - known)
    if unexpected:
        raise DatasetV2Error(f"unexpected reference: {unexpected[0]}")
    return {"excluded": excluded, "rows": rows}


def _reference_subs(values: Any, answer_id: str) -> list[dict]:
    if not isinstance(values, list):
        raise DatasetV2Error(f"invalid reference subquestion: {answer_id}")
    subs = []
    seen: set[str] = set()
    for row in values:
        if not isinstance(row, dict) or not isinstance(row.get("sub"), str) or not row["sub"].strip():
            raise DatasetV2Error(f"invalid reference subquestion: {answer_id}")
        sub = row["sub"]
        if sub in seen:
            raise DatasetV2Error(f"duplicate reference subquestion: {answer_id} {sub}")
        seen.add(sub)
        subs.append({"sub": sub, "score": _optional_number(row, "score", f"{answer_id} {sub}")})
    return subs


def _optional_number(row: dict, key: str, label: str) -> float | None:
    if key not in row or row[key] is None:
        return None
    number = optional_score(row[key])
    if number is None:
        raise DatasetV2Error(f"invalid score: {label}")
    return number


def _string_set(values: Any, label: str) -> set[str]:
    if not isinstance(values, list):
        raise DatasetV2Error(f"invalid {label}")
    found: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise DatasetV2Error(f"invalid {label}")
        if value in found:
            raise DatasetV2Error(f"duplicate {label}: {value}")
        found.append(value)
    return set(found)


def _observations(results_path: Path, side: dict, reference: dict) -> list[dict]:
    results = _load_json(results_path)
    if not isinstance(results, dict):
        raise DatasetV2Error("grading results must be an object")
    summary = results.get("summary")
    if not isinstance(summary, dict):
        raise DatasetV2Error("grading did not finish: missing summary")
    for name in ("pending", "failed"):
        if not _count_matches(summary.get(name), 0):
            raise DatasetV2Error(f"grading did not finish: {name}")
    if summary.get("stopped_reason"):
        raise DatasetV2Error(f"grading did not finish: {summary.get('stopped_reason')}")
    expected = results.get("expected_runs")
    runs = results.get("runs")
    if not isinstance(expected, list) or not isinstance(runs, list):
        raise DatasetV2Error("grading results are incomplete")
    by_request: dict[str, dict] = {}
    for run in runs:
        if not isinstance(run, dict) or not isinstance(run.get("request_id"), str):
            raise DatasetV2Error("invalid result id")
        request_id = run["request_id"]
        if request_id in by_request:
            raise DatasetV2Error(f"duplicate result: {request_id}")
        by_request[request_id] = run
    observations = []
    seen_pairs: set[tuple[str, int]] = set()
    seen_ids: set[str] = set()
    expected_ids: set[str] = set()
    for job in expected:
        if not isinstance(job, dict):
            raise DatasetV2Error("invalid result id")
        case_id = job.get("case_id")
        request_id = job.get("request_id")
        problem_id = job.get("problem_id")
        repetition = job.get("repetition")
        if not isinstance(case_id, str) or not isinstance(request_id, str) or not isinstance(problem_id, str):
            raise DatasetV2Error("invalid result id")
        if isinstance(repetition, bool) or not isinstance(repetition, int) or repetition < 1:
            raise DatasetV2Error(f"invalid result repetition: {case_id}")
        if (case_id, repetition) in seen_pairs:
            raise DatasetV2Error(f"duplicate result: {case_id}")
        seen_pairs.add((case_id, repetition))
        if request_id in expected_ids:
            raise DatasetV2Error(f"duplicate result: {request_id}")
        expected_ids.add(request_id)
        if request_id not in by_request:
            raise DatasetV2Error(f"missing result: {case_id}")
        if not case_id.startswith(problem_id + "_"):
            raise DatasetV2Error(f"result problem does not match answer id: {case_id}")
        writer, _image = _writer_and_image(case_id, problem_id)
        if writer not in side["writers"]:
            raise DatasetV2Error(f"writer is not in the split: {case_id}")
        if side["problems"] is not None and problem_id not in side["problems"]:
            raise DatasetV2Error(f"problem is not in the split: {case_id}")
        run = by_request[request_id]
        provenance = run.get("outputs", {}).get("experiment_case") if isinstance(run.get("outputs"), dict) else None
        if isinstance(provenance, dict) and (
            provenance.get("case_id") != case_id or provenance.get("repetition") != repetition
        ):
            raise DatasetV2Error(f"result does not match its case: {case_id}")
        grade = run.get("outputs", {}).get("grading") if isinstance(run.get("outputs"), dict) else None
        if not isinstance(grade, dict) or grade.get("status") not in {"validated", "review_required"}:
            raise DatasetV2Error(f"grading did not finish: {case_id}")
        excluded = case_id in reference["excluded"]
        row = reference["rows"].get(case_id)
        if row is not None and row["problem"] != problem_id:
            raise DatasetV2Error(f"reference problem does not match answer id: {case_id}")
        seen_ids.add(case_id)
        observations.append({
            "answer_id": case_id,
            "problem": problem_id,
            "repetition": repetition,
            "excluded": excluded,
            "predicted_total": _optional_number(grade, "total_score", case_id),
            "predicted_subs": _predicted_subs(grade.get("items", []), case_id),
            "reference_total": None if row is None else row["total"],
            "reference_subs": [] if row is None else row["subs"],
        })
    extra_requests = sorted(set(by_request) - expected_ids)
    if extra_requests:
        raise DatasetV2Error(f"unexpected result: {extra_requests[0]}")
    missing_ids = sorted(set(side["ids"]) - seen_ids)
    extra_ids = sorted(seen_ids - set(side["ids"]))
    if missing_ids:
        raise DatasetV2Error(f"missing result: {missing_ids[0]}")
    if extra_ids:
        raise DatasetV2Error(f"unexpected result: {extra_ids[0]}")
    for row in observations:
        if not row["excluded"] and row["reference_total"] is None and row["answer_id"] not in reference["rows"]:
            raise DatasetV2Error(f"missing reference: {row['answer_id']}")
    return observations


def _predicted_subs(values: Any, answer_id: str) -> dict[str, float | None]:
    if not isinstance(values, list):
        raise DatasetV2Error(f"invalid graded item: {answer_id}")
    found: dict[str, float | None] = {}
    for row in values:
        if not isinstance(row, dict) or not isinstance(row.get("item_id"), str) or not row["item_id"].strip():
            raise DatasetV2Error(f"invalid graded item: {answer_id}")
        item_id = row["item_id"]
        if item_id in found:
            raise DatasetV2Error(f"duplicate graded item: {answer_id} {item_id}")
        found[item_id] = _optional_number(row, "score", f"{answer_id} {item_id}")
    return found


def _metrics(split: str, observations: list[dict], excluded: set[str]) -> dict[str, Any]:
    evaluated = [row for row in observations if not row["excluded"]]
    per_answer = [_answer_row(row) for row in sorted(evaluated, key=lambda item: (item["answer_id"], item["repetition"]))]
    by_problem: dict[str, list[dict]] = {}
    for row in per_answer:
        by_problem.setdefault(row["problem"], []).append(row)
    return {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "split": split,
        "excluded_low_confidence": sorted(excluded & {row["answer_id"] for row in observations}),
        "n_graded": len(observations),
        "n_evaluated": len(evaluated),
        "n_excluded": len(observations) - len(evaluated),
        **_aggregate(per_answer),
        "per_problem": [
            {"problem": problem, "n_evaluated": len(rows), **_aggregate(rows)}
            for problem, rows in sorted(by_problem.items())
        ],
        "per_answer": per_answer,
    }


def _answer_row(row: dict) -> dict[str, Any]:
    predicted = row["predicted_subs"]
    subs = []
    seen: set[str] = set()
    for item in row["reference_subs"]:
        seen.add(item["sub"])
        subs.append({
            "sub": item["sub"],
            "predicted": predicted.get(item["sub"]),
            "reference": item["score"],
            "abs_error": _abs(predicted.get(item["sub"]), item["score"]),
        })
    for item_id in sorted(set(predicted) - seen):
        subs.append({
            "sub": item_id,
            "predicted": predicted[item_id],
            "reference": None,
            "abs_error": None,
        })
    return {
        "answer_id": row["answer_id"],
        "problem": row["problem"],
        "repetition": row["repetition"],
        "predicted_total": row["predicted_total"],
        "reference_total": row["reference_total"],
        "abs_error": _abs(row["predicted_total"], row["reference_total"]),
        "exact_total": _same(row["predicted_total"], row["reference_total"]),
        "subquestions": subs,
    }


def _aggregate(rows: list[dict]) -> dict[str, Any]:
    total_errors = [row["abs_error"] for row in rows if row["abs_error"] is not None]
    exact_rows = [row for row in rows if row["reference_total"] is not None]
    exact_count = sum(row["exact_total"] for row in exact_rows)
    sub_errors = [
        item["abs_error"]
        for row in rows
        for item in row["subquestions"]
        if item["abs_error"] is not None
    ]
    return {
        "total_mae": _mean(total_errors),
        "total_mae_count": len(total_errors),
        "exact_match_count": exact_count,
        "exact_match_denominator": len(exact_rows),
        "exact_match_rate": _mean([1.0 if row["exact_total"] else 0.0 for row in exact_rows]),
        "subquestion_mae": _mean(sub_errors),
        "subquestion_mae_count": len(sub_errors),
    }


def _abs(predicted: float | None, reference: float | None) -> float | None:
    if predicted is None or reference is None:
        return None
    return abs(predicted - reference)


def _same(predicted: float | None, reference: float | None) -> bool:
    return predicted is not None and reference is not None and abs(predicted - reference) <= 1e-6


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _result_code(result: dict) -> int:
    if result["stopped_reason"] == "quota":
        return 3
    if result["failed"]:
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("baseline", help="grade one split and write metrics")
    command.add_argument("--dataset-dir", type=Path, required=True)
    command.add_argument("--split", required=True)
    command.add_argument("--provider", required=True, choices=["gemma", "fixture"])
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--model")
    command.add_argument("--prompt-version", default=DEFAULT_PROMPT)
    command.add_argument("--repeats", type=int, default=1)
    command.add_argument("--temperature", type=float, default=0)
    command.add_argument("--seed", type=int)
    command.add_argument("--max-output-tokens", type=int)
    command.add_argument("--top-p", type=float)
    command.add_argument("--top-k", type=int)
    command.add_argument("--fixture-completion", type=Path)
    command.add_argument("--commit", help="implementation identity recorded with the run")
    args = parser.parse_args(argv)
    settings: dict[str, Any] = {"temperature": args.temperature}
    for name in ("seed", "max_output_tokens", "top_p", "top_k"):
        value = getattr(args, name.replace("-", "_"))
        if value is not None:
            settings[name] = value
    completion = None
    if args.fixture_completion is not None:
        completion = args.fixture_completion.read_text(encoding="utf-8")
    try:
        result = baseline(
            args.dataset_dir, args.split, args.out, provider=args.provider, model=args.model,
            prompt_version=args.prompt_version, generation_settings=settings, repeats=args.repeats,
            fixture_completion=completion, commit=args.commit,
        )
    except (DatasetV2Error, ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "completed": result["completed"],
        "failed": result["failed"],
        "pending": result["pending"],
        "skipped": result["skipped"],
        "stopped_reason": result["stopped_reason"],
        "metrics_path": result["metrics_path"],
    }, ensure_ascii=False))
    return _result_code(result)


if __name__ == "__main__":
    raise SystemExit(main())
