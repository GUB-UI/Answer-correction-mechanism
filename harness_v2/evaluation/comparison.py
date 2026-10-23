"""Fail-closed old/new result checks and benchmark accounting.

A completed comparison means both sides produced valid grades; it does not
mean their scores agree, or that either score agrees with a reference label.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from harness_v2.evaluation.independent import prepare_independent_input
from harness_v2.grading.rubric import rubric_from_data
from harness_v2.grading.scorer import ScoringParseError, items_from_completion
from harness_v2.grading.validator import validate


def load_result(path: Path) -> tuple[Any, str]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), ""
    except FileNotFoundError:
        return None, "missing result.json"
    except (OSError, ValueError, UnicodeError) as exc:
        return None, f"invalid result.json: {type(exc).__name__}"


def _invalid_result(payload: Any, side: str, case: dict[str, Any]) -> str:
    if payload is None:
        return "missing grading result"
    if not isinstance(payload, dict):
        return "invalid result object"
    if payload.get("failure_reason"):
        return f"result failure: {payload['failure_reason']}"
    if side == "old":
        if payload.get("status") != "parsed":
            return f"no valid grade: status={payload.get('status')}"
        grade = payload.get("data")
    else:
        if payload.get("status") != "ran":
            return f"no valid grade: status={payload.get('status')}"
        run = payload.get("run")
        outputs = run.get("outputs") if isinstance(run, dict) else None
        grade = outputs.get("grading") if isinstance(outputs, dict) else None
        if not isinstance(grade, dict) or grade.get("status") != "validated":
            return "missing or non-validated grading result"
    if not isinstance(grade, dict):
        return "missing grading object"
    if grade.get("failure_reason") or any(grade.get(flag) for flag in (
        "needs_review", "review_required", "grading_uncertain", "ocr_uncertain", "review_reason"
    )):
        return "grading requires review or reports a failure"
    try:
        view = prepare_independent_input(case)
        rubric = rubric_from_data(view.get("rubric") or [])
        if side == "old":
            maximum = grade.get("max_score")
            if (isinstance(maximum, bool) or not isinstance(maximum, (int, float))
                    or not math.isfinite(maximum)
                    or abs(maximum - sum(item.max_points for item in rubric)) > 1e-6):
                return "invalid legacy max_score"
            rows = grade.get("rubric_scores")
            required = {"item_id", "score", "max_score", "reason", "evidence"}
            if not isinstance(rows, list) or any(
                not isinstance(row, dict) or not required.issubset(row)
                or not isinstance(row["reason"], str) or not isinstance(row["evidence"], str)
                for row in rows
            ):
                return "invalid legacy rubric_scores schema"
            normalized = {"items": [
                {"item_id": row["item_id"], "score": row["score"], "max_points": row["max_score"],
                 "math_reason": row["reason"], "evidence_spans": [{"text": row["evidence"]}],
                 "review_reason": row.get("review_reason"), "uncertainty": row.get("uncertainty")}
                for row in rows
            ], "total_score": grade.get("score")}
        else:
            normalized = {"items": grade.get("items"), "total_score": grade.get("total_score")}
        # Each runner's canonical fields must be present. Cross-schema aliases
        # may never hide a missing/invalid old or new score or item array.
        items, total, _ = items_from_completion(json.dumps(normalized), rubric)
        answer = view.get("answer_text") or ""
        if case.get("ocr_import_path"):
            answer = prepare_independent_input({
                "answer_text": Path(case["ocr_import_path"]).read_bytes().decode("utf-8")
            })["answer_text"]
        checked = validate(rubric, items, total, answer)
    except (ScoringParseError, ValueError, TypeError, OSError, OverflowError) as exc:
        return f"invalid grading result: {exc}"
    if checked.status != "validated":
        return f"invalid grading result: {checked.failure_reason}"
    return ""


def evaluate_comparison(old: Any, new: Any, case: dict[str, Any]) -> dict[str, Any]:
    reasons = []
    for side, payload in (("old", old), ("new", new)):
        reason = _invalid_result(payload, side, case)
        if reason:
            reasons.append(f"{side}: {reason}")
    return {"success": not reasons, "status": "incomplete" if reasons else "complete",
            "reason": "; ".join(reasons)}


def summarize_comparisons(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # Incomplete cases remain in the denominator; empty benchmarks have no rate.
    success_count = sum(row.get("success") is True and row.get("status") == "complete"
                        and not row.get("reason") for row in rows)
    return {"rows": rows, "case_count": len(rows), "success_count": success_count,
            "incomplete_count": len(rows) - success_count,
            "success_rate": success_count / len(rows) if rows else None}


def experiment_index(payload: dict) -> tuple[dict, list[str]]:
    """Align repetitions by case, not by config-dependent request IDs."""
    reasons, expected, indexed = [], {}, {}
    if not isinstance(payload, dict):
        return {}, ['invalid experiment object']
    jobs, runs = payload.get('expected_runs'), payload.get('runs')
    if not isinstance(jobs, list) or not jobs:
        return {}, ['missing or empty expected cases']
    if not isinstance(runs, list):
        return {}, ['incomplete run list']
    for job in jobs:
        if not isinstance(job, dict) or not isinstance(job.get('case_id'), str) or not isinstance(job.get('repetition'), int):
            reasons.append('invalid expected case/repetition')
            continue
        key = (job['case_id'], job['repetition'])
        if key in expected:
            reasons.append('duplicate expected case/repetition')
        expected[key] = job
    by_request = {}
    for run in runs:
        rid = run.get('request_id') if isinstance(run, dict) else None
        if not rid or rid in by_request:
            reasons.append('duplicate or invalid run request id')
            continue
        by_request[rid] = run
    for key, job in expected.items():
        run = by_request.pop(job.get('request_id'), None)
        if run is None:
            reasons.append(f'incomplete case: {key}')
            continue
        if run.get('outputs', {}).get('grading', {}).get('status') not in {'validated', 'review_required', 'failed'}:
            reasons.append(f'invalid grading status: {key}')
        indexed[key] = run
    if by_request:
        reasons.append('unexpected cases in run list')
    return indexed, reasons



def _ocr_metadata_reasons(case: dict, outputs: dict) -> list[str]:
    """Check effective run provenance, not just agreement between manifests."""
    from harness_v2.recognition.run_ocr_cpu import MODEL_ID

    actual = outputs.get('ocr_metadata')
    # Failures before OCR have no imported provenance to validate.
    if (actual is None and not outputs.get('ocr_sha256')
            and outputs.get('grading', {}).get('status') == 'failed'):
        return []
    if not isinstance(actual, dict):
        return ['run OCR metadata is missing or invalid']
    declared = case.get('ocr_metadata', {})
    if not isinstance(declared, dict):
        return ['manifest OCR metadata is invalid']
    input_hash = case.get('grading_input', {}).get('sha256')
    expected = {'engine': str(declared.get('engine') or MODEL_ID),
                'revision': str(declared.get('revision') or declared.get('model_revision') or ''),
                'dtype': str(declared.get('dtype') or ''), 'source_sha256': input_hash}
    # Source locations are optional because portable datasets can move. An
    # explicitly declared source or sidecar digest must still be honored.
    for field in ('source', 'sidecar_sha256'):
        if field in declared:
            expected[field] = declared[field]
    reasons = [f'run OCR {field} differs from manifest or is missing'
               for field, value in expected.items() if field not in actual or actual[field] != value]
    if 'source_sha256' in declared and declared['source_sha256'] != input_hash:
        reasons.append('manifest OCR source_sha256 differs from input hash')
    digest = actual.get('sidecar_sha256')
    if 'sidecar_sha256' in actual and (not isinstance(digest, str) or len(digest) != 64
            or any(char not in '0123456789abcdef' for char in digest)):
        reasons.append('run OCR sidecar_sha256 is invalid')
    return reasons


def _provenance_reasons(payload: dict) -> list[str]:
    reasons = []
    manifest = payload.get('dataset', {})
    rows = manifest.get('cases', [])
    if not isinstance(rows, list) or any(not isinstance(c, dict) or not isinstance(c.get('case_id'), str) for c in rows):
        return ['invalid manifest case list']
    cases = {c['case_id']: c for c in rows}
    if not cases or len(cases) != len(rows):
        reasons.append('empty or duplicate manifest cases')
    jobs = payload.get('expected_runs', [])
    if not isinstance(jobs, list):
        return reasons + ['invalid expected cases']
    seen = set()
    indexed, _ = experiment_index(payload)
    for job in jobs:
        if not isinstance(job, dict):
            reasons.append('invalid expected case')
            continue
        case_id, rep = job.get('case_id'), job.get('repetition')
        if isinstance(rep, bool) or not isinstance(rep, int) or rep < 1:
            reasons.append('invalid repetition')
            continue
        seen.add((case_id, rep))
        case = cases.get(case_id)
        if case is None:
            reasons.append(f'unknown expected problem/case: {case_id}')
            continue
        expected = {'problem_id': case.get('problem_id'), 'input_version': case.get('input_version'),
                    'input_sha256': case.get('grading_input', {}).get('sha256'),
                    'rubric_sha256': case.get('rubric', {}).get('sha256')}
        for field, value in expected.items():
            if not value or job.get(field) != value:
                reasons.append(f'expected case {field} differs from manifest: {case_id}')
        run = indexed.get((case_id, rep))
        if run is None:
            continue
        outputs = run.get('outputs', {})
        actual = outputs.get('experiment_case')
        if not isinstance(actual, dict) or any(actual.get(k) != v for k, v in job.items()):
            reasons.append(f'run case provenance differs from expected case: {case_id}')
        reasons.extend(f'{reason}: {case_id}' for reason in _ocr_metadata_reasons(case, outputs))
        observed = outputs.get('ocr_sha256')
        if observed is not None and observed != expected['input_sha256']:
            reasons.append(f'run OCR/input hash differs from manifest: {case_id}')
        if outputs.get('grading', {}).get('status') != 'failed' and not observed:
            reasons.append(f'run OCR/input hash is missing: {case_id}')
    repetitions = {rep for _, rep in seen}
    if seen != {(case_id, rep) for case_id in cases for rep in repetitions}:
        reasons.append('incomplete problem/case/repetition plan')
    return reasons


def compatibility_reasons(baseline: dict, candidate: dict) -> list[str]:
    reasons = []
    for name, payload in (('baseline', baseline), ('candidate', candidate)):
        if not isinstance(payload, dict) or not isinstance(payload.get('config'), dict) or not isinstance(payload.get('dataset'), dict):
            reasons.append(f'{name}: missing configuration or dataset manifest')
    if reasons:
        return reasons
    for name, payload in (("baseline", baseline), ("candidate", candidate)):
        reasons.extend(f"{name}: {reason}" for reason in _provenance_reasons(payload))
    left, right = baseline['config'], candidate['config']
    for key, label in (('model', 'model'), ('provider', 'provider'),
                       ('rubric_version', 'rubric version'), ('generation_settings', 'generation settings')):
        if key not in left or key not in right or left[key] != right[key]:
            reasons.append(f'{label} differs or is missing')
    for key in ('dataset_id', 'version', 'references'):
        if baseline['dataset'].get(key) != candidate['dataset'].get(key) or key not in baseline['dataset']:
            reasons.append(f'dataset {key} differs or is missing')
    def cases(payload):
        rows = payload['dataset'].get('cases', [])
        if not isinstance(rows, list) or any(not isinstance(c, dict) or 'case_id' not in c for c in rows):
            return {}
        return {c['case_id']: c for c in rows}
    a, b = cases(baseline), cases(candidate)
    if not a or set(a) != set(b):
        reasons.append('problem/case sets differ or are empty')
    for key in set(a) & set(b):
        for field, label in (('problem_id', 'problem'), ('problem_text', 'problem text'),
                             ('input_version', 'input version'), ('grading_input', 'OCR/input hash'),
                             ('ocr_metadata', 'OCR provenance'), ('rubric', 'rubric hash'),
                             ('rubric_version', 'rubric version')):
            # Paths may differ in otherwise identical portable datasets.
            av, bv = a[key].get(field), b[key].get(field)
            if field in {'grading_input', 'rubric'}:
                av = av.get('sha256') if isinstance(av, dict) else None
                bv = bv.get('sha256') if isinstance(bv, dict) else None
            if av is None or bv is None or av != bv:
                reasons.append(f'{label} differs or is missing: {key}')
    ai, ar = experiment_index(baseline); bi, br = experiment_index(candidate)
    reasons.extend(f'baseline: {reason}' for reason in ar)
    reasons.extend(f'candidate: {reason}' for reason in br)
    if set(ai) != set(bi):
        reasons.append('case/repetition sets differ')
    for key in set(ai) & set(bi):
        x, y = ai[key], bi[key]
        if not x.get('model_id') or x.get('model_id') != y.get('model_id'):
            reasons.append(f'returned model differs or is missing: {key}')
        if x.get('rubric_version') != y.get('rubric_version') or x.get('rubric_version') != left.get('rubric_version'):
            reasons.append(f'run rubric version differs: {key}')
        for field, label in (('ocr_sha256', 'OCR/input hash'), ('grading_input_sha256', 'grading input hash')):
            xv, yv = x.get('outputs', {}).get(field), y.get('outputs', {}).get(field)
            if xv != yv:
                reasons.append(f'{label} differs: {key}')
        xm, ym = x.get('outputs', {}).get('ocr_metadata'), y.get('outputs', {}).get('ocr_metadata')
        if isinstance(xm, dict) and isinstance(ym, dict):
            for field in ('engine', 'revision', 'dtype', 'source_sha256', 'sidecar_sha256'):
                if xm.get(field) != ym.get(field):
                    reasons.append(f'run OCR {field} differs between counterparts: {key}')
        for run, cfg in ((x, left), (y, right)):
            settings = run.get('outputs', {}).get('settings', {})
            if run.get('outputs', {}).get('grading', {}).get('status') != 'failed':
                effective = {k: v for k, v in settings.items() if k != 'provider_policy'}
                if effective != cfg.get('generation_settings'):
                    reasons.append(f'effective generation settings differ: {key}')
    return list(dict.fromkeys(reasons))


def metric_deltas(baseline: dict, candidate: dict) -> dict:
    deltas = {}
    for key in baseline.keys() & candidate.keys():
        x, y = baseline[key], candidate[key]
        if isinstance(x, dict) and isinstance(y, dict):
            deltas[key] = metric_deltas(x, y)
        elif (isinstance(x, (int, float)) and not isinstance(x, bool)
              and isinstance(y, (int, float)) and not isinstance(y, bool)):
            deltas[key] = y - x
        elif x is None or y is None:
            deltas[key] = None
    return deltas


def compare_experiments(baseline: dict, candidate: dict, *, dataset=None) -> dict:
    from harness_v2.evaluation.dataset import load_dataset
    from harness_v2.evaluation.metrics import compute_metrics
    reasons = compatibility_reasons(baseline, candidate)
    if dataset is None and not reasons:
        try:
            dataset = load_dataset(Path(baseline['dataset_path']))
        except (ValueError, OSError, KeyError, TypeError) as exc:
            reasons.append(f'dataset unavailable for metrics: {exc}')
    if dataset is not None:
        import hashlib
        from harness_v2.storage.permissions import private_open
        derived = {}
        for case in dataset.cases:
            with private_open(case['ocr_import_path'], 'rb') as handle:
                raw = handle.read()
            if hashlib.sha256(raw).hexdigest() != case['input_sha256']:
                reasons.append(f"dataset input hash changed: {case['case_id']}")
            text = prepare_independent_input({'answer_text': raw.decode('utf-8')})['answer_text']
            derived[case['case_id']] = hashlib.sha256(text.encode('utf-8')).hexdigest()
        for name, payload in (('baseline', baseline), ('candidate', candidate)):
            if payload.get('dataset') != dataset.manifest:
                reasons.append(f'{name}: supplied dataset manifest differs')
            indexed, _ = experiment_index(payload)
            for (case_id, _), run in indexed.items():
                outputs = run.get('outputs', {})
                actual = outputs.get('grading_input_sha256')
                if (actual is not None or outputs.get('grading', {}).get('status') != 'failed') and actual != derived.get(case_id):
                    reasons.append(f'{name}: derived grading input hash differs: {case_id}')
    if reasons:
        return {'comparable': False, 'verdict': 'incomparable', 'reasons': reasons,
                'baseline': None, 'candidate': None, 'deltas': {}}
    before, after = compute_metrics(baseline, dataset), compute_metrics(candidate, dataset)
    new_wrong = sorted(set(after['wrong_units']) - set(before['wrong_units']))
    x, y = before['evidence_accuracy'], after['evidence_accuracy']
    lower_evidence = x is not None and (y is None or y < x)
    severe = {'new_wrong_grading': new_wrong, 'lower_evidence_accuracy': lower_evidence}
    # Quality determines the verdict; latency and usage remain reported tradeoffs.
    quality = [('mae', -1), ('item_agreement', 1), ('evidence_accuracy', 1),
               ('success_rate', 1), ('unscored_totals', -1), ('unscored_items', -1)]
    gains, losses = [], []
    for field, direction in quality:
        x, y = before[field], after[field]
        if x is None or y is None:
            if x is not None and y is None:
                losses.append(field)
            continue
        change = (y - x) * direction
        if change > 1e-12:
            gains.append(field)
        elif change < -1e-12:
            losses.append(field)
    for field in ('count', 'unjust_addition', 'unjust_deduction'):
        change = after['wrong_grading'][field] - before['wrong_grading'][field]
        if change < 0:
            gains.append('wrong_grading.' + field)
        elif change > 0:
            losses.append('wrong_grading.' + field)
    # False flags are measured only among correct grades; detection only among
    # wrong grades. No wrong grades means recall is undefined, not zero.
    for field, direction in (('recall', 1), ('false_flag_rate', -1)):
        x, y = before['review_detection'][field], after['review_detection'][field]
        if x is not None and y is not None:
            change = (y - x) * direction
            if change > 1e-12:
                gains.append('review_detection.' + field)
            elif change < -1e-12:
                losses.append('review_detection.' + field)
    verdict = ('regressed' if new_wrong or lower_evidence or losses else
               'improved' if gains else 'not_improved')
    return {'comparable': True, 'verdict': verdict, 'reasons': [],
            'baseline': before, 'candidate': after, 'deltas': metric_deltas(before, after),
            'severe_regressions': severe, 'quality_gains': gains, 'quality_losses': losses}
