"""Reference and execution metrics over saved runs, with explicit denominators."""
from __future__ import annotations

import statistics
from typing import Any

from harness_v2.evaluation.dataset import Dataset
from harness_v2.evaluation.independent import prepare_independent_input
from harness_v2.evaluation.reference import optional_score
from harness_v2.grading.validator import evidence_traceable


def _distribution(values: list[float]) -> dict:
    return {'mean': statistics.mean(values) if values else None,
            'median': statistics.median(values) if values else None, 'count': len(values)}


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _numeric(value: Any) -> bool:
    return optional_score(value) is not None


def _wrong_id(case_id: str, repetition: int, item: str) -> str:
    # JSON avoids ambiguity even when identifiers contain separators.
    import json
    return json.dumps([case_id, repetition, item], ensure_ascii=False, separators=(',', ':'))


def compute_metrics(experiment: dict, dataset: Dataset) -> dict:
    """Missing scores are unscored, never zero. Missing runs remain in denominators.

    MAE uses available finite totals with coverage reported separately. Item
    agreement includes every referenced item in the expected run set. Wrong
    additions/deductions use known item labels when present, otherwise totals.
    Explicit evidence absence has no cited span, so adds no evidence denominator.
    """
    cases = {case['case_id']: case for case in dataset.cases}
    answers = {}
    for key, case in cases.items():
        with open(case['ocr_import_path'], 'rb') as handle:
            raw = handle.read().decode('utf-8')
        answers[key] = prepare_independent_input({'answer_text': raw})['answer_text']
    jobs = experiment.get('expected_runs', [])
    by_id = {run['request_id']: run for run in experiment.get('runs', [])}
    errors, wrong_units, totals_latency = [], [], []
    item_matches = item_count = item_available = 0
    span_count = traceable_count = success = 0
    wrong_count = addition = deduction = flagged_wrong = false_flags = correct_count = 0
    unscored_totals = unscored_items = referenced_totals = 0
    for job in jobs:
        case_id, repetition = job['case_id'], job['repetition']
        if case_id not in cases:
            raise ValueError(f'unknown metric case: {case_id}')
        case = cases[case_id]
        run = by_id.get(job['request_id'], {})
        grade = run.get('outputs', {}).get('grading', {})
        status = grade.get('status', 'failed')
        success += status in {'validated', 'review_required'}
        if _numeric(run.get('latency_ms')):
            totals_latency.append(float(run['latency_ms']))
        ref = dataset.reference_for(case['problem_id'])
        total_ref, item_refs = optional_score(ref['reference_score']), ref['reference_items'] or {}
        total = optional_score(grade.get('total_score'))
        items = grade.get('items') or []
        predicted = {item['item_id']: optional_score(item.get('score')) for item in items}
        total_wrong = total_ref is not None and total is not None and total != total_ref
        if total_ref is not None:
            referenced_totals += 1
            if total is None:
                unscored_totals += 1
            else:
                errors.append(abs(total - total_ref))
        any_item_wrong = False
        all_items_scored = True
        for item_id, score_ref in item_refs.items():
            item_count += 1
            score = predicted.get(item_id)
            if score is None:
                unscored_items += 1
                all_items_scored = False
                continue
            item_available += 1
            if score == score_ref:
                item_matches += 1
            else:
                any_item_wrong = True
                addition += score > score_ref
                deduction += score < score_ref
                wrong_units.append(_wrong_id(case_id, repetition, item_id))
        if total_wrong and not any_item_wrong:
            # Matching or partial item labels cannot hide a known total error.
            # Count its direction once, without attributing it to an item.
            addition += total > total_ref
            deduction += total < total_ref
            wrong_units.append(_wrong_id(case_id, repetition, '__total__'))
        wrong = total_wrong or any_item_wrong
        flagged = status == 'review_required' or any(
            bool(str(item.get('review_reason') or '').strip())
            or (_numeric(item.get('uncertainty')) and item['uncertainty'] >= 0.5)
            for item in items)
        if wrong:
            wrong_count += 1
            flagged_wrong += flagged
        elif ((total_ref is not None or item_refs) and all_items_scored
              and (total_ref is None or total is not None) and status != 'failed'):
            correct_count += 1
            false_flags += flagged
        for item in items:
            for span in item.get('evidence_spans', []):
                span_count += 1
                text = span.get('text') if isinstance(span, dict) else None
                traceable_count += isinstance(text, str) and evidence_traceable(text, answers[case_id])

    # Usage and attempt timing include previous failed run attempts on resume,
    # not just the last successful row for each logical request.
    history = experiment.get('attempt_runs', experiment.get('runs', []))
    expected_ids = {job['request_id'] for job in jobs}
    seen, attempts_latency = set(), []
    requests = input_tokens = output_tokens = unknown_input = unknown_output = 0
    attempt_statuses: dict[str, int] = {}
    unknown_input_attempts = unknown_output_attempts = 0
    for run in history:
        if run.get('request_id') not in expected_ids:
            continue
        key = run.get('run_id') or (run['request_id'], run.get('outputs', {}).get('run_attempt', 1))
        if key in seen:
            continue
        seen.add(key)
        outputs = run.get('outputs', {})
        count = outputs.get('attempt_count', 1)
        count = count if isinstance(count, int) and not isinstance(count, bool) and count >= 0 else 0
        requests += count
        for name in ('input_tokens', 'output_tokens'):
            value = run.get(name)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                if name == 'input_tokens':
                    input_tokens += value
                else:
                    output_tokens += value
            elif count:
                if name == 'input_tokens':
                    unknown_input += 1
                else:
                    unknown_output += 1
        attempts = outputs.get('attempts', [])
        unknown_input_attempts += max(0, count - len(attempts))
        unknown_output_attempts += max(0, count - len(attempts))
        for attempt in attempts:
            unknown_input_attempts += not _numeric(attempt.get('input_tokens'))
            unknown_output_attempts += not _numeric(attempt.get('output_tokens'))
            if _numeric(attempt.get('latency_ms')):
                attempts_latency.append(float(attempt['latency_ms']))
            status = str(attempt.get('status') or 'unknown')
            attempt_statuses[status] = attempt_statuses.get(status, 0) + 1
    return {
        'run_count': len(jobs), 'success_count': success, 'success_rate': _ratio(success, len(jobs)),
        'mae': statistics.mean(errors) if errors else None,
        'mae_count': len(errors), 'reference_total_count': referenced_totals,
        'unscored_totals': unscored_totals, 'unscored_items': unscored_items,
        'item_agreement': _ratio(item_matches, item_count), 'item_reference_count': item_count,
        'item_scored_count': item_available, 'item_match_count': item_matches,
        'evidence_accuracy': _ratio(traceable_count, span_count),
        'evidence_spans': {'cited': span_count, 'traceable': traceable_count},
        'wrong_grading': {'count': wrong_count, 'unjust_addition': addition, 'unjust_deduction': deduction},
        'wrong_units': sorted(wrong_units),
        'review_detection': {'flagged_wrong': flagged_wrong, 'wrong_count': wrong_count,
            'recall': _ratio(flagged_wrong, wrong_count), 'false_flags': false_flags,
            'correct_count': correct_count, 'false_flag_rate': _ratio(false_flags, correct_count)},
        'latency': {'total_ms': _distribution(totals_latency),
                    'per_attempt_ms': _distribution(attempts_latency)},
        'usage': {'requests': requests, 'input_tokens': input_tokens, 'output_tokens': output_tokens,
                  'unknown_input_token_runs': unknown_input, 'unknown_output_token_runs': unknown_output,
                  'unknown_input_token_attempts': unknown_input_attempts,
                  'unknown_output_token_attempts': unknown_output_attempts,
                  'attempt_statuses': attempt_statuses},
    }
