"""Post-inference reference evaluation. This module never calls a provider."""
from __future__ import annotations

import math
import uuid
from typing import Any

from harness_v2 import __version__
from harness_v2.core.contracts import FailureCase


def optional_score(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(float(value)) else None


def reference_errors(grading: dict, reference_score: Any = None,
                     reference_items: dict | None = None) -> list[str]:
    errors = []
    total = optional_score(reference_score)
    if total is not None and optional_score(grading.get('total_score')) != total:
        errors.append('reference_total_mismatch')
    predicted = {item['item_id']: optional_score(item.get('score')) for item in grading.get('items', [])}
    for item_id, value in (reference_items or {}).items():
        reference = optional_score(value)
        if reference is not None and predicted.get(item_id) != reference:
            errors.append(f'reference_item_mismatch:{item_id}')
    return errors


def failure_from_run(run: Any, *, reference_score: Any = None,
                     reference_items: dict | None = None) -> FailureCase | None:
    """Accept a RunRecord or the dictionary returned by RunStore.list_runs()."""
    run = run.to_dict() if hasattr(run, 'to_dict') else run
    outputs = run.get('outputs', {})
    grading = outputs.get('grading', {})
    mismatch = reference_errors(grading, reference_score, reference_items)
    if grading.get('status') == 'validated' and not mismatch:
        return None
    errors = [check['name'] for check in run.get('checks', []) if not check['ok']]
    reason = run.get('failure_reason') or grading.get('failure_reason')
    if reason and reason not in errors:
        errors.insert(0, reason)
    errors.extend(mismatch)
    items = grading.get('items', [])
    uncertainties = [value for item in items if (value := optional_score(item.get('uncertainty'))) is not None]
    meta = outputs.get('case_metadata', {})
    return FailureCase(
        case_id=uuid.uuid4().hex,
        problem_id=str(meta.get('problem_id') or (run.get('input_ids') or [run.get('request_id', '')])[0]),
        input_version=str(meta.get('input_version') or ''),
        model_version=str(run.get('model_id') or grading.get('model_id') or ''),
        harness_version=__version__, rubric_version=str(run.get('rubric_version') or 'v0'),
        predicted_score=optional_score(grading.get('total_score')), reference_score=optional_score(reference_score),
        rubric_results=items, evidence=[span for item in items for span in item.get('evidence_spans', [])],
        errors=errors, uncertainty=max(uncertainties) if uncertainties else None,
        execution_log=[{'request_id': run.get('request_id', ''), 'experiment_id': run.get('experiment_id', ''),
                        'artifact': outputs.get('raw_response_artifact', ''), 'status': grading.get('status', 'failed')}],
    )
