"""Grokbot / GitHub handoff for proposal reviews (design §8 (d)).

No HTTP model API and no GitHub calls: the harness writes an Issue body, a PR
body, a branch name and plain-text commands for a human or Grokbot to run, and
later records the structured review result. Reviews are bound to the issued
request_id and the exact proposal snapshot; mismatches and replays are refused.
A review is a teacher_review event, never an approval.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from harness_v2.improvement.history import History, HistoryError
from harness_v2.providers.grokbot_exchange import (SchemaError, _check, _load_schema, _reject_images,
                                                   _blind_text, safe_request_id)
from harness_v2.storage.permissions import private_directory, private_write_text


def _validate(payload: dict, schema: str) -> None:
    errors = _check(payload, _load_schema(schema), "")
    if errors:
        raise SchemaError("; ".join(errors))
    _reject_images(payload)


def write_review_request(history: History, proposal_id: str, out_root: Path) -> Path:
    rec = history.state()["proposals"].get(proposal_id)
    if rec is None or not rec["evaluations"]:
        raise HistoryError("proposal needs an evaluation before review")
    latest = rec["evaluations"][-1]
    proposal = rec["proposal"]
    request = {"request_id": f"pr-{uuid.uuid4().hex}", "proposal_id": proposal_id,
               "snapshot_id": latest["snapshot_id"],
               "baseline_commit": latest["snapshot"]["baseline_commit"],
               "candidate_commit": latest["snapshot"]["candidate_commit"],
               "cause": proposal["cause"], "changed_components": list(proposal["changed_components"]),
               "expected_effect": proposal["expected_effect"], "verdict": latest["verdict"],
               "reasons": [str(r) for r in (latest["evaluation_results"].get("reasons") or [])]}
    request = _blind_text(request)  # no score text leaves the box
    _validate(request, "proposal_review_request.schema.json")
    rid = safe_request_id(request["request_id"])
    folder = private_directory(Path(out_root) / rid, managed=True)
    branch = f"harness/review-{proposal_id}"
    private_write_text(folder / "request.json", json.dumps(request, ensure_ascii=False, indent=2))
    private_write_text(folder / "branch.txt", branch + "\n")
    private_write_text(folder / "issue.md",
        f"# Review proposal {proposal_id}\n\nrequest_id: {rid}\nsnapshot_id: {request['snapshot_id']}\n"
        f"baseline: {request['baseline_commit']}\ncandidate: {request['candidate_commit']}\n"
        f"changed: {', '.join(request['changed_components'])}\nverdict: {request['verdict']}\n\n"
        f"cause: {request['cause']}\n\nexpected effect: {request['expected_effect']}\n\n"
        "Reply with a result JSON matching proposal_review_result.schema.json. "
        "A review is advisory; merging needs the owner's explicit approval.\n")
    private_write_text(folder / "pr.md",
        f"# Proposal {proposal_id} (draft)\n\nCandidate {request['candidate_commit']}, snapshot "
        f"{request['snapshot_id']}.\nDo not merge without the owner's approval recorded in the history.\n")
    private_write_text(folder / "commands.txt",
        "# Plain text for a human/Grokbot; not executed by the harness.\n"
        f"gh issue create --title 'Review proposal {proposal_id}' --body-file issue.md\n"
        f"git push origin {request['candidate_commit']}:refs/heads/{branch}\n"
        f"gh pr create --draft --head {branch} --body-file pr.md\n")
    history._append("review_requested", proposal_id, {"request_id": rid, "snapshot_id": request["snapshot_id"]})
    return folder


def record_review(history: History, result: dict) -> dict:
    _validate(result, "proposal_review_result.schema.json")
    return history._append("teacher_review", result["proposal_id"], dict(result))
