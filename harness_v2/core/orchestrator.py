"""Run a list of inputs. A quota error stops the loop and keeps finished rows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from harness_v2.core.contracts import RunRecord
from harness_v2.core.pipeline import Pipeline
from harness_v2.providers.gemma_ai_studio import QuotaExceeded


@dataclass
class ExperimentResult:
    records: list[RunRecord] = field(default_factory=list)
    stopped_reason: str = ""


class Orchestrator:
    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline

    def run(
        self,
        cases: list[dict[str, Any]],
        *,
        experiment_id: str,
        commit: str,
    ) -> ExperimentResult:
        result = ExperimentResult()
        for case in cases:
            try:
                result.records.append(
                    self.pipeline.run(case, experiment_id=experiment_id, commit=commit)
                )
            except QuotaExceeded as exc:
                result.records.append(
                    self.pipeline.record_quota(
                        case,
                        exc,
                        experiment_id=experiment_id,
                        commit=commit,
                    )
                )
                result.stopped_reason = "quota"
                break
        return result
