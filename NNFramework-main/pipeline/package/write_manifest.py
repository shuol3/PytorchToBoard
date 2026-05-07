"""写出流水线总清单。"""

from __future__ import annotations

from pathlib import Path

from ..types import PipelineResult
from ..utils.serde import write_json


# 写出整个流水线运行的总 manifest。
def write_pipeline_manifest(path: Path, result: PipelineResult) -> None:
    # manifest 是总入口，尽量覆盖最终决策和关键中间结果。
    payload = {
        "deployment_status": result.deployment_status,
        "failure_stage": result.failure_stage,
        "warnings": result.warnings,
        "errors": result.errors,
        "artifact_dir": result.artifact_dir,
        "input_bundle": result.input_bundle,
        "identity": result.identity,
        "process_contract": result.process_contract,
        "selected_precision": result.selection.selected_precision if result.selection else None,
        "selected_candidate_path": (
            result.selection.selected_candidate.model_path
            if result.selection and result.selection.selected_candidate
            else None
        ),
        "selection_reason": result.selection.selection_reason if result.selection else None,
        "selection_rejected": result.selection.rejected if result.selection else {},
        "candidates": result.candidates,
        "validation_report": result.validation_report,
    }
    write_json(path, payload)
