"""回写每个候选的最终清单。"""

from __future__ import annotations

from ..types import CandidateArtifact, SelectionResult
from ..utils.serde import write_json


# 为每个候选单独写出一份 manifest。
def write_candidate_manifests(
    candidates: list[CandidateArtifact],
    selection: SelectionResult | None,
    deployment_status: str,
) -> None:
    selected_precision = selection.selected_precision if selection else None
    selection_rejected = selection.rejected if selection else {}
    selection_reason = selection.selection_reason if selection else None

    for candidate in candidates:
        if candidate.manifest_path is None:
            continue
        is_selected = candidate.precision == selected_precision
        rejected_reason = selection_rejected.get(candidate.precision)
        not_selected_reason = None
        # 对未选中的候选补充原因，便于后续比较和调试。
        if not is_selected and rejected_reason is None and selected_precision is not None:
            not_selected_reason = f"Higher-priority candidate selected: {selected_precision}"
        # 候选 manifest 既保留导出细节，也附带最终分析与验证结果。
        payload = {
            "precision": candidate.precision,
            "selected": is_selected,
            "selection_reason": selection_reason if is_selected else None,
            "selection_rejected_reason": rejected_reason,
            "not_selected_reason": not_selected_reason,
            "pipeline_deployment_status": deployment_status if is_selected else None,
            "export_ok": candidate.export_ok,
            "export_error": candidate.export_error,
            "model_path": candidate.model_path,
            "manifest_path": candidate.manifest_path,
            "export_metadata": candidate.export_metadata,
            "warnings": candidate.warnings,
            "graph_info": candidate.graph_info,
            "capability_report": candidate.capability_report,
            "memory_estimate": candidate.memory_estimate,
            "validation_pass": candidate.validation_pass,
            "validation_metrics": candidate.validation_metrics,
        }
        payload.update(candidate.export_metadata)
        write_json(candidate.manifest_path, payload)
