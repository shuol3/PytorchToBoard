"""写出多精度候选对比结果。"""

from __future__ import annotations

from pathlib import Path

from ..types import CandidateArtifact, SelectionResult
from ..utils.serde import write_json


# 写出三种精度候选的横向对比结果。
def write_candidate_comparison(
    path: Path,
    candidates: list[CandidateArtifact],
    selection: SelectionResult | None,
) -> None:
    # 这里保留横向比较所需的核心字段，不重复写全部顶层清单。
    payload = {
        "selected_precision": selection.selected_precision if selection else None,
        "selection_reason": selection.selection_reason if selection else None,
        "rejected": selection.rejected if selection else {},
        "candidates": [
            {
                "precision": candidate.precision,
                "export_ok": candidate.export_ok,
                "export_error": candidate.export_error,
                "model_path": candidate.model_path,
                "manifest_path": candidate.manifest_path,
                "warnings": candidate.warnings,
                "graph_info": candidate.graph_info,
                "capability_report": candidate.capability_report,
                "validation_pass": candidate.validation_pass,
                "validation_metrics": candidate.validation_metrics,
            }
            for candidate in candidates
        ],
    }
    write_json(path, payload)
