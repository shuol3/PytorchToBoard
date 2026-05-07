"""写出能力分析与验证报告。"""

from __future__ import annotations

from pathlib import Path

from ..types import CandidateArtifact, ValidationReport
from ..utils.serde import write_json


# 写出候选加速覆盖情况报告。
def write_acceleration_report(path: Path, candidates: list[CandidateArtifact]) -> None:
    # 该报告侧重“每个候选能否部署、哪些部分被加速”。
    payload = {
        "candidates": [
            {
                "precision": candidate.precision,
                "export_ok": candidate.export_ok,
                "deployable": candidate.capability_report.deployable if candidate.capability_report else None,
                "accelerated": candidate.capability_report.accelerated if candidate.capability_report else None,
                "summary": candidate.capability_report.summary if candidate.capability_report else None,
                "preprocess_report": candidate.capability_report.preprocess_report if candidate.capability_report else [],
                "model_op_report": candidate.capability_report.model_op_report if candidate.capability_report else [],
            }
            for candidate in candidates
        ],
    }
    write_json(path, payload)


# 写出不支持项与未完全加速项报告。
def write_unsupported_report(path: Path, candidates: list[CandidateArtifact]) -> None:
    # unsupported 和 not_fully_accelerated 会被拆开写，便于后续消费。
    payload = {
        "candidates": [
            {
                "precision": candidate.precision,
                "deployable": candidate.capability_report.deployable if candidate.capability_report else None,
                "unsupported": _filter_support_records(candidate, only_unsupported=True),
                "not_fully_accelerated": _filter_support_records(candidate, only_unsupported=False),
            }
            for candidate in candidates
        ],
    }
    write_json(path, payload)


# 写出数值验证结果报告。
def write_validation_report(path: Path, report: ValidationReport | None) -> None:
    # 验证报告允许为空，方便早期切片按阶段渐进落地。
    payload = {
        "reference": report.reference if report else None,
        "thresholds": report.thresholds if report else {},
        "summary": report.summary if report else {},
        "results": report.results if report else [],
        "golden_dir": report.golden_dir if report else None,
    }
    write_json(path, payload)


# 根据支持级别筛选候选中的支持记录。
def _filter_support_records(
    candidate: CandidateArtifact,
    *,
    only_unsupported: bool,
) -> list[object]:
    # 统一从 preprocess 和 model_op 两类记录里筛选。
    if candidate.capability_report is None:
        return []
    records = [
        *candidate.capability_report.preprocess_report,
        *candidate.capability_report.model_op_report,
    ]
    if only_unsupported:
        return [record for record in records if record.support_level == "unsupported"]
    return [record for record in records if record.support_level != "accelerated"]
