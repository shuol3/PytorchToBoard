"""候选精度选择逻辑。"""

from __future__ import annotations

from .export.export_candidates import PRECISION_ORDER
from .types import CandidateArtifact, SelectionResult


def select_best_candidate(candidates: list[CandidateArtifact]) -> SelectionResult:
    """按优先级和验证闸门选择最终候选。

    仍然优先尝试 `int8 -> float16 -> float32`，但是否可选不再只看
    synthetic logit 的数值接近程度，而是同时结合：
    1. 导出是否成功
    2. 能力分析是否可部署
    3. synthetic smoke 是否能跑通
    4. 有真实特征时，任务指标是否通过
    """

    candidates_by_precision = {candidate.precision: candidate for candidate in candidates}
    rejected: dict[str, str] = {}

    for precision in PRECISION_ORDER:
        candidate = candidates_by_precision.get(precision)
        if candidate is None:
            rejected[precision] = "candidate was not produced"
            continue

        rejection_reason = _candidate_rejection_reason(candidate)
        if rejection_reason is None:
            return SelectionResult(
                selected_precision=precision,
                selected_candidate=candidate,
                rejected=rejected,
                selection_reason=_build_selection_reason(candidate),
            )
        rejected[precision] = rejection_reason

    return SelectionResult(
        selected_precision=None,
        selected_candidate=None,
        rejected=rejected,
        selection_reason="No precision candidate satisfied export, capability, and validation gates.",
    )


def _candidate_rejection_reason(candidate: CandidateArtifact) -> str | None:
    """返回候选被拒绝的主原因。"""

    if not candidate.export_ok:
        return candidate.export_error or "export failed"
    if candidate.capability_report is None:
        return "capability analysis was not produced"
    if not candidate.capability_report.deployable:
        unsupported = candidate.capability_report.summary.get("unsupported_items", [])
        if unsupported:
            return f"unsupported items detected: {', '.join(unsupported)}"
        return "candidate is not deployable for the target lowering rules"
    if candidate.validation_pass is not True:
        if candidate.validation_metrics.get("error"):
            return f"validation failed: {candidate.validation_metrics['error']}"
        if candidate.validation_metrics.get("synthetic_smoke_pass") is False:
            return "synthetic smoke validation did not pass"
        if candidate.validation_metrics.get("task_validation_available") and candidate.validation_metrics.get("task_validation_pass") is False:
            return "real-feature task validation did not pass"
        return "validation gate did not pass"
    return None


def _build_selection_reason(candidate: CandidateArtifact) -> str:
    """生成最终入选原因，便于写入 summary 和 manifest。"""

    if candidate.validation_metrics.get("task_validation_available"):
        return (
            "Highest-priority candidate that exported successfully, passed capability analysis, "
            "and passed real-feature task validation."
        )
    return (
        "Highest-priority candidate that exported successfully, passed capability analysis, "
        "and passed synthetic smoke validation only."
    )
