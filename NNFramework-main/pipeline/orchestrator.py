"""pipeline 各阶段编排逻辑。"""

from __future__ import annotations

from pathlib import Path

from .analyze.build_capability_report import analyze_candidates
from .export.export_candidates import export_all_candidates
from .exceptions import PipelineStageError
from .intake.inspect_pt import inspect_checkpoint
from .selection import select_best_candidate
from .recovery.recover_process_contract import recover_process_contract
from .recovery.restore_model import restore_model_from_bundle
from .types import InputBundle, ModelIdentity, ModelRestoreResult, PipelineResult, ProcessContract, SelectionResult
from .utils.logger import log_stage_fail, log_stage_ok, log_stage_start
from .validate.validate_candidates import validate_candidates


def run_identify(bundle: InputBundle) -> tuple[InputBundle, ModelIdentity]:
    """识别 checkpoint 类型和基础模型身份。"""

    started_at = log_stage_start("IDENTIFY", str(bundle.checkpoint))
    identity = inspect_checkpoint(bundle.checkpoint)
    log_stage_ok("IDENTIFY", started_at, identity.checkpoint_kind)
    return bundle, identity


def run_recover(
    bundle: InputBundle,
    identity: ModelIdentity,
) -> tuple[ModelRestoreResult, ProcessContract]:
    """恢复 PyTorch 模型并重建输入到输出的处理契约。"""

    started_at = log_stage_start("RESTORE", str(bundle.model_py))
    restore_result = restore_model_from_bundle(bundle, identity)
    log_stage_ok("RESTORE", started_at, restore_result.load_mode)

    started_at = log_stage_start("RECOVERY", str(bundle.config))
    process_contract = recover_process_contract(bundle, restore_result)
    log_stage_ok("RECOVERY", started_at, "process contract recovered")
    return restore_result, process_contract


def run_pipeline(
    bundle: InputBundle,
    output_dir: Path,
) -> PipelineResult:
    """执行完整 pipeline，并返回可写盘的汇总结果。"""

    identity: ModelIdentity | None = None
    process_contract: ProcessContract | None = None
    candidates = []
    selection = SelectionResult(
        selected_precision=None,
        selected_candidate=None,
        rejected={},
        selection_reason="Pipeline did not reach export.",
    )
    errors: list[dict[str, object]] = []
    failure_stage: str | None = None
    warnings: list[str] = []
    validation_report = None
    deployment_status = "recover_only"

    try:
        _bundle, identity = run_identify(bundle)
        restore_result, process_contract = run_recover(bundle, identity)

        # 先同时导出多个精度候选，后续再统一分析和筛选。
        started_at = log_stage_start("EXPORT", "int8,float16,float32")
        candidates = export_all_candidates(
            restore_result,
            process_contract,
            output_dir,
            config_path=bundle.config,
        )
        successful_candidates = [candidate for candidate in candidates if candidate.export_ok]
        if not successful_candidates:
            raise PipelineStageError(
                stage="export",
                reason="All precision candidates failed during export",
                details={
                    candidate.precision: candidate.export_error or "unknown export failure"
                    for candidate in candidates
                },
            )
        log_stage_ok("EXPORT", started_at, f"{len(successful_candidates)}/3 candidates exported")

        started_at = log_stage_start("ANALYZE", "candidate capability reports")
        candidates = analyze_candidates(process_contract, candidates)
        deployable_candidates = [
            candidate
            for candidate in candidates
            if candidate.capability_report is not None and candidate.capability_report.deployable
        ]
        log_stage_ok(
            "ANALYZE",
            started_at,
            f"{len(deployable_candidates)}/{len(successful_candidates)} exported candidates deployable",
        )

        # 验证阶段现在分成两层：
        # 1. synthetic smoke：确认候选能跑通
        # 2. real task：有真实数据时，用真实特征重放做任务指标闸门
        started_at = log_stage_start("VALIDATE", "synthetic_smoke+real_task")
        validation_report = validate_candidates(
            restore_result,
            process_contract,
            candidates,
            config_path=bundle.config,
        )
        validated_candidates = [
            candidate
            for candidate in candidates
            if candidate.validation_pass is True
        ]
        log_stage_ok(
            "VALIDATE",
            started_at,
            f"{len(validated_candidates)}/{len(successful_candidates)} exported candidates validated",
        )

        selection = select_best_candidate(candidates)
        if selection.selected_candidate is None:
            raise PipelineStageError(
                stage="selection",
                reason="No precision candidate satisfied export, capability, and validation requirements",
                details={"rejected": selection.rejected},
            )

        # success 表示可直接交付，partial_success 表示可以部署但仍带临时性风险。
        deployment_status = _derive_deployment_status(selection.selected_candidate)
        _append_selection_warnings(warnings, selection)
        if any(not candidate.export_ok for candidate in candidates):
            warnings.append("One or more lower-priority precision candidates failed to export.")
        if any(
            candidate.export_ok and candidate.capability_report is not None and not candidate.capability_report.deployable
            for candidate in candidates
        ):
            warnings.append("One or more exported candidates contain unsupported preprocess steps or model operators.")
        if any(candidate.export_ok and candidate.validation_pass is False for candidate in candidates):
            warnings.append("One or more exported candidates failed smoke or task validation gates.")
    except PipelineStageError as exc:
        log_stage_fail(exc.stage.upper(), exc.reason)
        failure_stage = exc.stage
        deployment_status = "failed"
        errors.append(exc.to_dict())
        if identity is None:
            identity = ModelIdentity(
                framework="torch",
                checkpoint_kind="unknown",
                reasons=["identify stage did not complete"],
            )

    return PipelineResult(
        input_bundle=bundle,
        identity=identity,
        process_contract=process_contract,
        candidates=candidates,
        selection=selection,
        validation_report=validation_report,
        artifact_dir=output_dir.resolve(),
        deployment_status=deployment_status,
        failure_stage=failure_stage,
        warnings=warnings,
        errors=errors,
    )


def _append_selection_warnings(
    warnings: list[str],
    selection: SelectionResult,
) -> None:
    """把影响可信度和板端落地的附加风险转成 warning。"""

    candidate = selection.selected_candidate
    if candidate is None:
        return

    if selection.selected_precision == "int8" and _uses_pseudo_calibration(candidate):
        fallback_reason = candidate.export_metadata.get("calibration_fallback_reason")
        if isinstance(fallback_reason, str) and fallback_reason:
            warnings.append(
                "The selected int8 candidate fell back to pseudo calibration. "
                f"{fallback_reason}"
            )
        else:
            warnings.append(
                "The selected int8 candidate uses pseudo calibration and should be treated as provisional until representative calibration data is provided."
            )

    if _uses_smoke_only_validation(candidate):
        warnings.append(
            "The selected candidate was validated with synthetic smoke checks only because real-feature task validation data was unavailable. Treat this result as provisional."
        )

    board_runtime_supported = True
    if candidate.capability_report is not None:
        board_runtime_supported = candidate.capability_report.summary.get("board_runtime_supported", True)
    if candidate.capability_report is not None and not board_runtime_supported:
        unsupported = candidate.capability_report.summary.get("board_runtime_unsupported_items", [])
        warnings.append(
            "The selected candidate passed export/analysis/validation, but the current generated firmware runner contract is still mismatched. "
            f"Blocking items: {', '.join(unsupported)}."
        )

    if candidate.capability_report is not None and candidate.capability_report.accelerated != "full":
        not_fully_accelerated = candidate.capability_report.summary.get("not_fully_accelerated_items", [])
        if not_fully_accelerated:
            warnings.append(
                "The selected candidate is deployable but not fully accelerator-backed. "
                f"Non-accelerated items: {', '.join(not_fully_accelerated)}."
            )


def _derive_deployment_status(candidate) -> str:
    """把候选状态折叠为 success / partial_success。"""

    provisional = _uses_pseudo_calibration(candidate) or _uses_smoke_only_validation(candidate)
    partially_accelerated = (
        candidate.capability_report is not None
        and candidate.capability_report.accelerated != "full"
    )
    board_runtime_mismatch = (
        candidate.capability_report is not None
        and not candidate.capability_report.summary.get("board_runtime_supported", True)
    )
    if provisional or partially_accelerated or board_runtime_mismatch:
        return "partial_success"
    return "success"


def _uses_pseudo_calibration(candidate) -> bool:
    """当前候选是否退回到了伪校准数据。"""

    return bool(candidate.export_metadata.get("pseudo_calibration"))


def _uses_smoke_only_validation(candidate) -> bool:
    """当前候选是否只做了 synthetic smoke，没有真实任务验证。"""

    return candidate.validation_metrics.get("task_validation_available") is False
