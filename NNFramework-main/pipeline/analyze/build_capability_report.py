"""构建候选模型的能力分析报告。"""

from __future__ import annotations

from ..board_runtime_contract import summarize_board_runtime_contract
from ..types import CandidateArtifact, CapabilityReport, ProcessContract, SupportRecord
from .capability_db import DB_METADATA, get_model_op_spec, get_preprocess_spec


# 为每个导出候选补充能力分析结果。
def analyze_candidates(
    process_contract: ProcessContract,
    candidates: list[CandidateArtifact],
) -> list[CandidateArtifact]:
    for candidate in candidates:
        if not candidate.export_ok:
            continue
        candidate.capability_report = build_capability_report(process_contract, candidate)
    return candidates


# 汇总前处理和模型算子支持情况，生成单个候选的能力报告。
def build_capability_report(
    process_contract: ProcessContract,
    candidate: CandidateArtifact,
) -> CapabilityReport:
    board_runtime = summarize_board_runtime_contract(process_contract)
    preprocess_report = _build_preprocess_report(process_contract, board_runtime)
    model_op_report = _build_model_op_report(candidate)
    all_records = [*preprocess_report, *model_op_report]

    # 分析结果同时区分“不支持”与“可运行但未完全加速”。
    unsupported_items = [
        f"{record.scope}:{record.name}"
        for record in all_records
        if record.support_level == "unsupported"
    ]
    not_fully_accelerated_items = [
        f"{record.scope}:{record.name}"
        for record in all_records
        if record.support_level != "accelerated"
    ]
    accelerated_items = [
        f"{record.scope}:{record.name}"
        for record in all_records
        if record.support_level == "accelerated"
    ]

    deployable = not unsupported_items
    acceleration_level = _derive_acceleration_level(all_records)

    summary = {
        "precision": candidate.precision,
        "database_versions": DB_METADATA,
        "deployable": deployable,
        "enabled_preprocess_steps": [step.name for step in process_contract.preprocess if step.enabled],
        "model_ops": [
            {
                "op_name": op.op_name,
                "count": op.count,
            }
            for op in (candidate.graph_info.ops if candidate.graph_info else [])
        ],
        "accelerated_items": accelerated_items,
        "not_fully_accelerated_items": not_fully_accelerated_items,
        "unsupported_items": unsupported_items,
        "accelerated_count": len(accelerated_items),
        "not_fully_accelerated_count": len(not_fully_accelerated_items),
        "unsupported_count": len(unsupported_items),
        # 将“理论 lowering 支持”与“当前生成固件运行时约束”并列写入 summary。
        "board_runtime_supported": board_runtime["supported"],
        "board_runtime_unsupported_items": board_runtime["unsupported_items"],
        "board_runtime_unsupported_reasons": board_runtime["unsupported_reasons"],
        "board_runtime_warnings": board_runtime["warnings"],
        "board_runtime_expected": board_runtime["expected_runtime"],
        "board_runtime_supported_input_shapes": board_runtime["supported_input_shapes"],
    }
    if candidate.precision == "float16":
        summary["notes"] = [
            "Float16 export is treated as deployable only through TFLM builtin kernels; CMSIS-NN acceleration is not assumed.",
        ]

    return CapabilityReport(
        deployable=bool(deployable),
        accelerated=acceleration_level,
        preprocess_report=preprocess_report,
        model_op_report=model_op_report,
        summary=summary,
    )


# 基于处理合同中的前处理步骤构建支持记录，并在 notes 中附带板端运行时限制提示。
def _build_preprocess_report(
    process_contract: ProcessContract,
    board_runtime: dict[str, object],
) -> list[SupportRecord]:
    records: list[SupportRecord] = []
    runtime_reasons = board_runtime.get("unsupported_reasons", {})
    for step in process_contract.preprocess:
        if not step.enabled:
            continue
        spec = get_preprocess_spec(step.name)
        if spec is None:
            records.append(
                SupportRecord(
                    scope="preprocess",
                    name=step.name,
                    support_level="unsupported",
                    unsupported_reason="No preprocess lowering rule is registered for this step.",
                )
            )
            continue

        notes = list(spec.notes)
        if step.params:
            notes.append(f"params={step.params}")
        step_key = f"preprocess:{step.name}"
        if step_key in runtime_reasons:
            notes.append(f"board_runtime={runtime_reasons[step_key]}")
        if step.name == "stft" and "preprocess:stft.center" in runtime_reasons:
            notes.append(f"board_runtime={runtime_reasons['preprocess:stft.center']}")
        if step.name == "normalize" and "preprocess:normalize.mode" in runtime_reasons:
            notes.append(f"board_runtime={runtime_reasons['preprocess:normalize.mode']}")
        records.append(
            SupportRecord(
                scope="preprocess",
                name=step.name,
                support_level=spec.support_level,
                accelerated_by=list(spec.accelerated_by),
                fallback=spec.fallback,
                unsupported_reason=spec.unsupported_reason,
                notes=notes,
            )
        )
    return records


# 基于候选图信息和精度能力表构建模型算子支持记录。
def _build_model_op_report(candidate: CandidateArtifact) -> list[SupportRecord]:
    if candidate.graph_info is None:
        return [
            SupportRecord(
                scope="model_op",
                name="graph_info_missing",
                support_level="unsupported",
                unsupported_reason="TFLite graph inspection did not produce operator metadata.",
            )
        ]

    records: list[SupportRecord] = []
    for op in candidate.graph_info.ops:
        # 这里按导出后的实际 TFLite 算子名匹配能力规则。
        spec = get_model_op_spec(candidate.precision, op.op_name)
        if spec is None:
            records.append(
                SupportRecord(
                    scope="model_op",
                    name=op.op_name,
                    support_level="unsupported",
                    unsupported_reason=f"No lowering rule is registered for {candidate.precision} operator {op.op_name}.",
                    notes=[f"count={op.count}"],
                )
            )
            continue

        notes = list(spec.notes)
        notes.append(f"count={op.count}")
        records.append(
            SupportRecord(
                scope="model_op",
                name=op.op_name,
                support_level=spec.support_level,
                accelerated_by=list(spec.accelerated_by),
                fallback=spec.fallback,
                unsupported_reason=spec.unsupported_reason,
                notes=notes,
            )
        )
    return records


# 从支持记录汇总整体加速覆盖等级。
def _derive_acceleration_level(records: list[SupportRecord]) -> str:
    if not records:
        return "none"
    accelerated_count = sum(1 for record in records if record.support_level == "accelerated")
    if accelerated_count == 0:
        return "none"
    if accelerated_count == len(records):
        return "full"
    return "partial"
