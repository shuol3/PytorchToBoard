"""定义 pipeline 各阶段共享的数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

# 支持的候选精度，默认按 int8 -> float16 -> float32 的顺序尝试选择。
Precision = Literal["int8", "float16", "float32"]

# 输入模型所属框架，当前主流程主要面向 torch。
Framework = Literal["torch", "tensorflow", "keras", "unknown"]

# checkpoint 的封装形式，会影响模型恢复路径。
CheckpointKind = Literal[
    "state_dict",
    "training_checkpoint",
    "full_module",
    "torchscript",
    "unknown",
]

# 单个预处理步骤或模型算子的支持级别。
SupportLevel = Literal["accelerated", "supported", "fallback", "unsupported"]

# 整个候选在目标侧的加速覆盖情况。
AccelerationLevel = Literal["full", "partial", "none"]


@dataclass
class InputBundle:
    """pipeline 运行所需的最小输入包。"""

    model_py: Path
    checkpoint: Path
    config: Path


@dataclass
class ModelIdentity:
    """identify 阶段得到的模型身份信息。"""

    framework: Framework
    checkpoint_kind: CheckpointKind
    model_class_name: str | None = None
    load_mode: str | None = None
    reasons: list[str] = field(default_factory=list)
    missing_requirements: list[str] = field(default_factory=list)


@dataclass
class PreprocessStep:
    """标准化后的单个预处理步骤描述。"""

    name: str
    enabled: bool
    params: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass
class ProcessContract:
    """从原始音频到模型输入输出的完整处理契约。"""

    raw_input: dict[str, Any]
    preprocess: list[PreprocessStep]
    model_input: dict[str, Any]
    model_output: dict[str, Any]
    labels: list[str] = field(default_factory=list)
    postprocess: list[dict[str, Any]] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)


@dataclass
class ModelRestoreResult:
    """restore 阶段的模型恢复结果。"""

    model: Any
    model_class_name: str | None
    model_kwargs: dict[str, Any]
    checkpoint_metadata: dict[str, Any]
    load_mode: str


@dataclass
class OpInfo:
    """单个算子在导出图中的摘要信息。"""

    op_name: str
    count: int = 1
    attrs: dict[str, Any] = field(default_factory=dict)
    input_shapes: list[list[int]] = field(default_factory=list)
    output_shapes: list[list[int]] = field(default_factory=list)
    dtypes: list[str] = field(default_factory=list)


@dataclass
class ModelGraphInfo:
    """导出后模型图的摘要。"""

    source_format: str
    ops: list[OpInfo]
    input_tensors: list[dict[str, Any]]
    output_tensors: list[dict[str, Any]]
    model_size_bytes: int | None = None


@dataclass
class SupportRecord:
    """预处理步骤或模型算子的支持性判断记录。"""

    scope: Literal["preprocess", "model_op"]
    name: str
    support_level: SupportLevel
    accelerated_by: list[str] = field(default_factory=list)
    fallback: str | None = None
    unsupported_reason: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class CapabilityReport:
    """单个候选的整体能力分析结果。"""

    deployable: bool
    accelerated: AccelerationLevel
    preprocess_report: list[SupportRecord]
    model_op_report: list[SupportRecord]
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryEstimate:
    """单个候选的内存与存储占用估计。"""

    precision: Precision
    model_flash_bytes: int
    runtime_flash_bytes: int
    tensor_arena_bytes: int
    preprocess_ram_bytes: int
    scratch_ram_bytes: int
    total_flash_bytes: int
    total_ram_bytes: int
    fits_board_limits: bool
    notes: list[str] = field(default_factory=list)


@dataclass
class CandidateArtifact:
    """单个精度候选在全流程中的统一状态容器。"""

    precision: Precision
    export_ok: bool
    export_error: str | None = None
    model_path: Path | None = None
    manifest_path: Path | None = None
    export_metadata: dict[str, Any] = field(default_factory=dict)
    graph_info: ModelGraphInfo | None = None
    precheck_report: CapabilityReport | None = None
    capability_report: CapabilityReport | None = None
    memory_estimate: MemoryEstimate | None = None
    validation_pass: bool | None = None
    validation_metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class SelectionResult:
    """候选选择阶段的最终结论。"""

    selected_precision: Precision | None
    selected_candidate: CandidateArtifact | None
    rejected: dict[str, str] = field(default_factory=dict)
    selection_reason: str | None = None


@dataclass
class ValidationCaseResult:
    """单个验证样本或验证分组的结果记录。"""

    group: str
    precision: Precision
    target: str
    passed: bool
    metrics: dict[str, Any]
    notes: list[str] = field(default_factory=list)


@dataclass
class ValidationThreshold:
    """旧式单精度误差阈值结构，保留用于兼容扩展。"""

    precision: Precision
    max_abs_error: float
    mean_abs_error: float
    require_top1_match: bool = True


@dataclass
class ValidationReport:
    """整个验证阶段的统一汇总报告。"""

    reference: str
    results: list[ValidationCaseResult]
    thresholds: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    golden_dir: Path | None = None


@dataclass
class PipelineResult:
    """一次完整 pipeline run 的顶层返回结果。"""

    input_bundle: InputBundle
    identity: ModelIdentity
    process_contract: ProcessContract | None
    candidates: list[CandidateArtifact]
    selection: SelectionResult | None
    validation_report: ValidationReport | None = None
    artifact_dir: Path | None = None
    deployment_status: str = "recover_only"
    failure_stage: str | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
