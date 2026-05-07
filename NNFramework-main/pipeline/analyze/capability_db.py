"""维护分析阶段使用的能力映射表。"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..types import Precision, SupportLevel


# 描述单个前处理步骤或模型算子的支持规格。
@dataclass(frozen=True)
class CapabilitySpec:
    support_level: SupportLevel
    accelerated_by: tuple[str, ...] = ()
    fallback: str | None = None
    unsupported_reason: str | None = None
    notes: tuple[str, ...] = ()

# 能力表版本信息，后续可替换为真实库版本。
DB_METADATA = {
    "preprocess": {
        "schema_version": "1.0",
        "library_version": "manual-baseline",
    },
    "cmsis_nn": {
        "schema_version": "1.0",
        "library_version": "manual-baseline",
    },
    "tflm_builtin": {
        "schema_version": "1.0",
        "library_version": "manual-baseline",
    },
}

# 前处理步骤的能力映射。
PREPROCESS_RULES: dict[str, CapabilitySpec] = {
    "bandpass": CapabilitySpec(
        support_level="fallback",
        fallback="Current deployment flow keeps bandpass as metadata only; no board-runtime bandpass stage is generated yet.",
        notes=("CMSIS-DSP biquad/FIR acceleration is feasible, but the lowering path is still TODO.",),
    ),
    "stft": CapabilitySpec(
        support_level="accelerated",
        accelerated_by=("CMSIS-DSP RFFT/vector kernels",),
        notes=("Generated board runner lowers windowing, RFFT, and power spectrum onto CMSIS-DSP helpers.",),
    ),
    "mel": CapabilitySpec(
        support_level="accelerated",
        accelerated_by=("CMSIS-DSP dot-product kernels",),
        notes=("Generated board runner uses offline sparse mel coefficients plus CMSIS-DSP dot products.",),
    ),
    "db_compression": CapabilitySpec(
        support_level="accelerated",
        accelerated_by=("CMSIS-DSP vector log/scale kernels",),
    ),
    "normalize": CapabilitySpec(
        support_level="accelerated",
        accelerated_by=("CMSIS-DSP statistics and scale kernels",),
    ),
    "preemphasis": CapabilitySpec(
        support_level="accelerated",
        accelerated_by=("CMSIS-DSP vector kernels",),
    ),
    "resample": CapabilitySpec(
        support_level="fallback",
        fallback="Resampling requires a dedicated polyphase/FIR lowering path.",
    ),
}

# int8 模型算子的能力映射，优先体现 CMSIS-NN 加速范围。
INT8_MODEL_RULES: dict[str, CapabilitySpec] = {
    "CONV_2D": CapabilitySpec(
        support_level="accelerated",
        accelerated_by=("CMSIS-NN",),
    ),
    "DEPTHWISE_CONV_2D": CapabilitySpec(
        support_level="accelerated",
        accelerated_by=("CMSIS-NN",),
    ),
    "FULLY_CONNECTED": CapabilitySpec(
        support_level="accelerated",
        accelerated_by=("CMSIS-NN",),
    ),
    "MAX_POOL_2D": CapabilitySpec(
        support_level="accelerated",
        accelerated_by=("CMSIS-NN",),
    ),
    "AVERAGE_POOL_2D": CapabilitySpec(
        support_level="accelerated",
        accelerated_by=("CMSIS-NN",),
    ),
    "SOFTMAX": CapabilitySpec(
        support_level="supported",
        fallback="Use TFLM builtin kernel; CMSIS-NN acceleration depends on the exact integration path.",
    ),
    "RESHAPE": CapabilitySpec(
        support_level="supported",
        fallback="Metadata-only op handled by TFLM runtime.",
    ),
    "STRIDED_SLICE": CapabilitySpec(
        support_level="supported",
        fallback="Use TFLM builtin kernel.",
    ),
    "QUANTIZE": CapabilitySpec(
        support_level="supported",
        fallback="Use TFLM builtin kernel.",
    ),
    "DEQUANTIZE": CapabilitySpec(
        support_level="supported",
        fallback="Use TFLM builtin kernel.",
    ),
    "MEAN": CapabilitySpec(
        support_level="supported",
        fallback="Use TFLM builtin/reference kernel.",
    ),
    "ADD": CapabilitySpec(
        support_level="supported",
        fallback="Use TFLM builtin kernel.",
    ),
    "CONCATENATION": CapabilitySpec(
        support_level="supported",
        fallback="Use TFLM builtin kernel.",
    ),
    "MUL": CapabilitySpec(
        support_level="supported",
        fallback="Use TFLM builtin kernel.",
    ),
    "RELU": CapabilitySpec(
        support_level="supported",
        fallback="Usually fused into adjacent ops; standalone case uses TFLM builtin kernel.",
    ),
}

# 浮点模型算子的能力映射，当前主要依赖 TFLM builtin。
FLOAT_MODEL_RULES: dict[str, CapabilitySpec] = {
    "CONV_2D": CapabilitySpec(
        support_level="supported",
        fallback="Use TFLM builtin float kernel.",
    ),
    "DEPTHWISE_CONV_2D": CapabilitySpec(
        support_level="supported",
        fallback="Use TFLM builtin float kernel.",
    ),
    "FULLY_CONNECTED": CapabilitySpec(
        support_level="supported",
        fallback="Use TFLM builtin float kernel.",
    ),
    "MAX_POOL_2D": CapabilitySpec(
        support_level="supported",
        fallback="Use TFLM builtin float kernel.",
    ),
    "AVERAGE_POOL_2D": CapabilitySpec(
        support_level="supported",
        fallback="Use TFLM builtin float kernel.",
    ),
    "MEAN": CapabilitySpec(
        support_level="supported",
        fallback="Use TFLM builtin float kernel.",
    ),
    "RESHAPE": CapabilitySpec(
        support_level="supported",
        fallback="Metadata-only op handled by TFLM runtime.",
    ),
    "STRIDED_SLICE": CapabilitySpec(
        support_level="supported",
        fallback="Use TFLM builtin kernel.",
    ),
    "DEQUANTIZE": CapabilitySpec(
        support_level="supported",
        fallback="Use TFLM builtin kernel to materialize float activations.",
    ),
    "QUANTIZE": CapabilitySpec(
        support_level="supported",
        fallback="Use TFLM builtin kernel.",
    ),
    "SOFTMAX": CapabilitySpec(
        support_level="supported",
        fallback="Use TFLM builtin float kernel.",
    ),
    "ADD": CapabilitySpec(
        support_level="supported",
        fallback="Use TFLM builtin float kernel.",
    ),
    "MUL": CapabilitySpec(
        support_level="supported",
        fallback="Use TFLM builtin float kernel.",
    ),
    "RELU": CapabilitySpec(
        support_level="supported",
        fallback="Use TFLM builtin float kernel or fused activation.",
    ),
    "PAD": CapabilitySpec(
        support_level="supported",
        fallback="Use TFLM builtin kernel.",
    ),
    "CONCATENATION": CapabilitySpec(
        support_level="supported",
        fallback="Use TFLM builtin kernel.",
    ),
    "LOGISTIC": CapabilitySpec(
        support_level="supported",
        fallback="Use TFLM builtin float kernel.",
    ),
}


# 查询前处理步骤对应的能力规格。
def get_preprocess_spec(step_name: str) -> CapabilitySpec | None:
    return PREPROCESS_RULES.get(step_name)


# 按候选精度查询模型算子的能力规格。
def get_model_op_spec(precision: Precision, op_name: str) -> CapabilitySpec | None:
    # int8 和浮点路径使用不同的能力表。
    if precision == "int8":
        return INT8_MODEL_RULES.get(op_name)
    return FLOAT_MODEL_RULES.get(op_name)
