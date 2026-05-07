"""解析 TFLite 模型的图结构摘要。"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ..exceptions import PipelineStageError
from ..types import ModelGraphInfo, OpInfo

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover - import guard
    tf = None


# 读取 TFLite 模型结构，提取输入输出与算子信息。
def inspect_tflite_model(model_path: Path) -> ModelGraphInfo:
    if tf is None:
        raise PipelineStageError(
            stage="export",
            reason="TensorFlow is required to inspect TFLite artifacts",
        )

    # 强制禁用默认 delegate，保证看到的是基础 TFLite 图信息。
    resolver_type = tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
    interpreter = tf.lite.Interpreter(
        model_path=str(model_path),
        experimental_op_resolver_type=resolver_type,
    )
    interpreter.allocate_tensors()

    # 当前只统计算子计数和输入输出张量信息，不展开中间张量。
    op_counts = Counter()
    for op_detail in interpreter._get_ops_details():
        op_name = op_detail.get("op_name") or "UNKNOWN"
        op_counts[op_name] += 1

    ops = [
        OpInfo(op_name=op_name, count=count)
        for op_name, count in sorted(op_counts.items())
    ]
    return ModelGraphInfo(
        source_format="tflite",
        ops=ops,
        input_tensors=[_serialize_tensor_detail(detail) for detail in interpreter.get_input_details()],
        output_tensors=[_serialize_tensor_detail(detail) for detail in interpreter.get_output_details()],
        model_size_bytes=model_path.stat().st_size,
    )


# 将 TensorFlow 的 tensor detail 规整为统一字典结构。
def _serialize_tensor_detail(detail: dict[str, Any]) -> dict[str, Any]:
    # 量化参数在 int8 验证和板端对接时都会复用。
    scale, zero_point = detail.get("quantization", (0.0, 0))
    return {
        "index": int(detail.get("index", -1)),
        "name": detail.get("name"),
        "shape": [int(value) for value in detail.get("shape", [])],
        "shape_signature": [int(value) for value in detail.get("shape_signature", [])],
        "dtype": _dtype_name(detail.get("dtype")),
        "quantization": {
            "scale": float(scale),
            "zero_point": int(zero_point),
        },
    }


# 归一化不同框架返回的 dtype 名称。
def _dtype_name(dtype: Any) -> str:
    if dtype is None:
        return "unknown"
    try:
        return np.dtype(dtype).name
    except Exception:
        return getattr(dtype, "name", str(dtype))
