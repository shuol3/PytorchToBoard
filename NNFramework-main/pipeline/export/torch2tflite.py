import os
import pathlib
import subprocess
import sys

import numpy as np
import tensorflow as tf
import torch

from pipeline.common.onnx_export_config import DEFAULT_ONNX_EXPORT_DYNAMO
#后续可以添加int8量化

VARIANT_PRIORITY = {
    "int8": 0,
    "float32": 1,
    "float16": 2,
}
ROOT = pathlib.Path(__file__).resolve().parents[2]


def _snapshot_tflite_files(tflite_path: pathlib.Path) -> dict[pathlib.Path, tuple[int, int]]:
    snapshot = {}
    for path in tflite_path.glob("*.tflite"):
        stat = path.stat()
        snapshot[path] = (stat.st_mtime_ns, stat.st_size)
    return snapshot


def _collect_conversion_candidates(
    tflite_path: pathlib.Path,
    before_snapshot: dict[pathlib.Path, tuple[int, int]],
) -> list[pathlib.Path]:
    current_snapshot = _snapshot_tflite_files(tflite_path)
    updated_files = []
    for path, state in current_snapshot.items():
        if before_snapshot.get(path) != state:
            updated_files.append(path)

    if updated_files:
        return updated_files

    return sorted(current_snapshot.keys())


def _classify_tflite_variant(model_path: pathlib.Path) -> str | None:
    lower_name = model_path.name.lower()
    if "int8" in lower_name:
        return "int8"
    if "float32" in lower_name or "fp32" in lower_name:
        return "float32"
    if "float16" in lower_name or "fp16" in lower_name:
        return "float16"

    try:
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
    except Exception:
        return None

    tensor_details = interpreter.get_tensor_details()
    has_quantized_int8_tensor = any(
        detail["dtype"] in (np.int8, np.uint8)
        and detail.get("quantization_parameters", {}).get("scales", np.array([])).size > 0
        for detail in tensor_details
    )
    if has_quantized_int8_tensor:
        return "int8"

    tensor_dtypes = {detail["dtype"] for detail in tensor_details}
    if np.float16 in tensor_dtypes:
        return "float16"
    if np.float32 in tensor_dtypes:
        return "float32"

    return None


def _select_tflite_model(candidates: list[pathlib.Path]) -> pathlib.Path:
    classified_candidates = []
    unreadable_candidates = []

    for path in candidates:
        variant = _classify_tflite_variant(path)
        if variant is None:
            unreadable_candidates.append(path.name)
            continue
        classified_candidates.append((path, variant))

    if not classified_candidates:
        candidate_list = ", ".join(path.name for path in candidates) if candidates else "none"
        unreadable_list = ", ".join(unreadable_candidates) if unreadable_candidates else "none"
        raise FileNotFoundError(
            "No readable TFLite model artifact was found after conversion. "
            f"Discovered candidates: {candidate_list}. "
            f"Unreadable or unclassified candidates: {unreadable_list}."
        )

    classified_candidates.sort(
        key=lambda item: (
            VARIANT_PRIORITY[item[1]],
            -item[0].stat().st_mtime_ns,
            item[0].name,
        )
    )
    return classified_candidates[0][0]


def convert_to_tflite(model: torch.nn.Module, input_shape: list[int]) -> bytes:
    onnx_path = ROOT / "intermediate" / "torch_model.onnx"
    tflite_path = ROOT / "intermediate" / "tflite_models"
    if not onnx_path.parent.exists():
        os.mkdir(onnx_path.parent)
    if not tflite_path.exists():
        os.mkdir(tflite_path)

   
    model = model.eval()
    torch.onnx.export(
        model,
        torch.randn(*input_shape),
        onnx_path,
        dynamo=DEFAULT_ONNX_EXPORT_DYNAMO,
    )

    before_snapshot = _snapshot_tflite_files(tflite_path)
    subprocess.run(
        [sys.executable, "-m", "onnx2tf", "-i", str(onnx_path), "-o", str(tflite_path)],
        check=True,
    )

    candidates = _collect_conversion_candidates(tflite_path, before_snapshot)
    selected_model_path = _select_tflite_model(candidates)

    with open(selected_model_path, "rb") as f:
        tflite_model = f.read()
    return tflite_model
