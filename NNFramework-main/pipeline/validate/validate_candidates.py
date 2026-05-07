"""候选验证逻辑：synthetic smoke + 真实特征任务验证。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..common.input_data import AudioWindowDataset, LogMelFeatureExtractor, build_split_bundle
from ..common.representative_calibration import load_training_config
from ..exceptions import PipelineStageError
from ..types import (
    CandidateArtifact,
    ModelRestoreResult,
    Precision,
    ProcessContract,
    ValidationCaseResult,
    ValidationReport,
)

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover - import guard
    tf = None

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - import guard
    torch = None
    nn = None


CASE_ZERO = "zeros"
CASE_NORMAL = "normal_seed_7"
CASE_UNIFORM = "uniform_seed_17"
REAL_VALIDATION_SPLIT_PRIORITY = ("test", "val", "train")
REFERENCE_BATCH_SIZE = 64


def validate_candidates(
    restore_result: ModelRestoreResult,
    process_contract: ProcessContract,
    candidates: list[CandidateArtifact],
    config_path: Path | None = None,
) -> ValidationReport:
    """按两层验证候选。

    1. `synthetic smoke`
       只确认候选能否在量化输入准备和反量化输出恢复后稳定跑通。
    2. `real-feature task validation`
       用真实音频窗口恢复特征后重放，按任务指标判断候选是否可接受。
    """

    model = _require_torch_model(restore_result)
    shape_nchw = _extract_input_shape(process_contract)
    thresholds = default_validation_thresholds()

    # synthetic smoke 的参考输出只需要跑一次，后续所有候选共享。
    smoke_cases = build_synthetic_cases(shape_nchw)
    smoke_reference_outputs = run_reference_model(model, smoke_cases)

    # 真实特征验证包同样只构建一次，避免每个候选重复切窗和提特征。
    task_bundle = build_real_task_validation_bundle(model, config_path)
    results: list[ValidationCaseResult] = []
    candidate_summary: dict[str, Any] = {}

    for candidate in candidates:
        if not candidate.export_ok:
            candidate.validation_pass = None
            candidate.validation_metrics = {
                "skipped": True,
                "reason": "candidate did not export successfully",
            }
            continue

        if candidate.model_path is None:
            candidate.validation_pass = False
            candidate.validation_metrics = {
                "error": "candidate model_path is missing",
            }
            results.append(
                ValidationCaseResult(
                    group="candidate_setup",
                    precision=candidate.precision,
                    target="candidate_setup",
                    passed=False,
                    metrics={"error": "candidate model_path is missing"},
                )
            )
            continue

        try:
            interpreter = _build_tflite_interpreter(candidate.model_path)
            smoke_case_results, synthetic_smoke = _run_synthetic_smoke(
                precision=candidate.precision,
                interpreter=interpreter,
                cases=smoke_cases,
                reference_outputs=smoke_reference_outputs,
                shape_nchw=shape_nchw,
            )
            task_case_result, task_validation = _run_real_task_validation(
                candidate=candidate,
                interpreter=interpreter,
                task_bundle=task_bundle,
                shape_nchw=shape_nchw,
            )
        except Exception as exc:
            candidate.validation_pass = False
            candidate.validation_metrics = {
                "error": f"{exc.__class__.__name__}: {exc}",
            }
            results.append(
                ValidationCaseResult(
                    group="candidate_setup",
                    precision=candidate.precision,
                    target="candidate_setup",
                    passed=False,
                    metrics={"error": f"{exc.__class__.__name__}: {exc}"},
                )
            )
            continue

        # 有真实数据时，最终闸门由真实任务指标决定；否则仅保留 smoke 结果，
        # 并在后续 selection / manifest 中明确标记为临时结论。
        validation_mode = (
            "real_task_with_synthetic_smoke"
            if task_validation["available"]
            else "synthetic_smoke_only"
        )
        candidate.validation_pass = bool(synthetic_smoke["pass"]) and (
            bool(task_validation["pass"]) if task_validation["available"] else True
        )
        candidate.validation_metrics = {
            "validation_mode": validation_mode,
            "synthetic_smoke_pass": synthetic_smoke["pass"],
            "synthetic_smoke": synthetic_smoke,
            "task_validation_available": task_validation["available"],
            "task_validation_reason": task_validation.get("reason"),
            "task_validation_pass": task_validation.get("pass"),
            "task_validation": task_validation,
        }
        candidate_summary[candidate.precision] = {
            "validation_pass": candidate.validation_pass,
            "validation_mode": validation_mode,
            "synthetic_smoke_pass": synthetic_smoke["pass"],
            "task_validation_available": task_validation["available"],
            "task_validation_pass": task_validation.get("pass"),
            "accuracy": task_validation.get("candidate_metrics", {}).get("accuracy"),
            "macro_f1": task_validation.get("candidate_metrics", {}).get("macro_f1"),
            "reference_top1_match_rate": task_validation.get("reference_top1_match_rate"),
            "worst_smoke_max_abs_error": synthetic_smoke["worst_max_abs_error"],
            "worst_smoke_mean_abs_error": synthetic_smoke["worst_mean_abs_error"],
        }
        results.extend(smoke_case_results)
        if task_case_result is not None:
            results.append(task_case_result)

    passed_precisions = [
        candidate.precision
        for candidate in candidates
        if candidate.validation_pass is True
    ]
    summary = {
        "candidate_count": len(candidates),
        "validated_precisions": passed_precisions,
        "validation_flow": {
            "synthetic_smoke": thresholds["synthetic_smoke"],
            "real_task": {
                "available": task_bundle["available"],
                "reason": task_bundle.get("reason"),
                "split": task_bundle.get("split_name"),
                "sample_count": task_bundle.get("sample_count", 0),
                "class_names": task_bundle.get("class_names", []),
                "reference_metrics": task_bundle.get("reference_metrics"),
                "thresholds": thresholds["task_metrics"],
            },
        },
        "candidate_summary": candidate_summary,
    }
    return ValidationReport(
        reference="restored_pytorch_model",
        results=results,
        thresholds=thresholds,
        summary=summary,
        golden_dir=None,
    )


def default_validation_thresholds() -> dict[str, Any]:
    """返回 smoke 阶段和真实任务阶段的阈值配置。"""

    return {
        "synthetic_smoke": {
            "case_names": [CASE_ZERO, CASE_NORMAL, CASE_UNIFORM],
            "require_finite_outputs": True,
            "require_shape_match": True,
        },
        "task_metrics": {
            "float32": {
                "max_accuracy_drop": 0.001,
                "max_macro_f1_drop": 0.001,
                "min_reference_top1_match_rate": 0.999,
            },
            "float16": {
                "max_accuracy_drop": 0.005,
                "max_macro_f1_drop": 0.005,
                "min_reference_top1_match_rate": 0.995,
            },
            "int8": {
                "max_accuracy_drop": 0.02,
                "max_macro_f1_drop": 0.02,
                "min_reference_top1_match_rate": 0.98,
            },
        },
    }


def build_synthetic_cases(shape_nchw: list[int]) -> dict[str, np.ndarray]:
    """构造固定 synthetic smoke 输入样本。"""

    normal_rng = np.random.default_rng(7)
    uniform_rng = np.random.default_rng(17)
    return {
        CASE_ZERO: np.zeros(shape_nchw, dtype=np.float32),
        CASE_NORMAL: normal_rng.normal(loc=0.0, scale=1.0, size=shape_nchw).astype(np.float32),
        CASE_UNIFORM: uniform_rng.uniform(low=-2.0, high=2.0, size=shape_nchw).astype(np.float32),
    }


def run_reference_model(
    model: nn.Module,
    cases: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """运行 PyTorch 参考模型，得到 synthetic smoke 对照输出。"""

    outputs: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for name, array in cases.items():
            tensor = torch.from_numpy(array).to(dtype=torch.float32)
            outputs[name] = model(tensor).detach().cpu().numpy().astype(np.float32, copy=False)
    return outputs


def build_real_task_validation_bundle(
    model: nn.Module,
    config_path: Path | None,
) -> dict[str, Any]:
    """预先构造真实特征验证包，供所有候选复用。"""

    if config_path is None:
        return {
            "available": False,
            "reason": "No config path was provided for real-feature task validation.",
        }
    if not config_path.is_file():
        return {
            "available": False,
            "reason": f"Config file does not exist: {config_path}",
        }

    try:
        config = load_training_config(config_path)
        split_bundle = build_split_bundle(config)
        split_name, records = _select_real_validation_records(split_bundle)
        feature_extractor = LogMelFeatureExtractor(config)
        dataset = AudioWindowDataset(
            records,
            feature_extractor,
            augment_cfg={},
            training=False,
        )
        if len(dataset) == 0:
            return {
                "available": False,
                "reason": f"Selected validation split is empty: {split_name}",
            }

        features: list[np.ndarray] = []
        labels: list[int] = []
        for index in range(len(dataset)):
            feature, label = dataset[index]
            # dataset 单样本已经是 CHW；这里直接堆叠成 [N, C, H, W]，
            # 到单候选回放时再补单样本 batch 维度。
            features.append(feature.detach().cpu().numpy().astype(np.float32, copy=False))
            labels.append(int(label.item()))

        feature_array = np.stack(features, axis=0).astype(np.float32, copy=False)
        reference_outputs = _run_reference_batches(model, feature_array)
        reference_predictions = np.argmax(reference_outputs, axis=1).astype(np.int64).tolist()
        class_names = list(config["data"]["classes"])
        reference_metrics = _compute_classification_metrics(
            labels,
            reference_predictions,
            class_names,
        )
        reference_metrics["logit_min"] = float(np.min(reference_outputs))
        reference_metrics["logit_max"] = float(np.max(reference_outputs))

        return {
            "available": True,
            "config_path": str(config_path.resolve()),
            "split_name": split_name,
            "sample_count": int(feature_array.shape[0]),
            "class_names": class_names,
            "features_nchw": feature_array,
            "labels": labels,
            "reference_outputs": reference_outputs.astype(np.float32, copy=False),
            "reference_predictions": reference_predictions,
            "reference_metrics": reference_metrics,
        }
    except Exception as exc:
        return {
            "available": False,
            "reason": f"{exc.__class__.__name__}: {exc}",
        }


def _run_synthetic_smoke(
    *,
    precision: Precision,
    interpreter,
    cases: dict[str, np.ndarray],
    reference_outputs: dict[str, np.ndarray],
    shape_nchw: list[int],
) -> tuple[list[ValidationCaseResult], dict[str, Any]]:
    """执行 synthetic smoke，只验证候选能否稳定跑通。"""

    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    results: list[ValidationCaseResult] = []
    all_finite = True

    for case_name, array_nchw in cases.items():
        prepared_input = _prepare_runtime_input(array_nchw, input_detail, shape_nchw)
        interpreter.set_tensor(int(input_detail["index"]), prepared_input)
        interpreter.invoke()
        candidate_output = _read_runtime_output(interpreter, output_detail)
        if not np.isfinite(candidate_output).all():
            all_finite = False

        reference_output = reference_outputs[case_name]
        metrics = _compute_metrics(reference_output, candidate_output)
        metrics["output_is_finite"] = bool(np.isfinite(candidate_output).all())
        passed = bool(metrics["output_is_finite"])
        notes = []
        if _dtype_name(input_detail.get("dtype")) in {"int8", "uint8", "int16"}:
            notes.append("Compared after quantized input preparation and dequantized output recovery.")
        notes.append(
            "Synthetic smoke only checks runtime viability; task gating happens on real features when available."
        )
        results.append(
            ValidationCaseResult(
                group="synthetic_smoke",
                precision=precision,
                target=case_name,
                passed=passed,
                metrics=metrics,
                notes=notes,
            )
        )

    worst_max_abs = max(float(result.metrics["max_abs_error"]) for result in results)
    worst_mean_abs = max(float(result.metrics["mean_abs_error"]) for result in results)
    all_top1_match = all(bool(result.metrics.get("top1_match", True)) for result in results)
    passed_case_count = sum(1 for result in results if result.passed)
    summary = {
        "pass": passed_case_count == len(results) and all_finite,
        "case_count": len(results),
        "passed_case_count": passed_case_count,
        "all_finite": all_finite,
        "all_top1_match": all_top1_match,
        "worst_max_abs_error": worst_max_abs,
        "worst_mean_abs_error": worst_mean_abs,
    }
    return results, summary


def _run_real_task_validation(
    *,
    candidate: CandidateArtifact,
    interpreter,
    task_bundle: dict[str, Any],
    shape_nchw: list[int],
) -> tuple[ValidationCaseResult | None, dict[str, Any]]:
    """重放真实特征，并按任务指标判定候选是否通过。"""

    if not task_bundle["available"]:
        return None, {
            "available": False,
            "reason": task_bundle.get("reason"),
        }

    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    candidate_outputs: list[np.ndarray] = []
    candidate_predictions: list[int] = []
    labels = list(task_bundle["labels"])

    for array_chw in task_bundle["features_nchw"]:
        sample_nchw = np.expand_dims(array_chw, axis=0)
        prepared_input = _prepare_runtime_input(sample_nchw, input_detail, shape_nchw)
        interpreter.set_tensor(int(input_detail["index"]), prepared_input)
        interpreter.invoke()
        candidate_output = _read_runtime_output(interpreter, output_detail).reshape(-1)
        if not np.isfinite(candidate_output).all():
            raise PipelineStageError(
                stage="validate",
                reason=f"Candidate produced non-finite outputs during real-feature validation: {candidate.precision}",
            )
        candidate_outputs.append(candidate_output.astype(np.float32, copy=False))
        candidate_predictions.append(int(np.argmax(candidate_output)))

    candidate_output_array = np.stack(candidate_outputs, axis=0).astype(np.float32, copy=False)
    reference_output_array = task_bundle["reference_outputs"]
    class_names = task_bundle["class_names"]
    candidate_metrics = _compute_classification_metrics(labels, candidate_predictions, class_names)
    reference_top1_match_rate = float(
        np.mean(
            np.array(candidate_predictions, dtype=np.int64)
            == np.array(task_bundle["reference_predictions"], dtype=np.int64)
        )
    )
    logit_diff = np.abs(reference_output_array - candidate_output_array)
    accuracy_drop = float(task_bundle["reference_metrics"]["accuracy"] - candidate_metrics["accuracy"])
    macro_f1_drop = float(task_bundle["reference_metrics"]["macro_f1"] - candidate_metrics["macro_f1"])
    threshold = default_validation_thresholds()["task_metrics"][candidate.precision]
    passed = _passes_task_thresholds(
        accuracy_drop=accuracy_drop,
        macro_f1_drop=macro_f1_drop,
        reference_top1_match_rate=reference_top1_match_rate,
        threshold=threshold,
    )

    metrics = {
        "split": task_bundle["split_name"],
        "sample_count": int(task_bundle["sample_count"]),
        "class_names": class_names,
        "candidate_metrics": candidate_metrics,
        "accuracy_drop_vs_reference": accuracy_drop,
        "macro_f1_drop_vs_reference": macro_f1_drop,
        "reference_top1_match_rate": reference_top1_match_rate,
        "mean_abs_logit_diff": float(np.mean(logit_diff)),
        "max_abs_logit_diff": float(np.max(logit_diff)),
        "threshold": threshold,
    }
    notes = [
        "Real-feature validation replays the recovered training frontend on actual audio windows.",
        "Task gating prefers accuracy/F1/top1 consistency over raw synthetic-logit parity.",
    ]
    return (
        ValidationCaseResult(
            group="real_task",
            precision=candidate.precision,
            target=task_bundle["split_name"],
            passed=passed,
            metrics=metrics,
            notes=notes,
        ),
        {
            "available": True,
            "pass": passed,
            "split": task_bundle["split_name"],
            "sample_count": int(task_bundle["sample_count"]),
            "candidate_metrics": candidate_metrics,
            "accuracy_drop_vs_reference": accuracy_drop,
            "macro_f1_drop_vs_reference": macro_f1_drop,
            "reference_top1_match_rate": reference_top1_match_rate,
            "mean_abs_logit_diff": float(np.mean(logit_diff)),
            "max_abs_logit_diff": float(np.max(logit_diff)),
            "threshold": threshold,
        },
    )


def _build_tflite_interpreter(model_path: Path | None):
    """构造不带默认 delegate 的 TFLite 解释器。"""

    if tf is None:
        raise PipelineStageError(
            stage="validate",
            reason="TensorFlow is required to validate TFLite candidates",
        )
    if model_path is None:
        raise PipelineStageError(
            stage="validate",
            reason="TFLite candidate path is missing",
        )

    resolver_type = tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
    interpreter = tf.lite.Interpreter(
        model_path=str(model_path),
        experimental_op_resolver_type=resolver_type,
    )
    interpreter.allocate_tensors()
    return interpreter


def _prepare_runtime_input(
    array_nchw: np.ndarray,
    input_detail: dict[str, Any],
    shape_nchw: list[int],
) -> np.ndarray:
    """按候选输入 dtype 和布局准备运行时输入。"""

    runtime_array = _transpose_to_runtime_layout(array_nchw, input_detail, shape_nchw)
    dtype_name = _dtype_name(input_detail.get("dtype"))
    if dtype_name in {"int8", "uint8", "int16"}:
        quant = input_detail.get("quantization", (0.0, 0))
        scale = float(quant[0]) if isinstance(quant, tuple) else float(quant.get("scale", 0.0))
        zero_point = int(quant[1]) if isinstance(quant, tuple) else int(quant.get("zero_point", 0))
        return _quantize_input(runtime_array, dtype_name, scale, zero_point)
    if dtype_name == "float16":
        return runtime_array.astype(np.float16, copy=False)
    return runtime_array.astype(np.float32, copy=False)


def _transpose_to_runtime_layout(
    array_nchw: np.ndarray,
    input_detail: dict[str, Any],
    shape_nchw: list[int],
) -> np.ndarray:
    """把参考 NCHW 输入映射到 TFLite 运行时实际布局。"""

    runtime_shape = [int(value) for value in input_detail.get("shape", [])]
    source_shape = list(array_nchw.shape)
    if runtime_shape == source_shape:
        return array_nchw

    if len(shape_nchw) == 4:
        expected_nhwc = [shape_nchw[0], shape_nchw[2], shape_nchw[3], shape_nchw[1]]
        if runtime_shape == expected_nhwc:
            return np.transpose(array_nchw, (0, 2, 3, 1))
    if len(shape_nchw) == 3:
        expected_nwc = [shape_nchw[0], shape_nchw[2], shape_nchw[1]]
        if runtime_shape == expected_nwc:
            return np.transpose(array_nchw, (0, 2, 1))

    raise PipelineStageError(
        stage="validate",
        reason="Unable to map model_input.shape_nchw to TFLite runtime input layout",
        details={
            "shape_nchw": shape_nchw,
            "runtime_shape": runtime_shape,
            "source_shape": source_shape,
        },
    )


def _quantize_input(
    runtime_array: np.ndarray,
    dtype_name: str,
    scale: float,
    zero_point: int,
) -> np.ndarray:
    """把浮点输入量化到候选期望的整型输入域。"""

    if scale <= 0.0:
        raise PipelineStageError(
            stage="validate",
            reason="Quantized TFLite input is missing a valid scale",
            details={"dtype": dtype_name, "scale": scale},
        )

    dtype = np.dtype(dtype_name)
    info = np.iinfo(dtype)
    quantized = np.round(runtime_array / scale + zero_point)
    return np.clip(quantized, info.min, info.max).astype(dtype, copy=False)


def _read_runtime_output(interpreter, output_detail: dict[str, Any]) -> np.ndarray:
    """读取候选输出，并在需要时做反量化恢复。"""

    raw = interpreter.get_tensor(int(output_detail["index"]))
    dtype_name = _dtype_name(output_detail.get("dtype"))
    if dtype_name in {"int8", "uint8", "int16"}:
        quant = output_detail.get("quantization", (0.0, 0))
        scale = float(quant[0]) if isinstance(quant, tuple) else float(quant.get("scale", 0.0))
        zero_point = int(quant[1]) if isinstance(quant, tuple) else int(quant.get("zero_point", 0))
        if scale <= 0.0:
            raise PipelineStageError(
                stage="validate",
                reason="Quantized TFLite output is missing a valid scale",
                details={"dtype": dtype_name, "scale": scale},
            )
        return (raw.astype(np.float32) - float(zero_point)) * scale
    return raw.astype(np.float32, copy=False)


def _compute_metrics(reference_output: np.ndarray, candidate_output: np.ndarray) -> dict[str, Any]:
    """计算 synthetic smoke 所需的基础误差指标。"""

    if reference_output.shape != candidate_output.shape:
        raise PipelineStageError(
            stage="validate",
            reason="Reference and candidate outputs have different shapes",
            details={
                "reference_shape": list(reference_output.shape),
                "candidate_shape": list(candidate_output.shape),
            },
        )

    diff = np.abs(reference_output.astype(np.float32) - candidate_output.astype(np.float32))
    ref_top1 = np.argmax(reference_output, axis=-1).reshape(-1).tolist()
    candidate_top1 = np.argmax(candidate_output, axis=-1).reshape(-1).tolist()
    return {
        "max_abs_error": float(np.max(diff)),
        "mean_abs_error": float(np.mean(diff)),
        "top1_match": ref_top1 == candidate_top1,
        "reference_top1": ref_top1,
        "candidate_top1": candidate_top1,
    }


def _passes_task_thresholds(
    *,
    accuracy_drop: float,
    macro_f1_drop: float,
    reference_top1_match_rate: float,
    threshold: dict[str, float],
) -> bool:
    """判断真实任务指标是否满足当前精度阈值。"""

    if accuracy_drop > float(threshold["max_accuracy_drop"]):
        return False
    if macro_f1_drop > float(threshold["max_macro_f1_drop"]):
        return False
    if reference_top1_match_rate < float(threshold["min_reference_top1_match_rate"]):
        return False
    return True


def _extract_input_shape(process_contract: ProcessContract) -> list[int]:
    """从 process contract 提取统一的参考输入形状。"""

    shape = process_contract.model_input.get("shape_nchw")
    if not isinstance(shape, list) or not shape:
        raise PipelineStageError(
            stage="validate",
            reason="Process contract does not provide model_input.shape_nchw",
        )
    return [int(value) for value in shape]


def _select_real_validation_records(split_bundle) -> tuple[str, list[Any]]:
    """按 test -> val -> train 的优先级挑选真实验证集。"""

    split_map = {
        "train": list(split_bundle.train),
        "val": list(split_bundle.val),
        "test": list(split_bundle.test),
    }
    for split_name in REAL_VALIDATION_SPLIT_PRIORITY:
        records = split_map[split_name]
        if records:
            return split_name, records
    raise RuntimeError("No records are available for real-feature validation.")


def _run_reference_batches(model: nn.Module, feature_array: np.ndarray) -> np.ndarray:
    """分批跑参考模型，避免真实验证样本过多时内存峰值过高。"""

    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start_index in range(0, int(feature_array.shape[0]), REFERENCE_BATCH_SIZE):
            batch = torch.from_numpy(
                feature_array[start_index:start_index + REFERENCE_BATCH_SIZE]
            ).to(dtype=torch.float32)
            logits = model(batch).detach().cpu().numpy().astype(np.float32, copy=False)
            outputs.append(logits)
    return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)


def _compute_classification_metrics(
    labels: list[int],
    predictions: list[int],
    class_names: list[str],
) -> dict[str, Any]:
    """计算真实任务验证使用的分类指标。"""

    class_count = len(class_names)
    confusion = [[0 for _ in range(class_count)] for _ in range(class_count)]
    for truth, pred in zip(labels, predictions, strict=True):
        confusion[truth][pred] += 1

    per_class = {}
    total_correct = 0
    for class_index, class_name in enumerate(class_names):
        true_positive = confusion[class_index][class_index]
        total_correct += true_positive
        predicted_positive = sum(confusion[row][class_index] for row in range(class_count))
        actual_positive = sum(confusion[class_index][col] for col in range(class_count))
        precision = true_positive / float(predicted_positive) if predicted_positive else 0.0
        recall = true_positive / float(actual_positive) if actual_positive else 0.0
        f1 = (
            2.0 * precision * recall / float(precision + recall)
            if (precision + recall) > 0.0
            else 0.0
        )
        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": actual_positive,
        }

    accuracy = total_correct / float(len(labels)) if labels else 0.0
    macro_precision = (
        sum(metrics["precision"] for metrics in per_class.values()) / float(class_count)
        if class_count
        else 0.0
    )
    macro_recall = (
        sum(metrics["recall"] for metrics in per_class.values()) / float(class_count)
        if class_count
        else 0.0
    )
    macro_f1 = (
        sum(metrics["f1"] for metrics in per_class.values()) / float(class_count)
        if class_count
        else 0.0
    )
    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "confusion_matrix": confusion,
        "per_class": per_class,
    }


def _require_torch_model(restore_result: ModelRestoreResult) -> nn.Module:
    """确认 restore 结果里确实拿到了可执行的 PyTorch 模型。"""

    if torch is None or nn is None:
        raise PipelineStageError(
            stage="validate",
            reason="PyTorch is required to validate exported candidates",
        )
    if not isinstance(restore_result.model, nn.Module):
        raise PipelineStageError(
            stage="validate",
            reason="Restored model is not a torch.nn.Module instance",
            details={"model_type": type(restore_result.model).__name__},
        )
    return restore_result.model.eval().cpu()


def _dtype_name(dtype: Any) -> str:
    """把 numpy / tensorflow dtype 统一转换成字符串名。"""

    if dtype is None:
        return "unknown"
    try:
        return np.dtype(dtype).name
    except Exception:
        return getattr(dtype, "name", str(dtype))
