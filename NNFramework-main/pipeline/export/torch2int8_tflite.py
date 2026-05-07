import argparse
import importlib.util
import inspect
import json
import pathlib
import re
import subprocess
import sys
from typing import Any

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.common.runtime_env import ensure_default_python_for_script

ensure_default_python_for_script(__file__)

from pipeline.common.checkpoint_loader import extract_state_dict, load_torch_checkpoint
from pipeline.common.calibration_targets import build_default_target_counts, resolve_calibration_sample_count
from pipeline.common.frontend_shape_utils import infer_feature_input_shape_nchw
from pipeline.common.onnx_export_config import DEFAULT_ONNX_EXPORT_DYNAMO
from pipeline.common.representative_calibration import generate_representative_calibration_npy

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn

try:
    import yaml
except ImportError:
    yaml = None

from pipeline.export.torch2tflite import (
    _classify_tflite_variant,
    _collect_conversion_candidates,
    _select_tflite_model,
    _snapshot_tflite_files,
)


ROOT = PROJECT_ROOT
DEFAULT_MODEL_PY = ROOT / "in" / "model.py"
DEFAULT_CHECKPOINT = ROOT / "in" / "model.pt"
DEFAULT_OUTPUT_DIR = ROOT / "intermediate" / "int8_tflite"
DEFAULT_INPUT_NAME = "input"
DEFAULT_OUTPUT_NAME = "output"
DEFAULT_FALLBACK_INPUT_SHAPE = [1, 1, 48, 94]


def natural_sort_key(text: str) -> list[Any]:
    parts = re.split(r"(\d+)", text)
    key: list[Any] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part)
    return key


def load_yaml_config(config_path: pathlib.Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    if yaml is None:
        raise ImportError(
            "PyYAML is not installed. Install it into D:\\work\\AnacondaEnvironment "
            "to load the YAML config passed by --config."
        )
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"配置文件不是有效的字典结构: {config_path}")
    return config


def load_python_module(py_path: pathlib.Path):
    spec = importlib.util.spec_from_file_location("user_model_module", py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法从 {py_path} 加载模型定义文件")

    module = importlib.util.module_from_spec(spec)
    sys.modules["user_model_module"] = module
    spec.loader.exec_module(module)
    return module


def find_model_class(module, preferred_class_name: str | None) -> type[nn.Module]:
    candidates: list[tuple[str, type[nn.Module]]] = []
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, nn.Module) and obj is not nn.Module:
            candidates.append((name, obj))

    if not candidates:
        raise LookupError("模型定义文件中没有找到 torch.nn.Module 子类")

    if preferred_class_name:
        for name, obj in candidates:
            if name == preferred_class_name:
                return obj
        names = ", ".join(name for name, _ in candidates)
        raise LookupError(f"未找到指定模型类 {preferred_class_name}，可选项: {names}")

    return candidates[0][1]


def find_nested_value(obj: Any, target_key: str) -> Any:
    if isinstance(obj, dict):
        if target_key in obj:
            return obj[target_key]
        for value in obj.values():
            found = find_nested_value(value, target_key)
            if found is not None:
                return found
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            found = find_nested_value(value, target_key)
            if found is not None:
                return found
    return None


def merge_optional_values(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def infer_channels_from_state_dict(state_dict: dict[str, Any]) -> list[int] | None:
    conv_weights: list[tuple[str, torch.Tensor]] = []
    for key, value in state_dict.items():
        if hasattr(value, "ndim") and value.ndim == 4 and key.endswith(".weight"):
            conv_weights.append((key, value))

    if not conv_weights:
        return None

    conv_weights.sort(key=lambda item: natural_sort_key(item[0]))
    return [int(weight.shape[0]) for _, weight in conv_weights]


def infer_hidden_dim_from_state_dict(state_dict: dict[str, Any]) -> int | None:
    linear_weights: list[tuple[str, torch.Tensor]] = []
    for key, value in state_dict.items():
        if hasattr(value, "ndim") and value.ndim == 2 and key.endswith(".weight"):
            linear_weights.append((key, value))

    if not linear_weights:
        return None

    linear_weights.sort(key=lambda item: natural_sort_key(item[0]))
    return int(linear_weights[0][1].shape[0])


def infer_num_classes_from_state_dict(state_dict: dict[str, Any]) -> int | None:
    linear_weights: list[tuple[str, torch.Tensor]] = []
    for key, value in state_dict.items():
        if hasattr(value, "ndim") and value.ndim == 2 and key.endswith(".weight"):
            linear_weights.append((key, value))

    if linear_weights:
        linear_weights.sort(key=lambda item: natural_sort_key(item[0]))
        return int(linear_weights[-1][1].shape[0])

    linear_biases: list[tuple[str, torch.Tensor]] = []
    for key, value in state_dict.items():
        if hasattr(value, "ndim") and value.ndim == 1 and key.endswith(".bias"):
            linear_biases.append((key, value))

    if not linear_biases:
        return None

    linear_biases.sort(key=lambda item: natural_sort_key(item[0]))
    return int(linear_biases[-1][1].shape[0])


def extract_checkpoint_metadata(checkpoint: Any) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        return {}

    metadata: dict[str, Any] = {}
    for key in (
        "classes",
        "channels",
        "hidden_dim",
        "dropout",
        "sample_rate",
        "window_sec",
        "win_length",
        "n_fft",
        "hop_length",
        "n_mels",
        "center",
    ):
        metadata[key] = find_nested_value(checkpoint, key)
    return metadata


def normalize_channels(value: Any) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and value:
        return [int(item) for item in value]
    return None


def build_model_kwargs(
    model_cls: type[nn.Module],
    yaml_config: dict[str, Any],
    checkpoint_metadata: dict[str, Any],
    state_dict: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    model_cfg = yaml_config.get("model", {})
    data_cfg = yaml_config.get("data", {})
    yaml_classes = data_cfg.get("classes")

    inferred_num_classes = infer_num_classes_from_state_dict(state_dict)
    inferred_channels = infer_channels_from_state_dict(state_dict)
    inferred_hidden_dim = infer_hidden_dim_from_state_dict(state_dict)

    known_values = {
        "num_classes": merge_optional_values(
            args.num_classes,
            len(yaml_classes) if isinstance(yaml_classes, list) else None,
            checkpoint_metadata.get("classes") and len(checkpoint_metadata["classes"]),
            inferred_num_classes,
        ),
        "channels": merge_optional_values(
            normalize_channels(args.channels),
            normalize_channels(model_cfg.get("channels")),
            normalize_channels(checkpoint_metadata.get("channels")),
            inferred_channels,
        ),
        "hidden_dim": merge_optional_values(
            args.hidden_dim,
            model_cfg.get("hidden_dim"),
            checkpoint_metadata.get("hidden_dim"),
            inferred_hidden_dim,
        ),
        "dropout": merge_optional_values(
            args.dropout,
            model_cfg.get("dropout"),
            checkpoint_metadata.get("dropout"),
            0.0,
        ),
    }

    kwargs: dict[str, Any] = {}
    signature = inspect.signature(model_cls)
    for name, parameter in signature.parameters.items():
        if name == "self":
            continue

        if name in known_values and known_values[name] is not None:
            kwargs[name] = known_values[name]
            continue

        if parameter.default is inspect._empty:
            raise KeyError(
                f"无法自动推导构造参数 {name}。"
                "请通过命令行显式传入，或提供可选的 --config YAML 辅助推导。"
            )

    return kwargs


def restore_model(
    model_py: pathlib.Path,
    checkpoint_path: pathlib.Path,
    config_path: pathlib.Path | None,
    model_class_name: str | None,
    args: argparse.Namespace,
) -> tuple[nn.Module, dict[str, Any], dict[str, Any]]:
    yaml_config = load_yaml_config(config_path)
    module = load_python_module(model_py)
    model_cls = find_model_class(module, model_class_name)

    checkpoint, checkpoint_load_mode = load_torch_checkpoint(
        checkpoint_path,
        map_location="cpu",
        model_module=module,
    )
    checkpoint_metadata = extract_checkpoint_metadata(checkpoint)

    if isinstance(checkpoint, nn.Module):
        resolved_load_mode = "torchscript" if checkpoint_load_mode == "torchscript" else "full_module"
        return checkpoint.eval().cpu(), yaml_config, {
            "load_mode": resolved_load_mode,
            "model_kwargs": {},
            "checkpoint_metadata": checkpoint_metadata,
        }

    state_dict, source = extract_state_dict(checkpoint)
    if state_dict is None:
        raise TypeError(
            "checkpoint 既不是完整 nn.Module，也没有识别到 state_dict/model_state_dict"
        )

    model_kwargs = build_model_kwargs(
        model_cls=model_cls,
        yaml_config=yaml_config,
        checkpoint_metadata=checkpoint_metadata,
        state_dict=state_dict,
        args=args,
    )
    model = model_cls(**model_kwargs)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "权重加载后仍有不匹配项: "
            f"missing={missing_keys}, unexpected={unexpected_keys}"
        )

    return model.eval().cpu(), yaml_config, {
        "load_mode": source,
        "model_kwargs": model_kwargs,
        "checkpoint_metadata": checkpoint_metadata,
    }


def infer_input_shape_from_metadata(
    yaml_config: dict[str, Any],
    restore_meta: dict[str, Any],
) -> list[int] | None:
    model_cfg = yaml_config.get("model", {})
    data_cfg = yaml_config.get("data", {})
    audio_cfg = yaml_config.get("audio", {})
    feature_cfg = yaml_config.get("feature", {})
    metadata = restore_meta.get("checkpoint_metadata", {})

    sample_rate = merge_optional_values(
        audio_cfg.get("sample_rate"),
        metadata.get("sample_rate"),
    )
    window_sec = merge_optional_values(
        audio_cfg.get("window_sec"),
        metadata.get("window_sec"),
    )
    n_mels = merge_optional_values(
        feature_cfg.get("n_mels"),
        metadata.get("n_mels"),
    )
    n_fft = merge_optional_values(
        feature_cfg.get("n_fft"),
        metadata.get("n_fft"),
        feature_cfg.get("win_length"),
        metadata.get("win_length"),
    )
    win_length = merge_optional_values(
        feature_cfg.get("win_length"),
        metadata.get("win_length"),
        feature_cfg.get("n_fft"),
        metadata.get("n_fft"),
    )
    hop_length = merge_optional_values(
        feature_cfg.get("hop_length"),
        metadata.get("hop_length"),
    )
    center = merge_optional_values(
        feature_cfg.get("center"),
        metadata.get("center"),
        False,
    )

    if None in (sample_rate, window_sec, n_mels, n_fft, win_length, hop_length):
        return None

    return infer_feature_input_shape_nchw(
        sample_rate_hz=int(sample_rate),
        window_sec=float(window_sec),
        hop_length=int(hop_length),
        n_mels=int(n_mels),
        center=bool(center),
        n_fft=int(n_fft),
        win_length=int(win_length),
    )


def parse_input_shape(
    input_shape_text: str | None,
    yaml_config: dict[str, Any],
    restore_meta: dict[str, Any],
) -> list[int]:
    if input_shape_text:
        values = [int(part.strip()) for part in input_shape_text.split(",") if part.strip()]
        if not values:
            raise ValueError("input shape 不能为空")
        return values

    inferred_shape = infer_input_shape_from_metadata(yaml_config, restore_meta)
    if inferred_shape is not None:
        return inferred_shape

    print(
        "未能从 checkpoint 或可选 YAML 自动推导输入形状，"
        f"将退回默认值 {DEFAULT_FALLBACK_INPUT_SHAPE}。"
    )
    return DEFAULT_FALLBACK_INPUT_SHAPE.copy()


def export_onnx_model(
    model: nn.Module,
    input_shape: list[int],
    onnx_path: pathlib.Path,
    input_name: str,
    output_name: str,
) -> None:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(*input_shape, dtype=torch.float32)

    # 导出前明确切到推理模式，避免 Dropout/BatchNorm 仍按训练态工作。
    model = model.eval().cpu()
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=[input_name],
        output_names=[output_name],
        dynamo=DEFAULT_ONNX_EXPORT_DYNAMO,
    )


def build_tf_calibration_shape(input_shape: list[int], calibration_samples: int) -> list[int]:
    if len(input_shape) == 4:
        _, channels, height, width = input_shape
        return [calibration_samples, height, width, channels]
    if len(input_shape) == 3:
        _, channels, width = input_shape
        return [calibration_samples, width, channels]
    if len(input_shape) == 2:
        _, width = input_shape
        return [calibration_samples, width]
    raise ValueError(f"暂不支持 rank={len(input_shape)} 的输入形状: {input_shape}")


def build_quant_mean_std(rank: int) -> tuple[str, str]:
    # onnx2tf 在 -oiqt + -cind 模式下要求提供 Float32 校准输入，
    # 并使用 (input - mean) / std 做归一化。
    # 这里先生成 [0,1] 范围的伪校准值，再借助 mean/std 映射到更像
    # 标准化特征图的分布，便于在不关心精度时先跑通 full-int8 链路。
    if rank == 4:
        mean = [[[[0.5]]]]
        std = [[[[0.25]]]]
    elif rank == 3:
        mean = [[[0.5]]]
        std = [[[0.25]]]
    elif rank == 2:
        mean = [[0.5]]
        std = [[0.25]]
    else:
        raise ValueError(f"暂不支持 rank={rank} 的 mean/std 生成")

    return json.dumps(mean), json.dumps(std)


def generate_pseudo_calibration_npy(
    input_shape: list[int],
    output_path: pathlib.Path,
    calibration_samples: int,
    seed: int,
) -> tuple[pathlib.Path, str, str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tf_shape = build_tf_calibration_shape(input_shape, calibration_samples)

    rng = np.random.default_rng(seed)
    # 这里采用“方式二”的伪校准：不使用真实样本，直接生成随机 Float32 数据。
    calibration_data = rng.random(tf_shape, dtype=np.float32)
    np.save(output_path, calibration_data.astype(np.float32, copy=False))

    mean_text, std_text = build_quant_mean_std(len(tf_shape))
    return output_path, mean_text, std_text


def resolve_calibration_request(
    config_path: pathlib.Path | None,
    calibration_samples: int | None,
) -> tuple[int, dict[str, int] | None]:
    if calibration_samples is not None:
        return resolve_calibration_sample_count((), calibration_samples), None

    if config_path is None or not config_path.is_file():
        return 1, None

    try:
        yaml_config = load_yaml_config(config_path)
    except Exception:
        return 1, None

    class_names = yaml_config.get("data", {}).get("classes")
    if not isinstance(class_names, list) or not class_names:
        return 1, None

    normalized_classes = [str(class_name) for class_name in class_names]
    target_counts = build_default_target_counts(normalized_classes)
    return resolve_calibration_sample_count(normalized_classes, None), target_counts


def prepare_int8_calibration(
    *,
    input_shape: list[int],
    output_dir: pathlib.Path,
    config_path: pathlib.Path | None,
    calibration_samples: int | None,
    seed: int,
) -> tuple[pathlib.Path, str, str, dict[str, Any], list[str]]:
    representative_path = output_dir / "representative_calibration_input.npy"
    pseudo_path = output_dir / "pseudo_calibration_input.npy"
    resolved_samples, resolved_target_counts = resolve_calibration_request(config_path, calibration_samples)

    if config_path is not None and config_path.is_file():
        try:
            calibration_path, calibration_mean, calibration_std, representative_metadata = (
                generate_representative_calibration_npy(
                    config_path=config_path,
                    input_shape_nchw=input_shape,
                    output_path=representative_path,
                    calibration_samples=calibration_samples,
                    seed=seed,
                )
            )
            calibration_metadata = {
                "calibration_path": str(calibration_path),
                "representative_calibration": True,
                "pseudo_calibration": False,
                "calibration_source": "representative",
                "calibration_samples": int(representative_metadata.get("actual_samples", resolved_samples)),
                "calibration_requested_samples": int(representative_metadata.get("requested_samples", resolved_samples)),
                "calibration_selection_mode": representative_metadata.get("selection_mode"),
                "calibration_manifest_path": representative_metadata.get("metadata_path"),
                "calibration_split": representative_metadata.get("split"),
                "calibration_class_counts": representative_metadata.get("class_counts"),
                "calibration_target_counts": representative_metadata.get("target_counts"),
                "calibration_config_path": representative_metadata.get("config_path"),
                "calibration_data_root": representative_metadata.get("data_root"),
                "int8_confidence": "higher",
            }
            return (
                calibration_path,
                calibration_mean,
                calibration_std,
                calibration_metadata,
                ["使用真实音频特征生成的代表性校准数据。"],
            )
        except Exception as exc:
            fallback_reason = f"代表性校准生成失败，已回退到伪校准: {exc.__class__.__name__}: {exc}"
    elif config_path is not None:
        fallback_reason = "已提供 --config，但对应文件不存在，已回退到伪校准。"
    else:
        fallback_reason = "未提供 --config，无法构建代表性校准，已回退到伪校准。"

    calibration_path, calibration_mean, calibration_std = generate_pseudo_calibration_npy(
        input_shape=input_shape,
        output_path=pseudo_path,
        calibration_samples=resolved_samples,
        seed=seed,
    )
    calibration_metadata = {
        "calibration_path": str(calibration_path),
        "representative_calibration": False,
        "pseudo_calibration": True,
        "calibration_source": "pseudo",
        "calibration_samples": resolved_samples,
        "calibration_requested_samples": resolved_samples,
        "calibration_selection_mode": "pseudo_fallback",
        "calibration_target_counts": resolved_target_counts,
        "calibration_fallback_reason": fallback_reason,
        "int8_confidence": "low",
    }
    return (
        calibration_path,
        calibration_mean,
        calibration_std,
        calibration_metadata,
        [fallback_reason, "伪校准仅作为兼容回退路径。"],
    )


def run_onnx2tf_full_int8(
    onnx_path: pathlib.Path,
    output_dir: pathlib.Path,
    input_name: str,
    calibration_npy: pathlib.Path,
    calibration_mean: str,
    calibration_std: str,
) -> pathlib.Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    before_snapshot = _snapshot_tflite_files(output_dir)

    command = [
        sys.executable,
        "-m",
        "onnx2tf",
        "-i",
        str(onnx_path),
        "-o",
        str(output_dir),
        "-oiqt",
        "-iqd",
        "int8",
        "-oqd",
        "int8",
        "-qt",
        "per-channel",
        "-cind",
        input_name,
        str(calibration_npy),
        calibration_mean,
        calibration_std,
    ]
    subprocess.run(command, check=True)

    candidates = _collect_conversion_candidates(output_dir, before_snapshot)
    if not candidates:
        raise FileNotFoundError("onnx2tf 执行完成，但没有找到任何新的 .tflite 文件")

    full_int8_candidates: list[pathlib.Path] = []
    for candidate in candidates:
        try:
            interpreter = tf.lite.Interpreter(model_path=str(candidate))
            interpreter.allocate_tensors()
            input_detail = interpreter.get_input_details()[0]
            output_detail = interpreter.get_output_details()[0]
        except Exception:
            continue

        input_is_int8 = np.dtype(input_detail["dtype"]) == np.dtype(np.int8)
        output_is_int8 = np.dtype(output_detail["dtype"]) == np.dtype(np.int8)
        input_scale = float(input_detail.get("quantization", (0.0, 0))[0])
        output_scale = float(output_detail.get("quantization", (0.0, 0))[0])
        if input_is_int8 and output_is_int8 and input_scale > 0.0 and output_scale > 0.0:
            full_int8_candidates.append(candidate)

    if not full_int8_candidates:
        candidate_names = ", ".join(path.name for path in candidates)
        raise RuntimeError(
            "未生成符合预期的 full-int8 TFLite 模型。"
            f" 可用产物: {candidate_names}"
        )

    full_int8_candidates.sort(key=lambda path: (-path.stat().st_mtime_ns, path.name))
    return full_int8_candidates[0]


def write_manifest(
    manifest_path: pathlib.Path,
    model_py: pathlib.Path,
    checkpoint_path: pathlib.Path,
    config_path: pathlib.Path | None,
    onnx_path: pathlib.Path,
    tflite_path: pathlib.Path,
    input_shape: list[int],
    calibration_metadata: dict[str, Any],
    calibration_notes: list[str],
    restore_meta: dict[str, Any],
) -> None:
    payload = {
        "step": "torch_to_int8_tflite",
        "model_py": str(model_py),
        "checkpoint_path": str(checkpoint_path),
        "config_path": str(config_path) if config_path else None,
        "onnx_path": str(onnx_path),
        "tflite_path": str(tflite_path),
        "input_shape_nchw": input_shape,
        "calibration_notes": calibration_notes,
        "load_mode": restore_meta.get("load_mode"),
        "model_kwargs": restore_meta.get("model_kwargs", {}),
        "checkpoint_metadata": restore_meta.get("checkpoint_metadata", {}),
    }
    payload.update(calibration_metadata)
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将 PyTorch 模型导出为 full-int8 TFLite。默认不依赖 YAML，优先从 checkpoint 反推参数。"
    )
    parser.add_argument("--model-py", default=str(DEFAULT_MODEL_PY), help="模型定义 .py 文件路径")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="PyTorch 权重 .pt 文件路径")
    parser.add_argument(
        "--config",
        help="可选的训练配置 .yaml 文件路径。默认不读取 YAML，只在需要额外辅助推导时使用。",
    )
    parser.add_argument("--model-class", help="模型类名；不传时自动取第一个 nn.Module 子类")
    parser.add_argument("--num-classes", type=int, help="显式指定类别数")
    parser.add_argument(
        "--channels",
        help="显式指定卷积通道列表，格式如 16,32,64；不传时尝试从 checkpoint 反推",
    )
    parser.add_argument("--hidden-dim", type=int, help="显式指定隐藏层维度")
    parser.add_argument(
        "--dropout",
        type=float,
        help="显式指定 dropout。若未提供且无法推断，则默认使用 0.0。",
    )
    parser.add_argument(
        "--input-shape",
        help="导出 ONNX 使用的输入形状，格式如 1,1,48,94；不传时先从 checkpoint 推导，再退回默认值",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="输出目录，默认写入 intermediate/int8_tflite",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        help="显式指定校准样本总数。不传时，如 --config 可用则按 prepare_calibration_data.py 的默认类别目标数取样，否则回退为 1。",
    )
    parser.add_argument("--seed", type=int, default=42, help="校准采样与伪校准随机种子")
    args = parser.parse_args()

    model_py = pathlib.Path(args.model_py).resolve()
    checkpoint_path = pathlib.Path(args.checkpoint).resolve()
    config_path = pathlib.Path(args.config).resolve() if args.config else None
    output_dir = pathlib.Path(args.output_dir).resolve()

    model, yaml_config, restore_meta = restore_model(
        model_py=model_py,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        model_class_name=args.model_class,
        args=args,
    )
    input_shape = parse_input_shape(args.input_shape, yaml_config, restore_meta)

    onnx_path = output_dir / "torch_model.onnx"
    manifest_path = output_dir / "torch2int8_tflite_manifest.json"

    export_onnx_model(
        model=model,
        input_shape=input_shape,
        onnx_path=onnx_path,
        input_name=DEFAULT_INPUT_NAME,
        output_name=DEFAULT_OUTPUT_NAME,
    )

    calibration_path, calibration_mean, calibration_std, calibration_metadata, calibration_notes = (
        prepare_int8_calibration(
            input_shape=input_shape,
            output_dir=output_dir,
            config_path=config_path,
            calibration_samples=args.calibration_samples,
            seed=args.seed,
        )
    )

    selected_tflite = run_onnx2tf_full_int8(
        onnx_path=onnx_path,
        output_dir=output_dir,
        input_name=DEFAULT_INPUT_NAME,
        calibration_npy=calibration_path,
        calibration_mean=calibration_mean,
        calibration_std=calibration_std,
    )

    write_manifest(
        manifest_path=manifest_path,
        model_py=model_py,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        onnx_path=onnx_path,
        tflite_path=selected_tflite,
        input_shape=input_shape,
        calibration_metadata=calibration_metadata,
        calibration_notes=calibration_notes,
        restore_meta=restore_meta,
    )

    print(f"ONNX 导出完成: {onnx_path}")
    print(f"校准数据: {calibration_path}")
    print(f"校准来源: {calibration_metadata.get('calibration_source')}")
    for note in calibration_notes:
        print(f"校准说明: {note}")
    print(f"INT8 TFLite: {selected_tflite}")
    print(f"转换清单: {manifest_path}")


if __name__ == "__main__":
    main()
