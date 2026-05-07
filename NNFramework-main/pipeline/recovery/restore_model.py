"""Restore PyTorch models from source code and checkpoint bundles."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import re
import sys
from typing import Any

from ..common.checkpoint_loader import extract_state_dict, load_torch_checkpoint

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - import guard
    torch = None
    nn = None

from ..exceptions import PipelineStageError
from ..intake.inspect_yaml import load_yaml_config
from ..types import InputBundle, ModelIdentity, ModelRestoreResult


def natural_sort_key(text: str) -> list[Any]:
    parts = re.split(r"(\d+)", text)
    key: list[Any] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part)
    return key


def load_python_module(py_path: Path):
    spec = importlib.util.spec_from_file_location("pipeline_user_model_module", py_path)
    if spec is None or spec.loader is None:
        raise PipelineStageError(
            stage="model_restore",
            reason=f"Unable to load model definition from {py_path}",
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_user_model_module"] = module
    spec.loader.exec_module(module)
    return module


def find_model_class(module, preferred_class_name: str | None) -> type[nn.Module]:
    if nn is None:
        raise PipelineStageError(
            stage="model_restore",
            reason="PyTorch is required to inspect the model class",
        )

    candidates: list[tuple[str, type[nn.Module]]] = []
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, nn.Module) and obj is not nn.Module:
            candidates.append((name, obj))

    if not candidates:
        raise PipelineStageError(
            stage="model_restore",
            reason="No torch.nn.Module subclass found in model.py",
        )

    if preferred_class_name:
        for name, obj in candidates:
            if name == preferred_class_name:
                return obj
        raise PipelineStageError(
            stage="model_restore",
            reason=f"Requested model class {preferred_class_name} was not found",
            details={"available_classes": [name for name, _obj in candidates]},
        )

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


def merge_optional_values(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def normalize_channels(value: Any) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and value:
        return [int(item) for item in value]
    return None


def infer_channels_from_state_dict(state_dict: dict[str, Any]) -> list[int] | None:
    if torch is None:
        return None

    conv_weights: list[tuple[str, torch.Tensor]] = []
    for key, value in state_dict.items():
        if hasattr(value, "ndim") and value.ndim == 4 and key.endswith(".weight"):
            conv_weights.append((key, value))
    if not conv_weights:
        return None
    conv_weights.sort(key=lambda item: natural_sort_key(item[0]))
    return [int(weight.shape[0]) for _key, weight in conv_weights]


def infer_hidden_dim_from_state_dict(state_dict: dict[str, Any]) -> int | None:
    if torch is None:
        return None

    linear_weights: list[tuple[str, torch.Tensor]] = []
    for key, value in state_dict.items():
        if hasattr(value, "ndim") and value.ndim == 2 and key.endswith(".weight"):
            linear_weights.append((key, value))
    if not linear_weights:
        return None
    linear_weights.sort(key=lambda item: natural_sort_key(item[0]))
    return int(linear_weights[0][1].shape[0])


def infer_num_classes_from_state_dict(state_dict: dict[str, Any]) -> int | None:
    if torch is None:
        return None

    linear_weights: list[tuple[str, torch.Tensor]] = []
    for key, value in state_dict.items():
        if hasattr(value, "ndim") and value.ndim == 2 and key.endswith(".weight"):
            linear_weights.append((key, value))
    if linear_weights:
        linear_weights.sort(key=lambda item: natural_sort_key(item[0]))
        return int(linear_weights[-1][1].shape[0])
    return None


def build_model_kwargs(
    model_cls: type[nn.Module],
    yaml_config: dict[str, Any],
    checkpoint_metadata: dict[str, Any],
    state_dict: dict[str, Any],
) -> dict[str, Any]:
    model_cfg = yaml_config.get("model", {})
    data_cfg = yaml_config.get("data", {})
    yaml_classes = data_cfg.get("classes")

    inferred_num_classes = infer_num_classes_from_state_dict(state_dict)
    inferred_channels = infer_channels_from_state_dict(state_dict)
    inferred_hidden_dim = infer_hidden_dim_from_state_dict(state_dict)

    known_values = {
        "num_classes": merge_optional_values(
            len(yaml_classes) if isinstance(yaml_classes, list) else None,
            checkpoint_metadata.get("classes") and len(checkpoint_metadata["classes"]),
            inferred_num_classes,
        ),
        "channels": merge_optional_values(
            normalize_channels(model_cfg.get("channels")),
            normalize_channels(checkpoint_metadata.get("channels")),
            inferred_channels,
        ),
        "hidden_dim": merge_optional_values(
            model_cfg.get("hidden_dim"),
            checkpoint_metadata.get("hidden_dim"),
            inferred_hidden_dim,
        ),
        "dropout": merge_optional_values(
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
            raise PipelineStageError(
                stage="model_restore",
                reason=f"Unable to infer required constructor argument: {name}",
                details={"model_class": model_cls.__name__},
            )
    return kwargs


def restore_model_from_bundle(
    bundle: InputBundle,
    identity: ModelIdentity,
) -> ModelRestoreResult:
    if torch is None or nn is None:
        raise PipelineStageError(
            stage="model_restore",
            reason="PyTorch is not installed in the current Python environment",
        )

    yaml_config = load_yaml_config(bundle.config)
    module = load_python_module(bundle.model_py)
    model_cls = find_model_class(module, identity.model_class_name)

    checkpoint, checkpoint_load_mode = load_torch_checkpoint(
        bundle.checkpoint,
        map_location="cpu",
        model_module=module,
    )
    checkpoint_metadata = extract_checkpoint_metadata(checkpoint)

    if isinstance(checkpoint, nn.Module):
        resolved_load_mode = "torchscript" if checkpoint_load_mode == "torchscript" else "full_module"
        identity.model_class_name = model_cls.__name__
        identity.load_mode = resolved_load_mode
        return ModelRestoreResult(
            model=checkpoint.eval().cpu(),
            model_class_name=model_cls.__name__,
            model_kwargs={},
            checkpoint_metadata=checkpoint_metadata,
            load_mode=resolved_load_mode,
        )

    state_dict, source = extract_state_dict(checkpoint)
    if state_dict is None:
        raise PipelineStageError(
            stage="model_restore",
            reason="Checkpoint is neither a full nn.Module nor a recognized state_dict payload",
        )

    model_kwargs = build_model_kwargs(
        model_cls=model_cls,
        yaml_config=yaml_config,
        checkpoint_metadata=checkpoint_metadata,
        state_dict=state_dict,
    )
    model = model_cls(**model_kwargs)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        raise PipelineStageError(
            stage="model_restore",
            reason="State dict load mismatch after model reconstruction",
            details={
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
            },
        )

    identity.model_class_name = model_cls.__name__
    identity.load_mode = source
    return ModelRestoreResult(
        model=model.eval().cpu(),
        model_class_name=model_cls.__name__,
        model_kwargs=model_kwargs,
        checkpoint_metadata=checkpoint_metadata,
        load_mode=source,
    )
