"""恢复前处理与模型输入输出合同。"""

from __future__ import annotations

from typing import Any

from ..common.frontend_shape_utils import infer_feature_input_shape_nchw

from ..board_runtime_contract import (
    build_model_input_contract_metadata,
    summarize_board_runtime_contract,
)
from ..exceptions import PipelineStageError
from ..intake.inspect_yaml import load_yaml_config
from ..types import InputBundle, ModelRestoreResult, PreprocessStep, ProcessContract
from .recover_labels import recover_labels


# 汇总恢复原始输入、前处理和模型 IO 合同。
def recover_process_contract(
    bundle: InputBundle,
    restore_result: ModelRestoreResult,
) -> ProcessContract:
    yaml_config = load_yaml_config(bundle.config)
    checkpoint_metadata = restore_result.checkpoint_metadata
    labels = recover_labels(yaml_config, checkpoint_metadata)

    # 当前约定优先从 YAML 恢复流程，再用 checkpoint 元数据补全缺失字段。
    raw_input = recover_raw_input_contract(yaml_config, checkpoint_metadata)
    preprocess = recover_preprocess_steps(yaml_config, checkpoint_metadata)
    model_input = recover_model_input_contract(yaml_config, checkpoint_metadata)
    contract = ProcessContract(
        raw_input=raw_input,
        preprocess=preprocess,
        model_input=model_input,
        model_output=recover_model_output_contract(restore_result, labels),
        labels=labels,
        postprocess=[],
        assumptions=_build_contract_assumptions(raw_input, preprocess, model_input),
    )

    # 将当前生成固件运行时的已知约束也落盘到 assumptions，便于后续包侧预检直接复用。
    runtime_summary = summarize_board_runtime_contract(contract)
    contract.assumptions.extend(runtime_summary["warnings"])
    return contract


# 恢复原始数据侧的域信息和采样参数。
def recover_raw_input_contract(
    yaml_config: dict[str, Any],
    checkpoint_metadata: dict[str, Any],
) -> dict[str, Any]:
    audio_cfg = yaml_config.get("audio", {})
    sample_rate_hz = _merge(audio_cfg.get("sample_rate"), checkpoint_metadata.get("sample_rate"))
    window_sec = _merge(audio_cfg.get("window_sec"), checkpoint_metadata.get("window_sec"))
    pad_short = audio_cfg.get("pad_short")

    missing = []
    if sample_rate_hz is None:
        missing.append("audio.sample_rate")
    if window_sec is None:
        missing.append("audio.window_sec")
    if pad_short is None:
        missing.append("audio.pad_short")
    if missing:
        raise PipelineStageError(
            stage="recovery",
            reason="Unable to recover raw input contract",
            details={"missing_fields": missing},
        )

    return {
        "domain": "audio",
        "sample_rate_hz": int(sample_rate_hz),
        "window_sec": float(window_sec),
        "pad_short": bool(pad_short),
    }


# 将 YAML 中的前处理配置规范化为步骤列表。
def recover_preprocess_steps(
    yaml_config: dict[str, Any],
    checkpoint_metadata: dict[str, Any],
) -> list[PreprocessStep]:
    del checkpoint_metadata
    audio_cfg = yaml_config.get("audio", {})
    feature_cfg = yaml_config.get("feature", {})
    bandpass_cfg = audio_cfg.get("bandpass", {})

    # 先按固定顺序整理前处理步骤，后续板端复刻直接复用该顺序。
    steps = [
        PreprocessStep(
            name="bandpass",
            enabled=bool(bandpass_cfg.get("enabled", False)),
            params={
                "low_hz": bandpass_cfg.get("low_hz"),
                "high_hz": bandpass_cfg.get("high_hz"),
                "order": bandpass_cfg.get("order"),
            },
            source="yaml",
        ),
        PreprocessStep(
            name="stft",
            enabled=True,
            params={
                "n_fft": feature_cfg.get("n_fft"),
                "win_length": feature_cfg.get("win_length"),
                "hop_length": feature_cfg.get("hop_length"),
                "center": feature_cfg.get("center"),
            },
            source="yaml",
        ),
        PreprocessStep(
            name="mel",
            enabled=True,
            params={
                "n_mels": feature_cfg.get("n_mels"),
                "fmin": feature_cfg.get("fmin"),
                "fmax": feature_cfg.get("fmax"),
                "power": feature_cfg.get("power"),
            },
            source="yaml",
        ),
        PreprocessStep(
            name="db_compression",
            enabled=True,
            params={"top_db": feature_cfg.get("top_db")},
            source="yaml",
        ),
        PreprocessStep(
            name="normalize",
            enabled=bool(feature_cfg.get("normalize", False)),
            params={"mode": "per_sample_standardize" if feature_cfg.get("normalize") else "disabled"},
            source="yaml",
        ),
    ]

    missing: list[str] = []
    for step in steps:
        if not step.enabled:
            continue
        for key, value in step.params.items():
            if value is None:
                missing.append(f"{step.name}.{key}")
    if missing:
        raise PipelineStageError(
            stage="recovery",
            reason="Unable to recover complete preprocess contract",
            details={"missing_fields": missing},
        )
    return steps


# 恢复模型输入的形状、语义和数据类型。
def recover_model_input_contract(
    yaml_config: dict[str, Any],
    checkpoint_metadata: dict[str, Any],
) -> dict[str, Any]:
    input_shape = infer_input_shape_from_metadata(yaml_config, checkpoint_metadata)
    if input_shape is None:
        raise PipelineStageError(
            stage="recovery",
            reason="Unable to infer model input shape from yaml and checkpoint metadata",
        )

    # 保留 shape_nchw 作为 canonical 合同，同时补齐布局和特征网格元数据。
    return {
        "shape_nchw": input_shape,
        "dtype": "float32",
        "semantic": "log_mel_feature",
        **build_model_input_contract_metadata(input_shape),
    }


# 恢复模型输出的任务类型与类别规模。
def recover_model_output_contract(
    restore_result: ModelRestoreResult,
    labels: list[str],
) -> dict[str, Any]:
    class_count = len(labels)
    if class_count <= 0:
        raise PipelineStageError(
            stage="recovery",
            reason="Labels recovered but class count is zero",
        )
    return {
        "type": "classification",
        "class_count": class_count,
        "load_mode": restore_result.load_mode,
    }


# 从 YAML 和检查点元数据推导模型输入形状。
def infer_input_shape_from_metadata(
    yaml_config: dict[str, Any],
    checkpoint_metadata: dict[str, Any],
) -> list[int] | None:
    audio_cfg = yaml_config.get("audio", {})
    feature_cfg = yaml_config.get("feature", {})
    sample_rate = _merge(audio_cfg.get("sample_rate"), checkpoint_metadata.get("sample_rate"))
    window_sec = _merge(audio_cfg.get("window_sec"), checkpoint_metadata.get("window_sec"))
    n_mels = _merge(feature_cfg.get("n_mels"), checkpoint_metadata.get("n_mels"))
    n_fft = _merge(
        feature_cfg.get("n_fft"),
        checkpoint_metadata.get("n_fft"),
        feature_cfg.get("win_length"),
        checkpoint_metadata.get("win_length"),
    )
    win_length = _merge(
        feature_cfg.get("win_length"),
        checkpoint_metadata.get("win_length"),
        feature_cfg.get("n_fft"),
        checkpoint_metadata.get("n_fft"),
    )
    hop_length = _merge(feature_cfg.get("hop_length"), checkpoint_metadata.get("hop_length"))
    center = _merge(feature_cfg.get("center"), checkpoint_metadata.get("center"), False)
    if None in (sample_rate, window_sec, n_mels, n_fft, win_length, hop_length):
        return None

    return infer_feature_input_shape_nchw(
        sample_rate_hz=int(sample_rate),
        window_sec=float(window_sec),
        n_fft=int(n_fft),
        win_length=int(win_length),
        hop_length=int(hop_length),
        n_mels=int(n_mels),
        center=bool(center),
    )


# 将当前恢复结果里的强假设和板端运行时约束整理为 assumptions。
def _build_contract_assumptions(
    raw_input: dict[str, Any],
    preprocess: list[PreprocessStep],
    model_input: dict[str, Any],
) -> list[str]:
    contract = ProcessContract(
        raw_input=raw_input,
        preprocess=preprocess,
        model_input=model_input,
        model_output={},
        labels=[],
        postprocess=[],
        assumptions=[],
    )
    runtime_summary = summarize_board_runtime_contract(contract)
    assumptions = [
        "Recovered frontend tensors use canonical single-channel NCHW layout before any runtime-specific layout adaptation.",
    ]
    if runtime_summary["unsupported_items"]:
        assumptions.append(
            "Current generated board runtime preflight would reject this contract until the unsupported items are resolved: "
            + ", ".join(runtime_summary["unsupported_items"])
        )
    return assumptions


# 从多个候选值中取第一个非空结果。
def _merge(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None
