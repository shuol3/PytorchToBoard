"""汇总当前生成固件运行时与处理合同之间的约束。"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from .types import ProcessContract


# 当前生成固件运行时可接受的输入布局标签。
SUPPORTED_RUNTIME_LAYOUTS = ["NCHW", "NHWC", "MATRIX", "FLAT"]

# 当前生成固件运行时显式覆盖的前处理步骤。
SUPPORTED_GENERATED_PREPROCESS = {"stft", "mel", "db_compression", "normalize"}

# normalize 目前只接受与生成 runner 一致的逐样本标准化模式。
SUPPORTED_NORMALIZE_MODES = {"per_sample_standardize", "disabled"}


# 将 dataclass 或字典形式的合同统一转换为普通映射，便于脚本和 pipeline 复用。
def _normalize_contract(contract: ProcessContract | dict[str, Any]) -> dict[str, Any]:
    if is_dataclass(contract):
        return asdict(contract)
    return dict(contract)


# 从恢复出的 NCHW 特征张量形状提取 [mel, frame] 语义维度。
def _feature_shape_from_shape_nchw(shape_nchw: Any) -> list[int] | None:
    if not isinstance(shape_nchw, list) or len(shape_nchw) != 4:
        return None
    try:
        normalized = [int(value) for value in shape_nchw]
    except Exception:
        return None
    if normalized[0] != 1 or normalized[1] != 1:
        return None
    if normalized[2] <= 0 or normalized[3] <= 0:
        return None
    return [normalized[2], normalized[3]]


# 为同一份 mel-frame 特征网格枚举生成 runner 当前可接受的运行时输入形状。
def build_supported_runtime_input_shapes(feature_shape_hw: list[int]) -> list[list[int]]:
    mel_bins = int(feature_shape_hw[0])
    frame_count = int(feature_shape_hw[1])
    feature_elements = mel_bins * frame_count
    return [
        [1, 1, mel_bins, frame_count],
        [1, mel_bins, frame_count, 1],
        [1, 1, frame_count, mel_bins],
        [1, frame_count, mel_bins, 1],
        [1, mel_bins, frame_count],
        [1, frame_count, mel_bins],
        [1, 1, feature_elements],
        [1, feature_elements, 1],
        [1, feature_elements],
    ]


# 为恢复出的 canonical NCHW 特征张量补齐可复用的布局元数据。
def build_model_input_contract_metadata(shape_nchw: list[int]) -> dict[str, Any]:
    feature_shape_hw = _feature_shape_from_shape_nchw(shape_nchw)
    if feature_shape_hw is None:
        return {
            "canonical_layout": "NCHW",
            "runtime_supported_layouts": SUPPORTED_RUNTIME_LAYOUTS.copy(),
        }

    feature_element_count = int(feature_shape_hw[0]) * int(feature_shape_hw[1])
    return {
        "canonical_layout": "NCHW",
        "canonical_shape_nchw": [int(value) for value in shape_nchw],
        "feature_shape_hw": feature_shape_hw,
        "feature_channels": 1,
        "feature_element_count": feature_element_count,
        "runtime_supported_layouts": SUPPORTED_RUNTIME_LAYOUTS.copy(),
        "runtime_supported_input_shapes": build_supported_runtime_input_shapes(feature_shape_hw),
    }


# 将样本数精确映射为整数毫秒；无法无损映射时返回 None。
def _exact_ms_from_samples(sample_count: Any, sample_rate_hz: Any) -> int | None:
    try:
        samples = int(sample_count)
        sample_rate = int(sample_rate_hz)
    except Exception:
        return None
    if samples <= 0 or sample_rate <= 0:
        return None
    numerator = samples * 1000
    if numerator % sample_rate != 0:
        return None
    return numerator // sample_rate


# 读取单个前处理步骤，统一为字典形式并保留 enabled 标志。
def _normalize_step(step: Any) -> dict[str, Any]:
    if is_dataclass(step):
        return asdict(step)
    if isinstance(step, dict):
        return dict(step)
    return {}


# 汇总当前生成固件运行时对合同的兼容性、期望参数和已知限制。
def summarize_board_runtime_contract(
    contract: ProcessContract | dict[str, Any],
) -> dict[str, Any]:
    payload = _normalize_contract(contract)
    raw_input = payload.get("raw_input", {}) or {}
    preprocess = [_normalize_step(step) for step in payload.get("preprocess", [])]
    model_input = payload.get("model_input", {}) or {}

    unsupported_reasons: dict[str, str] = {}
    warnings: list[str] = []

    sample_rate_hz = raw_input.get("sample_rate_hz")
    window_sec = raw_input.get("window_sec")
    if raw_input.get("domain") != "audio":
        unsupported_reasons["raw_input:domain"] = (
            "Generated board runtime currently starts from raw audio PCM and does not package non-audio domains."
        )

    feature_shape_hw = model_input.get("feature_shape_hw")
    if not isinstance(feature_shape_hw, list):
        feature_shape_hw = _feature_shape_from_shape_nchw(model_input.get("shape_nchw"))
    if feature_shape_hw is None:
        unsupported_reasons["model_input:shape"] = (
            "Generated board runtime expects a single-channel mel-frame feature grid recovered as canonical NCHW."
        )
        feature_shape_hw = []

    if model_input.get("semantic") not in (None, "log_mel_feature"):
        unsupported_reasons["model_input:semantic"] = (
            "Generated board runtime only emits log-mel feature tensors into the model input."
        )

    canonical_layout = model_input.get("canonical_layout")
    if canonical_layout not in (None, "NCHW"):
        unsupported_reasons["model_input:canonical_layout"] = (
            "Generated board runtime reconstructs canonical frontend tensors in NCHW order before layout adaptation."
        )

    shape_nchw = model_input.get("shape_nchw")
    if isinstance(shape_nchw, list):
        try:
            normalized_shape = [int(value) for value in shape_nchw]
        except Exception:
            normalized_shape = []
        if normalized_shape and (len(normalized_shape) != 4 or normalized_shape[0] != 1 or normalized_shape[1] != 1):
            unsupported_reasons["model_input:shape_nchw"] = (
                "Recovered canonical model_input.shape_nchw must stay in [1, 1, mel_bins, frame_count] form."
            )

    feature_element_count = 0
    if feature_shape_hw:
        feature_element_count = int(feature_shape_hw[0]) * int(feature_shape_hw[1])

    expected_runtime: dict[str, Any] = {
        "sample_rate_hz": int(sample_rate_hz) if isinstance(sample_rate_hz, int) or str(sample_rate_hz).isdigit() else sample_rate_hz,
        "capture_window_ms": None,
        "frame_length_ms": None,
        "frame_stride_ms": None,
        "fft_length": None,
        "mel_bin_count": int(feature_shape_hw[0]) if feature_shape_hw else None,
        "feature_frame_count": int(feature_shape_hw[1]) if feature_shape_hw else None,
        "feature_element_count": feature_element_count if feature_element_count > 0 else None,
        "mel_lower_edge_hz": None,
        "mel_upper_edge_hz": None,
        "top_db": None,
    }

    if isinstance(window_sec, (int, float)):
        capture_window_ms = int(round(float(window_sec) * 1000.0))
        expected_runtime["capture_window_ms"] = capture_window_ms
        if abs(float(window_sec) * 1000.0 - float(capture_window_ms)) > 1e-6:
            unsupported_reasons["raw_input:window_sec"] = (
                "Generated board runtime stores the capture window as an integer millisecond value."
            )

    step_by_name = {
        str(step.get("name")): step
        for step in preprocess
        if step.get("enabled") is True
    }

    for step in preprocess:
        if step.get("enabled") is not True:
            continue
        step_name = str(step.get("name"))
        if step_name == "bandpass":
            warnings.append(
                "audio.bandpass is kept as metadata only in the current deployment flow; "
                "the generated runtime does not apply an extra time-domain bandpass stage."
            )
            continue
        if step_name not in SUPPORTED_GENERATED_PREPROCESS:
            unsupported_reasons[f"preprocess:{step_name}"] = (
                "Generated board runtime has no lowering rule for this preprocess step."
            )

    stft_step = step_by_name.get("stft")
    if stft_step is not None:
        stft_params = stft_step.get("params", {}) or {}
        expected_runtime["fft_length"] = stft_params.get("n_fft")
        frame_length_ms = _exact_ms_from_samples(stft_params.get("win_length"), sample_rate_hz)
        frame_stride_ms = _exact_ms_from_samples(stft_params.get("hop_length"), sample_rate_hz)
        expected_runtime["frame_length_ms"] = frame_length_ms
        expected_runtime["frame_stride_ms"] = frame_stride_ms
        if frame_length_ms is None:
            unsupported_reasons["preprocess:stft.win_length"] = (
                "Generated board runtime stores frame length in whole milliseconds and cannot represent the recovered win_length exactly."
            )
        if frame_stride_ms is None:
            unsupported_reasons["preprocess:stft.hop_length"] = (
                "Generated board runtime stores frame stride in whole milliseconds and cannot represent the recovered hop_length exactly."
            )
        if bool(stft_params.get("center", False)):
            unsupported_reasons["preprocess:stft.center"] = (
                "Generated board runtime does not pad centered STFT windows and therefore requires center=False."
            )

    mel_step = step_by_name.get("mel")
    if mel_step is not None:
        mel_params = mel_step.get("params", {}) or {}
        expected_runtime["mel_bin_count"] = mel_params.get("n_mels")
        expected_runtime["mel_lower_edge_hz"] = mel_params.get("fmin")
        expected_runtime["mel_upper_edge_hz"] = mel_params.get("fmax")
        if feature_shape_hw and int(mel_params.get("n_mels", -1)) != int(feature_shape_hw[0]):
            unsupported_reasons["preprocess:mel.n_mels"] = (
                "Recovered mel bin count does not match the canonical model_input feature height."
            )

    db_step = step_by_name.get("db_compression")
    if db_step is not None:
        db_params = db_step.get("params", {}) or {}
        expected_runtime["top_db"] = db_params.get("top_db")

    normalize_step = step_by_name.get("normalize")
    if normalize_step is not None:
        normalize_params = normalize_step.get("params", {}) or {}
        normalize_mode = normalize_params.get("mode")
        if normalize_mode not in SUPPORTED_NORMALIZE_MODES:
            unsupported_reasons["preprocess:normalize.mode"] = (
                "Generated board runtime only supports disabled normalization or per-sample standardization."
            )

    supported_input_shapes = (
        build_supported_runtime_input_shapes(feature_shape_hw)
        if feature_shape_hw
        else []
    )

    if feature_shape_hw:
        warnings.append(
            "Generated board runtime reconstructs the frontend in canonical NCHW order and then adapts it into the TFLite input layout."
        )

    return {
        "supported": not unsupported_reasons,
        "unsupported_items": list(unsupported_reasons.keys()),
        "unsupported_reasons": unsupported_reasons,
        "warnings": warnings,
        "expected_runtime": expected_runtime,
        "supported_input_shapes": supported_input_shapes,
        "supported_layouts": SUPPORTED_RUNTIME_LAYOUTS.copy(),
    }
