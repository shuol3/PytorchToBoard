import argparse
import json
import math
import pathlib
import sys
from typing import Any

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.common.runtime_env import ensure_default_python_for_script

ensure_default_python_for_script(__file__)

import numpy as np
import tensorflow as tf

from pipeline.board_runtime_contract import summarize_board_runtime_contract
from pipeline.board.tflite_op_registry import build_resolver_lines
from pipeline.common.frontend_shape_utils import infer_feature_frame_count


DEFAULT_LABELS = ["eat", "drink", "other"]
DEFAULT_SAMPLE_RATE_HZ = 16000
DEFAULT_CAPTURE_WINDOW_MS = 1000
DEFAULT_FRAME_LENGTH_MS = 25
DEFAULT_FRAME_STRIDE_MS = 10
DEFAULT_FFT_LENGTH = 1024
DEFAULT_MEL_BIN_COUNT = 48
DEFAULT_FEATURE_FRAME_COUNT = 94
DEFAULT_MEL_LOWER_EDGE_HZ = 3000.0
DEFAULT_MEL_UPPER_EDGE_HZ = 7900.0
DEFAULT_TOP_DB = 80.0
DEFAULT_TENSOR_ARENA_SIZE = 120 * 1024
DEFAULT_OUTPUT_BASENAME = "audio_event_model"
DEFAULT_OUTPUT_FILES = {
    "config_h": "audio_event_model_config.h",
    "data_h": "audio_event_model_data.h",
    "data_cpp": "audio_event_model_data.cpp",
    "runner_h": "audio_event_model_runner.h",
    "runner_cpp": "audio_event_model_runner.cpp",
}


def _parse_labels(text: str | None) -> list[str]:
    if text is None:
        return DEFAULT_LABELS.copy()
    labels = [part.strip() for part in text.split(",") if part.strip()]
    if not labels:
        raise ValueError("labels must not be empty")
    return labels


def _parse_manifest(path: pathlib.Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_process_contract(path: pathlib.Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_quantization(detail: dict[str, Any]) -> tuple[float, int]:
    scale, zero_point = detail.get("quantization", (0.0, 0))
    return float(scale), int(zero_point)


def _tflite_array_literal(data: bytes) -> str:
    rows = []
    for start in range(0, len(data), 12):
        chunk = data[start:start + 12]
        rows.append("    " + ", ".join(f"0x{byte:02X}" for byte in chunk))
    return ",\n".join(rows)


def _float_literal(value: float) -> str:
    rounded = float(np.float32(value))
    if abs(rounded) < 1.0e-30:
        rounded = 0.0
    text = repr(rounded)
    if "e" not in text and "." not in text:
        text += ".0"
    return f"{text}f"


def _array_literal(values: list[Any], *, formatter, items_per_row: int) -> str:
    rows = []
    for start in range(0, len(values), items_per_row):
        chunk = values[start:start + items_per_row]
        rows.append("    " + ", ".join(formatter(value) for value in chunk))
    return ",\n".join(rows)


def _float_array_literal(values: list[float], items_per_row: int = 8) -> str:
    return _array_literal(values, formatter=_float_literal, items_per_row=items_per_row)


def _int_array_literal(values: list[int], items_per_row: int = 12) -> str:
    return _array_literal(values, formatter=lambda value: str(int(value)), items_per_row=items_per_row)


def _hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * ((10.0 ** (mel / 2595.0)) - 1.0)


def _build_hann_window(window_len_samples: int) -> list[float]:
    if window_len_samples <= 0:
        raise ValueError("window_len_samples must be positive")
    return [
        0.5 - 0.5 * math.cos((2.0 * math.pi * index) / float(window_len_samples))
        for index in range(window_len_samples)
    ]


def _build_sparse_mel_filterbank(sample_rate_hz: int,
                                 fft_length: int,
                                 mel_bin_count: int,
                                 mel_lower_edge_hz: float,
                                 mel_upper_edge_hz: float) -> dict[str, list[float] | list[int]]:
    fft_bin_count = (fft_length // 2) + 1
    fft_bin_hz = float(sample_rate_hz) / float(fft_length)
    mel_lo = _hz_to_mel(mel_lower_edge_hz)
    mel_hi = _hz_to_mel(mel_upper_edge_hz)
    mel_edges_hz = [
        _mel_to_hz(mel_lo + ((mel_hi - mel_lo) * float(index) / float(mel_bin_count + 1)))
        for index in range(mel_bin_count + 2)
    ]

    filter_positions: list[int] = []
    filter_lengths: list[int] = []
    filter_coefficients: list[float] = []

    for mel_idx in range(mel_bin_count):
        left = mel_edges_hz[mel_idx]
        center = mel_edges_hz[mel_idx + 1]
        right = mel_edges_hz[mel_idx + 2]

        left_bin = max(0, min(int(math.ceil(left / fft_bin_hz)), fft_bin_count - 1))
        center_bin = max(left_bin, min(int(round(center / fft_bin_hz)), fft_bin_count - 1))
        right_bin = max(center_bin, min(int(math.floor(right / fft_bin_hz)), fft_bin_count - 1))

        left_span_inv = 1.0 / max(center - left, 1.0e-6)
        right_span_inv = 1.0 / max(right - center, 1.0e-6)

        start_bin: int | None = None
        local_weights: list[float] = []
        for fft_bin in range(left_bin, right_bin + 1):
            hz = float(fft_bin) * fft_bin_hz
            if fft_bin <= center_bin:
                weight = (hz - left) * left_span_inv
            else:
                weight = (right - hz) * right_span_inv
            if weight > 0.0:
                if start_bin is None:
                    start_bin = fft_bin
                local_weights.append(weight)

        if start_bin is None:
            start_bin = left_bin

        filter_positions.append(start_bin)
        filter_lengths.append(len(local_weights))
        filter_coefficients.extend(local_weights)

    return {
        "positions": filter_positions,
        "lengths": filter_lengths,
        "coefficients": filter_coefficients,
    }


def _read_tflite_details(model_path: pathlib.Path) -> dict[str, Any]:
    resolver_type = tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
    interpreter = tf.lite.Interpreter(
        model_path=str(model_path),
        experimental_op_resolver_type=resolver_type,
    )
    interpreter.allocate_tensors()

    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    input_scale, input_zero_point = _normalize_quantization(input_detail)
    output_scale, output_zero_point = _normalize_quantization(output_detail)

    op_names: list[str] = []
    for op_detail in interpreter._get_ops_details():
        op_name = op_detail.get("op_name")
        if op_name and op_name not in op_names:
            op_names.append(op_name)

    return {
        "input_shape": input_detail["shape"].tolist(),
        "output_shape": output_detail["shape"].tolist(),
        "input_dtype": np.dtype(input_detail["dtype"]).name,
        "output_dtype": np.dtype(output_detail["dtype"]).name,
        "input_scale": input_scale,
        "input_zero_point": input_zero_point,
        "output_scale": output_scale,
        "output_zero_point": output_zero_point,
        "op_names": op_names,
    }


def _feature_frame_count(sample_rate_hz: int,
                         capture_window_ms: int,
                         fft_length: int,
                         frame_length_ms: int,
                         frame_stride_ms: int) -> int:
    frame_stride_samples = (sample_rate_hz * frame_stride_ms) // 1000
    return infer_feature_frame_count(
        sample_rate_hz=int(sample_rate_hz),
        window_sec=float(capture_window_ms) / 1000.0,
        hop_length=int(frame_stride_samples),
        center=False,
        n_fft=int(fft_length),
        win_length=int((sample_rate_hz * frame_length_ms) // 1000),
    )


def _validate_input_shape(input_shape: list[int], mel_bins: int, feature_frames: int) -> None:
    feature_elements = mel_bins * feature_frames
    valid_shapes = (
        [1, mel_bins, feature_frames, 1],
        [1, 1, mel_bins, feature_frames],
        [1, feature_frames, mel_bins, 1],
        [1, 1, feature_frames, mel_bins],
        [1, mel_bins, feature_frames],
        [1, feature_frames, mel_bins],
        [1, 1, feature_elements],
        [1, feature_elements, 1],
        [1, feature_elements],
    )
    if input_shape not in valid_shapes:
        raise ValueError(
            "Generated model input shape does not match the configured feature layout. "
            f"Expected one of {valid_shapes}, got {input_shape}"
        )


# 当 recover 阶段的 process_contract 可用时，优先用它约束当前生成 runner 的配置和输入形状，
# 避免训练产物、pipeline 恢复结果和板端打包契约之间发生静默漂移。
def _validate_process_contract(
    process_contract: dict[str, Any] | None,
    input_shape: list[int],
    *,
    sample_rate_hz: int,
    capture_window_ms: int,
    frame_length_ms: int,
    frame_stride_ms: int,
    fft_length: int,
    mel_bin_count: int,
    feature_frame_count: int,
    mel_lower_edge_hz: float,
    mel_upper_edge_hz: float,
    top_db: float,
) -> dict[str, Any] | None:
    if process_contract is None:
        return None

    runtime_summary = summarize_board_runtime_contract(process_contract)
    if not runtime_summary["supported"]:
        unsupported_reasons = runtime_summary["unsupported_reasons"]
        detail_text = "; ".join(
            f"{key}: {value}"
            for key, value in unsupported_reasons.items()
        )
        raise ValueError(
            "Recovered process contract is not compatible with the current generated firmware runner. "
            + detail_text
        )

    supported_input_shapes = runtime_summary["supported_input_shapes"]
    if supported_input_shapes and input_shape not in supported_input_shapes:
        raise ValueError(
            "Recovered process contract and generated firmware runner disagree on the packaged model input shape. "
            f"Supported shapes from the contract: {supported_input_shapes}; actual TFLite shape: {input_shape}"
        )

    expected = runtime_summary["expected_runtime"]
    requested = {
        "sample_rate_hz": sample_rate_hz,
        "capture_window_ms": capture_window_ms,
        "frame_length_ms": frame_length_ms,
        "frame_stride_ms": frame_stride_ms,
        "fft_length": fft_length,
        "mel_bin_count": mel_bin_count,
        "feature_frame_count": feature_frame_count,
        "mel_lower_edge_hz": mel_lower_edge_hz,
        "mel_upper_edge_hz": mel_upper_edge_hz,
        "top_db": top_db,
    }
    mismatches = []
    for key, actual_value in requested.items():
        expected_value = expected.get(key)
        if expected_value is None:
            continue
        if isinstance(actual_value, float):
            if abs(float(expected_value) - float(actual_value)) > 1e-6:
                mismatches.append(f"{key}: expected {expected_value}, got {actual_value}")
            continue
        if expected_value != actual_value:
            mismatches.append(f"{key}: expected {expected_value}, got {actual_value}")
    if mismatches:
        raise ValueError(
            "Recovered process contract does not match the requested generated firmware runtime arguments. "
            + "; ".join(mismatches)
        )
    return runtime_summary


def _config_header_text(labels: list[str],
                        sample_rate_hz: int,
                        capture_window_ms: int,
                        frame_length_ms: int,
                        frame_stride_ms: int,
                        fft_length: int,
                        mel_bin_count: int,
                        feature_frame_count: int,
                        mel_lower_edge_hz: float,
                        mel_upper_edge_hz: float,
                        top_db: float) -> str:
    label_lines = ",\n".join(f'    "{label}"' for label in labels)
    return f"""#pragma once

// 本文件由部署脚本自动生成，保存板端前处理和标签所需的固定配置。

namespace audio_event_model_config {{

constexpr int kSampleRateHz = {sample_rate_hz};
constexpr int kCaptureWindowMs = {capture_window_ms};
constexpr int kFrameLengthMs = {frame_length_ms};
constexpr int kFrameStrideMs = {frame_stride_ms};
constexpr int kFftLength = {fft_length};
constexpr int kFrontendMelBinCount = {mel_bin_count};
constexpr int kFrontendFeatureFrameCount = {feature_frame_count};
constexpr int kFrontendFeatureElementCount =
    kFrontendMelBinCount * kFrontendFeatureFrameCount;
constexpr int kMaxModelInputElementCount = kFrontendFeatureElementCount;
constexpr int kLabelCount = {len(labels)};
constexpr float kMelLowerEdgeHz = {mel_lower_edge_hz:.6f}f;
constexpr float kMelUpperEdgeHz = {mel_upper_edge_hz:.6f}f;
constexpr float kTopDb = {top_db:.6f}f;

inline constexpr const char *kLabels[kLabelCount] = {{
{label_lines}
}};

}}  // namespace audio_event_model_config
"""


def _data_header_text() -> str:
    return """#pragma once

// 本文件由部署脚本自动生成，声明嵌入式模型二进制数组。

#include <stddef.h>
#include <stdint.h>

namespace audio_event_model_data {

extern const unsigned char g_model[];
extern const size_t g_model_len;
extern const bool kPlaceholderModel;

}  // namespace audio_event_model_data
"""


def _data_cpp_text(model_bytes: bytes) -> str:
    array_literal = _tflite_array_literal(model_bytes)
    return f"""// 本文件由部署脚本自动生成，内联保存 TFLite 模型字节流。
#include "generated/audio_event_model_data.h"

namespace audio_event_model_data {{

const unsigned char g_model[] = {{
{array_literal}
}};

const size_t g_model_len = sizeof(g_model);
const bool kPlaceholderModel = false;

}}  // namespace audio_event_model_data
"""


def _runner_header_text() -> str:
    return """#pragma once

// 本文件由部署脚本自动生成，声明板端推理入口与结果结构体。

#include <stdint.h>

#include "generated/audio_event_model_config.h"

namespace audio_event_model_runner {

inline constexpr int kWindowSamples =
    (audio_event_model_config::kSampleRateHz *
     audio_event_model_config::kCaptureWindowMs) / 1000;

struct Result {
    float predicted_score;
    int predicted_label;
    int output_count;
    float scores[audio_event_model_config::kLabelCount];
    bool inference_ok;
};

bool model_is_placeholder();
unsigned int runtime_memory_size_bytes();
inline constexpr int window_samples()
{
    return kWindowSamples;
}
const char *last_error();
uint32_t last_feature_ms();
uint32_t last_invoke_ms();
Result run_classifier(const int16_t *samples, int sample_count);

}  // namespace audio_event_model_runner
"""


def _runner_cpp_text(tensor_arena_size: int, resolver_lines: list[str]) -> str:
    resolver_block = "\n".join(f"    {line}" for line in resolver_lines)
    resolver_capacity = max(len(resolver_lines), 1)
    return f"""#include "generated/audio_event_model_runner.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#include <algorithm>

#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>

#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/micro_profiler_interface.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "generated/audio_event_model_config.h"
#include "generated/audio_event_model_data.h"
#include "kiss_fftr.h"

namespace audio_event_model_runner {{
namespace {{

constexpr float kPi = 3.14159265358979323846f;
constexpr int kWindowSampleCount =
    (audio_event_model_config::kSampleRateHz *
     audio_event_model_config::kCaptureWindowMs) / 1000;
constexpr int kFftLength = audio_event_model_config::kFftLength;
constexpr int kWindowLenSamples =
    (audio_event_model_config::kSampleRateHz *
     audio_event_model_config::kFrameLengthMs) / 1000;
constexpr int kAnalysisFrameSamples = kFftLength;
constexpr int kFrameStrideSamples =
    (audio_event_model_config::kSampleRateHz *
     audio_event_model_config::kFrameStrideMs) / 1000;
constexpr int kFftBinCount = (kFftLength / 2) + 1;
constexpr float kFftBinHz =
    static_cast<float>(audio_event_model_config::kSampleRateHz) /
    static_cast<float>(kFftLength);
constexpr int kBaseMelBinCount = audio_event_model_config::kFrontendMelBinCount;
constexpr int kMelEdgeCount = kBaseMelBinCount + 2;
constexpr int kBaseFeatureFrameCount = audio_event_model_config::kFrontendFeatureFrameCount;
constexpr int kBaseFeatureElementCount = audio_event_model_config::kFrontendFeatureElementCount;
constexpr int kWindowPadLeft =
    (kAnalysisFrameSamples > kWindowLenSamples)
        ? ((kAnalysisFrameSamples - kWindowLenSamples) / 2)
        : 0;
constexpr int kFrameFftCopySamples =
    (kWindowLenSamples < kAnalysisFrameSamples) ? kWindowLenSamples : kAnalysisFrameSamples;
constexpr size_t kFftCfgStorageSize = 16384;
constexpr size_t kTensorArenaSize = {tensor_arena_size};
constexpr float kDefaultDecisionThreshold = 0.5f;

enum class InputLayout {{
    kFlat,
    kMatrix,
    kNhwc,
    kNchw,
    kUnsupported,
}};

struct ModelInputSpec {{
    InputLayout layout = InputLayout::kUnsupported;
    int rows = 0;
    int cols = 0;
    int channels = 0;
    int element_count = 0;
}};

alignas(16) static float g_window[kWindowLenSamples];
alignas(16) static float g_mel_edges_hz[kMelEdgeCount];
alignas(16) static int16_t g_mel_left_bins[kBaseMelBinCount];
alignas(16) static int16_t g_mel_center_bins[kBaseMelBinCount];
alignas(16) static int16_t g_mel_right_bins[kBaseMelBinCount];
alignas(16) static float g_mel_left_span_inv[kBaseMelBinCount];
alignas(16) static float g_mel_right_span_inv[kBaseMelBinCount];
alignas(16) static float g_fft_input[kFftLength];
alignas(16) static kiss_fft_cpx g_fft_output[kFftBinCount];
alignas(16) static float g_power_spectrum[kFftBinCount];
alignas(16) static uint8_t g_fft_cfg_storage[kFftCfgStorageSize];
alignas(16) static float g_base_feature_buf[kBaseFeatureElementCount];
alignas(16) static uint8_t g_tensor_arena[kTensorArenaSize];

static kiss_fftr_cfg g_fft_cfg = nullptr;
static const tflite::Model *g_model = nullptr;
static tflite::MicroInterpreter *g_interpreter = nullptr;
static TfLiteTensor *g_input = nullptr;
static TfLiteTensor *g_output = nullptr;
static ModelInputSpec g_input_spec;
static int g_output_count = 0;
static bool g_frontend_initialized = false;
static bool g_frontend_init_failed = false;
static bool g_model_initialized = false;
static bool g_model_init_attempted = false;
static const char *g_last_error = "not initialized";
static uint32_t g_last_feature_ms = 0;
static uint32_t g_last_invoke_ms = 0;

void set_last_error(const char *message)
{{
    g_last_error = message;
}}

float hz_to_mel(float hz)
{{
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}}

float mel_to_hz(float mel)
{{
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}}

int8_t quantize_to_int8(float value, float scale, int zero_point)
{{
    if (scale == 0.0f) {{
        return 0;
    }}

    int32_t q = static_cast<int32_t>(lroundf(value / scale)) + zero_point;
    if (q < -128) {{
        q = -128;
    }} else if (q > 127) {{
        q = 127;
    }}
    return static_cast<int8_t>(q);
}}

float dequantize_from_int8(int8_t value, float scale, int zero_point)
{{
    return (static_cast<int32_t>(value) - zero_point) * scale;
}}

float sigmoidf(float x)
{{
    if (x >= 0.0f) {{
        const float e = expf(-x);
        return 1.0f / (1.0f + e);
    }}

    const float e = expf(x);
    return e / (1.0f + e);
}}

void softmax_inplace(float *values, int count)
{{
    if (values == nullptr || count <= 0) {{
        return;
    }}

    float max_value = values[0];
    for (int i = 1; i < count; ++i) {{
        if (values[i] > max_value) {{
            max_value = values[i];
        }}
    }}

    float sum = 0.0f;
    for (int i = 0; i < count; ++i) {{
        values[i] = expf(values[i] - max_value);
        sum += values[i];
    }}

    if (sum <= 0.0f) {{
        return;
    }}

    for (int i = 0; i < count; ++i) {{
        values[i] /= sum;
    }}
}}

Result invalid_result()
{{
    Result result = {{}};
    result.predicted_score = 0.0f;
    result.predicted_label = 0;
    result.output_count = 0;
    for (int i = 0; i < audio_event_model_config::kLabelCount; ++i) {{
        result.scores[i] = 0.0f;
    }}
    result.inference_ok = false;
    return result;
}}

int tensor_element_count(const TfLiteTensor *tensor)
{{
    if (tensor == nullptr || tensor->dims == nullptr) {{
        return 0;
    }}

    int count = 1;
    for (int i = 0; i < tensor->dims->size; ++i) {{
        count *= tensor->dims->data[i];
    }}
    return count;
}}

int tensor_value_count(const TfLiteTensor *tensor)
{{
    if (tensor == nullptr) {{
        return 0;
    }}

    switch (tensor->type) {{
    case kTfLiteFloat32:
        return static_cast<int>(tensor->bytes / sizeof(float));
    case kTfLiteInt8:
        return static_cast<int>(tensor->bytes / sizeof(int8_t));
    default:
        return 0;
    }}
}}

ModelInputSpec infer_model_input_spec(const TfLiteTensor *tensor)
{{
    ModelInputSpec spec;
    if (tensor == nullptr || tensor->dims == nullptr) {{
        return spec;
    }}

    const TfLiteIntArray *dims = tensor->dims;
    spec.element_count = tensor_element_count(tensor);
    if (spec.element_count <= 0) {{
        return spec;
    }}

    if (dims->size == 4) {{
        const int d0 = dims->data[0];
        const int d1 = dims->data[1];
        const int d2 = dims->data[2];
        const int d3 = dims->data[3];

        if (d0 != 1) {{
            return spec;
        }}
        if (d3 == 1 && d1 > 0 && d2 > 0) {{
            spec.layout = InputLayout::kNhwc;
            spec.rows = d1;
            spec.cols = d2;
            spec.channels = d3;
            return spec;
        }}
        if (d1 == 1 && d2 > 0 && d3 > 0) {{
            spec.layout = InputLayout::kNchw;
            spec.rows = d2;
            spec.cols = d3;
            spec.channels = d1;
            return spec;
        }}
        return spec;
    }}

    if (dims->size == 3) {{
        if (dims->data[0] != 1 || dims->data[1] <= 0 || dims->data[2] <= 0) {{
            return spec;
        }}
        spec.layout = InputLayout::kMatrix;
        spec.rows = dims->data[1];
        spec.cols = dims->data[2];
        spec.channels = 1;
        return spec;
    }}

    if (dims->size == 2) {{
        if (dims->data[0] != 1 || dims->data[1] <= 0) {{
            return spec;
        }}
        spec.layout = InputLayout::kFlat;
        spec.rows = 1;
        spec.cols = dims->data[1];
        spec.channels = 1;
    }}

    return spec;
}}

bool matches_base_feature_matrix(int rows, int cols)
{{
    return rows == kBaseMelBinCount && cols == kBaseFeatureFrameCount;
}}

bool matches_transposed_feature_matrix(int rows, int cols)
{{
    return rows == kBaseFeatureFrameCount && cols == kBaseMelBinCount;
}}

bool matches_flat_feature_vector(int rows, int cols)
{{
    return (rows == 1 && cols == kBaseFeatureElementCount) ||
           (rows == kBaseFeatureElementCount && cols == 1);
}}

float feature_value_for_input_index(int index)
{{
    if (index < 0 || index >= kBaseFeatureElementCount) {{
        return 0.0f;
    }}

    if (g_input_spec.layout == InputLayout::kFlat) {{
        return g_base_feature_buf[index];
    }}

    if (matches_base_feature_matrix(g_input_spec.rows, g_input_spec.cols) ||
        matches_flat_feature_vector(g_input_spec.rows, g_input_spec.cols)) {{
        return g_base_feature_buf[index];
    }}

    if (matches_transposed_feature_matrix(g_input_spec.rows, g_input_spec.cols)) {{
        const int frame_idx = index / kBaseMelBinCount;
        const int mel_idx = index % kBaseMelBinCount;
        return g_base_feature_buf[(mel_idx * kBaseFeatureFrameCount) + frame_idx];
    }}

    return 0.0f;
}}

bool init_frontend_once()
{{
    if (g_frontend_initialized) {{
        return true;
    }}
    if (g_frontend_init_failed) {{
        return false;
    }}

    for (int i = 0; i < kWindowLenSamples; ++i) {{
        g_window[i] = 0.5f - 0.5f * cosf(
            2.0f * kPi * static_cast<float>(i) /
            static_cast<float>(kWindowLenSamples));
    }}

    const float mel_lo = hz_to_mel(audio_event_model_config::kMelLowerEdgeHz);
    const float mel_hi = hz_to_mel(audio_event_model_config::kMelUpperEdgeHz);
    for (int i = 0; i < kMelEdgeCount; ++i) {{
        const float t = static_cast<float>(i) /
                        static_cast<float>(kMelEdgeCount - 1);
        g_mel_edges_hz[i] = mel_to_hz(mel_lo + ((mel_hi - mel_lo) * t));
    }}

    for (int mel_idx = 0; mel_idx < kBaseMelBinCount; ++mel_idx) {{
        const float left = g_mel_edges_hz[mel_idx];
        const float center = g_mel_edges_hz[mel_idx + 1];
        const float right = g_mel_edges_hz[mel_idx + 2];

        int left_bin = static_cast<int>(ceilf(left / kFftBinHz));
        int center_bin = static_cast<int>(roundf(center / kFftBinHz));
        int right_bin = static_cast<int>(floorf(right / kFftBinHz));

        left_bin = std::max(0, std::min(left_bin, kFftBinCount - 1));
        center_bin = std::max(left_bin, std::min(center_bin, kFftBinCount - 1));
        right_bin = std::max(center_bin, std::min(right_bin, kFftBinCount - 1));

        g_mel_left_bins[mel_idx] = static_cast<int16_t>(left_bin);
        g_mel_center_bins[mel_idx] = static_cast<int16_t>(center_bin);
        g_mel_right_bins[mel_idx] = static_cast<int16_t>(right_bin);
        g_mel_left_span_inv[mel_idx] = 1.0f / std::max(center - left, 1e-6f);
        g_mel_right_span_inv[mel_idx] = 1.0f / std::max(right - center, 1e-6f);
    }}

    size_t needed_size = 0;
    (void)kiss_fftr_alloc(kFftLength, 0, nullptr, &needed_size);
    if (needed_size > sizeof(g_fft_cfg_storage)) {{
        printk("Generated frontend FFT cfg buffer too small: need=%u have=%u\\n",
               static_cast<uint32_t>(needed_size),
               static_cast<uint32_t>(sizeof(g_fft_cfg_storage)));
        g_frontend_init_failed = true;
        return false;
    }}

    g_fft_cfg = kiss_fftr_alloc(kFftLength, 0, g_fft_cfg_storage, &needed_size);
    if (g_fft_cfg == nullptr) {{
        printk("Generated frontend FFT cfg init failed\\n");
        g_frontend_init_failed = true;
        return false;
    }}

    g_frontend_initialized = true;
    return true;
}}

bool extract_base_log_mel_features(const int16_t *samples, int sample_count)
{{
    if (samples == nullptr || sample_count < kWindowSampleCount) {{
        return false;
    }}
    if (!init_frontend_once()) {{
        return false;
    }}

    std::fill(g_base_feature_buf, g_base_feature_buf + kBaseFeatureElementCount, 0.0f);

    int frames_written = 0;
    for (int start = 0;
         start + kAnalysisFrameSamples <= sample_count &&
         frames_written < kBaseFeatureFrameCount;
         start += kFrameStrideSamples) {{
        std::fill(g_fft_input, g_fft_input + kFftLength, 0.0f);

        for (int i = 0; i < kFrameFftCopySamples; ++i) {{
            const int fft_index = kWindowPadLeft + i;
            g_fft_input[fft_index] =
                (static_cast<float>(samples[start + fft_index]) / 32768.0f) * g_window[i];
        }}

        kiss_fftr(g_fft_cfg, g_fft_input, g_fft_output);

        for (int i = 0; i < kFftBinCount; ++i) {{
            const float real = g_fft_output[i].r;
            const float imag = g_fft_output[i].i;
            g_power_spectrum[i] = (real * real) + (imag * imag);
        }}

        for (int mel_idx = 0; mel_idx < kBaseMelBinCount; ++mel_idx) {{
            const float left = g_mel_edges_hz[mel_idx];
            const float right = g_mel_edges_hz[mel_idx + 2];
            const int left_bin = g_mel_left_bins[mel_idx];
            const int center_bin = g_mel_center_bins[mel_idx];
            const int right_bin = g_mel_right_bins[mel_idx];
            const float left_span_inv = g_mel_left_span_inv[mel_idx];
            const float right_span_inv = g_mel_right_span_inv[mel_idx];
            float mel_energy = 0.0f;

            for (int bin = left_bin; bin <= center_bin; ++bin) {{
                const float hz = static_cast<float>(bin) * kFftBinHz;
                const float weight = (hz - left) * left_span_inv;
                if (weight > 0.0f) {{
                    mel_energy += g_power_spectrum[bin] * weight;
                }}
            }}

            for (int bin = center_bin + 1; bin <= right_bin; ++bin) {{
                const float hz = static_cast<float>(bin) * kFftBinHz;
                const float weight = (right - hz) * right_span_inv;
                if (weight > 0.0f) {{
                    mel_energy += g_power_spectrum[bin] * weight;
                }}
            }}

            g_base_feature_buf[(mel_idx * kBaseFeatureFrameCount) + frames_written] =
                mel_energy;
        }}

        ++frames_written;
    }}

    if (frames_written != kBaseFeatureFrameCount) {{
        return false;
    }}

    float max_db = -1.0e30f;
    for (int i = 0; i < kBaseFeatureElementCount; ++i) {{
        const float db = 10.0f * log10f(g_base_feature_buf[i] + 1e-12f);
        g_base_feature_buf[i] = db;
        if (db > max_db) {{
            max_db = db;
        }}
    }}

    const float floor_db = max_db - audio_event_model_config::kTopDb;
    for (int i = 0; i < kBaseFeatureElementCount; ++i) {{
        g_base_feature_buf[i] = std::max(g_base_feature_buf[i], floor_db);
    }}

    float mean = 0.0f;
    for (int i = 0; i < kBaseFeatureElementCount; ++i) {{
        mean += g_base_feature_buf[i];
    }}
    mean /= static_cast<float>(kBaseFeatureElementCount);

    float variance = 0.0f;
    for (int i = 0; i < kBaseFeatureElementCount; ++i) {{
        const float delta = g_base_feature_buf[i] - mean;
        variance += delta * delta;
    }}
    variance /= static_cast<float>(kBaseFeatureElementCount);
    const float stddev = sqrtf(variance);

    for (int i = 0; i < kBaseFeatureElementCount; ++i) {{
        g_base_feature_buf[i] = (g_base_feature_buf[i] - mean) / (stddev + 1e-6f);
    }}

    return true;
}}

bool validate_model_input_layout()
{{
    if (g_input_spec.element_count != kBaseFeatureElementCount) {{
        return false;
    }}

    if (g_input_spec.layout == InputLayout::kFlat) {{
        return true;
    }}

    if (g_input_spec.layout == InputLayout::kMatrix) {{
        return matches_base_feature_matrix(g_input_spec.rows, g_input_spec.cols) ||
               matches_transposed_feature_matrix(g_input_spec.rows, g_input_spec.cols) ||
               matches_flat_feature_vector(g_input_spec.rows, g_input_spec.cols);
    }}

    if ((g_input_spec.layout == InputLayout::kNhwc ||
         g_input_spec.layout == InputLayout::kNchw) &&
        g_input_spec.channels == 1) {{
        return matches_base_feature_matrix(g_input_spec.rows, g_input_spec.cols) ||
               matches_transposed_feature_matrix(g_input_spec.rows, g_input_spec.cols) ||
               matches_flat_feature_vector(g_input_spec.rows, g_input_spec.cols);
    }}

    return false;
}}

bool init_model_once()
{{
    if (g_model_initialized) {{
        return true;
    }}
    if (g_model_init_attempted) {{
        printk("Generated model init previously failed: %s\\n", g_last_error);
        return false;
    }}
    g_model_init_attempted = true;

    if (audio_event_model_data::kPlaceholderModel) {{
        set_last_error("placeholder model still active");
        printk("Generated model placeholder is still in use\\n");
        return false;
    }}

    g_model = tflite::GetModel(audio_event_model_data::g_model);
    if (g_model == nullptr) {{
        set_last_error("model pointer is null");
        return false;
    }}

    if (g_model->version() != TFLITE_SCHEMA_VERSION) {{
        set_last_error("schema mismatch");
        printk("Generated model schema mismatch: model=%d runtime=%d\\n",
               g_model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }}

    g_model_operator_count = flatbuffer_operator_count();
    if (g_model_operator_count <= 0) {{
        set_last_error("model operator list is empty");
        return false;
    }}
    if (g_model_operator_count > kMaxProfiledOps) {{
        set_last_error("model has more ops than profiler buffer");
        printk("Generated model op count %d exceeds profiler capacity %d\\n",
               g_model_operator_count, kMaxProfiledOps);
        return false;
    }}
    std::fill(g_invoke_profile_cycle_sums,
              g_invoke_profile_cycle_sums + kMaxProfiledOps,
              0ULL);
    g_invoke_profile_sample_count = 0;
    g_invoke_profile_reported = false;
    g_invoke_profiler.Reset();

    static tflite::MicroMutableOpResolver<{resolver_capacity}> resolver;
{resolver_block}

    static tflite::MicroInterpreter static_interpreter(
        g_model, resolver, g_tensor_arena, kTensorArenaSize, nullptr, &g_invoke_profiler);
    g_interpreter = &static_interpreter;

    if (g_interpreter->AllocateTensors() != kTfLiteOk) {{
        set_last_error("AllocateTensors failed");
        g_interpreter = nullptr;
        return false;
    }}

    g_input = g_interpreter->input(0);
    g_output = g_interpreter->output(0);
    if (g_input == nullptr || g_output == nullptr) {{
        set_last_error("input/output tensor missing");
        return false;
    }}

    g_input_spec = infer_model_input_spec(g_input);
    if (g_input_spec.layout == InputLayout::kUnsupported) {{
        set_last_error("unsupported model input shape");
        return false;
    }}
    if (g_input_spec.channels != 1) {{
        set_last_error("multi-channel model input unsupported");
        return false;
    }}
    if (g_input_spec.element_count > audio_event_model_config::kMaxModelInputElementCount) {{
        set_last_error("model input exceeds feature buffer capacity");
        return false;
    }}

    g_output_count = tensor_value_count(g_output);
    if (g_output_count <= 0) {{
        set_last_error("unsupported output tensor type");
        return false;
    }}

    g_model_initialized = true;
    set_last_error("ok");
    printk("Generated model ready: input_elements=%d output_count=%d arena_used=%u B arena_reserved=%u B\\n",
           g_input_spec.element_count,
           g_output_count,
           static_cast<unsigned int>(g_interpreter->arena_used_bytes()),
           static_cast<unsigned int>(kTensorArenaSize));
    printk("Invoke profiler armed: average first %u runs, then print per-op breakdown once\\n",
           static_cast<unsigned int>(kInvokeProfileReportAfterInvocations));
    return true;
}}

}}  // namespace

bool model_is_placeholder()
{{
    return audio_event_model_data::kPlaceholderModel;
}}

unsigned int runtime_memory_size_bytes()
{{
    return static_cast<unsigned int>(
        sizeof(g_base_feature_buf) + sizeof(g_tensor_arena));
}}

const char *last_error()
{{
    return g_last_error;
}}

uint32_t last_feature_ms()
{{
    return g_last_feature_ms;
}}

uint32_t last_invoke_ms()
{{
    return g_last_invoke_ms;
}}

Result run_classifier(const int16_t *samples, int sample_count)
{{
    g_last_feature_ms = 0;
    g_last_invoke_ms = 0;

    if (samples == nullptr || sample_count != kWindowSampleCount) {{
        set_last_error("invalid audio window");
        return invalid_result();
    }}
    if (!init_model_once()) {{
        return invalid_result();
    }}

    const uint32_t feature_start_ms = k_uptime_get_32();
    if (!extract_base_log_mel_features(samples, sample_count)) {{
        set_last_error("frontend extraction failed");
        return invalid_result();
    }}
    if (!validate_model_input_layout()) {{
        set_last_error("model input layout mismatch");
        return invalid_result();
    }}
    g_last_feature_ms = k_uptime_get_32() - feature_start_ms;

    if (g_input->type == kTfLiteInt8) {{
        for (int i = 0; i < g_input_spec.element_count; ++i) {{
            const float feature_value = feature_value_for_input_index(i);
            g_input->data.int8[i] = quantize_to_int8(
                feature_value, g_input->params.scale, g_input->params.zero_point);
        }}
    }} else if (g_input->type == kTfLiteFloat32) {{
        for (int i = 0; i < g_input_spec.element_count; ++i) {{
            g_input->data.f[i] = feature_value_for_input_index(i);
        }}
    }} else {{
        set_last_error("unsupported input tensor type");
        return invalid_result();
    }}

    g_invoke_profiler.Reset();
    const uint32_t invoke_start_ms = k_uptime_get_32();
    if (g_interpreter->Invoke() != kTfLiteOk) {{
        set_last_error("Invoke failed");
        return invalid_result();
    }}
    g_last_invoke_ms = k_uptime_get_32() - invoke_start_ms;
    accumulate_invoke_profile();

    Result result = {{}};
    result.output_count = g_output_count;
    if (g_output->type == kTfLiteInt8) {{
        for (int i = 0; i < g_output_count; ++i) {{
            const float value = dequantize_from_int8(
                g_output->data.int8[i], g_output->params.scale, g_output->params.zero_point);
            result.scores[i] = value;
        }}
    }} else if (g_output->type == kTfLiteFloat32) {{
        for (int i = 0; i < g_output_count; ++i) {{
            result.scores[i] = g_output->data.f[i];
        }}
    }} else {{
        set_last_error("unsupported output tensor type");
        return invalid_result();
    }}

    if (g_output_count == 1) {{
        result.scores[0] = sigmoidf(result.scores[0]);
        result.predicted_score = result.scores[0];
        result.predicted_label = (result.predicted_score >= kDefaultDecisionThreshold) ? 1 : 0;
    }} else {{
        softmax_inplace(result.scores, g_output_count);
        result.predicted_label = 0;
        result.predicted_score = result.scores[0];
        for (int i = 1; i < g_output_count; ++i) {{
            if (result.scores[i] > result.predicted_score) {{
                result.predicted_score = result.scores[i];
                result.predicted_label = i;
            }}
        }}
    }}

    set_last_error("ok");
    result.inference_ok = true;
    return result;
}}

}}  // namespace audio_event_model_runner
"""


def _runner_cpp_text_accelerated(tensor_arena_size: int,
                                 resolver_lines: list[str],
                                 window_coefficients: list[float],
                                 mel_filter_positions: list[int],
                                 mel_filter_lengths: list[int],
                                 mel_filter_coefficients: list[float]) -> str:
    resolver_block = "\n".join(f"    {line}" for line in resolver_lines)
    resolver_capacity = max(len(resolver_lines), 1)
    window_literal = _float_array_literal(window_coefficients)
    mel_filter_pos_literal = _int_array_literal(mel_filter_positions)
    mel_filter_len_literal = _int_array_literal(mel_filter_lengths)
    mel_filter_coef_literal = _float_array_literal(mel_filter_coefficients)
    mel_filter_coef_count = len(mel_filter_coefficients)

    return f"""// 本文件由部署脚本自动生成，负责在板端复现音频前处理并调用 TFLM 推理。
#include "generated/audio_event_model_runner.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#include <algorithm>

#include <arm_math.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>

#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "generated/audio_event_model_config.h"
#include "generated/audio_event_model_data.h"

namespace audio_event_model_runner {{
namespace {{

// 这里集中定义生成 runner 的前端尺寸、量化辅助常量和固定缓冲区。
constexpr int kWindowSampleCount =
    (audio_event_model_config::kSampleRateHz *
     audio_event_model_config::kCaptureWindowMs) / 1000;
constexpr int kFftLength = audio_event_model_config::kFftLength;
constexpr int kWindowLenSamples =
    (audio_event_model_config::kSampleRateHz *
     audio_event_model_config::kFrameLengthMs) / 1000;
constexpr int kAnalysisFrameSamples = kFftLength;
constexpr int kFrameStrideSamples =
    (audio_event_model_config::kSampleRateHz *
     audio_event_model_config::kFrameStrideMs) / 1000;
constexpr int kFftBinCount = (kFftLength / 2) + 1;
constexpr int kBaseMelBinCount = audio_event_model_config::kFrontendMelBinCount;
constexpr int kBaseFeatureFrameCount = audio_event_model_config::kFrontendFeatureFrameCount;
constexpr int kBaseFeatureElementCount = audio_event_model_config::kFrontendFeatureElementCount;
constexpr int kWindowPadLeft =
    (kAnalysisFrameSamples > kWindowLenSamples)
        ? ((kAnalysisFrameSamples - kWindowLenSamples) / 2)
        : 0;
constexpr int kFrameFftCopySamples =
    (kWindowLenSamples < kAnalysisFrameSamples) ? kWindowLenSamples : kAnalysisFrameSamples;
constexpr int kMelFilterCoefficientCount = {mel_filter_coef_count};
constexpr size_t kTensorArenaSize = {tensor_arena_size};
constexpr float kPcmScale = 1.0f / 32768.0f;
constexpr float kNaturalLogToDbScale = 4.342944819032518f;
constexpr float kDefaultDecisionThreshold = 0.5f;
constexpr int kMaxProfiledOps = 32;
constexpr uint32_t kInvokeProfileReportAfterInvocations = 5;

enum class InputLayout {{
    kFlat,
    kMatrix,
    kNhwc,
    kNchw,
    kUnsupported,
}};

struct ModelInputSpec {{
    InputLayout layout = InputLayout::kUnsupported;
    int rows = 0;
    int cols = 0;
    int channels = 0;
    int element_count = 0;
}};

// 杩欎釜缁撴瀯鐢ㄤ簬璁板綍鍗曟 Invoke 鍐呴儴姣忎釜绠楀瓙鐨勫懆鏈熸暟锛屾柟渚胯瘖鏂ā鍨嬫湰浣撶摱棰堛€?
struct ProfiledInvokeEvent {{
    const char *tag = "";
    uint32_t start_cycles = 0;
    uint32_t elapsed_cycles = 0;
}};

// 浣跨敤 Zephyr 楂樼簿搴︾‖浠跺懆鏈熻鏁板櫒瀹炵幇 TFLM profiler 鎺ュ彛锛屽彧瀵筁nvoke 鍐呴儴绠楀瓙鎵撶偣銆?
class InvokeProfiler final : public tflite::MicroProfilerInterface {{
 public:
    void Reset()
    {{
        event_count_ = 0;
        overflowed_ = false;
    }}

    uint32_t BeginEvent(const char *tag) override
    {{
        if (event_count_ >= kMaxProfiledOps) {{
            overflowed_ = true;
            return static_cast<uint32_t>(kMaxProfiledOps);
        }}

        events_[event_count_].tag = tag;
        events_[event_count_].start_cycles = k_cycle_get_32();
        events_[event_count_].elapsed_cycles = 0;
        return static_cast<uint32_t>(event_count_++);
    }}

    void EndEvent(uint32_t event_handle) override
    {{
        if (event_handle >= static_cast<uint32_t>(event_count_)) {{
            return;
        }}
        events_[event_handle].elapsed_cycles =
            k_cycle_get_32() - events_[event_handle].start_cycles;
    }}

    int event_count() const
    {{
        return event_count_;
    }}

    bool overflowed() const
    {{
        return overflowed_;
    }}

    const ProfiledInvokeEvent &event(int index) const
    {{
        return events_[index];
    }}

 private:
    ProfiledInvokeEvent events_[kMaxProfiledOps] = {{}};
    int event_count_ = 0;
    bool overflowed_ = false;
}};

// 这些常量由 PC 侧离线生成，保证板端与导出参数严格一致。
alignas(16) static const float g_window[kWindowLenSamples] = {{
{window_literal}
}};
alignas(16) static const uint16_t g_mel_filter_pos[kBaseMelBinCount] = {{
{mel_filter_pos_literal}
}};
alignas(16) static const uint16_t g_mel_filter_len[kBaseMelBinCount] = {{
{mel_filter_len_literal}
}};
alignas(16) static const float g_mel_filter_coefs[kMelFilterCoefficientCount] = {{
{mel_filter_coef_literal}
}};

alignas(16) static float g_fft_input[kFftLength];
alignas(16) static float g_fft_output[2 * kFftBinCount];
alignas(16) static float g_power_spectrum[kFftBinCount];
alignas(16) static float g_base_feature_buf[kBaseFeatureElementCount];
alignas(16) static uint8_t g_tensor_arena[kTensorArenaSize];

static arm_rfft_fast_instance_f32 g_fft_instance;
static const tflite::Model *g_model = nullptr;
static tflite::MicroInterpreter *g_interpreter = nullptr;
static TfLiteTensor *g_input = nullptr;
static TfLiteTensor *g_output = nullptr;
static ModelInputSpec g_input_spec;
static InvokeProfiler g_invoke_profiler;
static int g_output_count = 0;
static int g_model_operator_count = 0;
static bool g_frontend_initialized = false;
static bool g_frontend_init_failed = false;
static bool g_model_initialized = false;
static bool g_model_init_attempted = false;
static bool g_invoke_profile_reported = false;
static const char *g_last_error = "not initialized";
static uint32_t g_last_feature_ms = 0;
static uint32_t g_last_invoke_ms = 0;
static uint32_t g_invoke_profile_sample_count = 0;
static uint64_t g_invoke_profile_cycle_sums[kMaxProfiledOps] = {{}};

void set_last_error(const char *message)
{{
    g_last_error = message;
}}

uint32_t cycles_to_us(uint32_t cycles)
{{
    const uint64_t cycle_hz = static_cast<uint64_t>(sys_clock_hw_cycles_per_sec());
    if (cycle_hz == 0U) {{
        return 0U;
    }}
    return static_cast<uint32_t>(
        ((static_cast<uint64_t>(cycles) * 1000000ULL) + (cycle_hz / 2ULL)) / cycle_hz);
}}

const tflite::SubGraph *primary_subgraph()
{{
    if (g_model == nullptr || g_model->subgraphs() == nullptr || g_model->subgraphs()->size() == 0) {{
        return nullptr;
    }}
    return g_model->subgraphs()->Get(0);
}}

int flatbuffer_operator_count()
{{
    const tflite::SubGraph *subgraph = primary_subgraph();
    if (subgraph == nullptr || subgraph->operators() == nullptr) {{
        return 0;
    }}
    return static_cast<int>(subgraph->operators()->size());
}}

const tflite::Operator *flatbuffer_operator_at(int op_index)
{{
    const tflite::SubGraph *subgraph = primary_subgraph();
    if (subgraph == nullptr || subgraph->operators() == nullptr) {{
        return nullptr;
    }}
    const int operator_count = static_cast<int>(subgraph->operators()->size());
    if (op_index < 0 || op_index >= operator_count) {{
        return nullptr;
    }}
    return subgraph->operators()->Get(op_index);
}}

const tflite::OperatorCode *flatbuffer_operator_code(const tflite::Operator *op)
{{
    if (g_model == nullptr || op == nullptr || g_model->operator_codes() == nullptr) {{
        return nullptr;
    }}
    const int opcode_index = op->opcode_index();
    const int opcode_count = static_cast<int>(g_model->operator_codes()->size());
    if (opcode_index < 0 || opcode_index >= opcode_count) {{
        return nullptr;
    }}
    return g_model->operator_codes()->Get(opcode_index);
}}

const char *flatbuffer_operator_name(const tflite::Operator *op)
{{
    const tflite::OperatorCode *opcode = flatbuffer_operator_code(op);
    if (opcode == nullptr) {{
        return "UNKNOWN";
    }}
    return tflite::EnumNameBuiltinOperator(opcode->builtin_code());
}}

const tflite::Tensor *flatbuffer_tensor_at(int tensor_index)
{{
    const tflite::SubGraph *subgraph = primary_subgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {{
        return nullptr;
    }}
    const int tensor_count = static_cast<int>(subgraph->tensors()->size());
    if (tensor_index < 0 || tensor_index >= tensor_count) {{
        return nullptr;
    }}
    return subgraph->tensors()->Get(tensor_index);
}}

void log_flatbuffer_tensor_shape(const tflite::Tensor *tensor)
{{
    if (tensor == nullptr || tensor->shape() == nullptr) {{
        printk("[]");
        return;
    }}

    const int dim_count = static_cast<int>(tensor->shape()->size());
    printk("[");
    for (int dim_index = 0; dim_index < dim_count; ++dim_index) {{
        if (dim_index > 0) {{
            printk("x");
        }}
        printk("%d", tensor->shape()->Get(dim_index));
    }}
    printk("]");
}}

void log_invoke_profile_summary(int event_count)
{{
    if (g_invoke_profile_sample_count == 0) {{
        return;
    }}

    const int profiled_event_count = std::min(event_count, kMaxProfiledOps);
    const int mapped_event_count = std::min(profiled_event_count, g_model_operator_count);
    uint64_t total_avg_cycles = 0;
    for (int event_index = 0; event_index < profiled_event_count; ++event_index) {{
        total_avg_cycles += g_invoke_profile_cycle_sums[event_index] / g_invoke_profile_sample_count;
    }}

    printk("Invoke breakdown: avg over %u runs, cycle_hz=%u\\n",
           g_invoke_profile_sample_count,
           static_cast<unsigned int>(sys_clock_hw_cycles_per_sec()));

    if (g_invoke_profiler.overflowed()) {{
        printk("  profiler warning: event buffer overflowed, only first %d events kept\\n",
               kMaxProfiledOps);
    }}
    if (profiled_event_count != g_model_operator_count) {{
        printk("  profiler note: event_count=%d, model_op_count=%d\\n",
               profiled_event_count,
               g_model_operator_count);
    }}

    for (int event_index = 0; event_index < profiled_event_count; ++event_index) {{
        const uint32_t avg_cycles = static_cast<uint32_t>(
            g_invoke_profile_cycle_sums[event_index] / g_invoke_profile_sample_count);
        const uint32_t avg_us = cycles_to_us(avg_cycles);
        const uint32_t share_permille = (total_avg_cycles > 0U)
            ? static_cast<uint32_t>((static_cast<uint64_t>(avg_cycles) * 1000ULL) / total_avg_cycles)
            : 0U;

        const char *op_name = g_invoke_profiler.event(event_index).tag;
        const tflite::Operator *op = nullptr;
        if (event_index < mapped_event_count) {{
            op = flatbuffer_operator_at(event_index);
            if (op != nullptr) {{
                op_name = flatbuffer_operator_name(op);
            }}
        }}

        printk("  op%02d %s avg=%u us share=%u.%u%%",
               event_index,
               op_name != nullptr ? op_name : "UNKNOWN",
               avg_us,
               share_permille / 10U,
               share_permille % 10U);

        if (op != nullptr && op->outputs() != nullptr && op->outputs()->size() > 0) {{
            printk(" out=");
            log_flatbuffer_tensor_shape(flatbuffer_tensor_at(op->outputs()->Get(0)));
        }}
        printk("\\n");
    }}
}}

void accumulate_invoke_profile()
{{
    if (g_invoke_profile_reported) {{
        return;
    }}

    const int event_count = std::min(g_invoke_profiler.event_count(), kMaxProfiledOps);
    if (event_count <= 0) {{
        return;
    }}

    for (int event_index = 0; event_index < event_count; ++event_index) {{
        g_invoke_profile_cycle_sums[event_index] +=
            g_invoke_profiler.event(event_index).elapsed_cycles;
    }}
    ++g_invoke_profile_sample_count;

    if (g_invoke_profile_sample_count >= kInvokeProfileReportAfterInvocations) {{
        log_invoke_profile_summary(event_count);
        g_invoke_profile_reported = true;
    }}
}}

int8_t quantize_to_int8(float value, float scale, int zero_point)
{{
    if (scale == 0.0f) {{
        return 0;
    }}

    int32_t q = static_cast<int32_t>(lroundf(value / scale)) + zero_point;
    if (q < -128) {{
        q = -128;
    }} else if (q > 127) {{
        q = 127;
    }}
    return static_cast<int8_t>(q);
}}

float dequantize_from_int8(int8_t value, float scale, int zero_point)
{{
    return (static_cast<int32_t>(value) - zero_point) * scale;
}}

float sigmoidf(float x)
{{
    if (x >= 0.0f) {{
        const float e = expf(-x);
        return 1.0f / (1.0f + e);
    }}

    const float e = expf(x);
    return e / (1.0f + e);
}}

void softmax_inplace(float *values, int count)
{{
    if (values == nullptr || count <= 0) {{
        return;
    }}

    float max_value = values[0];
    for (int i = 1; i < count; ++i) {{
        if (values[i] > max_value) {{
            max_value = values[i];
        }}
    }}

    float sum = 0.0f;
    for (int i = 0; i < count; ++i) {{
        values[i] = expf(values[i] - max_value);
        sum += values[i];
    }}

    if (sum <= 0.0f) {{
        return;
    }}

    for (int i = 0; i < count; ++i) {{
        values[i] /= sum;
    }}
}}

Result invalid_result()
{{
    Result result = {{}};
    result.predicted_score = 0.0f;
    result.predicted_label = 0;
    result.output_count = 0;
    for (int i = 0; i < audio_event_model_config::kLabelCount; ++i) {{
        result.scores[i] = 0.0f;
    }}
    result.inference_ok = false;
    return result;
}}

int tensor_element_count(const TfLiteTensor *tensor)
{{
    if (tensor == nullptr || tensor->dims == nullptr) {{
        return 0;
    }}

    int count = 1;
    for (int i = 0; i < tensor->dims->size; ++i) {{
        count *= tensor->dims->data[i];
    }}
    return count;
}}

int tensor_value_count(const TfLiteTensor *tensor)
{{
    if (tensor == nullptr) {{
        return 0;
    }}

    switch (tensor->type) {{
    case kTfLiteFloat32:
        return static_cast<int>(tensor->bytes / sizeof(float));
    case kTfLiteInt8:
        return static_cast<int>(tensor->bytes / sizeof(int8_t));
    default:
        return 0;
    }}
}}

ModelInputSpec infer_model_input_spec(const TfLiteTensor *tensor)
{{
    ModelInputSpec spec;
    if (tensor == nullptr || tensor->dims == nullptr) {{
        return spec;
    }}

    const TfLiteIntArray *dims = tensor->dims;
    spec.element_count = tensor_element_count(tensor);
    if (spec.element_count <= 0) {{
        return spec;
    }}

    if (dims->size == 4) {{
        const int d0 = dims->data[0];
        const int d1 = dims->data[1];
        const int d2 = dims->data[2];
        const int d3 = dims->data[3];

        if (d0 != 1) {{
            return spec;
        }}
        if (d3 == 1 && d1 > 0 && d2 > 0) {{
            spec.layout = InputLayout::kNhwc;
            spec.rows = d1;
            spec.cols = d2;
            spec.channels = d3;
            return spec;
        }}
        if (d1 == 1 && d2 > 0 && d3 > 0) {{
            spec.layout = InputLayout::kNchw;
            spec.rows = d2;
            spec.cols = d3;
            spec.channels = d1;
            return spec;
        }}
        return spec;
    }}

    if (dims->size == 3) {{
        if (dims->data[0] != 1 || dims->data[1] <= 0 || dims->data[2] <= 0) {{
            return spec;
        }}
        spec.layout = InputLayout::kMatrix;
        spec.rows = dims->data[1];
        spec.cols = dims->data[2];
        spec.channels = 1;
        return spec;
    }}

    if (dims->size == 2) {{
        if (dims->data[0] != 1 || dims->data[1] <= 0) {{
            return spec;
        }}
        spec.layout = InputLayout::kFlat;
        spec.rows = 1;
        spec.cols = dims->data[1];
        spec.channels = 1;
    }}

    return spec;
}}

bool matches_base_feature_matrix(int rows, int cols)
{{
    return rows == kBaseMelBinCount && cols == kBaseFeatureFrameCount;
}}

bool matches_transposed_feature_matrix(int rows, int cols)
{{
    return rows == kBaseFeatureFrameCount && cols == kBaseMelBinCount;
}}

bool matches_flat_feature_vector(int rows, int cols)
{{
    return (rows == 1 && cols == kBaseFeatureElementCount) ||
           (rows == kBaseFeatureElementCount && cols == 1);
}}

float feature_value_for_input_index(int index)
{{
    if (index < 0 || index >= kBaseFeatureElementCount) {{
        return 0.0f;
    }}

    if (g_input_spec.layout == InputLayout::kFlat) {{
        return g_base_feature_buf[index];
    }}

    if (matches_base_feature_matrix(g_input_spec.rows, g_input_spec.cols) ||
        matches_flat_feature_vector(g_input_spec.rows, g_input_spec.cols)) {{
        return g_base_feature_buf[index];
    }}

    if (matches_transposed_feature_matrix(g_input_spec.rows, g_input_spec.cols)) {{
        const int frame_idx = index / kBaseMelBinCount;
        const int mel_idx = index % kBaseMelBinCount;
        return g_base_feature_buf[(mel_idx * kBaseFeatureFrameCount) + frame_idx];
    }}

    return 0.0f;
}}

bool init_frontend_once()
{{
    if (g_frontend_initialized) {{
        return true;
    }}
    if (g_frontend_init_failed) {{
        return false;
    }}

    const arm_status fft_status = arm_rfft_fast_init_f32(&g_fft_instance, kFftLength);
    if (fft_status != ARM_MATH_SUCCESS) {{
        printk("Generated frontend FFT init failed: status=%d\\n", static_cast<int>(fft_status));
        g_frontend_init_failed = true;
        return false;
    }}

    g_frontend_initialized = true;
    return true;
}}

bool extract_base_log_mel_features(const int16_t *samples, int sample_count)
{{
    if (samples == nullptr || sample_count < kWindowSampleCount) {{
        return false;
    }}
    if (!init_frontend_once()) {{
        return false;
    }}

    std::fill(g_base_feature_buf, g_base_feature_buf + kBaseFeatureElementCount, 0.0f);

    int frames_written = 0;
    for (int start = 0;
         start + kAnalysisFrameSamples <= sample_count &&
         frames_written < kBaseFeatureFrameCount;
         start += kFrameStrideSamples) {{
        // 逐帧构造零填充输入，再只对有效窗口区间做缩放和加窗。
        std::fill(g_fft_input, g_fft_input + kFftLength, 0.0f);
        for (int i = 0; i < kFrameFftCopySamples; ++i) {{
            const int fft_index = kWindowPadLeft + i;
            g_fft_input[fft_index] = static_cast<float>(samples[start + fft_index]);
        }}
        arm_scale_f32(
            g_fft_input + kWindowPadLeft,
            kPcmScale,
            g_fft_input + kWindowPadLeft,
            kFrameFftCopySamples);
        arm_mult_f32(
            g_fft_input + kWindowPadLeft,
            g_window,
            g_fft_input + kWindowPadLeft,
            kFrameFftCopySamples);

        // CMSIS-DSP 的 RFFT 输出先解包成 [re, im] 复数对，再求功率谱。
        arm_rfft_fast_f32(&g_fft_instance, g_fft_input, g_fft_output, 0);
        g_fft_output[kFftLength] = g_fft_output[1];
        g_fft_output[kFftLength + 1] = 0.0f;
        g_fft_output[1] = 0.0f;
        arm_cmplx_mag_squared_f32(g_fft_output, g_power_spectrum, kFftBinCount);

        const float *mel_weights = g_mel_filter_coefs;
        for (int mel_idx = 0; mel_idx < kBaseMelBinCount; ++mel_idx) {{
            float mel_energy = 0.0f;
            const uint16_t filter_pos = g_mel_filter_pos[mel_idx];
            const uint16_t filter_len = g_mel_filter_len[mel_idx];
            if (filter_len > 0U) {{
                arm_dot_prod_f32(
                    g_power_spectrum + filter_pos,
                    mel_weights,
                    filter_len,
                    &mel_energy);
            }}
            mel_weights += filter_len;
            g_base_feature_buf[(mel_idx * kBaseFeatureFrameCount) + frames_written] = mel_energy;
        }}

        ++frames_written;
    }}

    if (frames_written != kBaseFeatureFrameCount) {{
        return false;
    }}

    // 先用自然对数向量化，再缩放成 dB，随后按 top_db 做截断。
    arm_offset_f32(
        g_base_feature_buf,
        1.0e-12f,
        g_base_feature_buf,
        kBaseFeatureElementCount);
    arm_vlog_f32(
        g_base_feature_buf,
        g_base_feature_buf,
        kBaseFeatureElementCount);
    arm_scale_f32(
        g_base_feature_buf,
        kNaturalLogToDbScale,
        g_base_feature_buf,
        kBaseFeatureElementCount);

    float max_db = 0.0f;
    uint32_t max_index = 0;
    arm_max_f32(g_base_feature_buf, kBaseFeatureElementCount, &max_db, &max_index);
    const float floor_db = max_db - audio_event_model_config::kTopDb;
    for (int i = 0; i < kBaseFeatureElementCount; ++i) {{
        g_base_feature_buf[i] = std::max(g_base_feature_buf[i], floor_db);
    }}

    // 这里保持与旧实现一致，使用总体标准差而不是 CMSIS 的 N-1 样本标准差。
    float mean = 0.0f;
    arm_mean_f32(g_base_feature_buf, kBaseFeatureElementCount, &mean);
    arm_offset_f32(
        g_base_feature_buf,
        -mean,
        g_base_feature_buf,
        kBaseFeatureElementCount);

    float centered_energy = 0.0f;
    arm_dot_prod_f32(
        g_base_feature_buf,
        g_base_feature_buf,
        kBaseFeatureElementCount,
        &centered_energy);
    const float variance = std::max(
        centered_energy / static_cast<float>(kBaseFeatureElementCount),
        0.0f);
    float stddev = 0.0f;
    if (arm_sqrt_f32(variance, &stddev) != ARM_MATH_SUCCESS) {{
        return false;
    }}

    const float normalize_scale = 1.0f / (stddev + 1.0e-6f);
    arm_scale_f32(
        g_base_feature_buf,
        normalize_scale,
        g_base_feature_buf,
        kBaseFeatureElementCount);
    return true;
}}

bool validate_model_input_layout()
{{
    if (g_input_spec.element_count != kBaseFeatureElementCount) {{
        return false;
    }}

    if (g_input_spec.layout == InputLayout::kFlat) {{
        return true;
    }}

    if (g_input_spec.layout == InputLayout::kMatrix) {{
        return matches_base_feature_matrix(g_input_spec.rows, g_input_spec.cols) ||
               matches_transposed_feature_matrix(g_input_spec.rows, g_input_spec.cols) ||
               matches_flat_feature_vector(g_input_spec.rows, g_input_spec.cols);
    }}

    if ((g_input_spec.layout == InputLayout::kNhwc ||
         g_input_spec.layout == InputLayout::kNchw) &&
        g_input_spec.channels == 1) {{
        return matches_base_feature_matrix(g_input_spec.rows, g_input_spec.cols) ||
               matches_transposed_feature_matrix(g_input_spec.rows, g_input_spec.cols) ||
               matches_flat_feature_vector(g_input_spec.rows, g_input_spec.cols);
    }}

    return false;
}}

bool init_model_once()
{{
    if (g_model_initialized) {{
        return true;
    }}
    if (g_model_init_attempted) {{
        printk("Generated model init previously failed: %s\\n", g_last_error);
        return false;
    }}
    g_model_init_attempted = true;

    if (audio_event_model_data::kPlaceholderModel) {{
        set_last_error("placeholder model still active");
        printk("Generated model placeholder is still in use\\n");
        return false;
    }}

    g_model = tflite::GetModel(audio_event_model_data::g_model);
    if (g_model == nullptr) {{
        set_last_error("model pointer is null");
        return false;
    }}

    if (g_model->version() != TFLITE_SCHEMA_VERSION) {{
        set_last_error("schema mismatch");
        printk("Generated model schema mismatch: model=%d runtime=%d\\n",
               g_model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }}

    g_model_operator_count = flatbuffer_operator_count();
    if (g_model_operator_count <= 0) {{
        set_last_error("model operator list is empty");
        return false;
    }}
    if (g_model_operator_count > kMaxProfiledOps) {{
        set_last_error("model has more ops than profiler buffer");
        printk("Generated model op count %d exceeds profiler capacity %d\\n",
               g_model_operator_count, kMaxProfiledOps);
        return false;
    }}
    std::fill(g_invoke_profile_cycle_sums,
              g_invoke_profile_cycle_sums + kMaxProfiledOps,
              0ULL);
    g_invoke_profile_sample_count = 0;
    g_invoke_profile_reported = false;
    g_invoke_profiler.Reset();

    static tflite::MicroMutableOpResolver<{resolver_capacity}> resolver;
{resolver_block}

    static tflite::MicroInterpreter static_interpreter(
        g_model, resolver, g_tensor_arena, kTensorArenaSize, nullptr, &g_invoke_profiler);
    g_interpreter = &static_interpreter;

    if (g_interpreter->AllocateTensors() != kTfLiteOk) {{
        set_last_error("AllocateTensors failed");
        g_interpreter = nullptr;
        return false;
    }}

    g_input = g_interpreter->input(0);
    g_output = g_interpreter->output(0);
    if (g_input == nullptr || g_output == nullptr) {{
        set_last_error("input/output tensor missing");
        return false;
    }}

    g_input_spec = infer_model_input_spec(g_input);
    if (g_input_spec.layout == InputLayout::kUnsupported) {{
        set_last_error("unsupported model input shape");
        return false;
    }}
    if (g_input_spec.channels != 1) {{
        set_last_error("multi-channel model input unsupported");
        return false;
    }}
    if (g_input_spec.element_count > audio_event_model_config::kMaxModelInputElementCount) {{
        set_last_error("model input exceeds feature buffer capacity");
        return false;
    }}

    g_output_count = tensor_value_count(g_output);
    if (g_output_count <= 0) {{
        set_last_error("unsupported output tensor type");
        return false;
    }}

    g_model_initialized = true;
    set_last_error("ok");
    printk("Generated model ready: input_elements=%d output_count=%d arena_used=%u B arena_reserved=%u B\\n",
           g_input_spec.element_count,
           g_output_count,
           static_cast<unsigned int>(g_interpreter->arena_used_bytes()),
           static_cast<unsigned int>(kTensorArenaSize));
    printk("Invoke profiler armed: average first %u runs, then print per-op breakdown once\\n",
           static_cast<unsigned int>(kInvokeProfileReportAfterInvocations));
    return true;
}}

}}  // namespace

bool model_is_placeholder()
{{
    return audio_event_model_data::kPlaceholderModel;
}}

unsigned int runtime_memory_size_bytes()
{{
    return static_cast<unsigned int>(
        sizeof(g_base_feature_buf) + sizeof(g_tensor_arena));
}}

const char *last_error()
{{
    return g_last_error;
}}

uint32_t last_feature_ms()
{{
    return g_last_feature_ms;
}}

uint32_t last_invoke_ms()
{{
    return g_last_invoke_ms;
}}

Result run_classifier(const int16_t *samples, int sample_count)
{{
    g_last_feature_ms = 0;
    g_last_invoke_ms = 0;

    if (samples == nullptr || sample_count != kWindowSampleCount) {{
        set_last_error("invalid audio window");
        return invalid_result();
    }}
    if (!init_model_once()) {{
        return invalid_result();
    }}

    const uint32_t feature_start_ms = k_uptime_get_32();
    if (!extract_base_log_mel_features(samples, sample_count)) {{
        set_last_error("frontend extraction failed");
        return invalid_result();
    }}
    if (!validate_model_input_layout()) {{
        set_last_error("model input layout mismatch");
        return invalid_result();
    }}
    g_last_feature_ms = k_uptime_get_32() - feature_start_ms;

    if (g_input->type == kTfLiteInt8) {{
        for (int i = 0; i < g_input_spec.element_count; ++i) {{
            const float feature_value = feature_value_for_input_index(i);
            g_input->data.int8[i] = quantize_to_int8(
                feature_value, g_input->params.scale, g_input->params.zero_point);
        }}
    }} else if (g_input->type == kTfLiteFloat32) {{
        for (int i = 0; i < g_input_spec.element_count; ++i) {{
            g_input->data.f[i] = feature_value_for_input_index(i);
        }}
    }} else {{
        set_last_error("unsupported input tensor type");
        return invalid_result();
    }}

    g_invoke_profiler.Reset();
    const uint32_t invoke_start_ms = k_uptime_get_32();
    if (g_interpreter->Invoke() != kTfLiteOk) {{
        set_last_error("Invoke failed");
        return invalid_result();
    }}
    g_last_invoke_ms = k_uptime_get_32() - invoke_start_ms;
    accumulate_invoke_profile();

    Result result = {{}};
    result.output_count = g_output_count;
    if (g_output->type == kTfLiteInt8) {{
        for (int i = 0; i < g_output_count; ++i) {{
            const float value = dequantize_from_int8(
                g_output->data.int8[i], g_output->params.scale, g_output->params.zero_point);
            result.scores[i] = value;
        }}
    }} else if (g_output->type == kTfLiteFloat32) {{
        for (int i = 0; i < g_output_count; ++i) {{
            result.scores[i] = g_output->data.f[i];
        }}
    }} else {{
        set_last_error("unsupported output tensor type");
        return invalid_result();
    }}

    if (g_output_count == 1) {{
        result.scores[0] = sigmoidf(result.scores[0]);
        result.predicted_score = result.scores[0];
        result.predicted_label = (result.predicted_score >= kDefaultDecisionThreshold) ? 1 : 0;
    }} else {{
        softmax_inplace(result.scores, g_output_count);
        result.predicted_label = 0;
        result.predicted_score = result.scores[0];
        for (int i = 1; i < g_output_count; ++i) {{
            if (result.scores[i] > result.predicted_score) {{
                result.predicted_score = result.scores[i];
                result.predicted_label = i;
            }}
        }}
    }}

    set_last_error("ok");
    result.inference_ok = true;
    return result;
}}

}}  // namespace audio_event_model_runner
"""


def write_nrf5340_artifacts(model_path: pathlib.Path,
                            output_dir: pathlib.Path,
                            labels: list[str],
                            sample_rate_hz: int,
                            capture_window_ms: int,
                            frame_length_ms: int,
                            frame_stride_ms: int,
                            fft_length: int,
                            mel_bin_count: int,
                            feature_frame_count: int,
                            mel_lower_edge_hz: float,
                            mel_upper_edge_hz: float,
                            top_db: float,
                            tensor_arena_size: int,
                            process_contract: dict[str, Any] | None = None) -> dict[str, pathlib.Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    details = _read_tflite_details(model_path)
    _validate_input_shape(details["input_shape"], mel_bin_count, feature_frame_count)
    _validate_process_contract(
        process_contract,
        details["input_shape"],
        sample_rate_hz=sample_rate_hz,
        capture_window_ms=capture_window_ms,
        frame_length_ms=frame_length_ms,
        frame_stride_ms=frame_stride_ms,
        fft_length=fft_length,
        mel_bin_count=mel_bin_count,
        feature_frame_count=feature_frame_count,
        mel_lower_edge_hz=mel_lower_edge_hz,
        mel_upper_edge_hz=mel_upper_edge_hz,
        top_db=top_db,
    )

    if len(labels) != details["output_shape"][-1]:
        raise ValueError(
            f"Label count {len(labels)} does not match model output width {details['output_shape'][-1]}"
        )

    resolver_lines = build_resolver_lines(details["op_names"])
    model_bytes = model_path.read_bytes()
    window_len_samples = (sample_rate_hz * frame_length_ms) // 1000
    window_coefficients = _build_hann_window(window_len_samples)
    mel_filterbank = _build_sparse_mel_filterbank(
        sample_rate_hz=sample_rate_hz,
        fft_length=fft_length,
        mel_bin_count=mel_bin_count,
        mel_lower_edge_hz=mel_lower_edge_hz,
        mel_upper_edge_hz=mel_upper_edge_hz,
    )

    files = {
        "config_h": output_dir / DEFAULT_OUTPUT_FILES["config_h"],
        "data_h": output_dir / DEFAULT_OUTPUT_FILES["data_h"],
        "data_cpp": output_dir / DEFAULT_OUTPUT_FILES["data_cpp"],
        "runner_h": output_dir / DEFAULT_OUTPUT_FILES["runner_h"],
        "runner_cpp": output_dir / DEFAULT_OUTPUT_FILES["runner_cpp"],
    }

    files["config_h"].write_text(
        _config_header_text(
            labels=labels,
            sample_rate_hz=sample_rate_hz,
            capture_window_ms=capture_window_ms,
            frame_length_ms=frame_length_ms,
            frame_stride_ms=frame_stride_ms,
            fft_length=fft_length,
            mel_bin_count=mel_bin_count,
            feature_frame_count=feature_frame_count,
            mel_lower_edge_hz=mel_lower_edge_hz,
            mel_upper_edge_hz=mel_upper_edge_hz,
            top_db=top_db,
        ),
        encoding="utf-8",
    )
    files["data_h"].write_text(_data_header_text(), encoding="utf-8")
    files["data_cpp"].write_text(_data_cpp_text(model_bytes), encoding="utf-8")
    files["runner_h"].write_text(_runner_header_text(), encoding="utf-8")
    files["runner_cpp"].write_text(
        _runner_cpp_text_accelerated(
            tensor_arena_size=tensor_arena_size,
            resolver_lines=resolver_lines,
            window_coefficients=window_coefficients,
            mel_filter_positions=[int(value) for value in mel_filterbank["positions"]],
            mel_filter_lengths=[int(value) for value in mel_filterbank["lengths"]],
            mel_filter_coefficients=[float(value) for value in mel_filterbank["coefficients"]],
        ),
        encoding="utf-8",
    )
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate audio_event_model_* files for the nRF5340 firmware package."
    )
    parser.add_argument("model_path", help="Path to the full-int8 .tflite model file")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where audio_event_model_* files will be written",
    )
    parser.add_argument("--manifest", help="Optional torch2int8_tflite manifest JSON")
    parser.add_argument("--process-contract", help="Optional recovered process_contract.json used for firmware preflight")
    parser.add_argument("--labels", help="Comma-separated label names")
    parser.add_argument("--sample-rate-hz", type=int, default=DEFAULT_SAMPLE_RATE_HZ)
    parser.add_argument("--capture-window-ms", type=int, default=DEFAULT_CAPTURE_WINDOW_MS)
    parser.add_argument("--frame-length-ms", type=int, default=DEFAULT_FRAME_LENGTH_MS)
    parser.add_argument("--frame-stride-ms", type=int, default=DEFAULT_FRAME_STRIDE_MS)
    parser.add_argument("--fft-length", type=int, default=DEFAULT_FFT_LENGTH)
    parser.add_argument("--mel-bin-count", type=int, default=DEFAULT_MEL_BIN_COUNT)
    parser.add_argument("--feature-frame-count", type=int)
    parser.add_argument("--mel-lower-edge-hz", type=float, default=DEFAULT_MEL_LOWER_EDGE_HZ)
    parser.add_argument("--mel-upper-edge-hz", type=float, default=DEFAULT_MEL_UPPER_EDGE_HZ)
    parser.add_argument("--top-db", type=float, default=DEFAULT_TOP_DB)
    parser.add_argument("--tensor-arena-size", type=int, default=DEFAULT_TENSOR_ARENA_SIZE)
    args = parser.parse_args()

    model_path = pathlib.Path(args.model_path).resolve()
    output_dir = pathlib.Path(args.output_dir).resolve()
    manifest = _parse_manifest(pathlib.Path(args.manifest).resolve() if args.manifest else None)
    process_contract = _parse_process_contract(
        pathlib.Path(args.process_contract).resolve() if args.process_contract else None
    )
    labels = _parse_labels(args.labels)

    feature_frame_count = args.feature_frame_count
    if feature_frame_count is None:
        feature_frame_count = _feature_frame_count(
            sample_rate_hz=args.sample_rate_hz,
            capture_window_ms=args.capture_window_ms,
            fft_length=args.fft_length,
            frame_length_ms=args.frame_length_ms,
            frame_stride_ms=args.frame_stride_ms,
        )

    written_files = write_nrf5340_artifacts(
        model_path=model_path,
        output_dir=output_dir,
        labels=labels,
        sample_rate_hz=args.sample_rate_hz,
        capture_window_ms=args.capture_window_ms,
        frame_length_ms=args.frame_length_ms,
        frame_stride_ms=args.frame_stride_ms,
        fft_length=args.fft_length,
        mel_bin_count=args.mel_bin_count,
        feature_frame_count=feature_frame_count,
        mel_lower_edge_hz=args.mel_lower_edge_hz,
        mel_upper_edge_hz=args.mel_upper_edge_hz,
        top_db=args.top_db,
        tensor_arena_size=args.tensor_arena_size,
        process_contract=process_contract,
    )

    summary = {
        "model_path": str(model_path),
        "output_dir": str(output_dir),
        "manifest": manifest,
        "process_contract": str(pathlib.Path(args.process_contract).resolve()) if args.process_contract else None,
        "labels": labels,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    for key, path in written_files.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
