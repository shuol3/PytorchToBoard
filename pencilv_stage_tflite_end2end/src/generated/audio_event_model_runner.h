#pragma once

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
