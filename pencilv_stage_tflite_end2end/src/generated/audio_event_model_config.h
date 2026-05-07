#pragma once

// 本文件由部署脚本自动生成，保存板端前处理和标签所需的固定配置。

namespace audio_event_model_config {

constexpr int kSampleRateHz = 16000;
constexpr int kCaptureWindowMs = 1000;
constexpr int kFrameLengthMs = 25;
constexpr int kFrameStrideMs = 10;
constexpr int kFftLength = 1024;
constexpr int kFrontendMelBinCount = 48;
constexpr int kFrontendFeatureFrameCount = 94;
constexpr int kFrontendFeatureElementCount =
    kFrontendMelBinCount * kFrontendFeatureFrameCount;
constexpr int kMaxModelInputElementCount = kFrontendFeatureElementCount;
constexpr int kLabelCount = 3;
constexpr float kMelLowerEdgeHz = 3000.000000f;
constexpr float kMelUpperEdgeHz = 7900.000000f;
constexpr float kTopDb = 80.000000f;

inline constexpr const char *kLabels[kLabelCount] = {
    "eat",
    "drink",
    "other"
};

}  // namespace audio_event_model_config
