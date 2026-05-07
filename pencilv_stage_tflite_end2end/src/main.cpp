#include <errno.h>
#include <stdint.h>
#include <string.h>

#include <zephyr/device.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/drivers/i2s.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>

#include "generated/audio_event_model_runner.h"
#include "generated/audio_event_model_config.h"

namespace {

#define I2S_RX_NODE DT_NODELABEL(i2s0)

#if DT_NODE_EXISTS(DT_ALIAS(mic_en))
static const struct gpio_dt_spec mic_enable =
    GPIO_DT_SPEC_GET(DT_ALIAS(mic_en), gpios);
constexpr bool kHasMicEnable = true;
#else
constexpr bool kHasMicEnable = false;
#endif

constexpr uint32_t kAudioSampleRate = audio_event_model_config::kSampleRateHz;
constexpr uint8_t kAudioWordSizeBits = 16;
constexpr size_t kI2sChannels = 2;
constexpr uint32_t kFrameMs = 100;
constexpr size_t kFramesPerBlock = (kAudioSampleRate * kFrameMs) / 1000;
constexpr size_t kWindowSamples = audio_event_model_runner::window_samples();
constexpr uint32_t kRingBufferMs = 2000;
constexpr size_t kRingSamples = (kAudioSampleRate * kRingBufferMs) / 1000;
constexpr size_t kBlockSamples = kFramesPerBlock * kI2sChannels;
constexpr size_t kBlockSize = kBlockSamples * sizeof(int16_t);
constexpr size_t kI2sSlabBlockCount = 4;
constexpr uint32_t kReadTimeoutMs = 200;
constexpr uint32_t kMicPowerUpDelayMs = 20;
constexpr size_t kCaptureThreadStackSize = 4096;
constexpr int kCaptureThreadPriority = 0;
constexpr int kClassifierThreadPriority = 1;

static_assert(kRingSamples >= kWindowSamples, "Ring buffer must hold at least one inference window");

K_MEM_SLAB_DEFINE_STATIC(audio_mem_slab, kBlockSize, kI2sSlabBlockCount, 4);
K_MUTEX_DEFINE(audio_ring_mutex);
K_SEM_DEFINE(audio_window_ready_sem, 0, 1);
K_THREAD_STACK_DEFINE(capture_thread_stack, kCaptureThreadStackSize);

static const struct device *i2s_dev;
static struct i2s_config i2s_cfg;
static struct k_thread capture_thread_data;
static int16_t audio_ring[kRingSamples];
static int16_t infer_buf[kWindowSamples];

struct AudioRingState {
    size_t write_idx;
    size_t read_idx;
    size_t count;
    uint32_t overrun_events;
    uint32_t dropped_samples;
};

static AudioRingState audio_ring_state;

void print_runtime_memory_summary_once()
{
    const uint32_t infer_buffer_bytes = sizeof(infer_buf);
    const uint32_t ring_buffer_bytes = sizeof(audio_ring);
    const uint32_t generated_runtime_bytes =
        audio_event_model_runner::runtime_memory_size_bytes();
    const uint32_t i2s_slab_bytes = kI2sSlabBlockCount * kBlockSize;
    const uint32_t capture_thread_stack_bytes =
        K_THREAD_STACK_SIZEOF(capture_thread_stack);
    const uint32_t known_runtime_bytes =
        infer_buffer_bytes + ring_buffer_bytes + generated_runtime_bytes +
        i2s_slab_bytes + capture_thread_stack_bytes;

    printk("Runtime memory: infer_buf=%u B ring_buf=%u B generated_runtime=%u B i2s_slab=%u B capture_stack=%u B known_total=%u B\n",
           infer_buffer_bytes,
           ring_buffer_bytes,
           generated_runtime_bytes,
           i2s_slab_bytes,
           capture_thread_stack_bytes,
           known_runtime_bytes);
}

int16_t select_mono_sample(int16_t left, int16_t right)
{
    if (left == 0) {
        return right;
    }
    if (right == 0) {
        return left;
    }

    return static_cast<int16_t>((static_cast<int32_t>(left) +
                                 static_cast<int32_t>(right)) / 2);
}

int enable_microphone_power(void)
{
    if (!kHasMicEnable) {
        return 0;
    }

    if (!gpio_is_ready_dt(&mic_enable)) {
        printk("MIC_EN GPIO not ready\n");
        return -ENODEV;
    }

    int ret = gpio_pin_configure_dt(&mic_enable, GPIO_OUTPUT_INACTIVE);
    if (ret < 0) {
        printk("Failed to configure MIC_EN: %d\n", ret);
        return ret;
    }

    ret = gpio_pin_set_dt(&mic_enable, 1);
    if (ret < 0) {
        printk("Failed to enable microphone rail: %d\n", ret);
        return ret;
    }

    k_sleep(K_MSEC(kMicPowerUpDelayMs));
    return 0;
}

int init_i2s(void)
{
    i2s_dev = DEVICE_DT_GET(I2S_RX_NODE);
    if (!device_is_ready(i2s_dev)) {
        printk("I2S device not ready\n");
        return -ENODEV;
    }

    memset(&i2s_cfg, 0, sizeof(i2s_cfg));
    i2s_cfg.word_size = kAudioWordSizeBits;
    i2s_cfg.channels = kI2sChannels;
    i2s_cfg.format = I2S_FMT_DATA_FORMAT_I2S;
    i2s_cfg.options = I2S_OPT_BIT_CLK_MASTER | I2S_OPT_FRAME_CLK_MASTER;
    i2s_cfg.frame_clk_freq = kAudioSampleRate;
    i2s_cfg.mem_slab = &audio_mem_slab;
    i2s_cfg.block_size = kBlockSize;
    i2s_cfg.timeout = kReadTimeoutMs;

    const int ret = i2s_configure(i2s_dev, I2S_DIR_RX, &i2s_cfg);
    if (ret < 0) {
        printk("Failed to configure I2S RX: %d\n", ret);
        return ret;
    }

    return 0;
}

int stop_i2s_capture(void)
{
    const int ret = i2s_trigger(i2s_dev, I2S_DIR_RX, I2S_TRIGGER_DROP);
    if (ret < 0 && ret != -EIO) {
        printk("Failed to drop I2S RX stream: %d\n", ret);
        return ret;
    }

    return 0;
}

size_t push_pcm_frames_to_ring(const int16_t *pcm, size_t frame_count)
{
    if (pcm == nullptr || frame_count == 0) {
        return 0;
    }

    size_t overwritten_samples = 0;
    bool window_ready = false;

    k_mutex_lock(&audio_ring_mutex, K_FOREVER);

    for (size_t i = 0; i < frame_count; ++i) {
        if (audio_ring_state.count == kRingSamples) {
            audio_ring_state.read_idx =
                (audio_ring_state.read_idx + 1) % kRingSamples;
            audio_ring_state.count--;
            overwritten_samples++;
        }

        const int16_t left = pcm[(i * kI2sChannels) + 0];
        const int16_t right = pcm[(i * kI2sChannels) + 1];
        audio_ring[audio_ring_state.write_idx] = select_mono_sample(left, right);
        audio_ring_state.write_idx =
            (audio_ring_state.write_idx + 1) % kRingSamples;
        audio_ring_state.count++;
    }

    if (overwritten_samples > 0) {
        audio_ring_state.overrun_events++;
        audio_ring_state.dropped_samples +=
            static_cast<uint32_t>(overwritten_samples);
    }

    window_ready = audio_ring_state.count >= kWindowSamples;
    k_mutex_unlock(&audio_ring_mutex);

    if (window_ready) {
        k_sem_give(&audio_window_ready_sem);
    }

    return overwritten_samples;
}

bool pop_inference_window(int16_t *dest,
                          size_t *buffered_samples_after_pop,
                          uint32_t *overrun_events,
                          uint32_t *dropped_samples)
{
    if (dest == nullptr) {
        return false;
    }

    bool has_window = false;

    k_mutex_lock(&audio_ring_mutex, K_FOREVER);
    if (audio_ring_state.count >= kWindowSamples) {
        const size_t first_chunk = MIN(kRingSamples - audio_ring_state.read_idx,
                                       kWindowSamples);
        memcpy(dest,
               &audio_ring[audio_ring_state.read_idx],
               first_chunk * sizeof(int16_t));
        if (first_chunk < kWindowSamples) {
            memcpy(dest + first_chunk,
                   audio_ring,
                   (kWindowSamples - first_chunk) * sizeof(int16_t));
        }

        audio_ring_state.read_idx =
            (audio_ring_state.read_idx + kWindowSamples) % kRingSamples;
        audio_ring_state.count -= kWindowSamples;
        has_window = true;
    }

    if (buffered_samples_after_pop != nullptr) {
        *buffered_samples_after_pop = audio_ring_state.count;
    }
    if (overrun_events != nullptr) {
        *overrun_events = audio_ring_state.overrun_events;
    }
    if (dropped_samples != nullptr) {
        *dropped_samples = audio_ring_state.dropped_samples;
    }
    k_mutex_unlock(&audio_ring_mutex);

    return has_window;
}

void reset_audio_ring(void)
{
    k_mutex_lock(&audio_ring_mutex, K_FOREVER);
    audio_ring_state.write_idx = 0;
    audio_ring_state.read_idx = 0;
    audio_ring_state.count = 0;
    k_mutex_unlock(&audio_ring_mutex);
}

void capture_thread(void *, void *, void *)
{
    while (1) {
        reset_audio_ring();

        int ret = i2s_trigger(i2s_dev, I2S_DIR_RX, I2S_TRIGGER_START);
        if (ret < 0) {
            printk("Failed to start I2S RX: %d\n", ret);
            k_sleep(K_MSEC(250));
            continue;
        }

        while (1) {
            void *buffer = nullptr;
            size_t block_size = 0;

            ret = i2s_read(i2s_dev, &buffer, &block_size);
            if (ret < 0) {
                printk("I2S read failed: %d\n", ret);
                stop_i2s_capture();
                k_sleep(K_MSEC(250));
                break;
            }

            if ((buffer == nullptr) ||
                (block_size < (kI2sChannels * sizeof(int16_t)))) {
                if (buffer != nullptr) {
                    k_mem_slab_free(&audio_mem_slab, buffer);
                }
                continue;
            }

            const int16_t *pcm = static_cast<const int16_t *>(buffer);
            const size_t sample_count = block_size / sizeof(int16_t);
            const size_t frame_count = sample_count / kI2sChannels;
            (void)push_pcm_frames_to_ring(pcm, frame_count);
            k_mem_slab_free(&audio_mem_slab, buffer);
        }
    }
}

}  // namespace

int main(void)
{
    k_sleep(K_SECONDS(10));
    printk("============================\n");
    printk("Audio Event detector start\n");
    printk("Target: pencilv nRF5340 over I2S\n");
    printk("============================\n");
    print_runtime_memory_summary_once();

    int ret = enable_microphone_power();
    if (ret < 0) {
        return ret;
    }

    ret = init_i2s();
    if (ret < 0) {
        return ret;
    }

    printk("labels:");
    for (int label_index = 0; label_index < audio_event_model_config::kLabelCount; ++label_index) {
        printk(" [%s]", audio_event_model_config::kLabels[label_index]);
    }
    printk("\n");
    if (audio_event_model_runner::model_is_placeholder()) {
        printk("Placeholder generated model detected. Export a real generated C++ model.\n");
    }
    printk("Listening continuously from I2S microphone...\n");
    printk("Ring buffer: %u samples (%u ms), inference window: %u samples (%u ms)\n\n",
           static_cast<uint32_t>(kRingSamples),
           kRingBufferMs,
           static_cast<uint32_t>(kWindowSamples),
           audio_event_model_config::kCaptureWindowMs);

    k_thread_create(&capture_thread_data,
                    capture_thread_stack,
                    K_THREAD_STACK_SIZEOF(capture_thread_stack),
                    capture_thread,
                    nullptr,
                    nullptr,
                    nullptr,
                    kCaptureThreadPriority,
                    0,
                    K_NO_WAIT);
    k_thread_priority_set(k_current_get(), kClassifierThreadPriority);

    uint32_t last_reported_overrun_events = 0;
    uint32_t last_reported_dropped_samples = 0;

    while (1) {
        k_sem_take(&audio_window_ready_sem, K_FOREVER);

        while (1) {
            size_t buffered_samples_after_pop = 0;
            uint32_t overrun_events = 0;
            uint32_t dropped_samples = 0;

            if (!pop_inference_window(infer_buf,
                                      &buffered_samples_after_pop,
                                      &overrun_events,
                                      &dropped_samples)) {
                break;
            }

            const uint32_t classify_start_ms = k_uptime_get_32();
            const audio_event_model_runner::Result result =
                audio_event_model_runner::run_classifier(
                    infer_buf, audio_event_model_runner::window_samples());
            const uint32_t classify_end_ms = k_uptime_get_32();

            printk("Timing: queued=%u ms feature=%u ms invoke=%u ms classify=%u ms\n",
                   static_cast<uint32_t>(
                       (buffered_samples_after_pop * 1000U) / kAudioSampleRate),
                   audio_event_model_runner::last_feature_ms(),
                   audio_event_model_runner::last_invoke_ms(),
                   classify_end_ms - classify_start_ms);

            if ((overrun_events != last_reported_overrun_events) ||
                (dropped_samples != last_reported_dropped_samples)) {
                printk("Audio ring overrun: events=%u dropped_samples=%u\n",
                       overrun_events,
                       dropped_samples);
                last_reported_overrun_events = overrun_events;
                last_reported_dropped_samples = dropped_samples;
            }

            if (!result.inference_ok) {
                printk("Generated classifier unavailable: %s\n",
                       audio_event_model_runner::last_error());
                continue;
            }

            const bool label_in_range =
                result.predicted_label >= 0 &&
                result.predicted_label < audio_event_model_config::kLabelCount;
            const char *label_name = label_in_range ?
                audio_event_model_config::kLabels[result.predicted_label] :
                "unknown";
            printk("detected label=%d [%s], prob=%.3f, outputs=%d\n",
                   result.predicted_label,
                   label_name,
                   (double)result.predicted_score,
                   result.output_count);
            printk("class probs:");
            for (int label_index = 0; label_index < audio_event_model_config::kLabelCount; ++label_index) {
                const float probability =
                    (label_index < result.output_count) ? result.scores[label_index] : 0.0f;
                printk(" [%s]=%.3f",
                       audio_event_model_config::kLabels[label_index],
                       (double)probability);
            }
            printk("\n");
        }
    }
}
