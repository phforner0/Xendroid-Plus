/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2020 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */
#include "xe_aaudio_audio_driver.h"
#include "audio_runtime.h"
#include "xe_wave5_perf_logic.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <thread>

#include "xenia/apu/apu_flags.h"
#include "xenia/apu/conversion.h"
#include "xenia/base/assert.h"
#include "xenia/base/cvar.h"
#include "xenia/base/logging.h"
#include "xenia/base/profiling.h"

DEFINE_uint32(
    apu_aaudio_buffer_bursts, 4,
    "Depth of the Android audio buffer, in device bursts. Higher rides out "
    "emulator slowdowns without dropping audio, at the cost of latency.",
    "APU");
DEFINE_bool(apu_aaudio_adaptive_buffer, false,
            "Grow the Android audio buffer one burst after underruns (up to three "
            "times apu_aaudio_buffer_bursts) and shrink it back one burst after "
            "30 s without any: fewer dropouts on a slow scene, the configured "
            "latency otherwise.",
            "APU");
DEFINE_bool(apu_aaudio_dynamic_rate, true,
            "Slow playback toward 0.9x as the audio queue drains, so a guest "
            "that cannot keep real time bends pitch instead of popping.",
            "APU");
DEFINE_bool(apu_aaudio_log_stats, false,
            "Log Android audio callback statistics (gaps, queue depth, "
            "underruns) once a second.",
            "APU");
DEFINE_bool(apu_audio_sanity_log, false,
            "Log periodic AudioSanity lines (RMS, peak, sustained silence, "
            "clipping, concealed blocks, xruns) once a second.",
            "APU");

namespace xe {
namespace apu {
namespace aaudio {

static constexpr uint32_t kStatsIntervalMs = 1000;

// Sink depth, in blocks, below which playback slows to buy the producer time.
static constexpr uint32_t kRateControlDepth = 3;
// Queue depth above which post-pause / catch-up backlog gently drains at 1.005x.
static constexpr uint32_t kHighQueueDrainDepth = 6;

// Loudness Guard constants (ported from XDroid360 bf84217bb, BSD-3-Clause):
// ~6 ms smooth ramp-in at 48 kHz after pause/resume or >250 ms callback stall,
// plus transparent RMS ceiling (0.45f) that engages only on transient blasts.
static constexpr uint32_t kPostStallRampFrames = 288;
static constexpr uint64_t kCallbackStallNs = 250'000'000ULL;
static constexpr float kLoudnessGuardMaxRms = 0.45f;

AAudioAudioDriver::AAudioAudioDriver(Memory* memory,
                                     xe::threading::Semaphore* semaphore,
                                     uint32_t frequency, uint32_t channels,
                                     bool need_format_conversion)
    : semaphore_(semaphore),
      frame_frequency_(frequency),
      frame_channels_(channels),
      need_format_conversion_(need_format_conversion),
      channel_samples_(channels == 6 ? 256 : 768),
      submit_samples_(channels * (channels == 6 ? 256 : 768)),
      host_block_samples_(host_frame_channels_ *
                          (channels == 6 ? 256 : 768)),
      renderer_(channels == 6 ? 256 : 768, channels),
      post_stall_ramp_pos_(kPostStallRampFrames) {
  assert_true(channels == 6 || channels == 2);
}

AAudioAudioDriver::~AAudioAudioDriver() = default;

bool AAudioAudioDriver::Initialize() {
  // The game's 5.1 or the media player's stereo: blocks of any other count
  // would be read past their end (the constructor's assert is gone in release).
  if (frame_channels_ != 6 && frame_channels_ != 2) {
    XELOGE("AAudio: {} channels, not 2 or 6", frame_channels_);
    return false;
  }

  ring_storage_.assign(size_t(kRingCapacity) * submit_samples_, 0.0f);
  ring_write_.store(0, std::memory_order_relaxed);
  ring_read_.store(0, std::memory_order_relaxed);

  {
    std::unique_lock<std::mutex> stream_guard(stream_mutex_);
    if (!BuildStream()) {
      return false;
    }
  }

  // Worker that rebuilds the stream on output-device changes.
  recovery_thread_ = std::thread(&AAudioAudioDriver::RecoveryThreadMain, this);
  return true;
}

bool AAudioAudioDriver::BuildStream() {
  // Open on the current default device (no setDeviceId); fall back to SHARED if
  // the new device won't grant an exclusive MMAP stream.
  const aaudio_sharing_mode_t modes[] = {AAUDIO_SHARING_MODE_EXCLUSIVE,
                                         AAUDIO_SHARING_MODE_SHARED};
  for (aaudio_sharing_mode_t mode : modes) {
    aaudio_result_t result = AAudio_createStreamBuilder(&builder_);
    if (result != AAUDIO_OK) {
      XELOGE("AAudio_createStreamBuilder failed: {}", result);
      return false;
    }

    AAudioStreamBuilder_setFormat(builder_, AAUDIO_FORMAT_PCM_FLOAT);
    AAudioStreamBuilder_setSampleRate(builder_, frame_frequency_);
    AAudioStreamBuilder_setChannelCount(builder_, host_frame_channels_);
    AAudioStreamBuilder_setFramesPerDataCallback(builder_, channel_samples_);
    AAudioStreamBuilder_setDataCallback(builder_, AudioCallback, this);
    AAudioStreamBuilder_setErrorCallback(builder_, AudioErrorCallback, this);
    AAudioStreamBuilder_setPerformanceMode(builder_,
                                           AAUDIO_PERFORMANCE_MODE_LOW_LATENCY);
    AAudioStreamBuilder_setSharingMode(builder_, mode);
    AAudioStreamBuilder_setBufferCapacityInFrames(
        builder_, channel_samples_ * (cvars::apu_aaudio_adaptive_buffer
                                          ? tuner_.max_bursts()
                                          : cvars::apu_aaudio_buffer_bursts));

    result = AAudioStreamBuilder_openStream(builder_, &stream_);
    if (result != AAUDIO_OK) {
      XELOGE("AAudioStreamBuilder_openStream ({}) failed: {}",
             mode == AAUDIO_SHARING_MODE_EXCLUSIVE ? "exclusive" : "shared",
             result);
      AAudioStreamBuilder_delete(builder_);
      builder_ = nullptr;
      continue;
    }

    // Publish active_stream_ and request a smooth fade-in BEFORE starting the
    // stream so any immediate callback or error callback sees valid state.
    active_stream_.store(stream_, std::memory_order_release);
    fade_request_.store(true, std::memory_order_release);

    result = AAudioStream_requestStart(stream_);
    if (result != AAUDIO_OK) {
      XELOGE("AAudioStream_requestStart failed: {}", result);
      active_stream_.store(nullptr, std::memory_order_release);
      AAudioStream_close(stream_);
      stream_ = nullptr;
      AAudioStreamBuilder_delete(builder_);
      builder_ = nullptr;
      continue;
    }

    // Size against the larger of burst and granted callback: a device can
    // burst at 96 yet call back for 256, leaving under two callbacks of
    // headroom if only the burst is used.
    const int32_t burst = AAudioStream_getFramesPerBurst(stream_);
    const int32_t callback_frames =
        AAudioStream_getFramesPerDataCallback(stream_);
    const int32_t drain = std::max(
        burst, callback_frames > 0 ? callback_frames
                                   : static_cast<int32_t>(channel_samples_));
    drain_frames_ = drain;
    if (drain > 0) {
      // A rebuilt stream keeps the depth the adaptive buffer had reached.
      AAudioStream_setBufferSizeInFrames(
          stream_, drain * (cvars::apu_aaudio_adaptive_buffer
                                ? tuner_.bursts()
                                : cvars::apu_aaudio_buffer_bursts));
    }

    XELOGI(
        "AAudio: {} stream, rate {}, {} ch, burst {}, callback {} (asked {}), "
        "buffer {}/{} frames, perf mode {}",
        mode == AAUDIO_SHARING_MODE_EXCLUSIVE ? "exclusive" : "shared",
        AAudioStream_getSampleRate(stream_),
        AAudioStream_getChannelCount(stream_), burst,
        AAudioStream_getFramesPerDataCallback(stream_), channel_samples_,
        AAudioStream_getBufferSizeInFrames(stream_),
        AAudioStream_getBufferCapacityInFrames(stream_),
        AAudioStream_getPerformanceMode(stream_));

    stream_initialized_ = true;
    xruns_published_ = 0;
    ae::RunStats().backend.store(1, std::memory_order_relaxed);
    return true;
  }
  return false;
}

void AAudioAudioDriver::Pause() {
  std::unique_lock<std::mutex> stream_guard(stream_mutex_);
  if (stream_initialized_ && stream_) {
    AAudioStream_requestPause(stream_);
  }
}

void AAudioAudioDriver::Resume() {
  std::unique_lock<std::mutex> stream_guard(stream_mutex_);
  // Resuming at the rate the drained queue asked for would play slow; the
  // next callback starts the rate over and smoothly fades in over ~6 ms.
  rate_reset_.store(true, std::memory_order_release);
  fade_request_.store(true, std::memory_order_release);
  stat_rate_milli_.store(1000, std::memory_order_relaxed);
  if (stream_initialized_ && stream_) {
    AAudioStream_requestStart(stream_);
  }
}

size_t AAudioAudioDriver::GetQueuedFrameCount() {
  const uint32_t w = ring_write_.load(std::memory_order_acquire);
  const uint32_t r = ring_read_.load(std::memory_order_acquire);
  return static_cast<size_t>(w - r);
}

void AAudioAudioDriver::SetVolume(float volume) {
  // AAudio has no per-stream volume; applied to the samples in the callback.
  driver_volume_.store(std::clamp(volume, 0.0f, 1.0f),
                       std::memory_order_relaxed);
}

void AAudioAudioDriver::ApplyLoudnessGuardAndRamp(float* stereo_out,
                                                  int32_t num_frames,
                                                  uint64_t now_ns) {
  if (num_frames <= 0) {
    return;
  }
  const bool stall_detected =
      last_callback_ns_ != 0 && (now_ns - last_callback_ns_) > kCallbackStallNs;
  last_callback_ns_ = now_ns;

  if (fade_request_.exchange(false, std::memory_order_acquire) ||
      stall_detected) {
    post_stall_ramp_pos_ = 0;
  }

  // Compute block RMS across stereo samples.
  const size_t sample_count = static_cast<size_t>(num_frames) * 2;
  float sum_sq = 0.0f;
  for (size_t i = 0; i < sample_count; ++i) {
    const float s = stereo_out[i];
    sum_sq += s * s;
  }
  const float rms = std::sqrt(sum_sq / static_cast<float>(sample_count));

  // Instant attack when RMS exceeds ceiling; gentle release back to unity.
  float target_limiter = 1.0f;
  if (rms > kLoudnessGuardMaxRms && rms > 1e-6f) {
    target_limiter = kLoudnessGuardMaxRms / rms;
  }
  const float start_limiter = limiter_gain_;
  if (target_limiter < limiter_gain_) {
    limiter_gain_ = target_limiter;
  } else {
    limiter_gain_ += 0.05f * (target_limiter - limiter_gain_);
    if (limiter_gain_ > 0.9995f) {
      limiter_gain_ = 1.0f;
    }
  }

  const bool need_limiter = (start_limiter < 0.9999f || limiter_gain_ < 0.9999f);
  const bool need_ramp = (post_stall_ramp_pos_ < kPostStallRampFrames);
  if (!need_limiter && !need_ramp) {
    return;
  }

  const float inv_frames = 1.0f / static_cast<float>(num_frames);
  const float limiter_step = (limiter_gain_ - start_limiter) * inv_frames;
  float cur_lim = start_limiter;

  for (int32_t f = 0; f < num_frames; ++f) {
    float env = cur_lim;
    cur_lim += limiter_step;
    if (post_stall_ramp_pos_ < kPostStallRampFrames) {
      env *= static_cast<float>(post_stall_ramp_pos_) /
             static_cast<float>(kPostStallRampFrames);
      ++post_stall_ramp_pos_;
    }
    stereo_out[size_t(f) * 2] *= env;
    stereo_out[size_t(f) * 2 + 1] *= env;
  }
}

aaudio_data_callback_result_t AAudioAudioDriver::AudioCallback(
    AAudioStream* stream,
    void* userdata,
    void* audioData,
    int32_t numFrames) {
  SCOPE_profile_cpu_f("apu");
  const auto callback_begin = std::chrono::steady_clock::now();
  const uint64_t begin_ns = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          callback_begin.time_since_epoch())
          .count());

  auto driver = static_cast<AAudioAudioDriver*>(userdata);
  float* output_buffer = reinterpret_cast<float*>(audioData);

  if (numFrames != static_cast<int32_t>(driver->channel_samples_)) {
    driver->stat_unexpected_frames_.store(numFrames, std::memory_order_relaxed);
  }

  driver->stat_callbacks_.fetch_add(1, std::memory_order_relaxed);

  if (driver->rate_reset_.exchange(false, std::memory_order_acquire)) {
    driver->rate_ = 1.0f;
  }

  // Lock-free SPSC queue depth read (acquire pairs with SubmitFrame release).
  const uint32_t w = driver->ring_write_.load(std::memory_order_acquire);
  const uint32_t r = driver->ring_read_.load(std::memory_order_relaxed);
  const uint32_t depth_now = w - r;

  // Dynamic rate control:
  // - Below kRateControlDepth (3 blocks): bend pitch down toward 0.90x instead
  //   of gapping/clicking.
  // - Above high_drain_depth (at least kHighQueueDrainDepth and above the
  //   adaptive target): gently drain backlog at 1.005x (+8.6 cents).
  float rate_target = 1.0f;
  if (cvars::apu_aaudio_dynamic_rate) {
    const uint32_t high_drain_depth = xendroid::ComputeAAudioHighDrainDepth(
        kHighQueueDrainDepth,
        driver->target_queue_depth_.load(std::memory_order_relaxed));
    if (depth_now < kRateControlDepth) {
      rate_target =
          std::max(0.90f, 1.0f - 0.05f * float(kRateControlDepth - depth_now));
    } else if (depth_now > high_drain_depth) {
      rate_target = 1.005f;
    }
  }
  driver->rate_ +=
      std::clamp(rate_target - driver->rate_, -0.003f, 0.003f);
  driver->stat_rate_milli_.store(
      static_cast<uint32_t>(driver->rate_ * 1000.0f + 0.5f),
      std::memory_order_relaxed);

  // AAudio has no stream volume control, so the gain is applied in software.
  const float gain = driver->driver_volume_.load(std::memory_order_relaxed) *
                     (float(ae::EffectiveVolume()) / 100.0f);
  const AudioBlockRenderer::Result result = driver->renderer_.Render(
      output_buffer, numFrames, driver->rate_, gain,
      [driver]() { return driver->NextGuestBlock(); },
      [driver](const float* block) { driver->ReturnGuestBlock(block); });
  if (result.clipped) {
    driver->stat_clipped_.fetch_add(result.clipped, std::memory_order_relaxed);
  }

  driver->ApplyLoudnessGuardAndRamp(output_buffer, numFrames, begin_ns);

  if (cvars::apu_audio_sanity_log && numFrames > 0) {
    driver->audio_sanity_.Observe(output_buffer, uint32_t(numFrames),
                                  driver->frame_frequency_);
    driver->sanity_frames_accum_ += uint32_t(numFrames);
    if (driver->sanity_frames_accum_ >= driver->frame_frequency_) {
      driver->sanity_frames_accum_ = 0;
      const auto snap = driver->audio_sanity_.SnapshotAndReset();
      driver->sanity_rms_milli_.store(
          uint32_t(std::clamp(snap.rms * 1000.0f, 0.0f, 999999.0f)),
          std::memory_order_relaxed);
      driver->sanity_peak_milli_.store(
          uint32_t(std::clamp(snap.peak * 1000.0f, 0.0f, 999999.0f)),
          std::memory_order_relaxed);
      driver->sanity_silence_ms_.store(snap.silence_ms,
                                       std::memory_order_relaxed);
      driver->sanity_clipped_.store(snap.clipped_samples,
                                    std::memory_order_relaxed);
      driver->sanity_nans_.store(snap.nan_samples, std::memory_order_relaxed);
      driver->sanity_ready_.store(true, std::memory_order_release);
    }
  }

  // One tick per block consumed, so pacing holds when the callback size is
  // not a whole block. Tick once on underrun to keep the guest engine running.
  uint32_t releases = result.blocks;
  if (releases == 0 && result.gaps) {
    releases = 1;
  }
  if (releases > 0 &&
      !driver->semaphore_->Release(static_cast<int>(releases), nullptr)) {
    for (uint32_t i = 0; i < releases; ++i) {
      driver->semaphore_->Release(1, nullptr);
    }
  }

  const uint32_t callback_us = static_cast<uint32_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - callback_begin)
          .count());
  driver->stat_callback_us_sum_.fetch_add(callback_us,
                                          std::memory_order_relaxed);
  if (callback_us >
      driver->stat_callback_us_max_.load(std::memory_order_relaxed)) {
    driver->stat_callback_us_max_.store(callback_us,
                                        std::memory_order_relaxed);
  }
  return AAUDIO_CALLBACK_RESULT_CONTINUE;
}

const float* AAudioAudioDriver::NextGuestBlock() {
  const uint32_t r = ring_read_.load(std::memory_order_relaxed);
  const uint32_t w = ring_write_.load(std::memory_order_acquire);
  const uint32_t depth = w - r;

  stat_queue_depth_sum_.fetch_add(depth, std::memory_order_relaxed);
  if (depth > stat_queue_depth_max_.load(std::memory_order_relaxed)) {
    stat_queue_depth_max_.store(depth, std::memory_order_relaxed);
  }
  if (depth < stat_queue_depth_min_.load(std::memory_order_relaxed)) {
    stat_queue_depth_min_.store(depth, std::memory_order_relaxed);
  }

  if (played_once_) {
    ae::RunStats().blocks.fetch_add(1, std::memory_order_relaxed);
  }
  if (depth == 0) {
    stat_gaps_.fetch_add(1, std::memory_order_relaxed);
    if (played_once_) {
      ae::RunStats().concealed.fetch_add(1, std::memory_order_relaxed);
    }
    return nullptr;
  }
  if (!played_once_) {
    played_once_ = true;
    ae::RunStats().blocks.fetch_add(1, std::memory_order_relaxed);
  }
  return &ring_storage_[size_t(r & kRingMask) * submit_samples_];
}

void AAudioAudioDriver::ReturnGuestBlock(const float* /*block*/) {
  const uint32_t r = ring_read_.load(std::memory_order_relaxed);
  ring_read_.store(r + 1, std::memory_order_release);
}

void AAudioAudioDriver::LogAndResetStats() {
  const uint64_t callbacks = stat_callbacks_.exchange(0, std::memory_order_relaxed);
  if (!callbacks) {
    return;
  }
  const uint64_t gaps = stat_gaps_.exchange(0, std::memory_order_relaxed);
  const uint64_t depth_sum =
      stat_queue_depth_sum_.exchange(0, std::memory_order_relaxed);
  const uint32_t depth_max =
      stat_queue_depth_max_.exchange(0, std::memory_order_relaxed);
  const uint32_t depth_min =
      stat_queue_depth_min_.exchange(UINT32_MAX, std::memory_order_relaxed);
  const int32_t odd_frames =
      stat_unexpected_frames_.exchange(0, std::memory_order_relaxed);
  const uint64_t clipped = stat_clipped_.exchange(0, std::memory_order_relaxed);
  const uint64_t callback_us_sum =
      stat_callback_us_sum_.exchange(0, std::memory_order_relaxed);
  const uint32_t callback_us_max =
      stat_callback_us_max_.exchange(0, std::memory_order_relaxed);
  const uint64_t played = (callbacks - gaps) * host_block_samples_;

  int32_t xruns = -1;
  {
    std::unique_lock<std::mutex> stream_guard(stream_mutex_);
    if (stream_initialized_ && stream_) {
      xruns = AAudioStream_getXRunCount(stream_);
    }
  }

  XELOGI(
      "AAudio: {} cb, {} gaps ({:.1f}%), queue avg {:.2f} max {} min {}, rate "
      "{:.3f}, xruns {}, clipped {} ({:.3f}%), callback {:.0f} us avg {} max{}",
      callbacks, gaps, 100.0 * double(gaps) / double(callbacks),
      double(depth_sum) / double(callbacks), depth_max,
      depth_min == UINT32_MAX ? 0 : depth_min,
      stat_rate_milli_.load(std::memory_order_relaxed) / 1000.0, xruns,
      clipped,
      played ? 100.0 * double(clipped) / double(played) : 0.0,
      double(callback_us_sum) / double(callbacks), callback_us_max,
      odd_frames ? fmt::format(", UNEXPECTED framesPerCallback {}", odd_frames)
                 : "");
}

void AAudioAudioDriver::AudioErrorCallback(
    AAudioStream* stream,
    void* userdata,
    aaudio_result_t error) {
  auto driver = static_cast<AAudioAudioDriver*>(userdata);
  // Ignore late error callbacks from an already-retired stream during BT/USB
  // route changes so a dying stream cannot tear down its newly-opened successor
  // (fixes Bluetooth reconnect race, XDroid360 b4cc8eb9c).
  if (stream != driver->active_stream_.load(std::memory_order_acquire)) {
    XELOGI("AAudio: ignoring stale error {} from retired stream", error);
    return;
  }
  XELOGW("AAudio stream error: {} - requesting stream rebuild", error);
  {
    std::lock_guard<std::mutex> lk(driver->recovery_mutex_);
    driver->restart_requested_ = true;
  }
  driver->recovery_cv_.notify_one();
}

void AAudioAudioDriver::DrainWhileDisconnectedLocked(uint32_t max_blocks) {
  // Called ONLY under stream_mutex_ when stream_initialized_ == false (so no
  // AAudio callback thread exists), preserving single-consumer ring_read_ safety.
  uint32_t r = ring_read_.load(std::memory_order_relaxed);
  const uint32_t w = ring_write_.load(std::memory_order_acquire);
  const uint32_t avail = w - r;
  const uint32_t to_drain = std::min(avail, max_blocks);
  if (to_drain > 0) {
    ring_read_.store(r + to_drain, std::memory_order_release);
    semaphore_->Release(static_cast<int>(to_drain), nullptr);
  }
}

void AAudioAudioDriver::RecoveryThreadMain() {
  bool retry_pending = false;
  // Blocks produced per 50 ms retry tick (~9 blocks at 48 kHz / 256 samples).
  const uint32_t blocks_per_retry_tick =
      std::max<uint32_t>(1u, (frame_frequency_ / 20u) / channel_samples_);

  for (;;) {
    {
      std::unique_lock<std::mutex> lk(recovery_mutex_);
      auto wake = [this] { return restart_requested_ || recovery_quit_; };
      if (retry_pending) {
        recovery_cv_.wait_for(lk, std::chrono::milliseconds(50), wake);
      } else {
        recovery_cv_.wait_for(lk, std::chrono::milliseconds(kStatsIntervalMs),
                              wake);
      }
      if (recovery_quit_) {
        return;
      }
      if (!restart_requested_ && !retry_pending) {
        lk.unlock();
        const uint32_t fresh_xruns = PublishXRuns();
        if (cvars::apu_aaudio_adaptive_buffer) {
          AdaptBuffer(fresh_xruns);
        }
        if (cvars::apu_aaudio_log_stats) {
          LogAndResetStats();
        }
        if (cvars::apu_audio_sanity_log &&
            sanity_ready_.exchange(false, std::memory_order_acquire)) {
          XELOGI(
              "AudioSanity: rms={:.3f} peak={:.3f} silence_ms={} clipped={} "
              "concealed={} xruns={} nans={}",
              double(sanity_rms_milli_.load(std::memory_order_relaxed)) / 1000.0,
              double(sanity_peak_milli_.load(std::memory_order_relaxed)) / 1000.0,
              sanity_silence_ms_.load(std::memory_order_relaxed),
              sanity_clipped_.load(std::memory_order_relaxed),
              ae::RunStats().concealed.load(std::memory_order_relaxed),
              xruns_published_,
              sanity_nans_.load(std::memory_order_relaxed));
        }
        continue;
      }
      // Clear BEFORE RestartStream so a new error arriving on the freshly built
      // stream during or immediately after BuildStream is never swallowed!
      restart_requested_ = false;
    }
    PublishXRuns();
    retry_pending = !RestartStream();
    if (retry_pending) {
      std::unique_lock<std::mutex> stream_guard(stream_mutex_);
      if (!stream_initialized_) {
        DrainWhileDisconnectedLocked(blocks_per_retry_tick);
      }
    }
  }
}

uint32_t AAudioAudioDriver::PublishXRuns() {
  std::unique_lock<std::mutex> stream_guard(stream_mutex_);
  if (!stream_initialized_ || !stream_) {
    return 0;
  }
  const int32_t xruns = AAudioStream_getXRunCount(stream_);
  if (xruns > xruns_published_) {
    const uint32_t fresh = uint32_t(xruns - xruns_published_);
    ae::RunStats().device_xruns.fetch_add(fresh, std::memory_order_relaxed);
    xruns_published_ = xruns;
    return fresh;
  }
  return 0;
}

void AAudioAudioDriver::AdaptBuffer(uint32_t fresh_xruns) {
  std::unique_lock<std::mutex> stream_guard(stream_mutex_);
  const uint32_t before = tuner_.bursts();
  const uint32_t after = tuner_.Second(fresh_xruns);
  if (after == before || !stream_initialized_ || !stream_ || drain_frames_ <= 0) {
    return;
  }
  const int32_t granted =
      AAudioStream_setBufferSizeInFrames(stream_, drain_frames_ * int32_t(after));
  XELOGI("AAudio: buffer {} bursts ({} frames granted) after {}", after, granted,
         fresh_xruns ? "underruns" : "30 s without one");
}

bool AAudioAudioDriver::RestartStream() {
  std::unique_lock<std::mutex> stream_guard(stream_mutex_);
  XELOGW("AAudio: rebuilding stream on the current default output device");
  active_stream_.store(nullptr, std::memory_order_release);
  if (stream_) {
    AAudioStream_requestStop(stream_);
    AAudioStream_close(stream_);
    stream_ = nullptr;
  }
  if (builder_) {
    AAudioStreamBuilder_delete(builder_);
    builder_ = nullptr;
  }
  stream_initialized_ = false;

  if (shutting_down_.load(std::memory_order_acquire)) {
    return true;
  }

  if (BuildStream()) {
    XELOGI("AAudio: stream rebuilt; audio output restored");
    return true;
  }
  XELOGE("AAudio: stream rebuild failed; will retry");
  return false;
}

void AAudioAudioDriver::SubmitFrame(float* samples) {
  const uint32_t w = ring_write_.load(std::memory_order_relaxed);
  const uint32_t r = ring_read_.load(std::memory_order_acquire);
  if (w - r >= kRingCapacity) {
    // Defensive guard if apu_max_queued_frames ever exceeds kRingCapacity:
    // drop the incoming block rather than corrupting an in-flight callback read.
    semaphore_->Release(1, nullptr);
    return;
  }
  float* dst = &ring_storage_[size_t(w & kRingMask) * submit_samples_];
  std::memcpy(dst, samples, submit_samples_ * sizeof(float));
  ring_write_.store(w + 1, std::memory_order_release);
}

void AAudioAudioDriver::Shutdown() {
  shutting_down_.store(true, std::memory_order_release);
  {
    std::lock_guard<std::mutex> lk(recovery_mutex_);
    recovery_quit_ = true;
  }
  recovery_cv_.notify_one();
  if (recovery_thread_.joinable()) {
    recovery_thread_.join();
  }

  {
    std::unique_lock<std::mutex> stream_guard(stream_mutex_);
    active_stream_.store(nullptr, std::memory_order_release);
    if (stream_) {
      AAudioStream_requestStop(stream_);
      AAudioStream_close(stream_);
      stream_ = nullptr;
    }

    if (builder_) {
      AAudioStreamBuilder_delete(builder_);
      builder_ = nullptr;
    }

    stream_initialized_ = false;
  }

  ring_write_.store(0, std::memory_order_relaxed);
  ring_read_.store(0, std::memory_order_relaxed);
  ring_storage_.clear();
}

}  // namespace aaudio
}  // namespace apu
}  // namespace xe
