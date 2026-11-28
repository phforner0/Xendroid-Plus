#pragma once
// Pure lock-free audio sanity probe (RMS, peak, sustained silence, clipping, NaNs).
// Callback thread only calls Observe() with zero allocations and zero locks.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace xendroid {

struct AudioSanitySnapshot {
  float rms = 0.0f;
  float peak = 0.0f;
  uint64_t silence_ms = 0;
  uint64_t total_frames = 0;
  uint64_t clipped_samples = 0;
  uint64_t nan_samples = 0;
};

class AudioSanityTracker {
 public:
  static constexpr float kSilenceThreshold = 1e-4f;

  void Observe(const float* stereo_samples,
               uint32_t num_frames,
               uint32_t sample_rate_hz) {
    if (!stereo_samples || num_frames == 0 || sample_rate_hz == 0) {
      return;
    }
    sample_rate_hz_ = sample_rate_hz;
    const size_t n = size_t(num_frames) * 2;
    float block_peak = 0.0f;
    for (size_t i = 0; i < n; ++i) {
      const float s = stereo_samples[i];
      if (!std::isfinite(s)) {
        ++nan_samples_;
        continue;
      }
      const float a = std::fabs(s);
      if (a > block_peak) {
        block_peak = a;
      }
      if (a >= 0.9999f) {
        ++clipped_samples_;
      }
      sum_sq_ += double(s) * double(s);
    }
    if (block_peak > peak_) {
      peak_ = block_peak;
    }
    total_frames_ += num_frames;
    if (block_peak < kSilenceThreshold) {
      consecutive_silent_frames_ += num_frames;
    } else {
      consecutive_silent_frames_ = 0;
    }
  }

  AudioSanitySnapshot SnapshotAndReset() {
    AudioSanitySnapshot out;
    const double denom = double(total_frames_) * 2.0;
    out.rms = denom > 0.0 ? float(std::sqrt(sum_sq_ / denom)) : 0.0f;
    out.peak = peak_;
    out.silence_ms = sample_rate_hz_
                         ? (consecutive_silent_frames_ * 1000ULL) / sample_rate_hz_
                         : 0;
    out.total_frames = total_frames_;
    out.clipped_samples = clipped_samples_;
    out.nan_samples = nan_samples_;

    sum_sq_ = 0.0;
    peak_ = 0.0f;
    total_frames_ = 0;
    clipped_samples_ = 0;
    nan_samples_ = 0;
    return out;
  }

 private:
  double sum_sq_ = 0.0;
  float peak_ = 0.0f;
  uint64_t total_frames_ = 0;
  uint64_t consecutive_silent_frames_ = 0;
  uint64_t clipped_samples_ = 0;
  uint64_t nan_samples_ = 0;
  uint32_t sample_rate_hz_ = 48000;
};

}  // namespace xendroid
