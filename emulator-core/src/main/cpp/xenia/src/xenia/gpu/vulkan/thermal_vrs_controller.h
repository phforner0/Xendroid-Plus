/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_VULKAN_THERMAL_VRS_CONTROLLER_H_
#define XENIA_GPU_VULKAN_THERMAL_VRS_CONTROLLER_H_

#include <cstdint>

#include "xenia/base/android_thermal.h"

namespace xe {
namespace gpu {
namespace vulkan {

inline uint32_t ClampShadingRateToMaxFragmentSize(uint32_t rate,
                                                  uint32_t max_width,
                                                  uint32_t max_height) {
  uint32_t width = (rate & 1u) ? 2u : 1u;
  uint32_t height = (rate & 2u) ? 2u : 1u;
  if (width > max_width) {
    width = 1u;
  }
  if (height > max_height) {
    height = 1u;
  }
  return ((width == 2u) ? 1u : 0u) | ((height == 2u) ? 2u : 0u);
}

inline uint32_t SelectDrawShadingRate(int32_t manual_rate,
                                      bool auto_vrs_enabled,
                                      uint32_t thermal_rate, bool is_1x_rt,
                                      bool memexport_written,
                                      uint32_t max_width,
                                      uint32_t max_height) {
  if (is_1x_rt || memexport_written) {
    return 0u;
  }
  uint32_t requested = 0u;
  const uint32_t masked_manual = static_cast<uint32_t>(manual_rate) & 3u;
  if (masked_manual != 0u) {
    requested = masked_manual;
  } else if (auto_vrs_enabled) {
    requested = thermal_rate & 3u;
  }
  return ClampShadingRateToMaxFragmentSize(requested, max_width, max_height);
}

class ThermalVrsController {
 public:
  static constexpr uint64_t kEscalateDwellMs = 1000;
  static constexpr uint64_t kCooldownDwellMs = 4000;
  static constexpr float kHeadroomEscalate2x1 = 0.85f;
  static constexpr float kHeadroomEscalate2x2 = 0.95f;
  static constexpr float kHeadroomCooldown2x1 = 0.82f;
  static constexpr float kHeadroomCooldown1x1 = 0.70f;

  // Android 16 SystemHealthManager GPU headroom (inverted [0..100] scale:
  // lower means less compute/power headroom remaining). Coupled to thermal
  // baseline so VRS never false-triggers on a cool device under heavy GPU load.
  static constexpr float kGpuHeadroomEscalate2x1 = 15.0f;
  static constexpr float kGpuHeadroomCriticalSustained = 8.0f;
  static constexpr uint32_t kGpuHeadroomSustainedSamples = 2u;
  static constexpr float kGpuHeadroomEscalate2x2 = 5.0f;
  static constexpr float kGpuHeadroomCooldown2x1 = 18.0f;
  static constexpr float kGpuHeadroomCooldown1x1 = 25.0f;
  static constexpr float kThermalBaselineWarmHeadroom = 0.70f;
  static constexpr float kThermalBaselineModerateHeadroom = 0.85f;

  void Reset() {
    current_rate_ = 0;
    pending_target_rate_ = 0;
    pending_since_ms_ = 0;
    has_pending_transition_ = false;
    low_gpu_headroom_streak_ = 0;
  }

  void Update(uint64_t now_ms, const xe::ThermalSample& sample,
              bool enable_compute_headroom =
                  xe::IsAndroidComputeHeadroomEnabled()) {
    const bool use_gpu_headroom =
        enable_compute_headroom && sample.has_gpu_headroom;
    if (!sample.has_status && !sample.has_headroom && !use_gpu_headroom) {
      Reset();
      return;
    }

    if (use_gpu_headroom &&
        sample.gpu_headroom <= kGpuHeadroomCriticalSustained) {
      if (low_gpu_headroom_streak_ < kGpuHeadroomSustainedSamples) {
        ++low_gpu_headroom_streak_;
      }
    } else {
      low_gpu_headroom_streak_ = 0;
    }

    if (sample.has_status && sample.status >= xe::kThermalStatusCritical) {
      current_rate_ = 3;
      pending_target_rate_ = 3;
      pending_since_ms_ = 0;
      has_pending_transition_ = false;
      return;
    }

    const bool thermal_baseline_warm =
        (sample.has_status && sample.status >= xe::kThermalStatusLight) ||
        (sample.has_headroom &&
         sample.headroom >= kThermalBaselineWarmHeadroom);
    const bool thermal_baseline_moderate =
        (sample.has_status && sample.status >= xe::kThermalStatusModerate) ||
        (sample.has_headroom &&
         sample.headroom >= kThermalBaselineModerateHeadroom);

    const bool gpu_headroom_want_2x2 =
        use_gpu_headroom && sample.gpu_headroom <= kGpuHeadroomEscalate2x2 &&
        thermal_baseline_moderate;
    const bool gpu_headroom_want_2x1 =
        gpu_headroom_want_2x2 ||
        (use_gpu_headroom &&
         ((sample.gpu_headroom <= kGpuHeadroomEscalate2x1 &&
           thermal_baseline_warm) ||
          (sample.gpu_headroom <= kGpuHeadroomCriticalSustained &&
           low_gpu_headroom_streak_ >= kGpuHeadroomSustainedSamples)));
    const bool gpu_headroom_cool_to_2x1 =
        !use_gpu_headroom || sample.gpu_headroom >= kGpuHeadroomCooldown2x1;
    const bool gpu_headroom_cool_to_1x1 =
        !use_gpu_headroom || sample.gpu_headroom >= kGpuHeadroomCooldown1x1;

    const bool want_2x2 =
        (sample.has_status && sample.status >= xe::kThermalStatusSevere) ||
        (sample.has_headroom && sample.headroom >= kHeadroomEscalate2x2) ||
        gpu_headroom_want_2x2;
    const bool want_2x1 =
        want_2x2 ||
        (sample.has_status && sample.status >= xe::kThermalStatusModerate) ||
        (sample.has_headroom && sample.headroom >= kHeadroomEscalate2x1) ||
        gpu_headroom_want_2x1;
    const bool cool_to_2x1 =
        (!sample.has_status || sample.status <= xe::kThermalStatusModerate) &&
        (!sample.has_headroom || sample.headroom <= kHeadroomCooldown2x1) &&
        gpu_headroom_cool_to_2x1;
    const bool cool_to_1x1 =
        (!sample.has_status || sample.status <= xe::kThermalStatusLight) &&
        (!sample.has_headroom || sample.headroom <= kHeadroomCooldown1x1) &&
        gpu_headroom_cool_to_1x1;

    uint32_t next_rate = current_rate_;
    uint64_t required_dwell = 0;

    if (current_rate_ == 0) {
      if (want_2x1) {
        next_rate = 1;
        required_dwell = kEscalateDwellMs;
      }
    } else if (current_rate_ == 1) {
      if (want_2x2) {
        next_rate = 3;
        required_dwell = kEscalateDwellMs;
      } else if (cool_to_1x1) {
        next_rate = 0;
        required_dwell = kCooldownDwellMs;
      }
    } else {
      if (cool_to_2x1) {
        next_rate = 1;
        required_dwell = kCooldownDwellMs;
      }
    }

    if (next_rate == current_rate_) {
      has_pending_transition_ = false;
      pending_target_rate_ = current_rate_;
      return;
    }

    if (!has_pending_transition_ || pending_target_rate_ != next_rate) {
      has_pending_transition_ = true;
      pending_target_rate_ = next_rate;
      pending_since_ms_ = now_ms;
      return;
    }

    if (now_ms < pending_since_ms_) {
      pending_since_ms_ = now_ms;
      return;
    }

    if ((now_ms - pending_since_ms_) >= required_dwell) {
      current_rate_ = next_rate;
      has_pending_transition_ = false;
      pending_target_rate_ = current_rate_;
    }
  }

  uint32_t current_rate() const { return current_rate_; }

 private:
  uint32_t current_rate_ = 0;
  uint32_t pending_target_rate_ = 0;
  uint64_t pending_since_ms_ = 0;
  bool has_pending_transition_ = false;
  uint32_t low_gpu_headroom_streak_ = 0;
};

}  // namespace vulkan
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_VULKAN_THERMAL_VRS_CONTROLLER_H_
