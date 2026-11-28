/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_KERNEL_DELAY0_BACKOFF_H_
#define XENIA_KERNEL_DELAY0_BACKOFF_H_

#include <algorithm>
#include <cstdint>

namespace xe {
namespace kernel {

class Delay0BackoffState {
 public:
  static constexpr uint32_t kFreeYieldSpins = 2;
  static constexpr uint32_t kMinSleepUs = 100;
  static constexpr uint32_t kMaxSleepUs = 500;
  static constexpr uint64_t kSpinGapNs = 50'000ull;
  static constexpr uint64_t kLateToleranceNs = 500'000ull;
  static constexpr uint32_t kLateThresholdCount = 2;
  static constexpr uint32_t kLateCooldownCalls = 32;

  static constexpr uint32_t PopCount8(uint8_t value) {
    uint32_t count = 0;
    for (uint8_t v = value; v != 0; v = static_cast<uint8_t>(v & (v - 1u))) {
      ++count;
    }
    return count;
  }

  uint32_t NextSleepUs(uint64_t now_ns, bool nothing_else_to_run) {
    if (!nothing_else_to_run) {
      streak_ = 0;
      last_idle_end_ns_ = 0;
      return 0;
    }

    if (last_idle_end_ns_ != 0 && now_ns > last_idle_end_ns_ &&
        (now_ns - last_idle_end_ns_) > kSpinGapNs) {
      streak_ = 0;
    }

    if (cooldown_remaining_ > 0) {
      --cooldown_remaining_;
      streak_ = 0;
      last_idle_end_ns_ = now_ns;
      return 0;
    }

    ++streak_;
    last_idle_end_ns_ = now_ns;
    if (streak_ <= kFreeYieldSpins) {
      return 0;
    }

    const uint32_t exp = std::min(streak_ - kFreeYieldSpins - 1u, 4u);
    const uint32_t sleep_us = kMinSleepUs << exp;
    return std::min(sleep_us, kMaxSleepUs);
  }

  void RecordSleepResult(uint64_t after_sleep_ns, uint32_t target_us,
                         uint64_t actual_ns) {
    last_idle_end_ns_ = after_sleep_ns;
    const uint64_t target_ns = static_cast<uint64_t>(target_us) * 1000ull;
    const bool late = actual_ns > target_ns + kLateToleranceNs;
    late_history_ =
        static_cast<uint8_t>((late_history_ << 1u) | (late ? 1u : 0u));
    if (late && PopCount8(late_history_) >= kLateThresholdCount) {
      cooldown_remaining_ = kLateCooldownCalls;
      late_history_ = 0;
      streak_ = 0;
    }
  }

  void Reset() {
    streak_ = 0;
    last_idle_end_ns_ = 0;
  }

  uint32_t streak() const { return streak_; }
  uint32_t cooldown_remaining() const { return cooldown_remaining_; }
  uint8_t late_history() const { return late_history_; }

 private:
  uint32_t streak_ = 0;
  uint32_t cooldown_remaining_ = 0;
  uint64_t last_idle_end_ns_ = 0;
  uint8_t late_history_ = 0;
};

}  // namespace kernel
}  // namespace xe

#endif  // XENIA_KERNEL_DELAY0_BACKOFF_H_
