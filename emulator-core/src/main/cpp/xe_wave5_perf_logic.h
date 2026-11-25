#ifndef XENDROID_XE_WAVE5_PERF_LOGIC_H_
#define XENDROID_XE_WAVE5_PERF_LOGIC_H_

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace xendroid {

// Packs selected 128-bit float constant registers indexed by the 4x64-bit
// shader constant bitmap into contiguous destination memory using run-length
// spans, preserving ascending register order and avoiding one memcpy per bit.
inline void PackFloatConstantsByRuns(uint8_t* dst, const uint32_t* regs_base,
                                     const uint64_t map[4]) {
  for (uint32_t i = 0; i < 4; ++i) {
    uint64_t m = map[i];
    const uint32_t bank_offset = i << 8;
    while (m != 0) {
      const uint32_t start = static_cast<uint32_t>(std::countr_zero(m));
      const uint64_t shifted = m >> start;
      const uint32_t run = static_cast<uint32_t>(std::countr_one(shifted));
      const size_t bytes = size_t(run) * (sizeof(float) * 4);
      std::memcpy(dst, regs_base + bank_offset + (start << 2), bytes);
      dst += bytes;
      if (start + run >= 64) {
        m = 0;
      } else {
        const uint64_t span_mask =
            (run == 64 ? ~uint64_t(0) : ((uint64_t(1) << run) - 1)) << start;
        m &= ~span_mask;
      }
    }
  }
}

struct PixelSpecCacheState {
  const void* shader = nullptr;
  uint32_t flags_key = UINT32_MAX;
  uint32_t fetch_up_to_date = 0;
};

inline bool EvaluatePixelSpecCacheHit(const PixelSpecCacheState& state,
                                      const void* shader, uint32_t flags_key,
                                      uint32_t used_texture_mask) {
  if (!shader || state.shader != shader || state.flags_key != flags_key) {
    return false;
  }
  return (used_texture_mask & ~state.fetch_up_to_date) == 0;
}

inline void InvalidatePixelSpecCacheFetches(PixelSpecCacheState& state,
                                            uint32_t slot_clear_mask) {
  state.fetch_up_to_date &= slot_clear_mask;
}

struct AudioAdaptiveState {
  size_t extra = 0;
  size_t submitted_blocks = 0;
  uint64_t since_ms = 0;
  uint64_t grown_ms = 0;
  bool filled = false;
};

struct AudioAdaptiveStepResult {
  size_t target_blocks = 4;
  bool grew = false;
  bool shrunk = false;
};

// Advances adaptive audio queue depth across 5.33 ms worker pumps.
// Marks filled once either the queue reaches min(topup_base, 2) OR at least 4
// blocks have been submitted (prevents heavy-boot titles like RDR from never
// latching filled when initial depth stays at 1..3), grows +2 every 500 ms on
// stalls/near-empty queue up to `deepest`, and shrinks -1 after 20 s calm.
inline AudioAdaptiveStepResult StepAudioAdaptiveQueue(
    AudioAdaptiveState& state, bool enabled, bool has_valid_deadline,
    bool submitted, size_t depth, size_t topup_base, size_t deepest,
    uint64_t stall_us, uint64_t pump_interval_us, uint64_t now_ms) {
  AudioAdaptiveStepResult out;
  if (!has_valid_deadline) {
    state = AudioAdaptiveState{};
    out.target_blocks = std::min(topup_base, deepest);
    return out;
  }
  if (!enabled) {
    state = AudioAdaptiveState{};
    out.target_blocks = std::min(topup_base, deepest);
    return out;
  }
  if (submitted) {
    ++state.submitted_blocks;
  }
  if (depth >= std::min<size_t>(topup_base, 2) || state.submitted_blocks >= 4) {
    state.filled = true;
  }
  const size_t max_extra = deepest > topup_base ? (deepest - topup_base) : 0;
  if (state.extra > max_extra) {
    state.extra = max_extra;
  }
  const size_t current_target =
      std::min(deepest, topup_base + state.extra);
  const bool stressed =
      submitted && state.filled &&
      (depth <= 1 || stall_us * 2 >= current_target * pump_interval_us);

  if (stressed && current_target < deepest &&
      (state.grown_ms == 0 || now_ms - state.grown_ms >= 500)) {
    state.extra = std::min(state.extra + 2, max_extra);
    state.grown_ms = now_ms;
    state.since_ms = now_ms;
    out.grew = true;
  } else if (stressed) {
    state.since_ms = now_ms;
  } else if (state.extra > 0 && now_ms - state.since_ms >= 20000) {
    --state.extra;
    state.since_ms = now_ms;
    out.shrunk = true;
  }

  out.target_blocks =
      std::max(topup_base, std::min(deepest, topup_base + state.extra));
  return out;
}

inline uint32_t ComputeAAudioHighDrainDepth(uint32_t base_high_drain_depth,
                                            uint32_t target_queue_depth) {
  const uint32_t target_headroom =
      target_queue_depth > 0 ? (target_queue_depth + 2) : 0;
  return std::max(base_high_drain_depth, target_headroom);
}

inline bool ShouldRunEmergencySecondAudioTopup(size_t depth_after_first,
                                              size_t target_blocks,
                                              uint64_t total_exec_us,
                                              uint64_t pump_interval_us) {
  return depth_after_first <= 1 && depth_after_first < target_blocks &&
         total_exec_us < pump_interval_us * 2;
}

}  // namespace xendroid

#endif  // XENDROID_XE_WAVE5_PERF_LOGIC_H_
