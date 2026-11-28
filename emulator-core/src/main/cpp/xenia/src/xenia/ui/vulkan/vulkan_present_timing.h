/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_UI_VULKAN_VULKAN_PRESENT_TIMING_H_
#define XENIA_UI_VULKAN_VULKAN_PRESENT_TIMING_H_

#include <cstddef>
#include <cstdint>

namespace xe {
namespace ui {
namespace vulkan {

inline constexpr uint32_t kPresentTimingQueueSize = 16u;

// Time-domain values matching Vulkan spec VkTimeDomainKHR /
// VK_EXT_present_timing so host tests can exercise pure logic without Vulkan.
inline constexpr uint32_t kTimeDomainDevice = 0u;
inline constexpr uint32_t kTimeDomainClockMonotonic = 1u;
inline constexpr uint32_t kTimeDomainClockMonotonicRaw = 2u;
inline constexpr uint32_t kTimeDomainPresentStageLocal = 1000208000u;
inline constexpr uint32_t kTimeDomainSwapchainLocal = 1000208001u;

// Present stage flag bits matching VkPresentStageFlagBitsEXT (spec v3).
inline constexpr uint32_t kPresentStageQueueOpsEnd = 0x00000001u;
inline constexpr uint32_t kPresentStageRequestDequeued = 0x00000002u;
inline constexpr uint32_t kPresentStageFirstPixelOut = 0x00000004u;
inline constexpr uint32_t kPresentStageFirstPixelVisible = 0x00000008u;

enum class PresentTimingBackend : uint32_t {
  kNone = 0,
  kGoogleDisplayTiming = 1,
  kExtPresentTiming = 2,
};

enum class PresentInfoTimingChain : uint32_t {
  kNone = 0,
  kGoogleOnly = 1,
  kExtOnly = 2,
};

struct SelectedTimeDomain {
  bool valid = false;
  uint32_t time_domain = 0;
  uint64_t time_domain_id = 0;
  // True for either CLOCK_MONOTONIC or CLOCK_MONOTONIC_RAW.
  bool is_monotonic_clock = false;
  // True strictly for CLOCK_MONOTONIC (matches std::chrono::steady_clock on
  // Linux/Android; never mixes CLOCK_MONOTONIC_RAW with steady_clock).
  bool is_clock_monotonic = false;
  // True strictly for CLOCK_MONOTONIC_RAW.
  bool is_clock_monotonic_raw = false;
};

struct PresentStageSample {
  uint32_t stage = 0;
  uint64_t time_ns = 0;
};

struct UnifiedPastPresentationTiming {
  uint64_t present_id = 0;
  uint64_t target_time_ns = 0;
  uint64_t actual_present_time_ns = 0;
  uint64_t earliest_present_time_ns = 0;
  uint64_t refresh_duration_ns = 0;
  uint32_t chosen_stage = 0;
  uint32_t time_domain = 0;
  bool report_complete = false;
  bool valid = false;
};

enum class LateSwapAction : uint32_t {
  kBypassKeepPaced = 0,
  kKeepScheduledTarget = 1,
  kPresentImmediate = 2,
};

struct LateSwapDecision {
  LateSwapAction action = LateSwapAction::kBypassKeepPaced;
  uint64_t effective_target_ns = 0;
};

// Strict '0' | '1' parser for Android debug.xendroid.* boolean properties.
inline bool ParseStrictBoolProperty(const char* value, bool& out_value) {
  if (!value || (value[0] != '0' && value[0] != '1') || value[1] != '\0') {
    return false;
  }
  out_value = (value[0] == '1');
  return true;
}

// Prefers VK_EXT_present_timing only when cvar, device feature, surface
// capability and all 4 PFNs are ready; otherwise falls back cleanly to
// VK_GOOGLE_display_timing and then kNone.
inline PresentTimingBackend SelectPresentTimingBackend(
    bool ext_enabled_cvar, bool ext_device_feature, bool ext_surface_supported,
    bool ext_pfns_ready, bool google_supported) {
  if (ext_enabled_cvar && ext_device_feature && ext_surface_supported &&
      ext_pfns_ready) {
    return PresentTimingBackend::kExtPresentTiming;
  }
  if (google_supported) {
    return PresentTimingBackend::kGoogleDisplayTiming;
  }
  return PresentTimingBackend::kNone;
}

// Vulkan 1.4.350 spec v3 VUID dependency gate for VK_EXT_present_timing at
// logical device creation: requires VK_KHR_calibrated_timestamps and
// VK_KHR_present_id2 extensions plus presentTiming and a valid presentId
// feature (presentId2 || presentId).
inline bool CanEnablePresentTimingExtOnDevice(
    bool ext_present_timing_advertised, bool ext_calibrated_timestamps,
    bool ext_present_id2, bool feature_present_timing, bool feature_present_id2,
    bool feature_present_id) {
  return ext_present_timing_advertised && ext_calibrated_timestamps &&
         ext_present_id2 && feature_present_timing &&
         (feature_present_id2 || feature_present_id);
}

// Strict Vulkan 1.4.350 spec v3 device gate: requires VK_KHR_present_id2
// extension AND presentId2 feature (v1 VK_KHR_present_id without
// VK_SWAPCHAIN_CREATE_PRESENT_ID_2_BIT_KHR is rejected).
inline bool CanEnablePresentTimingExtOnDeviceV3(
    bool ext_present_timing_advertised, bool ext_calibrated_timestamps,
    bool ext_present_id2, bool feature_present_timing,
    bool feature_present_id2) {
  return CanEnablePresentTimingExtOnDevice(
      ext_present_timing_advertised, ext_calibrated_timestamps, ext_present_id2,
      feature_present_timing, feature_present_id2, false);
}

// True only when the selected swapchain time domain matches
// std::chrono::steady_clock (CLOCK_MONOTONIC on Linux/Android). Distinct from
// CLOCK_MONOTONIC_RAW so the two clock domains are never mixed.
inline bool TimeDomainMatchesSteadyClock(const SelectedTimeDomain& domain) {
  return domain.valid && domain.is_clock_monotonic &&
         domain.time_domain == kTimeDomainClockMonotonic;
}

// Pre-swapchain gate for ZeroFG egress: ZeroFG pacing requires absolute target
// scheduling (presentAtAbsoluteTime) plus presentId2 and stage queries; without
// them, fall back to VK_GOOGLE_display_timing before creating the swapchain.
inline bool CanUseExtPresentTimingForZeroFG(bool ext_enabled_cvar,
                                            bool ext_device_feature,
                                            bool ext_surface_supported,
                                            bool ext_pfns_ready,
                                            bool ext_present_at_absolute_time,
                                            bool ext_present_id2_ready,
                                            uint32_t stage_query_mask) {
  return ext_enabled_cvar && ext_device_feature && ext_surface_supported &&
         ext_pfns_ready && ext_present_at_absolute_time &&
         ext_present_id2_ready && stage_query_mask != 0u;
}

// Pre-swapchain gate for native VulkanPresenter: requires presentId2 and at
// least one supported stage query before creating a PRESENT_TIMING_BIT_EXT
// swapchain.
inline bool CanUseExtPresentTimingForPresenter(bool ext_enabled_cvar,
                                               bool ext_device_feature,
                                               bool ext_surface_supported,
                                               bool ext_pfns_ready,
                                               bool ext_present_id2_ready,
                                               uint32_t stage_query_mask) {
  return ext_enabled_cvar && ext_device_feature && ext_surface_supported &&
         ext_pfns_ready && ext_present_id2_ready && stage_query_mask != 0u;
}

// Enforces mutual exclusion on VkPresentInfoKHR::pNext: never chain both
// VkPresentTimingsInfoEXT and VkPresentTimesInfoGOOGLE simultaneously.
inline PresentInfoTimingChain SelectPresentInfoTimingChain(
    PresentTimingBackend active_backend, bool want_timing_for_this_present) {
  if (!want_timing_for_this_present) {
    return PresentInfoTimingChain::kNone;
  }
  switch (active_backend) {
    case PresentTimingBackend::kExtPresentTiming:
      return PresentInfoTimingChain::kExtOnly;
    case PresentTimingBackend::kGoogleDisplayTiming:
      return PresentInfoTimingChain::kGoogleOnly;
    case PresentTimingBackend::kNone:
    default:
      return PresentInfoTimingChain::kNone;
  }
}

inline int TimeDomainPreferenceRank(uint32_t time_domain) {
  switch (time_domain) {
    case kTimeDomainClockMonotonic:
      return 0;
    case kTimeDomainClockMonotonicRaw:
      return 1;
    case kTimeDomainPresentStageLocal:
      return 2;
    case kTimeDomainSwapchainLocal:
      return 3;
    default:
      return 4;
  }
}

inline SelectedTimeDomain SelectPreferredTimeDomain(
    const uint32_t* time_domains, const uint64_t* time_domain_ids,
    size_t count) {
  SelectedTimeDomain best;
  if (!time_domains || !time_domain_ids || count == 0) {
    return best;
  }
  int best_rank = 100;
  for (size_t i = 0; i < count; ++i) {
    const int rank = TimeDomainPreferenceRank(time_domains[i]);
    if (!best.valid || rank < best_rank) {
      best.valid = true;
      best.time_domain = time_domains[i];
      best.time_domain_id = time_domain_ids[i];
      best.is_clock_monotonic = (time_domains[i] == kTimeDomainClockMonotonic);
      best.is_clock_monotonic_raw =
          (time_domains[i] == kTimeDomainClockMonotonicRaw);
      best.is_monotonic_clock =
          best.is_clock_monotonic || best.is_clock_monotonic_raw;
      best_rank = rank;
    }
  }
  return best;
}

inline uint32_t SelectPreferredPresentStageQueryMask(
    uint32_t supported_stage_queries) {
  const uint32_t display_stages =
      supported_stage_queries &
      (kPresentStageFirstPixelVisible | kPresentStageFirstPixelOut);
  if (display_stages != 0u) {
    return display_stages;
  }
  if (supported_stage_queries & kPresentStageRequestDequeued) {
    return kPresentStageRequestDequeued;
  }
  if (supported_stage_queries & kPresentStageQueueOpsEnd) {
    return kPresentStageQueueOpsEnd;
  }
  return 0u;
}

inline uint32_t SelectTargetTimePresentStage(uint32_t supported_stage_queries) {
  if (supported_stage_queries & kPresentStageFirstPixelVisible) {
    return kPresentStageFirstPixelVisible;
  }
  if (supported_stage_queries & kPresentStageFirstPixelOut) {
    return kPresentStageFirstPixelOut;
  }
  if (supported_stage_queries & kPresentStageRequestDequeued) {
    return kPresentStageRequestDequeued;
  }
  if (supported_stage_queries & kPresentStageQueueOpsEnd) {
    return kPresentStageQueueOpsEnd;
  }
  return 0u;
}

inline UnifiedPastPresentationTiming ConvertPastPresentationTiming(
    uint64_t present_id, uint64_t target_time_ns,
    const PresentStageSample* stages, size_t stage_count, bool report_complete,
    uint64_t refresh_duration_ns) {
  UnifiedPastPresentationTiming out;
  out.present_id = present_id;
  out.target_time_ns = target_time_ns;
  out.refresh_duration_ns = refresh_duration_ns;
  out.report_complete = report_complete;
  if (!stages || stage_count == 0) {
    return out;
  }

  uint64_t visible_ns = 0;
  uint64_t first_out_ns = 0;
  uint64_t dequeued_ns = 0;
  uint64_t queue_end_ns = 0;
  for (size_t i = 0; i < stage_count; ++i) {
    const uint64_t t = stages[i].time_ns;
    if (t == 0) {
      continue;
    }
    if (stages[i].stage == kPresentStageFirstPixelVisible) {
      visible_ns = t;
    } else if (stages[i].stage == kPresentStageFirstPixelOut) {
      first_out_ns = t;
    } else if (stages[i].stage == kPresentStageRequestDequeued) {
      dequeued_ns = t;
    } else if (stages[i].stage == kPresentStageQueueOpsEnd) {
      queue_end_ns = t;
    }
  }

  if (visible_ns != 0) {
    out.actual_present_time_ns = visible_ns;
    out.chosen_stage = kPresentStageFirstPixelVisible;
  } else if (first_out_ns != 0) {
    out.actual_present_time_ns = first_out_ns;
    out.chosen_stage = kPresentStageFirstPixelOut;
  } else if (dequeued_ns != 0) {
    out.actual_present_time_ns = dequeued_ns;
    out.chosen_stage = kPresentStageRequestDequeued;
  } else if (queue_end_ns != 0) {
    out.actual_present_time_ns = queue_end_ns;
    out.chosen_stage = kPresentStageQueueOpsEnd;
  }

  if (out.actual_present_time_ns != 0) {
    uint64_t earliest = out.actual_present_time_ns;
    if (first_out_ns != 0 && first_out_ns <= earliest) {
      earliest = first_out_ns;
    } else if (dequeued_ns != 0 && dequeued_ns <= earliest) {
      earliest = dequeued_ns;
    } else if (queue_end_ns != 0 && queue_end_ns <= earliest) {
      earliest = queue_end_ns;
    }
    out.earliest_present_time_ns = earliest;
    out.valid = true;
  }
  return out;
}

inline bool ShouldRefreshSwapchainTimingMetadata(
    uint64_t cached_timing_props_counter,
    uint64_t reported_timing_props_counter,
    uint64_t cached_time_domains_counter,
    uint64_t reported_time_domains_counter, bool queue_full_error) {
  return queue_full_error ||
         (cached_timing_props_counter != reported_timing_props_counter) ||
         (cached_time_domains_counter != reported_time_domains_counter);
}

#ifdef VK_EXT_present_timing
inline SelectedTimeDomain SelectPreferredVkTimeDomain(
    const VkTimeDomainKHR* time_domains, const uint64_t* time_domain_ids,
    size_t count) {
  SelectedTimeDomain best;
  if (!time_domains || !time_domain_ids || count == 0) {
    return best;
  }
  int best_rank = 100;
  for (size_t i = 0; i < count; ++i) {
    const uint32_t dom = static_cast<uint32_t>(time_domains[i]);
    const int rank = TimeDomainPreferenceRank(dom);
    if (!best.valid || rank < best_rank) {
      best.valid = true;
      best.time_domain = dom;
      best.time_domain_id = time_domain_ids[i];
      best.is_clock_monotonic = (dom == kTimeDomainClockMonotonic);
      best.is_clock_monotonic_raw = (dom == kTimeDomainClockMonotonicRaw);
      best.is_monotonic_clock =
          best.is_clock_monotonic || best.is_clock_monotonic_raw;
      best_rank = rank;
    }
  }
  return best;
}

inline UnifiedPastPresentationTiming ConvertVkPastPresentationTimingEXT(
    const VkPastPresentationTimingEXT& timing, uint64_t refresh_duration_ns) {
  PresentStageSample stage_buf[8];
  const size_t n =
      (timing.pPresentStages != nullptr)
          ? (timing.presentStageCount < 8u ? timing.presentStageCount : 8u)
          : 0u;
  for (size_t i = 0; i < n; ++i) {
    stage_buf[i].stage = static_cast<uint32_t>(timing.pPresentStages[i].stage);
    stage_buf[i].time_ns = timing.pPresentStages[i].time;
  }
  UnifiedPastPresentationTiming out = ConvertPastPresentationTiming(
      timing.presentId, timing.targetTime, stage_buf, n,
      timing.reportComplete != VK_FALSE, refresh_duration_ns);
  out.time_domain = static_cast<uint32_t>(timing.timeDomain);
  return out;
}
#endif

// Rejects incomplete reports (report_complete == false) so partial stage
// timestamps are never consumed as completed presentation samples.
inline bool IsCompletePastPresentationTiming(
    const UnifiedPastPresentationTiming& timing) {
  return timing.valid && timing.report_complete &&
         timing.actual_present_time_ns != 0u;
}

// Requires both report_complete == true and strict CLOCK_MONOTONIC domain
// alignment with std::chrono::steady_clock.
inline bool IsCompleteMonotonicPastPresentationTiming(
    const UnifiedPastPresentationTiming& timing,
    const SelectedTimeDomain& swapchain_domain) {
  return IsCompletePastPresentationTiming(timing) &&
         TimeDomainMatchesSteadyClock(swapchain_domain) &&
         timing.time_domain == kTimeDomainClockMonotonic;
}

// Resolves the actual targetTime / desiredPresentTime passed to Vulkan on the
// native presenter path. Without a VSync-lattice quantizer (such as ZeroFG's
// QuantizeToVsync), passing an unquantized future timestamp (e.g.
// native_last_present_ns_ + refresh_ns) on Android FIFO causes SurfaceFlinger
// to retain early frames across VSync boundaries and drop them when a later
// overdue frame submits desiredPresentTime == 0. Always emit 0 (ASAP) unless a
// caller explicitly provides a VSync-quantized target.
inline uint64_t ResolveNativePresentTargetTimeNs(
    const LateSwapDecision& decision, bool is_fifo,
    bool can_use_absolute_target, bool target_is_vsync_quantized = false) {
  if (!is_fifo || !can_use_absolute_target || !target_is_vsync_quantized) {
    return 0ull;
  }
  if (decision.action == LateSwapAction::kKeepScheduledTarget) {
    return decision.effective_target_ns;
  }
  return 0ull;
}

// Timing records must only be enqueued when the present succeeded AND timing
// metadata remained attached to VkPresentInfoKHR::pNext (never after a
// QUEUE_FULL_EXT retry that stripped pNext).
inline bool ShouldEnqueueTimingRecord(PresentInfoTimingChain timing_chain,
                                      bool presented,
                                      bool retried_without_pnext) {
  return presented && timing_chain != PresentInfoTimingChain::kNone &&
         !retried_without_pnext;
}

// Wrap-safe comparison for overtaken timing records: uses full 64-bit signed
// distance for VK_KHR_present_id2 / VK_EXT_present_timing and 32-bit signed
// distance for VK_GOOGLE_display_timing.
inline bool IsOlderPresentId(uint64_t candidate_id, uint64_t reference_id,
                             bool is_64bit_id) {
  if (is_64bit_id) {
    return static_cast<int64_t>(candidate_id - reference_id) < 0;
  }
  return static_cast<int32_t>(static_cast<uint32_t>(candidate_id) -
                              static_cast<uint32_t>(reference_id)) < 0;
}

struct NativeLatencyTimingRecord {
  uint64_t present_id = 0;
  uint64_t source_id = 0;
  uint64_t issue_ns = 0;
};

struct NativeLatencyPollResult {
  bool matched = false;
  bool sample_valid = false;
  uint64_t latency_ns = 0;
  size_t retired_older_count = 0;
};

// Matches a completed present timing against the native latency ring and
// retires any strictly older unresolved records (wrap-safe via
// IsOlderPresentId) that were dropped/superseded by the compositor so they do
// not remain stuck in the ring until overwritten.
template <typename RecordT>
inline NativeLatencyPollResult MatchAndRetireNativeLatencyRecord(
    RecordT* records, size_t record_count, uint64_t present_id,
    uint64_t actual_present_time_ns, uint64_t observation_ns,
    bool is_64bit_id) {
  NativeLatencyPollResult result;
  if (!records || record_count == 0 || !present_id) {
    return result;
  }
  for (size_t i = 0; i < record_count; ++i) {
    RecordT& record = records[i];
    if (record.present_id && record.present_id == present_id) {
      result.matched = true;
      if (record.source_id && record.issue_ns &&
          actual_present_time_ns > record.issue_ns &&
          actual_present_time_ns <= observation_ns) {
        result.sample_valid = true;
        result.latency_ns = actual_present_time_ns - record.issue_ns;
      }
      record = {};
      break;
    }
  }
  if (!result.matched) {
    return result;
  }
  for (size_t i = 0; i < record_count; ++i) {
    RecordT& older = records[i];
    if (older.present_id &&
        IsOlderPresentId(older.present_id, present_id, is_64bit_id)) {
      older = {};
      ++result.retired_older_count;
    }
  }
  return result;
}

// Maps a monotonically increasing 64-bit sequence ID to a non-zero 32-bit
// presentID for VK_GOOGLE_display_timing.
inline uint32_t NormalizeGooglePresentId32(uint64_t sequence_id) {
  const uint32_t id = static_cast<uint32_t>(sequence_id);
  return id != 0u ? id : 1u;
}

// Late-swap policy for native presentation.
// Safety invariant: explicitly disabled whenever ZeroFG is active so
// ZeroFGMainSurfaceEgress::QuantizeToVsync and Real/Synthetic spacing remain
// untouched.
inline LateSwapDecision EvaluateLateSwapPresent(uint64_t now_ns,
                                                uint64_t scheduled_target_ns,
                                                uint64_t refresh_period_ns,
                                                bool zerofg_active,
                                                bool late_swap_enabled) {
  (void)refresh_period_ns;
  if (zerofg_active || !late_swap_enabled || scheduled_target_ns == 0) {
    return {LateSwapAction::kBypassKeepPaced, scheduled_target_ns};
  }
  if (now_ns >= scheduled_target_ns) {
    return {LateSwapAction::kPresentImmediate, 0u};
  }
  return {LateSwapAction::kKeepScheduledTarget, scheduled_target_ns};
}

// CommandProcessor::ThrottlePresentation anchor update policy.
// When late-swap is enabled outside ZeroFG, caps carry-over debt to <= 0.5 *
// target_duration_ticks so an overdue frame is released immediately without
// forcing a follow-up vblank skip or bursting back-to-back.
inline uint64_t ComputeNextThrottleAnchorTicks(uint64_t last_swap_ticks,
                                               uint64_t current_ticks,
                                               uint64_t target_duration_ticks,
                                               bool zerofg_active,
                                               bool late_swap_enabled) {
  if (current_ticks <= last_swap_ticks) {
    return current_ticks;
  }
  const uint64_t time_delta = current_ticks - last_swap_ticks;
  if (time_delta > target_duration_ticks * 2u) {
    return current_ticks;
  }
  const uint64_t stepped = last_swap_ticks + target_duration_ticks;
  if (zerofg_active || !late_swap_enabled || current_ticks <= stepped) {
    return stepped;
  }
  const uint64_t lateness = current_ticks - stepped;
  const uint64_t max_carry_over = target_duration_ticks / 2u;
  if (lateness > max_carry_over) {
    return current_ticks - max_carry_over;
  }
  return stepped;
}

}  // namespace vulkan
}  // namespace ui
}  // namespace xe

#endif  // XENIA_UI_VULKAN_VULKAN_PRESENT_TIMING_H_
