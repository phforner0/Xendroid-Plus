#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#ifndef XE_DISABLE_ANDROID_DLOPEN
#define XE_DISABLE_ANDROID_DLOPEN 1
#endif

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES 1
#endif

#include "third_party/Vulkan-Headers/include/vulkan/vulkan_core.h"
#include "xenia/ui/vulkan/vulkan_present_timing.h"

namespace {

#define W7_1C_CHECK(cond)                                                     \
  do {                                                                        \
    if (!(cond)) {                                                            \
      std::fprintf(stderr, "W7_1C check failed (%s:%d): %s\n", __FILE__,      \
                   __LINE__, #cond);                                          \
      std::abort();                                                           \
    }                                                                         \
  } while (false)

namespace fs = std::filesystem;
namespace vkui = xe::ui::vulkan;

std::string ReadFileText(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  W7_1C_CHECK(in.good());
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

// Mirror of android_linker_ns.cpp AArch64 BTI/BL/page-span pure logic so host
// tests validate exact instruction encodings and 4 KB / 16 KB page geometry.
constexpr bool IsAArch64CallLandingPad(uint32_t instr) {
  return instr == 0xd503245fu ||  // BTI c
         instr == 0xd50324dfu ||  // BTI jc
         instr == 0xd503233fu ||  // PACIASP
         instr == 0xd503237fu;    // PACIBSP
}

struct AArch64BranchLinked {
  uint32_t raw = 0;
  bool Verify() const { return (raw >> 26) == 0x25u; }
  int32_t Offset() const { return static_cast<int32_t>(raw << 6) >> 6; }
  static AArch64BranchLinked Encode(uint8_t sig, int32_t offset) {
    return {(static_cast<uint32_t>(sig & 0x3fu) << 26) |
            (static_cast<uint32_t>(offset) & 0x03ffffffu)};
  }
};
static_assert(sizeof(AArch64BranchLinked) == 4, "BL instruction must be 4 bytes");

const AArch64BranchLinked* ScanBoundedBlTarget(
    const AArch64BranchLinked* wrapper, int max_instrs) {
  for (int i = 0; i < max_instrs; ++i, ++wrapper) {
    if (wrapper->Verify()) {
      return wrapper + wrapper->Offset();
    }
  }
  return nullptr;
}

uintptr_t AlignDownPage(uintptr_t addr, size_t page_size) {
  return addr & ~(static_cast<uintptr_t>(page_size) - 1);
}

size_t ComputeWrapperPageSpan(uintptr_t start_addr, size_t wrapper_bytes,
                              size_t page_size) {
  const uintptr_t aligned_start = AlignDownPage(start_addr, page_size);
  const uintptr_t end_addr = start_addr + wrapper_bytes;
  return static_cast<size_t>(((end_addr - aligned_start) + page_size - 1) &
                             ~(static_cast<uintptr_t>(page_size) - 1));
}

void TestRisk1_DevicePresentId2StrictGate() {
  using vkui::CanEnablePresentTimingExtOnDeviceV3;

  // Requires present_timing + calibrated_timestamps + present_id2 ext +
  // presentTiming feature + presentId2 feature.
  W7_1C_CHECK(CanEnablePresentTimingExtOnDeviceV3(true, true, true, true, true));

  // Rejects v1-only (feature_present_id2 == false) or any missing dependency.
  W7_1C_CHECK(
      !CanEnablePresentTimingExtOnDeviceV3(true, true, true, true, false));
  W7_1C_CHECK(
      !CanEnablePresentTimingExtOnDeviceV3(true, true, false, true, true));
  W7_1C_CHECK(
      !CanEnablePresentTimingExtOnDeviceV3(true, false, true, true, true));
  W7_1C_CHECK(
      !CanEnablePresentTimingExtOnDeviceV3(false, true, true, true, true));
  W7_1C_CHECK(
      !CanEnablePresentTimingExtOnDeviceV3(true, true, true, false, true));
}

void TestRisk2And4_SwapchainPreFlightGates() {
  using vkui::CanUseExtPresentTimingForPresenter;
  using vkui::CanUseExtPresentTimingForZeroFG;
  using vkui::PresentTimingBackend;
  using vkui::SelectPresentTimingBackend;

  const uint32_t kMask = vkui::kPresentStageFirstPixelVisible;

  // Presenter requires presentId2 + non-zero stage mask before swapchain create.
  W7_1C_CHECK(
      CanUseExtPresentTimingForPresenter(true, true, true, true, true, kMask));
  W7_1C_CHECK(
      !CanUseExtPresentTimingForPresenter(true, true, true, true, false, kMask));
  W7_1C_CHECK(
      !CanUseExtPresentTimingForPresenter(true, true, true, true, true, 0u));

  // ZeroFG additionally requires presentAtAbsoluteTime; otherwise falls back to
  // VK_GOOGLE_display_timing BEFORE creating a PRESENT_TIMING_BIT_EXT swapchain.
  W7_1C_CHECK(CanUseExtPresentTimingForZeroFG(true, true, true, true, true,
                                              true, kMask));
  const bool zfg_no_abs = CanUseExtPresentTimingForZeroFG(
      true, true, true, true, /*ext_present_at_absolute_time=*/false, true,
      kMask);
  W7_1C_CHECK(!zfg_no_abs);
  W7_1C_CHECK(SelectPresentTimingBackend(zfg_no_abs, true, true, true,
                                         /*google_supported=*/true) ==
              PresentTimingBackend::kGoogleDisplayTiming);

  // ZeroFG without presentId2 or without stage queries also falls back before
  // swapchain creation.
  W7_1C_CHECK(!CanUseExtPresentTimingForZeroFG(true, true, true, true, true,
                                               false, kMask));
  W7_1C_CHECK(!CanUseExtPresentTimingForZeroFG(true, true, true, true, true,
                                               true, 0u));
}

void TestRisk3_ClockMonotonicDomainSeparation() {
  using vkui::SelectPreferredTimeDomain;
  using vkui::TimeDomainMatchesSteadyClock;

  // CLOCK_MONOTONIC matches std::chrono::steady_clock.
  const uint32_t d_mono[] = {vkui::kTimeDomainClockMonotonicRaw,
                             vkui::kTimeDomainClockMonotonic};
  const uint64_t ids_mono[] = {10, 20};
  const auto sel_mono = SelectPreferredTimeDomain(d_mono, ids_mono, 2);
  W7_1C_CHECK(sel_mono.valid);
  W7_1C_CHECK(sel_mono.time_domain == vkui::kTimeDomainClockMonotonic);
  W7_1C_CHECK(sel_mono.is_monotonic_clock);
  W7_1C_CHECK(sel_mono.is_clock_monotonic);
  W7_1C_CHECK(!sel_mono.is_clock_monotonic_raw);
  W7_1C_CHECK(TimeDomainMatchesSteadyClock(sel_mono));

  // CLOCK_MONOTONIC_RAW is distinct from CLOCK_MONOTONIC and MUST NOT be mixed
  // with std::chrono::steady_clock timestamps!
  const uint32_t d_raw[] = {vkui::kTimeDomainSwapchainLocal,
                            vkui::kTimeDomainClockMonotonicRaw};
  const uint64_t ids_raw[] = {11, 22};
  const auto sel_raw = SelectPreferredTimeDomain(d_raw, ids_raw, 2);
  W7_1C_CHECK(sel_raw.valid);
  W7_1C_CHECK(sel_raw.time_domain == vkui::kTimeDomainClockMonotonicRaw);
  W7_1C_CHECK(sel_raw.is_monotonic_clock);
  W7_1C_CHECK(!sel_raw.is_clock_monotonic);
  W7_1C_CHECK(sel_raw.is_clock_monotonic_raw);
  W7_1C_CHECK(!TimeDomainMatchesSteadyClock(sel_raw));

  // Local / device domains also rejected for steady_clock comparisons.
  const uint32_t d_local[] = {vkui::kTimeDomainPresentStageLocal};
  const uint64_t ids_local[] = {33};
  const auto sel_local = SelectPreferredTimeDomain(d_local, ids_local, 1);
  W7_1C_CHECK(sel_local.valid);
  W7_1C_CHECK(!sel_local.is_monotonic_clock);
  W7_1C_CHECK(!TimeDomainMatchesSteadyClock(sel_local));
}

void TestRisk4_NativeLateSwapTargetResolution() {
  using vkui::EvaluateLateSwapPresent;
  using vkui::LateSwapAction;
  using vkui::ResolveNativePresentTargetTimeNs;

  constexpr uint64_t kRefresh60 = 16'666'667ull;
  constexpr uint64_t kTarget = 2'000'000'000ull;

  // Early native FIFO frame with vulkan_late_swap_present=true and
  // now < last_present_ns + refresh_ns: without ZeroFG's VSync quantizer,
  // native path MUST keep desiredPresentTime == 0 (ASAP) to prevent
  // SurfaceFlinger frame drops and +1 VSync jumps.
  const auto early =
      EvaluateLateSwapPresent(kTarget - 4'000'000ull, kTarget, kRefresh60,
                              /*zerofg_active=*/false,
                              /*late_swap_enabled=*/true);
  W7_1C_CHECK(early.action == LateSwapAction::kKeepScheduledTarget);
  W7_1C_CHECK(ResolveNativePresentTargetTimeNs(early, /*is_fifo=*/true,
                                               /*can_use_absolute_target=*/true) ==
              0ull);
  W7_1C_CHECK(ResolveNativePresentTargetTimeNs(
                  early, /*is_fifo=*/true, /*can_use_absolute_target=*/true,
                  /*target_is_vsync_quantized=*/false) == 0ull);

  // Only an explicitly VSync-quantized FIFO target with absolute target
  // support may emit a non-zero targetTime.
  W7_1C_CHECK(ResolveNativePresentTargetTimeNs(
                  early, /*is_fifo=*/true, /*can_use_absolute_target=*/true,
                  /*target_is_vsync_quantized=*/true) == kTarget);

  // Without absolute target capability (e.g. EXT without presentAtAbsoluteTime),
  // must stay 0 (VUID safety) even if quantized.
  W7_1C_CHECK(ResolveNativePresentTargetTimeNs(
                  early, /*is_fifo=*/true,
                  /*can_use_absolute_target=*/false,
                  /*target_is_vsync_quantized=*/true) == 0ull);

  // Non-FIFO (IMMEDIATE / MAILBOX) must stay 0 (never artificially hold frames).
  W7_1C_CHECK(ResolveNativePresentTargetTimeNs(
                  early, /*is_fifo=*/false,
                  /*can_use_absolute_target=*/true,
                  /*target_is_vsync_quantized=*/true) == 0ull);

  // Overdue frame -> immediate (0), never pushed to vsync + 1.
  const auto late =
      EvaluateLateSwapPresent(kTarget + 2'000'000ull, kTarget, kRefresh60,
                              /*zerofg_active=*/false,
                              /*late_swap_enabled=*/true);
  W7_1C_CHECK(late.action == LateSwapAction::kPresentImmediate);
  W7_1C_CHECK(ResolveNativePresentTargetTimeNs(late, true, true) == 0ull);
  W7_1C_CHECK(ResolveNativePresentTargetTimeNs(late, true, true, true) == 0ull);

  // ZeroFG active or late-swap disabled -> bypass (native path emits 0).
  const auto zfg =
      EvaluateLateSwapPresent(kTarget - 4'000'000ull, kTarget, kRefresh60,
                              /*zerofg_active=*/true,
                              /*late_swap_enabled=*/true);
  W7_1C_CHECK(zfg.action == LateSwapAction::kBypassKeepPaced);
  W7_1C_CHECK(ResolveNativePresentTargetTimeNs(zfg, true, true) == 0ull);
  W7_1C_CHECK(ResolveNativePresentTargetTimeNs(zfg, true, true, true) == 0ull);

  const auto disabled =
      EvaluateLateSwapPresent(kTarget - 4'000'000ull, kTarget, kRefresh60,
                              /*zerofg_active=*/false,
                              /*late_swap_enabled=*/false);
  W7_1C_CHECK(disabled.action == LateSwapAction::kBypassKeepPaced);
  W7_1C_CHECK(ResolveNativePresentTargetTimeNs(disabled, true, true) == 0ull);
  W7_1C_CHECK(ResolveNativePresentTargetTimeNs(disabled, true, true, true) ==
              0ull);
}

void TestNativeLatencyRingRetirement() {
  using vkui::MatchAndRetireNativeLatencyRecord;
  using vkui::NativeLatencyTimingRecord;
  using vkui::NormalizeGooglePresentId32;

  constexpr size_t kRingSize = 32;

  // 1) Dropped frames are retired when a newer frame completes, leaving newer
  //    in-flight records intact and preventing ring overwrites across >32
  //    frames (overwritten == 0, reliable == true).
  {
    std::array<NativeLatencyTimingRecord, kRingSize> ring = {};
    ring[0] = {10u, 100u, 1'000'000ull};  // dropped by SurfaceFlinger
    ring[1] = {11u, 101u, 2'000'000ull};  // dropped by SurfaceFlinger
    ring[2] = {12u, 102u, 3'000'000ull};  // displayed
    ring[3] = {13u, 103u, 4'000'000ull};  // still in-flight

    const auto res = MatchAndRetireNativeLatencyRecord(
        ring.data(), ring.size(), /*present_id=*/12u,
        /*actual_present_time_ns=*/23'000'000ull,
        /*observation_ns=*/25'000'000ull, /*is_64bit_id=*/false);
    W7_1C_CHECK(res.matched);
    W7_1C_CHECK(res.sample_valid);
    W7_1C_CHECK(res.latency_ns == 20'000'000ull);
    W7_1C_CHECK(res.retired_older_count == 2u);
    W7_1C_CHECK(ring[0].present_id == 0u);
    W7_1C_CHECK(ring[1].present_id == 0u);
    W7_1C_CHECK(ring[2].present_id == 0u);
    W7_1C_CHECK(ring[3].present_id == 13u);

    // Unmatched ID must not corrupt or retire live pending records.
    const auto unmatched = MatchAndRetireNativeLatencyRecord(
        ring.data(), ring.size(), /*present_id=*/999u,
        /*actual_present_time_ns=*/26'000'000ull,
        /*observation_ns=*/27'000'000ull, /*is_64bit_id=*/false);
    W7_1C_CHECK(!unmatched.matched);
    W7_1C_CHECK(unmatched.retired_older_count == 0u);
    W7_1C_CHECK(ring[3].present_id == 13u);

    // Untagged repaint (present_id == 0) is a clean no-op.
    const auto untagged = MatchAndRetireNativeLatencyRecord(
        ring.data(), ring.size(), /*present_id=*/0u,
        /*actual_present_time_ns=*/26'000'000ull,
        /*observation_ns=*/27'000'000ull, /*is_64bit_id=*/false);
    W7_1C_CHECK(!untagged.matched);
    W7_1C_CHECK(ring[3].present_id == 13u);
  }

  // 2) 32-bit VK_GOOGLE_display_timing wraparound (0xFFFFFFFE -> 0xFFFFFFFF ->
  //    1 -> 2 -> 3): dropped pre-wrap frames are cleanly retired when post-wrap
  //    id=2 completes, keeping id=3 alive and generating 0 overwrites.
  {
    std::array<NativeLatencyTimingRecord, kRingSize> ring = {};
    ring[0] = {0xFFFFFFFEu, 500u, 10'000'000ull};  // dropped
    ring[1] = {0xFFFFFFFFu, 501u, 20'000'000ull};  // dropped
    ring[2] = {1u, 502u, 30'000'000ull};           // dropped
    ring[3] = {2u, 503u, 40'000'000ull};           // displayed
    ring[4] = {3u, 504u, 50'000'000ull};           // in-flight

    const auto wrap_res = MatchAndRetireNativeLatencyRecord(
        ring.data(), ring.size(), /*present_id=*/2u,
        /*actual_present_time_ns=*/62'000'000ull,
        /*observation_ns=*/65'000'000ull, /*is_64bit_id=*/false);
    W7_1C_CHECK(wrap_res.matched);
    W7_1C_CHECK(wrap_res.sample_valid);
    W7_1C_CHECK(wrap_res.latency_ns == 22'000'000ull);
    W7_1C_CHECK(wrap_res.retired_older_count == 3u);
    W7_1C_CHECK(ring[0].present_id == 0u);
    W7_1C_CHECK(ring[1].present_id == 0u);
    W7_1C_CHECK(ring[2].present_id == 0u);
    W7_1C_CHECK(ring[3].present_id == 0u);
    W7_1C_CHECK(ring[4].present_id == 3u);
  }

  // 3) Sustained stream of 96 frames across 32-bit wraparound with 1-in-3
  //    compositor drops: 0 overwritten slots, reliable stays true.
  {
    std::array<NativeLatencyTimingRecord, kRingSize> ring = {};
    size_t next_slot = 0;
    uint64_t seq = 0xFFFFFFFDull;
    uint64_t overwritten = 0;
    uint64_t samples = 0;
    uint64_t retired_total = 0;
    for (uint64_t frame = 1; frame <= 96; ++frame) {
      ++seq;
      if (static_cast<uint32_t>(seq) == 0u) {
        ++seq;  // avoid duplicate id=1 at exact 2^32 roll-over
      }
      const uint32_t pid = NormalizeGooglePresentId32(seq);
      const uint64_t issue_ns = frame * 16'000'000ull;
      if (ring[next_slot].present_id != 0u) {
        ++overwritten;
      }
      ring[next_slot] = {uint64_t(pid), frame, issue_ns};
      next_slot = (next_slot + 1) % kRingSize;

      // Simulate compositor dropping every 3rd frame (frame % 3 == 1).
      if (frame % 3 != 1) {
        const auto res = MatchAndRetireNativeLatencyRecord(
            ring.data(), ring.size(), uint64_t(pid),
            issue_ns + 20'000'000ull, issue_ns + 22'000'000ull,
            /*is_64bit_id=*/false);
        W7_1C_CHECK(res.matched && res.sample_valid);
        W7_1C_CHECK(res.latency_ns == 20'000'000ull);
        ++samples;
        retired_total += res.retired_older_count;
      }
    }
    W7_1C_CHECK(overwritten == 0u);
    W7_1C_CHECK(samples == 64u);
    W7_1C_CHECK(retired_total == 32u);
  }

  // 4) 64-bit VK_EXT_present_timing crossing UINT32_MAX retires older IDs
  //    without 32-bit wrap confusion.
  {
    std::array<NativeLatencyTimingRecord, kRingSize> ring = {};
    ring[0] = {0xFFFFFFFFull, 1u, 1'000'000ull};
    ring[1] = {0x100000000ull, 2u, 2'000'000ull};
    ring[2] = {0x100000001ull, 3u, 3'000'000ull};
    ring[3] = {0x100000002ull, 4u, 4'000'000ull};

    const auto ext_res = MatchAndRetireNativeLatencyRecord(
        ring.data(), ring.size(), /*present_id=*/0x100000001ull,
        /*actual_present_time_ns=*/19'000'000ull,
        /*observation_ns=*/20'000'000ull, /*is_64bit_id=*/true);
    W7_1C_CHECK(ext_res.matched);
    W7_1C_CHECK(ext_res.sample_valid);
    W7_1C_CHECK(ext_res.latency_ns == 16'000'000ull);
    W7_1C_CHECK(ext_res.retired_older_count == 2u);
    W7_1C_CHECK(ring[0].present_id == 0u);
    W7_1C_CHECK(ring[1].present_id == 0u);
    W7_1C_CHECK(ring[2].present_id == 0u);
    W7_1C_CHECK(ring[3].present_id == 0x100000002ull);
  }
}

void TestRisk5_QueueFullRetryNoTimingRecord() {
  using vkui::PresentInfoTimingChain;
  using vkui::ShouldEnqueueTimingRecord;

  // Normal presented frame with timing attached -> enqueue.
  W7_1C_CHECK(ShouldEnqueueTimingRecord(PresentInfoTimingChain::kExtOnly, true,
                                        false));
  W7_1C_CHECK(ShouldEnqueueTimingRecord(PresentInfoTimingChain::kGoogleOnly,
                                        true, false));

  // QUEUE_FULL_EXT retry stripped pNext -> MUST NOT enqueue a ghost record!
  W7_1C_CHECK(!ShouldEnqueueTimingRecord(PresentInfoTimingChain::kExtOnly, true,
                                         /*retried_without_pnext=*/true));

  // Unpresented or untimed -> never enqueue.
  W7_1C_CHECK(!ShouldEnqueueTimingRecord(PresentInfoTimingChain::kExtOnly,
                                         false, false));
  W7_1C_CHECK(!ShouldEnqueueTimingRecord(PresentInfoTimingChain::kNone, true,
                                         false));
}

void TestRisk6_CompleteReportAnd64BitPresentId() {
  using vkui::IsCompleteMonotonicPastPresentationTiming;
  using vkui::IsCompletePastPresentationTiming;
  using vkui::IsOlderPresentId;
  using vkui::NormalizeGooglePresentId32;
  using vkui::SelectPreferredTimeDomain;

  const uint32_t mono_dom[1] = {vkui::kTimeDomainClockMonotonic};
  const uint64_t mono_id[1] = {1};
  const auto swapchain_mono = SelectPreferredTimeDomain(mono_dom, mono_id, 1);

  const uint32_t raw_dom[1] = {vkui::kTimeDomainClockMonotonicRaw};
  const auto swapchain_raw = SelectPreferredTimeDomain(raw_dom, mono_id, 1);

#ifdef VK_EXT_present_timing
  // 64-bit presentId above UINT32_MAX must survive without 32-bit truncation.
  constexpr uint64_t kHighId1 = (1ull << 33) + 42ull;
  constexpr uint64_t kHighId2 = (1ull << 33) + 43ull;

  VkPresentStageTimeEXT stages[1] = {
      {VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_VISIBLE_BIT_EXT, 900'000'000ull},
  };
  VkPastPresentationTimingEXT timing = {};
  timing.sType = VK_STRUCTURE_TYPE_PAST_PRESENTATION_TIMING_EXT;
  timing.presentId = kHighId1;
  timing.targetTime = 899'000'000ull;
  timing.presentStageCount = 1;
  timing.pPresentStages = stages;
  timing.timeDomain = VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR;
  timing.timeDomainId = 1;
  timing.reportComplete = VK_FALSE;

  // reportComplete == VK_FALSE MUST be rejected!
  const auto partial =
      vkui::ConvertVkPastPresentationTimingEXT(timing, 16'666'667ull);
  W7_1C_CHECK(partial.present_id == kHighId1);
  W7_1C_CHECK(!partial.report_complete);
  W7_1C_CHECK(!IsCompletePastPresentationTiming(partial));
  W7_1C_CHECK(!IsCompleteMonotonicPastPresentationTiming(partial, swapchain_mono));

  // Complete + CLOCK_MONOTONIC -> accepted with exact 64-bit present_id!
  timing.reportComplete = VK_TRUE;
  const auto complete =
      vkui::ConvertVkPastPresentationTimingEXT(timing, 16'666'667ull);
  W7_1C_CHECK(complete.present_id == kHighId1);
  W7_1C_CHECK(IsCompletePastPresentationTiming(complete));
  W7_1C_CHECK(IsCompleteMonotonicPastPresentationTiming(complete, swapchain_mono));

  // Post-refresh non-CLOCK_MONOTONIC swapchain domain rejects report!
  W7_1C_CHECK(!IsCompleteMonotonicPastPresentationTiming(complete, swapchain_raw));

  // Report claiming CLOCK_MONOTONIC_RAW is also rejected!
  timing.timeDomain = VK_TIME_DOMAIN_CLOCK_MONOTONIC_RAW_KHR;
  const auto raw_report =
      vkui::ConvertVkPastPresentationTimingEXT(timing, 16'666'667ull);
  W7_1C_CHECK(
      !IsCompleteMonotonicPastPresentationTiming(raw_report, swapchain_mono));

  // 64-bit overtaken ordering works cleanly across UINT32_MAX boundary.
  W7_1C_CHECK(IsOlderPresentId(kHighId1, kHighId2, true));
  W7_1C_CHECK(!IsOlderPresentId(kHighId2, kHighId1, true));
  W7_1C_CHECK(IsOlderPresentId(0xFFFFFFFFull, 0x100000000ull, true));
  W7_1C_CHECK(!IsOlderPresentId(0x100000000ull, 0xFFFFFFFFull, true));
#endif

  // 32-bit GOOGLE wrap normalization never emits 0 and preserves signed order.
  W7_1C_CHECK(NormalizeGooglePresentId32(0ull) == 1u);
  W7_1C_CHECK(NormalizeGooglePresentId32(0x100000000ull) == 1u);
  W7_1C_CHECK(NormalizeGooglePresentId32(0x100000005ull) == 5u);
  W7_1C_CHECK(IsOlderPresentId(0xFFFFFFFFu, 1u, false));
  W7_1C_CHECK(!IsOlderPresentId(1u, 0xFFFFFFFFu, false));
}

void TestAndroid17BtiAnd16KbGeometry() {
  // Valid AArch64 BTI indirect-call (BLR) landing pads.
  W7_1C_CHECK(IsAArch64CallLandingPad(0xd503245fu));  // bti c
  W7_1C_CHECK(IsAArch64CallLandingPad(0xd50324dfu));  // bti jc
  W7_1C_CHECK(IsAArch64CallLandingPad(0xd503233fu));  // paciasp
  W7_1C_CHECK(IsAArch64CallLandingPad(0xd503237fu));  // pacibsp

  // Invalid indirect-call targets (must refuse indirect BLR when BTI active!).
  W7_1C_CHECK(!IsAArch64CallLandingPad(0xd503249fu));  // bti j (BR only, not BLR!)
  W7_1C_CHECK(!IsAArch64CallLandingPad(0xd503201fu));  // nop
  W7_1C_CHECK(!IsAArch64CallLandingPad(0xa9bf7bfdu));  // stp x29, x30, [sp, #-16]!
  W7_1C_CHECK(!IsAArch64CallLandingPad(0xaa1e03e2u));  // mov x2, x30

  // Bounded BL scan: finds BL within 32 instructions with signed offset.
  std::array<AArch64BranchLinked, 40> code = {};
  for (auto& ins : code) {
    ins.raw = 0xd503201fu;  // nop
  }
  code[3] = AArch64BranchLinked::Encode(0x25u, 10);
  W7_1C_CHECK(ScanBoundedBlTarget(code.data(), 32) == &code[13]);

  // Negative signed 26-bit offset works too.
  code[3] = AArch64BranchLinked::Encode(0x25u, -2);
  W7_1C_CHECK(ScanBoundedBlTarget(code.data(), 32) == &code[1]);

  // No BL within 32 instructions -> returns nullptr safely (no OOB read!).
  code[3].raw = 0xd503201fu;
  code[35] = AArch64BranchLinked::Encode(0x25u, 1);
  W7_1C_CHECK(ScanBoundedBlTarget(code.data(), 32) == nullptr);

  // Page alignment & span for 4 KB and 16 KB Android page sizes.
  W7_1C_CHECK(AlignDownPage(0x12345678u, 4096u) == 0x12345000u);
  W7_1C_CHECK(AlignDownPage(0x12345678u, 16384u) == 0x12344000u);
  // Wrapper starting 16 bytes before a 16 KB boundary spans 2 pages (32768 B).
  W7_1C_CHECK(ComputeWrapperPageSpan(16384u - 16u, 32u * 4u, 16384u) == 32768u);
  // Wrapper well inside a 16 KB page spans 1 page (16384 B).
  W7_1C_CHECK(ComputeWrapperPageSpan(16384u + 64u, 32u * 4u, 16384u) == 16384u);
}

void TestSourceContracts(const fs::path& repo_root) {
  const fs::path cpp_dir = repo_root / "emulator-core/src/main/cpp";
  const std::string linker_ns = ReadFileText(
      cpp_dir /
      "xenia/third_party/libadrenotools/lib/linkernsbypass/android_linker_ns.cpp");
  const std::string dev_cc =
      ReadFileText(cpp_dir / "xenia/src/xenia/ui/vulkan/vulkan_device.cc");
  const std::string pres_h =
      ReadFileText(cpp_dir / "xenia/src/xenia/ui/vulkan/vulkan_presenter.h");
  const std::string pres_cc =
      ReadFileText(cpp_dir / "xenia/src/xenia/ui/vulkan/vulkan_presenter.cc");
  const std::string zfg_h = ReadFileText(
      cpp_dir / "xenia/src/xenia/ui/vulkan/zerofg_main_surface_egress.h");
  const std::string zfg_cc = ReadFileText(
      cpp_dir / "xenia/src/xenia/ui/vulkan/zerofg_main_surface_egress.cc");
  const std::string audio_cc =
      ReadFileText(cpp_dir / "xenia/src/xenia/apu/audio_system.cc");
  const std::string toml = ReadFileText(
      repo_root / "emulator-core/src/main/assets/config/default_config.toml");
  const std::string schema_kt = ReadFileText(
      repo_root / "app/src/main/java/xendroid/compose/settings/SettingsSchema.kt");
  const std::string strings_en =
      ReadFileText(repo_root / "app/src/main/res/values/strings_settings.xml");
  const std::string strings_pt = ReadFileText(
      repo_root / "app/src/main/res/values-pt-rBR/strings_settings.xml");
  const std::string headroom_kt = ReadFileText(
      repo_root / "app/src/main/java/xendroid/compose/core/SystemHeadroomReader.kt");
  const std::string host_kt = ReadFileText(
      repo_root / "app/src/main/java/xendroid/compose/EmulatorHostActivity.kt");

  // 1) Android 17 BTI contracts in android_linker_ns.cpp.
  W7_1C_CHECK(linker_ns.find("__attribute__((constructor))") ==
              std::string::npos);
  W7_1C_CHECK(linker_ns.find("PROT_WRITE") == std::string::npos);
  W7_1C_CHECK(linker_ns.find("&loader_dlopen") == std::string::npos);
  W7_1C_CHECK(linker_ns.find("sysconf(_SC_PAGESIZE)") != std::string::npos);
  W7_1C_CHECK(linker_ns.find("HWCAP2_BTI") != std::string::npos);
  W7_1C_CHECK(linker_ns.find("is_call_landing_pad") != std::string::npos);
  W7_1C_CHECK(linker_ns.find("MaxWrapperInstrs") != std::string::npos);
  W7_1C_CHECK(linker_ns.find("lazy_android_get_exported_namespace") !=
              std::string::npos);

  // 2) VK_EXT_present_timing 6-risk contracts.
  W7_1C_CHECK(dev_cc.find("ext_present_id2_enabled") == std::string::npos);
  W7_1C_CHECK(dev_cc.find("CanEnablePresentTimingExtOnDeviceV3") !=
              std::string::npos);
  W7_1C_CHECK(dev_cc.find("erase_enabled_extension(VK_KHR_PRESENT_ID_2_EXTENSION_NAME)") !=
              std::string::npos);
  W7_1C_CHECK(pres_cc.find("std::call_once") != std::string::npos);
  W7_1C_CHECK(pres_cc.find("ResolveNativePresentTargetTimeNs") !=
              std::string::npos);
  W7_1C_CHECK(pres_cc.find("/*target_is_vsync_quantized=*/false") !=
              std::string::npos);
  W7_1C_CHECK(pres_cc.find("MatchAndRetireNativeLatencyRecord") !=
              std::string::npos);
  W7_1C_CHECK(pres_cc.find("ShouldEnqueueTimingRecord") != std::string::npos);
  W7_1C_CHECK(pres_cc.find("IsCompleteMonotonicPastPresentationTiming") !=
              std::string::npos);
  W7_1C_CHECK(pres_cc.find("VK_PAST_PRESENTATION_TIMING_ALLOW_PARTIAL_RESULTS_BIT_EXT") ==
              std::string::npos);
  W7_1C_CHECK(pres_h.find("uint64_t native_latency_present_id_") !=
              std::string::npos);
  W7_1C_CHECK(zfg_h.find("uint64_t present_id = 0;") != std::string::npos);
  W7_1C_CHECK(zfg_cc.find("CanUseExtPresentTimingForZeroFG") !=
              std::string::npos);
  W7_1C_CHECK(zfg_cc.find("ShouldEnqueueTimingRecord") != std::string::npos);
  W7_1C_CHECK(zfg_cc.find("IsCompleteMonotonicPastPresentationTiming") !=
              std::string::npos);
  W7_1C_CHECK(zfg_cc.find("VK_PAST_PRESENTATION_TIMING_ALLOW_PARTIAL_RESULTS_BIT_EXT") ==
              std::string::npos);
  W7_1C_CHECK(zfg_cc.find("IsOlderPresentId") != std::string::npos);

  // 3) C++ <-> Kotlin <-> TOML parity & strings contracts.
  // Note: audio_adpf_logic_test.cc:431-437 (Onda 1.1 #55) explicitly forbids
  // UPDATE_from_bool(apu_performance_hint, ...) so user TOML opt-ins stay reversible.
  W7_1C_CHECK(audio_cc.find("apu_performance_hint, false,") !=
              std::string::npos);
  W7_1C_CHECK(audio_cc.find("UPDATE_from_bool(apu_performance_hint") ==
              std::string::npos);
  W7_1C_CHECK(toml.find("apu_performance_hint = false") != std::string::npos);
  W7_1C_CHECK(toml.find("a64_wx_dual_mapping = true") != std::string::npos);
  W7_1C_CHECK(toml.find("writable_executable_memory = true") !=
              std::string::npos);
  W7_1C_CHECK(schema_kt.find("\"apu_performance_hint\", \"Audio performance hint (ADPF)\", false") !=
              std::string::npos);
  W7_1C_CHECK(strings_en.find("audio worker and XMA decoder threads") !=
              std::string::npos);
  W7_1C_CHECK(strings_pt.find("Audio Worker e decodificador XMA") !=
              std::string::npos);

  // 4) SystemHeadroomReader & watchThermalHeadroom cleanup contracts.
  W7_1C_CHECK(headroom_kt.find("resolveDefaultParamsOnce") != std::string::npos);
  W7_1C_CHECK(headroom_kt.find("cpuParamsResolved") != std::string::npos);
  W7_1C_CHECK(host_kt.find("GPU|vulkan_auto_vrs_thermal = true") !=
              std::string::npos);
  W7_1C_CHECK(host_kt.find("recordSystemHeadroom(null, null)") !=
              std::string::npos);
}

}  // namespace

int main(int argc, char** argv) {
  TestRisk1_DevicePresentId2StrictGate();
  TestRisk2And4_SwapchainPreFlightGates();
  TestRisk3_ClockMonotonicDomainSeparation();
  TestRisk4_NativeLateSwapTargetResolution();
  TestNativeLatencyRingRetirement();
  TestRisk5_QueueFullRetryNoTimingRecord();
  TestRisk6_CompleteReportAnd64BitPresentId();
  TestAndroid17BtiAnd16KbGeometry();
  if (argc >= 2 && argv[1] && argv[1][0] != '\0') {
    TestSourceContracts(fs::path(argv[1]));
  }
  std::printf("wave7_onda1c_platform_present_test: PASS\n");
  return 0;
}
