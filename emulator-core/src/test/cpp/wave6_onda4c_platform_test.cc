#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#ifndef XE_DISABLE_ANDROID_DLOPEN
#define XE_DISABLE_ANDROID_DLOPEN 1
#endif

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES 1
#endif

#include "third_party/Vulkan-Headers/include/vulkan/vulkan_core.h"
#include "xenia/base/android_thermal.h"
#include "xenia/gpu/vulkan/thermal_vrs_controller.h"
#include "xenia/ui/vulkan/vulkan_present_timing.h"

namespace {

#define ONDA4C_CHECK(cond)                                                 \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "ONDA4C check failed (%s:%d): %s\n", __FILE__,  \
                   __LINE__, #cond);                                       \
      std::abort();                                                        \
    }                                                                      \
  } while (false)

namespace fs = std::filesystem;
namespace vkui = xe::ui::vulkan;
using xe::gpu::vulkan::ThermalVrsController;

std::string ReadSourceText(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  ONDA4C_CHECK(in.good());
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

void TestSelectPresentTimingBackend() {
  using vkui::PresentTimingBackend;
  using vkui::SelectPresentTimingBackend;

  // All 4 EXT requirements + cvar true -> prefer VK_EXT_present_timing.
  ONDA4C_CHECK(SelectPresentTimingBackend(true, true, true, true, true) ==
               PresentTimingBackend::kExtPresentTiming);
  ONDA4C_CHECK(SelectPresentTimingBackend(true, true, true, true, false) ==
               PresentTimingBackend::kExtPresentTiming);

  // Missing any single EXT requirement falls back cleanly to GOOGLE.
  ONDA4C_CHECK(SelectPresentTimingBackend(false, true, true, true, true) ==
               PresentTimingBackend::kGoogleDisplayTiming);
  ONDA4C_CHECK(SelectPresentTimingBackend(true, false, true, true, true) ==
               PresentTimingBackend::kGoogleDisplayTiming);
  ONDA4C_CHECK(SelectPresentTimingBackend(true, true, false, true, true) ==
               PresentTimingBackend::kGoogleDisplayTiming);
  ONDA4C_CHECK(SelectPresentTimingBackend(true, true, true, false, true) ==
               PresentTimingBackend::kGoogleDisplayTiming);

  // Missing GOOGLE as well falls back to kNone (untimed).
  ONDA4C_CHECK(SelectPresentTimingBackend(false, true, true, true, false) ==
               PresentTimingBackend::kNone);
  ONDA4C_CHECK(SelectPresentTimingBackend(true, false, true, true, false) ==
               PresentTimingBackend::kNone);
  ONDA4C_CHECK(SelectPresentTimingBackend(true, true, false, true, false) ==
               PresentTimingBackend::kNone);
  ONDA4C_CHECK(SelectPresentTimingBackend(true, true, true, false, false) ==
               PresentTimingBackend::kNone);
  ONDA4C_CHECK(SelectPresentTimingBackend(false, false, false, false, false) ==
               PresentTimingBackend::kNone);

  // VUID device dependency gate (Fix #4): requires calibrated_timestamps,
  // present_id2 extension, presentTiming feature, and (presentId2 || presentId).
  using vkui::CanEnablePresentTimingExtOnDevice;
  ONDA4C_CHECK(CanEnablePresentTimingExtOnDevice(true, true, true, true, true,
                                                 false));
  ONDA4C_CHECK(CanEnablePresentTimingExtOnDevice(true, true, true, true, false,
                                                 true));
  ONDA4C_CHECK(!CanEnablePresentTimingExtOnDevice(false, true, true, true, true,
                                                  true));
  ONDA4C_CHECK(!CanEnablePresentTimingExtOnDevice(true, false, true, true, true,
                                                  true));
  ONDA4C_CHECK(!CanEnablePresentTimingExtOnDevice(true, true, false, true, true,
                                                  true));
  ONDA4C_CHECK(!CanEnablePresentTimingExtOnDevice(true, true, true, false, true,
                                                  true));
  ONDA4C_CHECK(!CanEnablePresentTimingExtOnDevice(true, true, true, true, false,
                                                  false));
}

void TestTimeDomainAndPastTimingConversion() {
  using vkui::ConvertPastPresentationTiming;
  using vkui::ConvertVkPastPresentationTimingEXT;
  using vkui::PresentStageSample;
  using vkui::PresentTimingBackend;
  using vkui::PresentInfoTimingChain;
  using vkui::SelectPreferredPresentStageQueryMask;
  using vkui::SelectPreferredTimeDomain;
  using vkui::SelectPresentInfoTimingChain;
  using vkui::ShouldRefreshSwapchainTimingMetadata;

  // Empty / null domain list -> invalid.
  ONDA4C_CHECK(!SelectPreferredTimeDomain(nullptr, nullptr, 0).valid);
  const uint32_t empty_dom[1] = {vkui::kTimeDomainDevice};
  const uint64_t empty_id[1] = {1};
  ONDA4C_CHECK(!SelectPreferredTimeDomain(empty_dom, empty_id, 0).valid);

  // Prefers CLOCK_MONOTONIC_KHR over RAW, PRESENT_STAGE_LOCAL, SWAPCHAIN_LOCAL.
  const uint32_t domains_all[] = {
      vkui::kTimeDomainDevice,
      vkui::kTimeDomainSwapchainLocal,
      vkui::kTimeDomainPresentStageLocal,
      vkui::kTimeDomainClockMonotonicRaw,
      vkui::kTimeDomainClockMonotonic,
  };
  const uint64_t ids_all[] = {10, 20, 30, 40, 50};
  const auto picked_mono = SelectPreferredTimeDomain(domains_all, ids_all, 5);
  ONDA4C_CHECK(picked_mono.valid);
  ONDA4C_CHECK(picked_mono.time_domain == vkui::kTimeDomainClockMonotonic);
  ONDA4C_CHECK(picked_mono.time_domain_id == 50);
  ONDA4C_CHECK(picked_mono.is_monotonic_clock);

  // Without CLOCK_MONOTONIC, prefers CLOCK_MONOTONIC_RAW, then
  // PRESENT_STAGE_LOCAL_EXT.
  const uint32_t domains_no_mono[] = {
      vkui::kTimeDomainDevice,
      vkui::kTimeDomainPresentStageLocal,
      vkui::kTimeDomainClockMonotonicRaw,
  };
  const uint64_t ids_no_mono[] = {11, 22, 33};
  const auto picked_raw =
      SelectPreferredTimeDomain(domains_no_mono, ids_no_mono, 3);
  ONDA4C_CHECK(picked_raw.valid);
  ONDA4C_CHECK(picked_raw.time_domain == vkui::kTimeDomainClockMonotonicRaw);
  ONDA4C_CHECK(picked_raw.time_domain_id == 33);
  ONDA4C_CHECK(picked_raw.is_monotonic_clock);

  const uint32_t domains_local[] = {
      vkui::kTimeDomainDevice,
      vkui::kTimeDomainSwapchainLocal,
      vkui::kTimeDomainPresentStageLocal,
  };
  const uint64_t ids_local[] = {1, 2, 3};
  const auto picked_local =
      SelectPreferredTimeDomain(domains_local, ids_local, 3);
  ONDA4C_CHECK(picked_local.valid);
  ONDA4C_CHECK(picked_local.time_domain == vkui::kTimeDomainPresentStageLocal);
  ONDA4C_CHECK(picked_local.time_domain_id == 3);
  ONDA4C_CHECK(!picked_local.is_monotonic_clock);

  // Stage query mask selection.
  ONDA4C_CHECK(
      SelectPreferredPresentStageQueryMask(
          vkui::kPresentStageQueueOpsEnd | vkui::kPresentStageFirstPixelOut |
          vkui::kPresentStageFirstPixelVisible) ==
      (vkui::kPresentStageFirstPixelOut |
       vkui::kPresentStageFirstPixelVisible));
  ONDA4C_CHECK(SelectPreferredPresentStageQueryMask(
                   vkui::kPresentStageFirstPixelOut) ==
               vkui::kPresentStageFirstPixelOut);
  ONDA4C_CHECK(SelectPreferredPresentStageQueryMask(
                   vkui::kPresentStageRequestDequeued) ==
               vkui::kPresentStageRequestDequeued);
  ONDA4C_CHECK(SelectPreferredPresentStageQueryMask(0) == 0);

  // Target stage selection picks a single bit matching priority.
  ONDA4C_CHECK(
      vkui::SelectTargetTimePresentStage(
          vkui::kPresentStageFirstPixelOut |
          vkui::kPresentStageFirstPixelVisible) ==
      vkui::kPresentStageFirstPixelVisible);
  ONDA4C_CHECK(
      vkui::SelectTargetTimePresentStage(vkui::kPresentStageFirstPixelOut) ==
      vkui::kPresentStageFirstPixelOut);
  ONDA4C_CHECK(vkui::SelectTargetTimePresentStage(0) == 0);

  // Pure conversion prefers FIRST_PIXEL_VISIBLE over FIRST_PIXEL_OUT.
  const PresentStageSample samples[] = {
      {vkui::kPresentStageQueueOpsEnd, 100'000'000ull},
      {vkui::kPresentStageFirstPixelOut, 108'000'000ull},
      {vkui::kPresentStageFirstPixelVisible, 110'000'000ull},
  };
  const auto conv = ConvertPastPresentationTiming(
      42, 109'000'000ull, samples, 3, true, 16'666'667ull);
  ONDA4C_CHECK(conv.valid);
  ONDA4C_CHECK(conv.present_id == 42);
  ONDA4C_CHECK(conv.target_time_ns == 109'000'000ull);
  ONDA4C_CHECK(conv.actual_present_time_ns == 110'000'000ull);
  ONDA4C_CHECK(conv.earliest_present_time_ns == 108'000'000ull);
  ONDA4C_CHECK(conv.refresh_duration_ns == 16'666'667ull);
  ONDA4C_CHECK(conv.chosen_stage == vkui::kPresentStageFirstPixelVisible);
  ONDA4C_CHECK(conv.report_complete);

  // Zero timestamp for FIRST_PIXEL_VISIBLE falls back to FIRST_PIXEL_OUT.
  const PresentStageSample partial_samples[] = {
      {vkui::kPresentStageFirstPixelVisible, 0ull},
      {vkui::kPresentStageFirstPixelOut, 205'000'000ull},
  };
  const auto conv_partial = ConvertPastPresentationTiming(
      7, 200'000'000ull, partial_samples, 2, false, 8'333'333ull);
  ONDA4C_CHECK(conv_partial.valid);
  ONDA4C_CHECK(conv_partial.actual_present_time_ns == 205'000'000ull);
  ONDA4C_CHECK(conv_partial.chosen_stage == vkui::kPresentStageFirstPixelOut);

  // Real Vulkan struct adapter (#ifdef VK_EXT_present_timing).
#ifdef VK_EXT_present_timing
  VkPresentStageTimeEXT vk_stages[2] = {
      {VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_OUT_BIT_EXT, 500'000'000ull},
      {VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_VISIBLE_BIT_EXT, 502'000'000ull},
  };
  VkPastPresentationTimingEXT vk_past = {};
  vk_past.sType = VK_STRUCTURE_TYPE_PAST_PRESENTATION_TIMING_EXT;
  vk_past.presentId = 99;
  vk_past.targetTime = 501'000'000ull;
  vk_past.presentStageCount = 2;
  vk_past.pPresentStages = vk_stages;
  vk_past.timeDomain = VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR;
  vk_past.timeDomainId = 50;
  vk_past.reportComplete = VK_TRUE;

  const auto vk_conv = ConvertVkPastPresentationTimingEXT(vk_past, 8'333'333ull);
  ONDA4C_CHECK(vk_conv.valid);
  ONDA4C_CHECK(vk_conv.present_id == 99);
  ONDA4C_CHECK(vk_conv.actual_present_time_ns == 502'000'000ull);
  ONDA4C_CHECK(vk_conv.earliest_present_time_ns == 500'000'000ull);
  ONDA4C_CHECK(vk_conv.refresh_duration_ns == 8'333'333ull);
  ONDA4C_CHECK(vk_conv.report_complete);
#endif

  // Mutual exclusion for VkPresentInfoKHR::pNext.
  ONDA4C_CHECK(
      SelectPresentInfoTimingChain(PresentTimingBackend::kExtPresentTiming,
                                   true) == PresentInfoTimingChain::kExtOnly);
  ONDA4C_CHECK(
      SelectPresentInfoTimingChain(PresentTimingBackend::kGoogleDisplayTiming,
                                   true) ==
      PresentInfoTimingChain::kGoogleOnly);
  ONDA4C_CHECK(SelectPresentInfoTimingChain(PresentTimingBackend::kNone,
                                            true) ==
               PresentInfoTimingChain::kNone);
  ONDA4C_CHECK(
      SelectPresentInfoTimingChain(PresentTimingBackend::kExtPresentTiming,
                                   false) == PresentInfoTimingChain::kNone);
  ONDA4C_CHECK(
      SelectPresentInfoTimingChain(PresentTimingBackend::kGoogleDisplayTiming,
                                   false) == PresentInfoTimingChain::kNone);

  // Counter / queue-full refresh trigger.
  ONDA4C_CHECK(!ShouldRefreshSwapchainTimingMetadata(5, 5, 2, 2, false));
  ONDA4C_CHECK(ShouldRefreshSwapchainTimingMetadata(5, 6, 2, 2, false));
  ONDA4C_CHECK(ShouldRefreshSwapchainTimingMetadata(5, 5, 2, 3, false));
  ONDA4C_CHECK(ShouldRefreshSwapchainTimingMetadata(5, 5, 2, 2, true));
}

void TestStrictBoolPropertyParsing() {
  bool v = true;
  ONDA4C_CHECK(vkui::ParseStrictBoolProperty("0", v));
  ONDA4C_CHECK(v == false);
  ONDA4C_CHECK(vkui::ParseStrictBoolProperty("1", v));
  ONDA4C_CHECK(v == true);

  // Invalid inputs must return false and leave out value unmodified.
  v = false;
  ONDA4C_CHECK(!vkui::ParseStrictBoolProperty(nullptr, v));
  ONDA4C_CHECK(!vkui::ParseStrictBoolProperty("", v));
  ONDA4C_CHECK(!vkui::ParseStrictBoolProperty("2", v));
  ONDA4C_CHECK(!vkui::ParseStrictBoolProperty("true", v));
  ONDA4C_CHECK(!vkui::ParseStrictBoolProperty("false", v));
  ONDA4C_CHECK(!vkui::ParseStrictBoolProperty("10", v));
  ONDA4C_CHECK(!vkui::ParseStrictBoolProperty("01", v));
  ONDA4C_CHECK(v == false);
}

void TestLateSwapPolicyAndZeroFGBypass() {
  using vkui::ComputeNextThrottleAnchorTicks;
  using vkui::EvaluateLateSwapPresent;
  using vkui::LateSwapAction;

  constexpr uint64_t kRefresh60 = 16'666'667ull;
  constexpr uint64_t kTarget = 1'000'000'000ull;

  // Absolute ZeroFG safety rule: zerofg_active == true ALWAYS bypasses.
  const uint64_t now_cases[] = {
      kTarget - uint64_t{5'000'000},
      kTarget,
      kTarget + uint64_t{3'000'000},
      kTarget + uint64_t{25'000'000},
  };
  for (uint64_t now : now_cases) {
    const auto d =
        EvaluateLateSwapPresent(now, kTarget, kRefresh60,
                                /*zerofg_active=*/true,
                                /*late_swap_enabled=*/true);
    ONDA4C_CHECK(d.action == LateSwapAction::kBypassKeepPaced);
    ONDA4C_CHECK(d.effective_target_ns == kTarget);
  }

  // Disabled toggle -> bypass.
  {
    const auto d = EvaluateLateSwapPresent(
        kTarget + 2'000'000ull, kTarget, kRefresh60, false, false);
    ONDA4C_CHECK(d.action == LateSwapAction::kBypassKeepPaced);
    ONDA4C_CHECK(d.effective_target_ns == kTarget);
  }

  // Native early frame -> keep scheduled target.
  {
    const auto d = EvaluateLateSwapPresent(
        kTarget - 4'000'000ull, kTarget, kRefresh60, false, true);
    ONDA4C_CHECK(d.action == LateSwapAction::kKeepScheduledTarget);
    ONDA4C_CHECK(d.effective_target_ns == kTarget);
  }

  // Native late frame within half-refresh window -> present immediate (0).
  {
    const auto d = EvaluateLateSwapPresent(
        kTarget + 3'000'000ull, kTarget, kRefresh60, false, true);
    ONDA4C_CHECK(d.action == LateSwapAction::kPresentImmediate);
    ONDA4C_CHECK(d.effective_target_ns == 0ull);
  }

  // Native frame at exact target or heavily overdue -> present immediate (0).
  {
    const auto d_exact =
        EvaluateLateSwapPresent(kTarget, kTarget, kRefresh60, false, true);
    ONDA4C_CHECK(d_exact.action == LateSwapAction::kPresentImmediate);
    ONDA4C_CHECK(d_exact.effective_target_ns == 0ull);

    const auto d_very_late = EvaluateLateSwapPresent(
        kTarget + 25'000'000ull, kTarget, kRefresh60, false, true);
    ONDA4C_CHECK(d_very_late.action == LateSwapAction::kPresentImmediate);
    ONDA4C_CHECK(d_very_late.effective_target_ns == 0ull);
  }

  // CommandProcessor::ThrottlePresentation carry-over policy.
  constexpr uint64_t kDur = 1000ull;
  constexpr uint64_t kLast = 10000ull;

  // 1) Small lateness (0.2x) <= 0.5x: advances anchor normally.
  ONDA4C_CHECK(ComputeNextThrottleAnchorTicks(kLast, 11200ull, kDur, false,
                                              true) == 11000ull);
  // 2) Moderate lateness (0.8x): native late-swap caps carry-over at 0.5x (500).
  const uint64_t capped =
      ComputeNextThrottleAnchorTicks(kLast, 11800ull, kDur, false, true);
  ONDA4C_CHECK(capped == 11300ull);
  ONDA4C_CHECK(11800ull - capped == kDur / 2);
  // Under ZeroFG (or late-swap disabled), legacy anchor is preserved (11000).
  ONDA4C_CHECK(ComputeNextThrottleAnchorTicks(kLast, 11800ull, kDur, true,
                                              true) == 11000ull);
  ONDA4C_CHECK(ComputeNextThrottleAnchorTicks(kLast, 11800ull, kDur, false,
                                              false) == 11000ull);
  // 3) Stall > 2x target duration: resets cleanly to current time.
  ONDA4C_CHECK(ComputeNextThrottleAnchorTicks(kLast, 12500ull, kDur, false,
                                              true) == 12500ull);
  ONDA4C_CHECK(ComputeNextThrottleAnchorTicks(kLast, 12500ull, kDur, true,
                                              true) == 12500ull);
}

void TestAndroidHeadroomAtomicsAndTtl() {
  xe::ResetAndroidComputeHeadroom();
  ONDA4C_CHECK(!xe::IsAndroidComputeHeadroomEnabled());

  // Empty initially.
  {
    const xe::ThermalSample s = xe::ReadAndroidThermalSampleAt(10, 5000);
    ONDA4C_CHECK(!s.has_cpu_headroom);
    ONDA4C_CHECK(!s.has_gpu_headroom);
    ONDA4C_CHECK(s.cpu_headroom < 0.0f);
    ONDA4C_CHECK(s.gpu_headroom < 0.0f);
  }

  // Out-of-range and NaN values are filtered cleanly.
  xe::SetAndroidComputeHeadroom(std::numeric_limits<float>::quiet_NaN(), 101.0f,
                                5000);
  {
    const xe::ThermalSample s = xe::ReadAndroidThermalSampleAt(10, 5100);
    ONDA4C_CHECK(!s.has_cpu_headroom);
    ONDA4C_CHECK(!s.has_gpu_headroom);
  }

  // Valid values within [0, 100] are visible up to TTL = 10000 ms inclusive.
  xe::SetAndroidComputeHeadroom(42.5f, 12.0f, 10000);
  {
    const xe::ThermalSample s0 = xe::ReadAndroidThermalSampleAt(10, 10000);
    ONDA4C_CHECK(s0.has_cpu_headroom && s0.cpu_headroom == 42.5f);
    ONDA4C_CHECK(s0.has_gpu_headroom && s0.gpu_headroom == 12.0f);

    const xe::ThermalSample s_edge = xe::ReadAndroidThermalSampleAt(10, 20000);
    ONDA4C_CHECK(s_edge.has_cpu_headroom && s_edge.cpu_headroom == 42.5f);
    ONDA4C_CHECK(s_edge.has_gpu_headroom && s_edge.gpu_headroom == 12.0f);

    // Expired at age = 10001 ms.
    const xe::ThermalSample s_expired =
        xe::ReadAndroidThermalSampleAt(10, 20001);
    ONDA4C_CHECK(!s_expired.has_cpu_headroom);
    ONDA4C_CHECK(!s_expired.has_gpu_headroom);

    // Clock regression (< sample timestamp) must not treat sample as valid.
    const xe::ThermalSample s_regressed =
        xe::ReadAndroidThermalSampleAt(10, 9999);
    ONDA4C_CHECK(!s_regressed.has_cpu_headroom);
    ONDA4C_CHECK(!s_regressed.has_gpu_headroom);
  }

  // Partial availability (GPU valid, CPU unavailable).
  xe::SetAndroidComputeHeadroom(-1.0f, 0.0f, 30000);
  {
    const xe::ThermalSample s = xe::ReadAndroidThermalSampleAt(10, 31000);
    ONDA4C_CHECK(!s.has_cpu_headroom);
    ONDA4C_CHECK(s.has_gpu_headroom && s.gpu_headroom == 0.0f);
  }

  xe::ResetAndroidComputeHeadroom();
}

void TestThermalVrsWithSystemHeadroom() {
  ThermalVrsController controller;

  // Helper to build a ThermalSample with compute headroom.
  const auto make_sample = [](int status, float thermal_hr, float gpu_hr,
                              bool has_gpu = true) {
    xe::ThermalSample s;
    s.has_status = status >= xe::kThermalStatusNone;
    s.status = s.has_status ? status : xe::kThermalStatusUnavailable;
    s.has_headroom = thermal_hr >= 0.0f;
    s.headroom = s.has_headroom ? thermal_hr : -1.0f;
    s.has_gpu_headroom = has_gpu;
    s.gpu_headroom = has_gpu ? gpu_hr : -1.0f;
    return s;
  };

  // 1) Toggle OFF (default): low gpu_headroom is ignored completely.
  {
    const auto warm_low_gpu = make_sample(xe::kThermalStatusLight, 0.50f, 4.0f);
    controller.Update(1000, warm_low_gpu, /*enable_compute_headroom=*/false);
    controller.Update(3000, warm_low_gpu, /*enable_compute_headroom=*/false);
    controller.Update(5000, warm_low_gpu, /*enable_compute_headroom=*/false);
    ONDA4C_CHECK(controller.current_rate() == 0u);
  }

  // 2) Cold phone + gpu_headroom = 12% MUST NEVER trigger VRS!
  controller.Reset();
  {
    const auto cold_heavy_gpu =
        make_sample(xe::kThermalStatusNone, 0.25f, 12.0f);
    for (uint64_t t = 1000; t <= 15000; t += 2000) {
      controller.Update(t, cold_heavy_gpu, /*enable_compute_headroom=*/true);
      ONDA4C_CHECK(controller.current_rate() == 0u);
    }
  }

  // 3) Cold phone + single transient gpu_headroom <= 8% spike does NOT trigger.
  controller.Reset();
  {
    const auto cold_spike = make_sample(xe::kThermalStatusNone, 0.20f, 6.0f);
    const auto cold_ok = make_sample(xe::kThermalStatusNone, 0.20f, 40.0f);
    controller.Update(1000, cold_spike, true);
    ONDA4C_CHECK(controller.current_rate() == 0u);
    controller.Update(3000, cold_ok, true);
    ONDA4C_CHECK(controller.current_rate() == 0u);
    controller.Update(5000, cold_spike, true);
    ONDA4C_CHECK(controller.current_rate() == 0u);
  }

  // 4) Cold phone + sustained gpu_headroom <= 8% (2+ consecutive samples +
  //    dwell) escalates to 2x1 ONLY (never 2x2 on a cold phone!).
  controller.Reset();
  {
    const auto cold_extreme = make_sample(xe::kThermalStatusNone, 0.20f, 4.0f);
    controller.Update(1000, cold_extreme, true);  // streak 1
    ONDA4C_CHECK(controller.current_rate() == 0u);
    controller.Update(3000, cold_extreme, true);  // streak 2 -> starts dwell
    ONDA4C_CHECK(controller.current_rate() == 0u);
    controller.Update(4000, cold_extreme, true);  // dwell 1000ms -> 2x1
    ONDA4C_CHECK(controller.current_rate() == 1u);
    // Even if kept at 4% on a cold phone, NEVER escalates to 2x2!
    controller.Update(6000, cold_extreme, true);
    controller.Update(8000, cold_extreme, true);
    ONDA4C_CHECK(controller.current_rate() == 1u);
  }

  // 5) Warm phone (Light) + gpu_headroom = 12% -> escalates to 2x1 (1u),
  //    then Moderate + gpu_headroom = 4% -> escalates to 2x2 (3u),
  //    then hysteresis cooldown:
  //    - 2x2 -> 2x1 blocked at gpu_headroom = 15%, allowed at >= 18% after 4s;
  //    - 2x1 -> 1x1 blocked at gpu_headroom = 20%, allowed at >= 25% after 4s.
  controller.Reset();
  {
    const auto light_12 = make_sample(xe::kThermalStatusLight, 0.50f, 12.0f);
    controller.Update(10000, light_12, true);
    ONDA4C_CHECK(controller.current_rate() == 0u);
    controller.Update(11000, light_12, true);
    ONDA4C_CHECK(controller.current_rate() == 1u);

    // Moderate + gpu_headroom = 4% -> 2x2 after dwell.
    const auto mod_4 = make_sample(xe::kThermalStatusModerate, 0.80f, 4.0f);
    controller.Update(12000, mod_4, true);
    ONDA4C_CHECK(controller.current_rate() == 1u);
    controller.Update(13000, mod_4, true);
    ONDA4C_CHECK(controller.current_rate() == 3u);

    // Deadband at 2x2: gpu_headroom = 15% (< 18%) prevents cooldown.
    const auto mod_15 = make_sample(xe::kThermalStatusModerate, 0.75f, 15.0f);
    controller.Update(14000, mod_15, true);
    controller.Update(20000, mod_15, true);
    ONDA4C_CHECK(controller.current_rate() == 3u);

    // Recovery to gpu_headroom = 18% cools 2x2 -> 2x1 after 4000ms.
    const auto mod_18 = make_sample(xe::kThermalStatusModerate, 0.75f, 18.0f);
    controller.Update(21000, mod_18, true);
    ONDA4C_CHECK(controller.current_rate() == 3u);
    controller.Update(24999, mod_18, true);
    ONDA4C_CHECK(controller.current_rate() == 3u);
    controller.Update(25000, mod_18, true);
    ONDA4C_CHECK(controller.current_rate() == 1u);

    // Deadband at 2x1: gpu_headroom = 20% (< 25%) prevents cooldown to 1x1.
    const auto light_20 = make_sample(xe::kThermalStatusLight, 0.60f, 20.0f);
    controller.Update(26000, light_20, true);
    controller.Update(32000, light_20, true);
    ONDA4C_CHECK(controller.current_rate() == 1u);

    // Recovery to gpu_headroom = 25% cools 2x1 -> 1x1 after 4000ms.
    const auto light_25 = make_sample(xe::kThermalStatusLight, 0.60f, 25.0f);
    controller.Update(33000, light_25, true);
    ONDA4C_CHECK(controller.current_rate() == 1u);
    controller.Update(36999, light_25, true);
    ONDA4C_CHECK(controller.current_rate() == 1u);
    controller.Update(37000, light_25, true);
    ONDA4C_CHECK(controller.current_rate() == 0u);
  }

  // 6) Thermal headroom baseline (None + thermal_headroom >= 0.70) also
  //    qualifies gpu_headroom <= 15% for 2x1; >= 0.85 qualifies <= 5% for 2x2.
  controller.Reset();
  {
    const auto hr70_gpu14 = make_sample(xe::kThermalStatusNone, 0.72f, 14.0f);
    controller.Update(40000, hr70_gpu14, true);
    controller.Update(41000, hr70_gpu14, true);
    ONDA4C_CHECK(controller.current_rate() == 1u);

    const auto hr85_gpu5 = make_sample(xe::kThermalStatusNone, 0.86f, 5.0f);
    controller.Update(42000, hr85_gpu5, true);
    controller.Update(43000, hr85_gpu5, true);
    ONDA4C_CHECK(controller.current_rate() == 3u);

    // TTL expiry (has_gpu_headroom == false) + cool thermal allows recovery.
    const auto expired_cool =
        make_sample(xe::kThermalStatusLight, 0.50f, -1.0f, /*has_gpu=*/false);
    controller.Update(44000, expired_cool, true);
    controller.Update(48000, expired_cool, true);
    ONDA4C_CHECK(controller.current_rate() == 1u);
    controller.Update(49000, expired_cool, true);
    controller.Update(53000, expired_cool, true);
    ONDA4C_CHECK(controller.current_rate() == 0u);
  }
}

void TestSourceContracts(const fs::path& cpp_dir) {
  const std::string dev_h =
      ReadSourceText(cpp_dir / "xenia/src/xenia/ui/vulkan/vulkan_device.h");
  const std::string dev_cc =
      ReadSourceText(cpp_dir / "xenia/src/xenia/ui/vulkan/vulkan_device.cc");
  const std::string pres_cc =
      ReadSourceText(cpp_dir / "xenia/src/xenia/ui/vulkan/vulkan_presenter.cc");
  const std::string zfg_cc = ReadSourceText(
      cpp_dir / "xenia/src/xenia/ui/vulkan/zerofg_main_surface_egress.cc");
  const std::string cp_cc =
      ReadSourceText(cpp_dir / "xenia/src/xenia/gpu/command_processor.cc");
  const std::string jni_cpp =
      ReadSourceText(cpp_dir / "emulator_xendroid.cpp");

  // 1) All 3 new cvars default to false and have zero UPDATE_from_*.
  ONDA4C_CHECK(dev_cc.find("vulkan_present_timing_ext, false,") !=
               std::string::npos);
  ONDA4C_CHECK(dev_cc.find("UPDATE_from_bool(vulkan_present_timing_ext") ==
               std::string::npos);
  ONDA4C_CHECK(pres_cc.find("vulkan_late_swap_present, false,") !=
               std::string::npos);
  ONDA4C_CHECK(pres_cc.find("UPDATE_from_bool(vulkan_late_swap_present") ==
               std::string::npos);
  ONDA4C_CHECK(pres_cc.find("vulkan_auto_vrs_headroom, false,") !=
               std::string::npos);
  ONDA4C_CHECK(pres_cc.find("UPDATE_from_bool(vulkan_auto_vrs_headroom") ==
               std::string::npos);

  // 2) Live debug.xendroid.* properties wired with strict parsing.
  ONDA4C_CHECK(dev_cc.find("debug.xendroid.present_timing_ext") !=
               std::string::npos);
  ONDA4C_CHECK(pres_cc.find("debug.xendroid.late_swap") != std::string::npos);
  ONDA4C_CHECK(cp_cc.find("debug.xendroid.late_swap") != std::string::npos);
  ONDA4C_CHECK(jni_cpp.find("debug.xendroid.auto_vrs_headroom") !=
               std::string::npos);

  // 3) VK_EXT_present_timing negotiation & 4 PFNs guarded by #ifdef.
  ONDA4C_CHECK(dev_h.find("#ifdef VK_EXT_present_timing") != std::string::npos);
  ONDA4C_CHECK(dev_h.find("vkSetSwapchainPresentTimingQueueSizeEXT") !=
               std::string::npos);
  ONDA4C_CHECK(dev_h.find("vkGetSwapchainTimingPropertiesEXT") !=
               std::string::npos);
  ONDA4C_CHECK(dev_h.find("vkGetSwapchainTimeDomainPropertiesEXT") !=
               std::string::npos);
  ONDA4C_CHECK(dev_h.find("vkGetPastPresentationTimingEXT") !=
               std::string::npos);
  ONDA4C_CHECK(dev_cc.find("VK_EXT_PRESENT_TIMING_EXTENSION_NAME") !=
               std::string::npos);
  ONDA4C_CHECK(dev_cc.find("VK_KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME") !=
               std::string::npos);
  ONDA4C_CHECK(dev_h.find("ext_KHR_calibrated_timestamps") !=
               std::string::npos);
  ONDA4C_CHECK(dev_cc.find("CanEnablePresentTimingExtOnDevice") !=
               std::string::npos);
  ONDA4C_CHECK(dev_cc.find("erase_enabled_extension(VK_EXT_PRESENT_TIMING_EXTENSION_NAME)") !=
               std::string::npos);
  ONDA4C_CHECK(zfg_cc.find("(!ext_use_present_id2_ && !ext_use_present_id1_)") !=
               std::string::npos);
  ONDA4C_CHECK(dev_cc.find("VkPhysicalDevicePresentTimingFeaturesEXT") !=
               std::string::npos);

  // 4) Swapchain & present mutual exclusion in Presenter and ZeroFG egress.
  ONDA4C_CHECK(pres_cc.find("SelectPresentTimingBackend") != std::string::npos);
  ONDA4C_CHECK(pres_cc.find("SelectPresentInfoTimingChain") !=
               std::string::npos);
  ONDA4C_CHECK(pres_cc.find("VK_SWAPCHAIN_CREATE_PRESENT_TIMING_BIT_EXT") !=
               std::string::npos);
  ONDA4C_CHECK(zfg_cc.find("SelectPresentTimingBackend") != std::string::npos);
  ONDA4C_CHECK(zfg_cc.find("SelectPresentInfoTimingChain") !=
               std::string::npos);
  ONDA4C_CHECK(zfg_cc.find("VK_SWAPCHAIN_CREATE_PRESENT_TIMING_BIT_EXT") !=
               std::string::npos);

  // 5) Late-swap is explicitly disabled under ZeroFG!
  ONDA4C_CHECK(zfg_cc.find("EvaluateLateSwapPresent") == std::string::npos);
  ONDA4C_CHECK(pres_cc.find("EvaluateLateSwapPresent") != std::string::npos);
  ONDA4C_CHECK(cp_cc.find("ComputeNextThrottleAnchorTicks") !=
               std::string::npos);
  ONDA4C_CHECK(cp_cc.find("IsZeroFGMainSurfaceActive") != std::string::npos);

  // 6) Android 16 SystemHealthManager Headroom JNI bridge.
  ONDA4C_CHECK(jni_cpp.find("\"set_system_headroom\", \"(FF)V\"") !=
               std::string::npos);
  ONDA4C_CHECK(jni_cpp.find("SetAndroidComputeHeadroom") != std::string::npos);
}

}  // namespace

int main(int argc, char** argv) {
  TestSelectPresentTimingBackend();
  TestTimeDomainAndPastTimingConversion();
  TestStrictBoolPropertyParsing();
  TestLateSwapPolicyAndZeroFGBypass();
  TestAndroidHeadroomAtomicsAndTtl();
  TestThermalVrsWithSystemHeadroom();
  if (argc >= 2 && argv[1] && argv[1][0] != '\0') {
    TestSourceContracts(fs::path(argv[1]));
  }
  std::printf("wave6_onda4c_platform_test: PASS\n");
  return 0;
}
