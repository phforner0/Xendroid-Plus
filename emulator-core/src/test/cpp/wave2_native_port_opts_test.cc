#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#ifndef XE_DISABLE_ANDROID_DLOPEN
#define XE_DISABLE_ANDROID_DLOPEN 1
#endif

#include "xenia/base/android_thermal.h"
#include "xenia/gpu/vulkan/deferred_resolve_clear.h"
#include "xenia/gpu/vulkan/thermal_vrs_controller.h"
#include "xenia/kernel/adpf_tid_registry.h"
#include "xenia/kernel/delay0_backoff.h"

namespace {

#define WAVE2_CHECK(cond)                                                  \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "WAVE2_OPT check failed (%s:%d): %s\n",         \
                   __FILE__, __LINE__, #cond);                             \
      std::abort();                                                        \
    }                                                                      \
  } while (false)

using xe::gpu::vulkan::CanDeferResolveClear;
using xe::gpu::vulkan::ClampShadingRateToMaxFragmentSize;
using xe::gpu::vulkan::DeferredClearAspect;
using xe::gpu::vulkan::DeferredClearCoversExtent;
using xe::gpu::vulkan::DeferredClearFoldMode;
using xe::gpu::vulkan::DeferredClearRect;
using xe::gpu::vulkan::DeferredClearRectCovers;
using xe::gpu::vulkan::DeferredClearRectFitsExtent;
using xe::gpu::vulkan::DeferredClearRectsOverlap;
using xe::gpu::vulkan::DeferredClearTileRangesOverlap;
using xe::gpu::vulkan::DeferredClearValue;
using xe::gpu::vulkan::ExpandDrawnExtentForClear;
using xe::gpu::vulkan::PendingResolveClear;
using xe::gpu::vulkan::SelectDeferredClearFoldMode;
using xe::gpu::vulkan::SelectDrawShadingRate;
using xe::gpu::vulkan::ThermalVrsController;
using xe::kernel::CpuPerformanceHintSession;
using xe::kernel::Delay0BackoffState;
using xe::kernel::FilterAliveTids;
using xe::kernel::GuestCpuTidRegistry;

std::string ReadSourceText(const std::filesystem::path& path) {
  std::ifstream in(path);
  WAVE2_CHECK(in.good());
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

// --- Block A: Deferred Resolve Clear Folding ---

void TestCanDeferResolveClear() {
  WAVE2_CHECK(CanDeferResolveClear(true, true, false, false, false));
  WAVE2_CHECK(!CanDeferResolveClear(false, true, false, false, false));
  WAVE2_CHECK(!CanDeferResolveClear(true, false, false, false, false));
  WAVE2_CHECK(!CanDeferResolveClear(true, true, true, false, false));
  WAVE2_CHECK(!CanDeferResolveClear(true, true, false, true, false));
  WAVE2_CHECK(!CanDeferResolveClear(true, true, false, false, true));
}

void TestSelectDeferredClearFoldMode() {
  // Dynamic rendering + full RT + not in pass -> LOAD_OP_CLEAR.
  uint32_t dont_care = 0b00001u;
  WAVE2_CHECK(SelectDeferredClearFoldMode(true, false, true, false, 0b00011u,
                                          dont_care, 0b00001u) ==
              DeferredClearFoldMode::kLoadOpClear);
  WAVE2_CHECK(dont_care == 0u);

  // Sub-rect clear -> always CmdClearAttachments (never LOAD_OP_CLEAR).
  dont_care = 0b00010u;
  WAVE2_CHECK(SelectDeferredClearFoldMode(true, false, false, false, 0b00011u,
                                          dont_care, 0b00001u) ==
              DeferredClearFoldMode::kCmdClearAttachments);
  WAVE2_CHECK(dont_care == 0b00010u);

  // Traditional VkRenderPass -> always CmdClearAttachments.
  dont_care = 0b00010u;
  WAVE2_CHECK(SelectDeferredClearFoldMode(false, false, true, false, 0b00011u,
                                          dont_care, 0b00001u) ==
              DeferredClearFoldMode::kCmdClearAttachments);
  WAVE2_CHECK(dont_care == 0b00010u);

  // Already in render pass -> always CmdClearAttachments.
  dont_care = 0b00010u;
  WAVE2_CHECK(SelectDeferredClearFoldMode(true, true, true, false, 0b00011u,
                                          dont_care, 0b00001u) ==
              DeferredClearFoldMode::kCmdClearAttachments);

  // Mali < r38p1 mixed load op normalization:
  // Depth cleared via LOAD_OP_CLEAR (bit 0) + Color0 DONT_CARE (bit 1) ->
  // must clear dont_care so Color0 uses LOAD instead of mixing DONT_CARE.
  dont_care = 0b00011u;
  WAVE2_CHECK(SelectDeferredClearFoldMode(true, false, true, true, 0b00011u,
                                          dont_care, 0b00001u) ==
              DeferredClearFoldMode::kLoadOpClear);
  WAVE2_CHECK(dont_care == 0u);

  // Without Mali quirk, Color0 can stay DONT_CARE while Depth uses CLEAR.
  dont_care = 0b00011u;
  WAVE2_CHECK(SelectDeferredClearFoldMode(true, false, true, false, 0b00011u,
                                          dont_care, 0b00001u) ==
              DeferredClearFoldMode::kLoadOpClear);
  WAVE2_CHECK(dont_care == 0b00010u);

  // If all used attachments are cleared via LOAD_OP_CLEAR, dont_care is 0.
  dont_care = 0b00011u;
  WAVE2_CHECK(SelectDeferredClearFoldMode(true, false, true, true, 0b00011u,
                                          dont_care, 0b00011u) ==
              DeferredClearFoldMode::kLoadOpClear);
  WAVE2_CHECK(dont_care == 0u);
}

void TestExpandDrawnExtentAndRects() {
  uint32_t drawn_w = 100;
  uint32_t drawn_h = 100;
  ExpandDrawnExtentForClear(drawn_w, drawn_h,
                            DeferredClearRect{0, 0, 1280, 720});
  WAVE2_CHECK(drawn_w == 1280 && drawn_h == 720);

  drawn_w = 100;
  drawn_h = 100;
  ExpandDrawnExtentForClear(drawn_w, drawn_h,
                            DeferredClearRect{320, 180, 640, 360});
  WAVE2_CHECK(drawn_w == 960 && drawn_h == 540);

  // Smaller rect does not shrink existing drawn bounds.
  ExpandDrawnExtentForClear(drawn_w, drawn_h,
                            DeferredClearRect{10, 10, 100, 100});
  WAVE2_CHECK(drawn_w == 960 && drawn_h == 540);

  // Negative offset clamps origin to 0 like AccumulateDrawnRect.
  drawn_w = 0;
  drawn_h = 0;
  ExpandDrawnExtentForClear(drawn_w, drawn_h,
                            DeferredClearRect{-16, -8, 128, 64});
  WAVE2_CHECK(drawn_w == 128 && drawn_h == 64);

  // CoversExtent & FitsExtent.
  const DeferredClearRect full{0, 0, 1280, 720};
  const DeferredClearRect sub{64, 64, 256, 256};
  const DeferredClearRect oversize{0, 0, 1920, 1080};
  const DeferredClearRect neg_origin{-4, 0, 1280, 720};
  WAVE2_CHECK(DeferredClearCoversExtent(full, 1280, 720));
  WAVE2_CHECK(!DeferredClearCoversExtent(sub, 1280, 720));
  WAVE2_CHECK(DeferredClearCoversExtent(oversize, 1280, 720));
  WAVE2_CHECK(!DeferredClearCoversExtent(full, 0, 720));
  WAVE2_CHECK(DeferredClearRectFitsExtent(full, 1280, 720));
  WAVE2_CHECK(DeferredClearRectFitsExtent(sub, 1280, 720));
  WAVE2_CHECK(!DeferredClearRectFitsExtent(oversize, 1280, 720));
  WAVE2_CHECK(!DeferredClearRectFitsExtent(neg_origin, 1280, 720));

  // RectCovers & RectsOverlap.
  WAVE2_CHECK(DeferredClearRectCovers(full, sub));
  WAVE2_CHECK(DeferredClearRectCovers(full, full));
  WAVE2_CHECK(!DeferredClearRectCovers(sub, full));
  WAVE2_CHECK(DeferredClearRectsOverlap(full, sub));
  WAVE2_CHECK(!DeferredClearRectsOverlap(
      DeferredClearRect{0, 0, 128, 128}, DeferredClearRect{128, 0, 128, 128}));
  WAVE2_CHECK(DeferredClearRectsOverlap(
      DeferredClearRect{0, 0, 128, 128}, DeferredClearRect{127, 127, 64, 64}));
}

void TestTileRangesOverlapWrap() {
  // Non-wrapping cases.
  WAVE2_CHECK(!DeferredClearTileRangesOverlap(0, 0, 0, 10));
  WAVE2_CHECK(!DeferredClearTileRangesOverlap(0, 10, 10, 10));
  WAVE2_CHECK(DeferredClearTileRangesOverlap(0, 10, 9, 10));
  WAVE2_CHECK(DeferredClearTileRangesOverlap(100, 50, 120, 10));
  WAVE2_CHECK(DeferredClearTileRangesOverlap(120, 10, 100, 50));

  // Wrapping around 2048: [2040, 2048) U [0, 20) == base 2040, count 28.
  WAVE2_CHECK(DeferredClearTileRangesOverlap(2040, 28, 0, 5));
  WAVE2_CHECK(DeferredClearTileRangesOverlap(2040, 28, 19, 5));
  WAVE2_CHECK(!DeferredClearTileRangesOverlap(2040, 28, 20, 5));
  WAVE2_CHECK(!DeferredClearTileRangesOverlap(2040, 28, 2035, 5));
  WAVE2_CHECK(DeferredClearTileRangesOverlap(2040, 28, 2035, 6));
  WAVE2_CHECK(DeferredClearTileRangesOverlap(15, 10, 2040, 28));
  WAVE2_CHECK(!DeferredClearTileRangesOverlap(20, 10, 2040, 28));

  // Full EDRAM span.
  WAVE2_CHECK(DeferredClearTileRangesOverlap(500, 2048, 1500, 1));
}

void TestPendingClearStagingLifecycle() {
  // Verify POD properties & trivial copyability (needed for -Wnontrivial-memcall).
  static_assert(std::is_trivially_copyable_v<PendingResolveClear>);
  static_assert(std::is_trivially_copyable_v<DeferredClearRect>);
  static_assert(std::is_trivially_copyable_v<DeferredClearValue>);

  PendingResolveClear clear{};
  WAVE2_CHECK(!clear.active);
  WAVE2_CHECK(clear.aspect == DeferredClearAspect::kNone);

  clear.rt_key = 0x12345678ULL;
  clear.base_tiles = 100;
  clear.tile_count = 64;
  clear.rect = DeferredClearRect{0, 0, 1280, 720};
  clear.value.color.float32[0] = 0.25f;
  clear.value.color.float32[1] = 0.50f;
  clear.value.color.float32[2] = 0.75f;
  clear.value.color.float32[3] = 1.00f;
  clear.aspect = DeferredClearAspect::kColor;
  clear.active = true;

  // Simulate speculative Update() -> abort -> PendingResolveClear stays intact.
  const bool full_rt = DeferredClearCoversExtent(clear.rect, 1280, 720);
  WAVE2_CHECK(full_rt);
  uint32_t staged_mask = 1u << 1;
  uint32_t load_op_mask = 0u;
  // Abort before pass begin: clear.active must still be true.
  WAVE2_CHECK(clear.active);

  // Re-stage and commit via LOAD_OP_CLEAR on pass begin.
  uint32_t dont_care = 0u;
  const DeferredClearFoldMode mode = SelectDeferredClearFoldMode(
      true, false, full_rt, false, 0b00010u, dont_care, staged_mask);
  WAVE2_CHECK(mode == DeferredClearFoldMode::kLoadOpClear);
  load_op_mask = staged_mask;
  // Pass begin consumes only load_op_mask bits.
  uint32_t drawn_w = 0, drawn_h = 0;
  if (load_op_mask & (1u << 1)) {
    ExpandDrawnExtentForClear(drawn_w, drawn_h, clear.rect);
    clear.active = false;
  }
  WAVE2_CHECK(!clear.active);
  WAVE2_CHECK(drawn_w == 1280 && drawn_h == 720);

  // Simulate partial overlap replacement decision:
  PendingResolveClear sub_clear{};
  sub_clear.rect = DeferredClearRect{64, 64, 256, 256};
  sub_clear.active = true;
  const DeferredClearRect covering_rect{0, 0, 1280, 720};
  const DeferredClearRect partial_rect{128, 128, 512, 512};
  WAVE2_CHECK(DeferredClearRectCovers(covering_rect, sub_clear.rect));
  WAVE2_CHECK(!DeferredClearRectCovers(partial_rect, sub_clear.rect));
}

void TestBlockAWiring(const std::filesystem::path& cpp_dir) {
  const std::string dcb_h = ReadSourceText(
      cpp_dir / "xenia/src/xenia/gpu/vulkan/deferred_command_buffer.h");
  WAVE2_CHECK(dcb_h.find("ExpandRenderAreaToRect(") != std::string::npos);

  const std::string rtc_h = ReadSourceText(
      cpp_dir / "xenia/src/xenia/gpu/vulkan/vulkan_render_target_cache.h");
  WAVE2_CHECK(rtc_h.find("DECLARE_bool(vulkan_defer_resolve_clears);") !=
              std::string::npos);
  WAVE2_CHECK(rtc_h.find("StageDeferredResolveClearsForLastUpdate(") !=
              std::string::npos);
  WAVE2_CHECK(rtc_h.find("ConsumeDeferredResolveClearsOnPassBegin(") !=
              std::string::npos);
  WAVE2_CHECK(rtc_h.find("EncodePendingDeferredResolveClears(") !=
              std::string::npos);
  WAVE2_CHECK(rtc_h.find("FlushAllDeferredResolveClears(") !=
              std::string::npos);
  WAVE2_CHECK(rtc_h.find("last_update_load_op_clear_mask(") !=
              std::string::npos);
  WAVE2_CHECK(rtc_h.find("deferred_clear_render_targets_") !=
              std::string::npos);

  const std::string rtc_cc = ReadSourceText(
      cpp_dir / "xenia/src/xenia/gpu/vulkan/vulkan_render_target_cache.cc");
  WAVE2_CHECK(
      rtc_cc.find("DEFINE_bool(\n    vulkan_defer_resolve_clears, true") !=
          std::string::npos ||
      rtc_cc.find("vulkan_defer_resolve_clears, true") != std::string::npos);
  WAVE2_CHECK(rtc_cc.find("CanDeferResolveClear(") != std::string::npos);
  WAVE2_CHECK(rtc_cc.find("SelectDeferredClearFoldMode(") != std::string::npos);
  WAVE2_CHECK(rtc_cc.find("ExpandRenderAreaToRect(") != std::string::npos);
  WAVE2_CHECK(rtc_cc.find("DiscardAllDeferredResolveClears()") !=
              std::string::npos);
  WAVE2_CHECK(rtc_cc.find("view_color_transfer_separate()") !=
              std::string::npos);
  WAVE2_CHECK(rtc_cc.find("StageDeferredResolveClearsForLastUpdate();") !=
              std::string::npos);
  WAVE2_CHECK(rtc_cc.find("FlushDeferredResolveClearsForTransfers(") !=
              std::string::npos);
  WAVE2_CHECK(rtc_cc.find("FlushOverlappingDeferredResolveClears(") !=
              std::string::npos);
  WAVE2_CHECK(rtc_cc.find("UnregisterPendingResolveClear(this);") !=
              std::string::npos);

  // Verify mandatory flush boundaries on dump / readback / tiling replication.
  const auto dump_pos =
      rtc_cc.find("void VulkanRenderTargetCache::DumpRenderTargets(");
  WAVE2_CHECK(dump_pos != std::string::npos);
  const auto dump_flush =
      rtc_cc.find("FlushAllDeferredResolveClears();", dump_pos);
  const auto dump_rects =
      rtc_cc.find("GetResolveCopyRectanglesToDump(", dump_pos);
  WAVE2_CHECK(dump_flush != std::string::npos &&
              dump_rects != std::string::npos && dump_flush < dump_rects);

  const auto dump_all_pos =
      rtc_cc.find("void VulkanRenderTargetCache::DumpAllRenderTargetsToEdram(");
  WAVE2_CHECK(dump_all_pos != std::string::npos);
  const auto dump_all_flush =
      rtc_cc.find("FlushAllDeferredResolveClears();", dump_all_pos);
  WAVE2_CHECK(dump_all_flush != std::string::npos &&
              dump_all_flush - dump_all_pos < 200);

  const auto readback_pos =
      rtc_cc.find("bool VulkanRenderTargetCache::BeginEdramSnapshotReadback(");
  WAVE2_CHECK(readback_pos != std::string::npos);
  const auto readback_flush =
      rtc_cc.find("FlushAllDeferredResolveClears();", readback_pos);
  WAVE2_CHECK(readback_flush != std::string::npos &&
              readback_flush - readback_pos < 200);

  const auto tiling_pos = rtc_cc.find(
      "void VulkanRenderTargetCache::ReplicateTilingBandRenderTargetRows(");
  WAVE2_CHECK(tiling_pos != std::string::npos);
  const auto tiling_flush =
      rtc_cc.find("FlushDeferredResolveClearForTarget(&rt);", tiling_pos);
  WAVE2_CHECK(tiling_flush != std::string::npos &&
              tiling_flush - tiling_pos < 250);

  // Verify discard boundaries without GPU work (RestoreEdramSnapshot)
  // and preservation across EndSubmission.
  const auto restore_pos =
      rtc_cc.find("void VulkanRenderTargetCache::RestoreEdramSnapshot(");
  WAVE2_CHECK(restore_pos != std::string::npos);
  const auto restore_discard =
      rtc_cc.find("DiscardAllDeferredResolveClears();", restore_pos);
  WAVE2_CHECK(restore_discard != std::string::npos &&
              restore_discard - restore_pos < 120);

  const auto end_sub_pos =
      rtc_cc.find("void VulkanRenderTargetCache::EndSubmission()");
  WAVE2_CHECK(end_sub_pos != std::string::npos);
  const auto end_sub_end = rtc_cc.find("\n}", end_sub_pos);
  WAVE2_CHECK(end_sub_end != std::string::npos);
  const std::string end_sub_body =
      rtc_cc.substr(end_sub_pos, end_sub_end - end_sub_pos);
  WAVE2_CHECK(end_sub_body.find("DeferredResolveClear") == std::string::npos);

  // Verify NormalizeKeyForCacheLookup remains intact.
  WAVE2_CHECK(rtc_h.find("NormalizeKeyForCacheLookup(RenderPassKey key)") !=
              std::string::npos);
  WAVE2_CHECK(rtc_h.find("key.depth_and_color_load_dont_care = 0;") !=
              std::string::npos);

  // Verify VulkanCommandProcessor wiring for Block A:
  const std::string vk_cp_h = ReadSourceText(
      cpp_dir / "xenia/src/xenia/gpu/vulkan/vulkan_command_processor.h");
  WAVE2_CHECK(vk_cp_h.find("FlushDeferredResolveClearsIfPending()") !=
              std::string::npos);

  const std::string vk_cp_cc = ReadSourceText(
      cpp_dir / "xenia/src/xenia/gpu/vulkan/vulkan_command_processor.cc");
  WAVE2_CHECK(vk_cp_cc.find("debug.xendroid.defer_resolve_clears") !=
              std::string::npos);
  WAVE2_CHECK(vk_cp_cc.find("FlushDeferredResolveClearsIfPending()") !=
              std::string::npos);

  const auto enter_2arg = vk_cp_cc.find(
      "void VulkanCommandProcessor::"
      "SubmitBarriersAndEnterRenderTargetCacheRenderPass(");
  WAVE2_CHECK(enter_2arg != std::string::npos);
  const auto enter_4arg = vk_cp_cc.find(
      "void VulkanCommandProcessor::"
      "SubmitBarriersAndEnterRenderTargetCacheRenderPass(",
      enter_2arg + 1);
  WAVE2_CHECK(enter_4arg != std::string::npos);
  const auto enter_4arg_end = vk_cp_cc.find("\nvoid VulkanCommandProcessor::",
                                            enter_4arg + 1);
  WAVE2_CHECK(enter_4arg_end != std::string::npos);

  const std::string body_2arg =
      vk_cp_cc.substr(enter_2arg, enter_4arg - enter_2arg);
  const std::string body_4arg =
      vk_cp_cc.substr(enter_4arg, enter_4arg_end - enter_4arg);

  // 2-arg overload: both early returns + pass begin must encode/consume.
  const auto enc1 = body_2arg.find("EncodePendingDeferredResolveClears();");
  const auto enc2 =
      body_2arg.find("EncodePendingDeferredResolveClears();", enc1 + 1);
  const auto consume =
      body_2arg.find("ConsumeDeferredResolveClearsOnPassBegin(");
  const auto enc3 =
      body_2arg.find("EncodePendingDeferredResolveClears();", consume + 1);
  WAVE2_CHECK(enc1 != std::string::npos && enc2 != std::string::npos &&
              consume != std::string::npos && enc3 != std::string::npos);
  WAVE2_CHECK(body_2arg.find("last_update_load_op_clear_mask()") !=
              std::string::npos);

  // 4-arg transfer-pass overload MUST NEVER consume/encode draw-pass clears.
  WAVE2_CHECK(body_4arg.find("ConsumeDeferredResolveClearsOnPassBegin") ==
              std::string::npos);
  WAVE2_CHECK(body_4arg.find("EncodePendingDeferredResolveClears") ==
              std::string::npos);
}

// --- Block B: Shared AThermal + Opt-In Thermal Auto-VRS ---

void TestAndroidThermalHostStub() {
  WAVE2_CHECK(xe::kThermalStatusUnavailable == -1);
  WAVE2_CHECK(xe::kThermalStatusNone == 0);
  WAVE2_CHECK(xe::kThermalStatusLight == 1);
  WAVE2_CHECK(xe::kThermalStatusModerate == 2);
  WAVE2_CHECK(xe::kThermalStatusSevere == 3);
  WAVE2_CHECK(xe::kThermalStatusCritical == 4);
  WAVE2_CHECK(xe::kThermalStatusEmergency == 5);
  WAVE2_CHECK(xe::kThermalStatusShutdown == 6);

  WAVE2_CHECK(xe::AcquireAndroidThermalManager() == nullptr);
  WAVE2_CHECK(xe::ReadAndroidThermalStatus() == xe::kThermalStatusUnavailable);
  WAVE2_CHECK(std::isnan(xe::ReadAndroidThermalHeadroom(10)));

  const xe::ThermalSample sample = xe::ReadAndroidThermalSample(10);
  WAVE2_CHECK(sample.status == xe::kThermalStatusUnavailable);
  WAVE2_CHECK(sample.has_status == false);
  WAVE2_CHECK(sample.has_headroom == false);
  WAVE2_CHECK(sample.headroom < 0.0f);
}

void TestThermalVrsHysteresis() {
  ThermalVrsController controller;
  WAVE2_CHECK(controller.current_rate() == 0u);

  // Unavailable thermal signal holds 1x1 safely.
  xe::ThermalSample unavailable{};
  controller.Update(1000, unavailable);
  controller.Update(3000, unavailable);
  WAVE2_CHECK(controller.current_rate() == 0u);

  // Brief borderline spike (< 1000ms dwell) must not oscillate into 2x1.
  xe::ThermalSample moderate{xe::kThermalStatusModerate, 0.80f, true, true};
  xe::ThermalSample light{xe::kThermalStatusLight, 0.50f, true, true};
  controller.Update(10000, moderate);
  WAVE2_CHECK(controller.current_rate() == 0u);
  controller.Update(10500, light);
  WAVE2_CHECK(controller.current_rate() == 0u);

  // Sustained MODERATE >= 1000ms escalates 1x1 (0) -> 2x1 (1).
  controller.Update(12000, moderate);
  WAVE2_CHECK(controller.current_rate() == 0u);
  controller.Update(13000, moderate);
  WAVE2_CHECK(controller.current_rate() == 1u);

  // Brief SEVERE (< 1000ms) does not escalate immediately to 2x2.
  xe::ThermalSample severe{xe::kThermalStatusSevere, 0.90f, true, true};
  controller.Update(14000, severe);
  WAVE2_CHECK(controller.current_rate() == 1u);
  controller.Update(14500, moderate);
  WAVE2_CHECK(controller.current_rate() == 1u);

  // Sustained SEVERE >= 1000ms escalates 2x1 (1) -> 2x2 (3).
  controller.Update(15000, severe);
  WAVE2_CHECK(controller.current_rate() == 1u);
  controller.Update(16000, severe);
  WAVE2_CHECK(controller.current_rate() == 3u);

  // Deadband: MODERATE with headroom > 0.82f does not step down from 2x2.
  xe::ThermalSample moderate_warm{xe::kThermalStatusModerate, 0.84f, true, true};
  controller.Update(17000, moderate_warm);
  controller.Update(23000, moderate_warm);
  WAVE2_CHECK(controller.current_rate() == 3u);

  // Cooldown 2x2 -> 2x1 requires status <= MODERATE and headroom <= 0.82f for >= 4000ms.
  xe::ThermalSample moderate_cool{xe::kThermalStatusModerate, 0.80f, true, true};
  controller.Update(24000, moderate_cool);
  WAVE2_CHECK(controller.current_rate() == 3u);
  controller.Update(27999, moderate_cool);
  WAVE2_CHECK(controller.current_rate() == 3u);
  controller.Update(28000, moderate_cool);
  WAVE2_CHECK(controller.current_rate() == 1u);

  // Deadband at 2x1: LIGHT with headroom 0.72f (> 0.70f) stays at 2x1.
  xe::ThermalSample light_warm{xe::kThermalStatusLight, 0.72f, true, true};
  controller.Update(29000, light_warm);
  controller.Update(35000, light_warm);
  WAVE2_CHECK(controller.current_rate() == 1u);

  // Cooldown 2x1 -> 1x1 requires status <= LIGHT and headroom <= 0.70f for >= 4000ms.
  xe::ThermalSample light_cool{xe::kThermalStatusLight, 0.65f, true, true};
  controller.Update(36000, light_cool);
  WAVE2_CHECK(controller.current_rate() == 1u);
  controller.Update(39999, light_cool);
  WAVE2_CHECK(controller.current_rate() == 1u);
  controller.Update(40000, light_cool);
  WAVE2_CHECK(controller.current_rate() == 0u);

  // Headroom-only path (has_status == false): >= 0.85 -> 2x1, >= 0.95 -> 2x2.
  xe::ThermalSample headroom_mid{xe::kThermalStatusUnavailable, 0.86f, false,
                                 true};
  xe::ThermalSample headroom_high{xe::kThermalStatusUnavailable, 0.96f, false,
                                  true};
  controller.Update(41000, headroom_mid);
  controller.Update(42000, headroom_mid);
  WAVE2_CHECK(controller.current_rate() == 1u);
  controller.Update(43000, headroom_high);
  controller.Update(44000, headroom_high);
  WAVE2_CHECK(controller.current_rate() == 3u);

  // Unavailable signal after escalation falls back to 1x1 safely.
  controller.Update(45000, unavailable);
  WAVE2_CHECK(controller.current_rate() == 0u);

  // Critical thermal status escalates immediately without waiting 1000ms dwell.
  xe::ThermalSample critical{xe::kThermalStatusCritical, 1.0f, true, true};
  controller.Update(46000, critical);
  WAVE2_CHECK(controller.current_rate() == 3u);

  controller.Reset();
  WAVE2_CHECK(controller.current_rate() == 0u);
}

void TestClampAndSelectDrawShadingRate() {
  // maxFragmentSize = 2x2
  WAVE2_CHECK(ClampShadingRateToMaxFragmentSize(0, 2, 2) == 0u);
  WAVE2_CHECK(ClampShadingRateToMaxFragmentSize(1, 2, 2) == 1u);
  WAVE2_CHECK(ClampShadingRateToMaxFragmentSize(2, 2, 2) == 2u);
  WAVE2_CHECK(ClampShadingRateToMaxFragmentSize(3, 2, 2) == 3u);

  // maxFragmentSize = 2x1
  WAVE2_CHECK(ClampShadingRateToMaxFragmentSize(0, 2, 1) == 0u);
  WAVE2_CHECK(ClampShadingRateToMaxFragmentSize(1, 2, 1) == 1u);
  WAVE2_CHECK(ClampShadingRateToMaxFragmentSize(2, 2, 1) == 0u);
  WAVE2_CHECK(ClampShadingRateToMaxFragmentSize(3, 2, 1) == 1u);

  // maxFragmentSize = 1x2
  WAVE2_CHECK(ClampShadingRateToMaxFragmentSize(0, 1, 2) == 0u);
  WAVE2_CHECK(ClampShadingRateToMaxFragmentSize(1, 1, 2) == 0u);
  WAVE2_CHECK(ClampShadingRateToMaxFragmentSize(2, 1, 2) == 2u);
  WAVE2_CHECK(ClampShadingRateToMaxFragmentSize(3, 1, 2) == 2u);

  // maxFragmentSize = 1x1
  WAVE2_CHECK(ClampShadingRateToMaxFragmentSize(0, 1, 1) == 0u);
  WAVE2_CHECK(ClampShadingRateToMaxFragmentSize(1, 1, 1) == 0u);
  WAVE2_CHECK(ClampShadingRateToMaxFragmentSize(2, 1, 1) == 0u);
  WAVE2_CHECK(ClampShadingRateToMaxFragmentSize(3, 1, 1) == 0u);

  // Strict guards: 1x RT and memexport ALWAYS return 0 (1x1).
  WAVE2_CHECK(SelectDrawShadingRate(3, true, 3, true, false, 2, 2) == 0u);
  WAVE2_CHECK(SelectDrawShadingRate(0, true, 3, true, false, 2, 2) == 0u);
  WAVE2_CHECK(SelectDrawShadingRate(3, true, 3, false, true, 2, 2) == 0u);
  WAVE2_CHECK(SelectDrawShadingRate(0, true, 3, false, true, 2, 2) == 0u);

  // Disabled both manual and auto returns 0.
  WAVE2_CHECK(SelectDrawShadingRate(0, false, 3, false, false, 2, 2) == 0u);

  // Manual vulkan_shading_rate overrides auto-VRS and respects maxFragmentSize.
  WAVE2_CHECK(SelectDrawShadingRate(1, true, 3, false, false, 2, 2) == 1u);
  WAVE2_CHECK(SelectDrawShadingRate(3, false, 0, false, false, 2, 1) == 1u);
  WAVE2_CHECK(SelectDrawShadingRate(2, true, 1, false, false, 2, 1) == 0u);

  // Auto-VRS active on multisampled non-memexport scene respects maxFragmentSize.
  WAVE2_CHECK(SelectDrawShadingRate(0, true, 1, false, false, 2, 2) == 1u);
  WAVE2_CHECK(SelectDrawShadingRate(0, true, 3, false, false, 2, 2) == 3u);
  WAVE2_CHECK(SelectDrawShadingRate(0, true, 3, false, false, 2, 1) == 1u);
  WAVE2_CHECK(SelectDrawShadingRate(0, true, 3, false, false, 1, 1) == 0u);
}

void TestBlockBWiring(const std::filesystem::path& cpp_dir) {
  const std::string lsfg_cmake =
      ReadSourceText(cpp_dir / "third_party/lsfg/CMakeLists.txt");
  WAVE2_CHECK(lsfg_cmake.find("xenia-base") == std::string::npos);
  WAVE2_CHECK(lsfg_cmake.find("../../xenia/src") != std::string::npos);

  const std::string lsfg_gov =
      ReadSourceText(cpp_dir / "third_party/lsfg/lsfg_governor.cpp");
  WAVE2_CHECK(lsfg_gov.find("xenia/base/android_thermal.h") !=
              std::string::npos);
  WAVE2_CHECK(lsfg_gov.find("xe::ReadAndroidThermalStatus()") !=
              std::string::npos);
  WAVE2_CHECK(lsfg_gov.find("AThermal_acquireManager") == std::string::npos);

  const std::string vk_device_h =
      ReadSourceText(cpp_dir / "xenia/src/xenia/ui/vulkan/vulkan_device.h");
  WAVE2_CHECK(vk_device_h.find("maxFragmentSize") != std::string::npos);

  const std::string vk_device_cc =
      ReadSourceText(cpp_dir / "xenia/src/xenia/ui/vulkan/vulkan_device.cc");
  WAVE2_CHECK(
      vk_device_cc.find("ReadBoolCvarOrDefault(\"vulkan_auto_vrs_thermal\", "
                        "false)") != std::string::npos);
  WAVE2_CHECK(vk_device_cc.find("DECLARE_bool(vulkan_auto_vrs_thermal)") ==
              std::string::npos);
  WAVE2_CHECK(vk_device_cc.find("maxFragmentSize = fsr.maxFragmentSize") !=
              std::string::npos);

  const std::string vk_cp_cc = ReadSourceText(
      cpp_dir / "xenia/src/xenia/gpu/vulkan/vulkan_command_processor.cc");
  WAVE2_CHECK(vk_cp_cc.find("vulkan_auto_vrs_thermal") != std::string::npos);
  WAVE2_CHECK(vk_cp_cc.find("debug.xendroid.auto_vrs_thermal") !=
              std::string::npos);
  WAVE2_CHECK(vk_cp_cc.find("!pixel_shader->memexport_eM_written()") !=
              std::string::npos);
  WAVE2_CHECK(vk_cp_cc.find("SelectDrawShadingRate(") != std::string::npos);
}

// --- Block C: Delay(0) Backoff + kLateGuard + ADPF Guest CPU TIDs ---

void TestDelay0BackoffState() {
  Delay0BackoffState state;
  uint64_t now_ns = 1'000'000'000ULL;

  // 1. nothing_else_to_run == false ALWAYS returns 0 and resets streak.
  WAVE2_CHECK(state.NextSleepUs(now_ns, false) == 0u);
  WAVE2_CHECK(state.streak() == 0u);

  // 2. Progressive ramp on continuous idle spins (<= 50 us gap):
  //    0 -> 0 -> 100 -> 200 -> 400 -> 500 -> 500.
  WAVE2_CHECK(state.NextSleepUs(now_ns, true) == 0u);
  WAVE2_CHECK(state.streak() == 1u);

  now_ns += 10'000ULL;
  WAVE2_CHECK(state.NextSleepUs(now_ns, true) == 0u);
  WAVE2_CHECK(state.streak() == 2u);

  now_ns += 10'000ULL;
  uint32_t s = state.NextSleepUs(now_ns, true);
  WAVE2_CHECK(s == 100u);
  now_ns += uint64_t(s) * 1000ULL;
  state.RecordSleepResult(now_ns, s, uint64_t(s) * 1000ULL);

  now_ns += 10'000ULL;
  s = state.NextSleepUs(now_ns, true);
  WAVE2_CHECK(s == 200u);
  now_ns += uint64_t(s) * 1000ULL;
  state.RecordSleepResult(now_ns, s, uint64_t(s) * 1000ULL);

  now_ns += 10'000ULL;
  s = state.NextSleepUs(now_ns, true);
  WAVE2_CHECK(s == 400u);
  now_ns += uint64_t(s) * 1000ULL;
  state.RecordSleepResult(now_ns, s, uint64_t(s) * 1000ULL);

  // Exact 50us boundary does NOT reset streak.
  now_ns += 50'000ULL;
  s = state.NextSleepUs(now_ns, true);
  WAVE2_CHECK(s == 500u);
  now_ns += uint64_t(s) * 1000ULL;
  state.RecordSleepResult(now_ns, s, uint64_t(s) * 1000ULL);

  now_ns += 10'000ULL;
  s = state.NextSleepUs(now_ns, true);
  WAVE2_CHECK(s == 500u);

  // 3. nothing_else_to_run == false resets streak immediately even mid-ramp.
  now_ns += 10'000ULL;
  WAVE2_CHECK(state.NextSleepUs(now_ns, false) == 0u);
  WAVE2_CHECK(state.streak() == 0u);

  // Ramp restarts from 0 -> 0 -> 100.
  now_ns += 5'000ULL;
  WAVE2_CHECK(state.NextSleepUs(now_ns, true) == 0u);
  now_ns += 5'000ULL;
  WAVE2_CHECK(state.NextSleepUs(now_ns, true) == 0u);
  now_ns += 5'000ULL;
  WAVE2_CHECK(state.NextSleepUs(now_ns, true) == 100u);

  // 4. Active gap (> 50 us) resets streak back to 1 (returns 0).
  now_ns += 50'001ULL;
  WAVE2_CHECK(state.NextSleepUs(now_ns, true) == 0u);
  WAVE2_CHECK(state.streak() == 1u);

  // Reset() clears streak.
  state.Reset();
  WAVE2_CHECK(state.streak() == 0u);

  // 5. kLateGuard: 1 late sleep in 8 does NOT disarm; 2 late sleeps in 8-bit
  //    window trigger 32-call cooldown and then recover cleanly.
  Delay0BackoffState late_state;
  now_ns = 2'000'000'000ULL;
  WAVE2_CHECK(late_state.NextSleepUs(now_ns, true) == 0u);
  now_ns += 5'000ULL;
  WAVE2_CHECK(late_state.NextSleepUs(now_ns, true) == 0u);
  now_ns += 5'000ULL;
  s = late_state.NextSleepUs(now_ns, true);
  WAVE2_CHECK(s == 100u);

  // 1st late sleep (+600 us overshoot > 500 us tolerance).
  now_ns += uint64_t(s) * 1000ULL + 600'000ULL;
  late_state.RecordSleepResult(now_ns, s, uint64_t(s) * 1000ULL + 600'000ULL);
  WAVE2_CHECK(late_state.cooldown_remaining() == 0u);
  WAVE2_CHECK(late_state.late_history() == 0x01u);

  // 7 on-time sleeps push the 1st late bit out after 8 samples total.
  for (int i = 0; i < 7; ++i) {
    now_ns += 5'000ULL;
    s = late_state.NextSleepUs(now_ns, true);
    WAVE2_CHECK(s >= 100u);
    now_ns += uint64_t(s) * 1000ULL + 100'000ULL;  // <= 500 us tolerance
    late_state.RecordSleepResult(now_ns, s,
                                 uint64_t(s) * 1000ULL + 100'000ULL);
    WAVE2_CHECK(late_state.cooldown_remaining() == 0u);
  }
  WAVE2_CHECK(late_state.late_history() == 0x80u);

  // Another single late sleep still only has 1 bit in the 8-bit window.
  now_ns += 5'000ULL;
  s = late_state.NextSleepUs(now_ns, true);
  WAVE2_CHECK(s == 500u);
  now_ns += uint64_t(s) * 1000ULL + 700'000ULL;
  late_state.RecordSleepResult(now_ns, s, uint64_t(s) * 1000ULL + 700'000ULL);
  WAVE2_CHECK(late_state.cooldown_remaining() == 0u);

  // A 2nd late sleep within the 8-sample window triggers 32-call cooldown.
  now_ns += 5'000ULL;
  s = late_state.NextSleepUs(now_ns, true);
  WAVE2_CHECK(s == 500u);
  now_ns += uint64_t(s) * 1000ULL + 700'000ULL;
  late_state.RecordSleepResult(now_ns, s, uint64_t(s) * 1000ULL + 700'000ULL);
  WAVE2_CHECK(late_state.cooldown_remaining() == 32u);

  // Next 32 calls return 0 during cooldown.
  for (int i = 0; i < 32; ++i) {
    now_ns += 5'000ULL;
    WAVE2_CHECK(late_state.NextSleepUs(now_ns, true) == 0u);
    WAVE2_CHECK(late_state.cooldown_remaining() == uint32_t(31 - i));
  }

  // After 32 calls, backoff recovers cleanly at 0 -> 0 -> 100.
  now_ns += 5'000ULL;
  WAVE2_CHECK(late_state.NextSleepUs(now_ns, true) == 0u);
  now_ns += 5'000ULL;
  WAVE2_CHECK(late_state.NextSleepUs(now_ns, true) == 0u);
  now_ns += 5'000ULL;
  WAVE2_CHECK(late_state.NextSleepUs(now_ns, true) == 100u);
}

void TestFilterAliveTidsAndRegistry() {
  const std::vector<int32_t> mixed = {101, 102, 0, -5, 102, 103, 101};
  const auto filtered = FilterAliveTids(mixed, [](int32_t tid) {
    return tid == 101 || tid == 103;
  });
  WAVE2_CHECK(filtered.size() == 2u);
  WAVE2_CHECK(filtered[0] == 101);
  WAVE2_CHECK(filtered[1] == 103);

  const auto empty_filtered =
      FilterAliveTids(mixed, [](int32_t) { return false; });
  WAVE2_CHECK(empty_filtered.empty());

  GuestCpuTidRegistry registry;
  WAVE2_CHECK(registry.generation() == 0u);

  registry.RegisterTid(0);
  registry.RegisterTid(-42);
  WAVE2_CHECK(registry.generation() == 0u);

  registry.RegisterTid(101);
  const uint64_t gen1 = registry.generation();
  WAVE2_CHECK(gen1 == 1u);

  registry.RegisterTid(102);
  const uint64_t gen2 = registry.generation();
  WAVE2_CHECK(gen2 == 2u);

  registry.RegisterTid(103);
  const uint64_t gen3 = registry.generation();
  WAVE2_CHECK(gen3 == 3u);

  // Duplicate registration does not duplicate entries.
  registry.RegisterTid(101);
  const auto live_all = registry.SnapshotLiveTids([](int32_t) { return true; });
  WAVE2_CHECK(live_all.size() == 3u);

  const auto live_partial =
      registry.SnapshotLiveTids([](int32_t tid) { return tid != 102; });
  WAVE2_CHECK(live_partial.size() == 2u);
  WAVE2_CHECK(live_partial[0] == 101);
  WAVE2_CHECK(live_partial[1] == 103);

  registry.UnregisterTid(999);  // absent: no-op
  WAVE2_CHECK(registry.generation() == gen3);

  registry.UnregisterTid(102);
  WAVE2_CHECK(registry.generation() > gen3);
  const auto after_unreg =
      registry.SnapshotLiveTids([](int32_t) { return true; });
  WAVE2_CHECK(after_unreg.size() == 2u);
  WAVE2_CHECK(after_unreg[0] == 101);
  WAVE2_CHECK(after_unreg[1] == 103);

  // Empty live set skip and disabled ADPF safety.
  CpuPerformanceHintSession session;
  session.OnFrameTick(/*enabled=*/false, 16'666'667ULL, registry);
  WAVE2_CHECK(!session.active());
  session.Shutdown();
  WAVE2_CHECK(!session.active());
}

void TestBlockCWiring(const std::filesystem::path& cpp_dir) {
  const std::string kernel_flags_h =
      ReadSourceText(cpp_dir / "xenia/src/xenia/kernel/kernel_flags.h");
  const std::string xthread_h =
      ReadSourceText(cpp_dir / "xenia/src/xenia/kernel/xthread.h");
  const std::string xthread_cc =
      ReadSourceText(cpp_dir / "xenia/src/xenia/kernel/xthread.cc");
  const std::string cpu_flags_h =
      ReadSourceText(cpp_dir / "xenia/src/xenia/cpu/cpu_flags.h");
  const std::string cpu_flags_cc =
      ReadSourceText(cpp_dir / "xenia/src/xenia/cpu/cpu_flags.cc");
  const std::string scheduler_h =
      ReadSourceText(cpp_dir / "xenia/src/xenia/kernel/guest_scheduler.h");
  const std::string scheduler_cc =
      ReadSourceText(cpp_dir / "xenia/src/xenia/kernel/guest_scheduler.cc");
  const std::string cp_cc =
      ReadSourceText(cpp_dir / "xenia/src/xenia/gpu/command_processor.cc");

  WAVE2_CHECK(kernel_flags_h.find("DECLARE_bool(kernel_delay0_backoff);") !=
              std::string::npos);
  WAVE2_CHECK(xthread_h.find("Delay0BackoffState delay0_backoff_;") !=
              std::string::npos);
  WAVE2_CHECK(
      xthread_cc.find("DEFINE_bool(\n    kernel_delay0_backoff, true") !=
          std::string::npos ||
      xthread_cc.find("DEFINE_bool(kernel_delay0_backoff, true") !=
          std::string::npos);
  WAVE2_CHECK(xthread_cc.find("scheduler->YieldCurrentThread(false)") !=
              std::string::npos);
  WAVE2_CHECK(
      xthread_cc.find("__atomic_load_n(&cvars::kernel_delay0_backoff") !=
      std::string::npos);
  WAVE2_CHECK(xthread_cc.find("delay0_backoff_.NextSleepUs") !=
              std::string::npos);
  WAVE2_CHECK(xthread_cc.find("RegisterTid(") != std::string::npos);
  WAVE2_CHECK(xthread_cc.find("UnregisterTid(") != std::string::npos);

  WAVE2_CHECK(cpu_flags_h.find("DECLARE_bool(adpf_cpu_performance_hint);") !=
              std::string::npos);
  WAVE2_CHECK(
      cpu_flags_cc.find("DEFINE_bool(adpf_cpu_performance_hint, false") !=
          std::string::npos ||
      cpu_flags_cc.find(
          "DEFINE_bool(\n    adpf_cpu_performance_hint, false") !=
          std::string::npos);

  WAVE2_CHECK(scheduler_h.find("GuestCpuTidRegistry cpu_tid_registry_;") !=
              std::string::npos);
  WAVE2_CHECK(scheduler_h.find("CpuPerformanceHintSession adpf_session_;") !=
              std::string::npos);
  WAVE2_CHECK(scheduler_cc.find("RegisterTid(") != std::string::npos);
  WAVE2_CHECK(scheduler_cc.find("UnregisterTid(") != std::string::npos);
  WAVE2_CHECK(scheduler_cc.find("adpf_session_.Shutdown()") !=
              std::string::npos);

  WAVE2_CHECK(cp_cc.find("debug.xendroid.delay0_backoff") != std::string::npos);
  WAVE2_CHECK(cp_cc.find("__atomic_store_n(&cvars::kernel_delay0_backoff") !=
              std::string::npos);
  WAVE2_CHECK(cp_cc.find("adpf_cpu_performance_hint") != std::string::npos);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "Usage: %s <emulator-core/src/main/cpp>\n", argv[0]);
    return 2;
  }
  const std::filesystem::path cpp_dir(argv[1]);

  TestCanDeferResolveClear();
  TestSelectDeferredClearFoldMode();
  TestExpandDrawnExtentAndRects();
  TestTileRangesOverlapWrap();
  TestPendingClearStagingLifecycle();
  TestBlockAWiring(cpp_dir);

  TestAndroidThermalHostStub();
  TestThermalVrsHysteresis();
  TestClampAndSelectDrawShadingRate();
  TestBlockBWiring(cpp_dir);

  TestDelay0BackoffState();
  TestFilterAliveTidsAndRegistry();
  TestBlockCWiring(cpp_dir);

  std::puts("wave2_native_port_opts_test: PASS");
  return 0;
}
