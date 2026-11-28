// Pure-logic and source-contract tests for Onda 2C GPU resolve optimizations:
// 1. In-pass resolve for float RTs when effective exp_bias == 0
//    (vulkan_in_pass_resolve_float)
// 2. In-pass resolve for 2x MSAA with mandatory cross-sample BY_REGION barrier,
//    no ROAA pipeline flag on MSAA, fixed 8px 2x-MSAA tile-walk Y step, and
//    tiling-band source offset (vulkan_in_pass_resolve_msaa2x)
// 3. Direct host resolve for k_8_8_8_8_GAMMA -> k_8_8_8_8 re-alias with PWL
//    gamma encoding from UNORM16 linear storage
//    (vulkan_direct_host_resolve_gamma_alias)
// 4. Tracked stencil-zero rect for skipping unused stencil fetches in direct
//    host depth resolves (vulkan_skip_unused_stencil_resolve)
// 5. 64x1 stride-1 compute resolve workgroup specialization + synchronized
//    dispatch calculations (vulkan_resolve_compute_64x1)

#define XENIA_GPU_VULKAN_RESOLVE_OPTS_PURE_LOGIC_ONLY
#include "xenia/gpu/vulkan/vulkan_render_target_cache.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

namespace {

using namespace xe::gpu::vulkan::resolve_opts;

constexpr uint32_t kMsaa1x = 0;
constexpr uint32_t kMsaa2x = 1;
constexpr uint32_t kMsaa4x = 2;

constexpr uint32_t kSample0 = 0;
constexpr uint32_t kSample1 = 1;
constexpr uint32_t kSample2 = 2;
constexpr uint32_t kSample3 = 3;
constexpr uint32_t kSample01 = 4;
constexpr uint32_t kSample23 = 5;
constexpr uint32_t kSample0123 = 6;

constexpr uint32_t kFmt8888 = 0;
constexpr uint32_t kFmt8888Gamma = 1;
constexpr uint32_t kFmt2101010Float = 3;
constexpr uint32_t kFmt1616Float = 6;
constexpr uint32_t kFmt16161616Float = 7;
constexpr uint32_t kFmt32Float = 14;
constexpr uint32_t kFmt3232Float = 15;

constexpr uint32_t kStencilKeep = 0;
constexpr uint32_t kStencilZero = 1;
constexpr uint32_t kStencilReplace = 2;
constexpr uint32_t kStencilIncrClamp = 3;
constexpr uint32_t kStencilDecrClamp = 4;
constexpr uint32_t kStencilInvert = 5;
constexpr uint32_t kStencilIncrWrap = 6;
constexpr uint32_t kStencilDecrWrap = 7;

std::string ReadFileText(const std::string& path) {
  std::ifstream ifs(path);
  assert(ifs.is_open() && "Failed to open source file for contract test");
  std::ostringstream oss;
  oss << ifs.rdbuf();
  return oss.str();
}

void TestItem1InPassFloatResolve() {
  assert(IsInPassFloatColorFormat(kFmt1616Float));
  assert(IsInPassFloatColorFormat(kFmt16161616Float));
  assert(IsInPassFloatColorFormat(kFmt32Float));
  assert(IsInPassFloatColorFormat(kFmt3232Float));
  assert(!IsInPassFloatColorFormat(kFmt8888));
  assert(!IsInPassFloatColorFormat(kFmt8888Gamma));
  assert(!IsInPassFloatColorFormat(kFmt2101010Float));

  // Non-float always passes through regardless of flag/bias.
  assert(EvaluateInPassFloatResolve(false, false, 3) ==
         InPassFloatDecision::kNotFloat);

  // Disabled by default -> reject specials not preserved.
  assert(EvaluateInPassFloatResolve(true, false, 0) ==
         InPassFloatDecision::kRejectDisabled);
  assert(EvaluateInPassFloatResolve(true, false, -2) ==
         InPassFloatDecision::kRejectDisabled);

  // Enabled + zero effective exp_bias -> allowed bitwise-preserving.
  assert(EvaluateInPassFloatResolve(true, true, 0) ==
         InPassFloatDecision::kAllowed);

  // Enabled + nonzero exp_bias (or debug exp_bias) -> rejected.
  assert(EvaluateInPassFloatResolve(true, true, 1) ==
         InPassFloatDecision::kRejectNonZeroExpBias);
  assert(EvaluateInPassFloatResolve(true, true, -1) ==
         InPassFloatDecision::kRejectNonZeroExpBias);
}

void TestItem2InPassMsaa2xAndSyncAndTileWalk() {
  // 1x MSAA works regardless of vulkan_in_pass_resolve_msaa2x.
  assert(EvaluateInPassMsaaResolve(kMsaa1x, kMsaa1x, false, false, false,
                                   kSample0, false) ==
         InPassMsaaDecision::kAllowed1x);
  assert(EvaluateInPassMsaaResolve(kMsaa1x, kMsaa1x, false, false, false,
                                   kSample01, false) ==
         InPassMsaaDecision::kRejectUnsupportedSampleSelect);
  assert(EvaluateInPassMsaaResolve(kMsaa1x, kMsaa1x, false, false, true,
                                   kSample0, false) ==
         InPassMsaaDecision::kRejectPassMismatch);

  // 2x MSAA rejected when cvar disabled.
  assert(EvaluateInPassMsaaResolve(kMsaa2x, kMsaa2x, false, false, false,
                                   kSample01, false) ==
         InPassMsaaDecision::kRejectDisabled);

  // 2x MSAA enabled: only mathematically exact sample_select modes k0, k1, k01.
  assert(EvaluateInPassMsaaResolve(kMsaa2x, kMsaa2x, false, false, false,
                                   kSample0, true) ==
         InPassMsaaDecision::kAllowed2x);
  assert(EvaluateInPassMsaaResolve(kMsaa2x, kMsaa2x, false, false, false,
                                   kSample1, true) ==
         InPassMsaaDecision::kAllowed2x);
  assert(EvaluateInPassMsaaResolve(kMsaa2x, kMsaa2x, false, false, false,
                                   kSample01, true) ==
         InPassMsaaDecision::kAllowed2x);
  // Invalid sample selects for 2x MSAA must be rejected.
  assert(EvaluateInPassMsaaResolve(kMsaa2x, kMsaa2x, false, false, false,
                                   kSample2, true) ==
         InPassMsaaDecision::kRejectUnsupportedSampleSelect);
  assert(EvaluateInPassMsaaResolve(kMsaa2x, kMsaa2x, false, false, false,
                                   kSample3, true) ==
         InPassMsaaDecision::kRejectUnsupportedSampleSelect);
  assert(EvaluateInPassMsaaResolve(kMsaa2x, kMsaa2x, false, false, false,
                                   kSample23, true) ==
         InPassMsaaDecision::kRejectUnsupportedSampleSelect);
  assert(EvaluateInPassMsaaResolve(kMsaa2x, kMsaa2x, false, false, false,
                                   kSample0123, true) ==
         InPassMsaaDecision::kRejectUnsupportedSampleSelect);

  // Pass mismatch / host_2x / single-sample storage / transfer format guards.
  assert(EvaluateInPassMsaaResolve(kMsaa2x, kMsaa1x, false, false, false,
                                   kSample01, true) ==
         InPassMsaaDecision::kRejectPassMismatch);
  assert(EvaluateInPassMsaaResolve(kMsaa2x, kMsaa2x, true, false, false,
                                   kSample01, true) ==
         InPassMsaaDecision::kRejectPassMismatch);
  assert(EvaluateInPassMsaaResolve(kMsaa2x, kMsaa2x, false, true, false,
                                   kSample01, true) ==
         InPassMsaaDecision::kRejectPassMismatch);
  assert(EvaluateInPassMsaaResolve(kMsaa2x, kMsaa2x, false, false, true,
                                   kSample01, true) ==
         InPassMsaaDecision::kRejectPassMismatch);

  // 4x MSAA stays rejected.
  assert(EvaluateInPassMsaaResolve(kMsaa4x, kMsaa4x, false, false, false,
                                   kSample0123, true) ==
         InPassMsaaDecision::kRejectUnsupportedMsaa);

  // ROAA synchronization root-cause fix:
  // 1x MSAA with ROAA skips barrier and sets ROAA flag.
  assert(!ShouldEmitInPassLocalReadBarrier(true, false));
  assert(ShouldEnableInPassRoaaPipelineFlag(true, false));
  // 1x MSAA without ROAA emits barrier and omits ROAA flag.
  assert(ShouldEmitInPassLocalReadBarrier(false, false));
  assert(!ShouldEnableInPassRoaaPipelineFlag(false, false));
  // 2x MSAA MUST ALWAYS emit BY_REGION barrier and MUST NOT set ROAA flag,
  // even when rasterization_order_color_access is true (cross-sample
  // subpassLoad is not ordered by ROAA on Turnip/Adreno).
  assert(ShouldEmitInPassLocalReadBarrier(true, true));
  assert(!ShouldEnableInPassRoaaPipelineFlag(true, true));
  assert(ShouldEmitInPassLocalReadBarrier(false, true));
  assert(!ShouldEnableInPassRoaaPipelineFlag(false, true));

  // Destination number format guard under accurate_resolve_number_formats:
  assert(IsInPassDestNumberFormatSupported(kFmt8888, /*UNORM=*/0, true));
  assert(!IsInPassDestNumberFormatSupported(kFmt8888, /*SNORM=*/1, true));
  assert(!IsInPassDestNumberFormatSupported(kFmt8888, /*UINT=*/2, true));
  assert(IsInPassDestNumberFormatSupported(kFmt8888, /*SNORM=*/1, false));
  assert(IsInPassDestNumberFormatSupported(kFmt1616Float, /*FLOAT=*/7, true));
  assert(IsInPassDestNumberFormatSupported(kFmt1616Float, /*UNORM=*/0, true));
  assert(!IsInPassDestNumberFormatSupported(kFmt1616Float, /*SNORM=*/1, true));
  assert(IsInPassDestNumberFormatSupported(/*16_16=*/4, /*SNORM=*/1, true));

  // Affine tile-walk fix for 2x MSAA:
  // At 2x MSAA, tile height is 16 samples = 8 pixels.
  // Case A: clean 2x MSAA 1280x512 rect with matching pitch=16 tiles.
  {
    auto res = ComputeInPassAffineOffset(
        /*is_64bpp=*/false, /*is_2x_msaa=*/true, /*first_sample_index=*/0,
        /*pixel_x0=*/0, /*pixel_y0=*/0, /*width=*/1280, /*height=*/512,
        /*dump_base=*/0, /*dump_pitch=*/16, /*rt_base_tiles=*/0,
        /*rt_pitch_tiles=*/16, /*band_tile_offset=*/0);
    assert(res.ok);
    assert(res.offset_x == 0);
    assert(res.offset_y == 0);
  }
  // Case B: odd tile row discontinuity at py=8 (tile_y=1) when dump_pitch !=
  // rt_pitch_tiles, while py=0 (tile_y=0) and py=16 (tile_y=2) happen to match!
  // With dump_pitch=8, rt_pitch=16, width=80, height=16:
  //   py=0 -> tile_y=0 -> linear=0 -> rt=(0, 0) -> host_y=0, offset_y=0
  //   py=8 -> tile_y=1 -> linear=8 -> rt=(8, 0) -> host_y=0, offset_y=8 (NON-AFFINE!)
  //   py=15 -> tile_y=1 -> linear=8 -> rt=(8, 0) -> host_y=7, offset_y=8
  // Stepping by 16px instead of 8px would miss py=8 for height=17!
  // With height=17 (py=0 -> offset 0, py=8 -> offset 8, py=16 -> linear=16 ->
  // rt=(0,1) -> host_y=8 -> offset_y=8? Wait: py=16 -> sy=32 -> tile_y=2 ->
  // linear=16 -> rt_tile_y=1, rt_tile_x=0 -> host_y=8 -> py-host_y = 8 != 0;
  // check height=24 where py=0 has offset (0,0), py=8 has offset (-640, +8),
  // py=16 has offset (0, +8) -> guaranteed non-affine!).
  // Even sharper: dump_pitch=8, rt_pitch=16, height=17:
  // For an exact test of odd-row check at py=8, use width=80, height=17 where
  // only py=8 hits rt_tile_x=8 while py=0 and py=16 have rt_tile_x=0!
  {
    auto res = ComputeInPassAffineOffset(
        /*is_64bpp=*/false, /*is_2x_msaa=*/true, /*first_sample_index=*/0,
        /*pixel_x0=*/0, /*pixel_y0=*/0, /*width=*/80, /*height=*/16,
        /*dump_base=*/0, /*dump_pitch=*/8, /*rt_base_tiles=*/0,
        /*rt_pitch_tiles=*/16, /*band_tile_offset=*/0);
    assert(!res.ok);
  }
  // Case C: merge_tiling_bands band_tile_offset for 2x MSAA second band
  // (band 1 starts at guest row 512 -> 1024 samples -> 64 tile rows ->
  // band_tile_offset = 64 * 16 = 1024 tiles).
  {
    auto res = ComputeInPassAffineOffset(
        /*is_64bpp=*/false, /*is_2x_msaa=*/true, /*first_sample_index=*/0,
        /*pixel_x0=*/0, /*pixel_y0=*/0, /*width=*/1280, /*height=*/208,
        /*dump_base=*/0, /*dump_pitch=*/16, /*rt_base_tiles=*/0,
        /*rt_pitch_tiles=*/16, /*band_tile_offset=*/1024);
    assert(res.ok);
    assert(res.offset_x == 0);
    // host_y = 512 + py -> offset_y = py - host_y = -512!
    assert(res.offset_y == -512);
  }
  // Case D: tile before RT base with underflow safety.
  {
    auto res = ComputeInPassAffineOffset(
        /*is_64bpp=*/false, /*is_2x_msaa=*/true, /*first_sample_index=*/0,
        /*pixel_x0=*/0, /*pixel_y0=*/0, /*width=*/1280, /*height=*/64,
        /*dump_base=*/0, /*dump_pitch=*/16, /*rt_base_tiles=*/100,
        /*rt_pitch_tiles=*/16, /*band_tile_offset=*/0);
    assert(!res.ok);
  }
}

void TestItem3DirectHostGammaAlias() {
  // Exact format match: always matched, no PWL encode needed.
  {
    auto m = EvaluateDirectHostColorFormatMatch(
        kFmt8888, kFmt8888, kFmt8888, kFmt8888Gamma, true, false);
    assert(m.matched);
    assert(!m.encode_pwl_gamma_from_unorm16);
  }
  // RDR case: owning RT is k_8_8_8_8_GAMMA, resolve declares k_8_8_8_8.
  // Disabled by default -> rejected.
  {
    auto m = EvaluateDirectHostColorFormatMatch(
        kFmt8888Gamma, kFmt8888, kFmt8888, kFmt8888Gamma, true, false);
    assert(!m.matched);
    assert(!m.encode_pwl_gamma_from_unorm16);
  }
  // Enabled + gamma_render_target_as_unorm16 = true -> matched + PWL encode!
  {
    auto m = EvaluateDirectHostColorFormatMatch(
        kFmt8888Gamma, kFmt8888, kFmt8888, kFmt8888Gamma, true, true);
    assert(m.matched);
    assert(m.encode_pwl_gamma_from_unorm16);
  }
  // Enabled + gamma_render_target_as_unorm16 = false -> matched, already PWL
  // encoded in UNORM8 so no extra PWL encode needed.
  {
    auto m = EvaluateDirectHostColorFormatMatch(
        kFmt8888Gamma, kFmt8888, kFmt8888, kFmt8888Gamma, false, true);
    assert(m.matched);
    assert(!m.encode_pwl_gamma_from_unorm16);
  }
  // Unrelated mismatch must still be rejected even when enabled.
  {
    auto m = EvaluateDirectHostColorFormatMatch(
        kFmt2101010Float, kFmt8888, kFmt8888, kFmt8888Gamma, true, true);
    assert(!m.matched);
  }
  {
    auto m = EvaluateDirectHostColorFormatMatch(
        kFmt8888Gamma, kFmt2101010Float, kFmt8888, kFmt8888Gamma, true, true);
    assert(!m.matched);
  }
}

void TestItem4TrackedStencilSkip() {
  // Default constructed rect is invalid (conservative).
  StencilZeroRect rect{};
  assert(!rect.valid);
  assert(!rect.Covers(0, 0, 800, 800));

  // Clear RDR shadowmap 800x800 with stencil=0 (inside 800x3280 host RT).
  UpdateStencilZeroRectOnDepthResolveClear(rect, /*had_clear_transfers=*/false,
                                           0, 0, 800, 800, /*stencil=*/0);
  assert(rect.valid);
  assert(rect.Covers(0, 0, 800, 800));
  assert(rect.Covers(16, 16, 256, 256));
  assert(!rect.Covers(0, 0, 800, 808));  // Outside cleared rect!

  // Resolve check for 800x800 1x depth (pitch=10 tiles, base=0).
  assert(IsDepthResolveCoveredByStencilZero(
      rect, kMsaa1x, 0, 0, 800, 800, 0, 10, 0, 10, 0));
  // Resolve slightly taller -> NOT covered.
  assert(!IsDepthResolveCoveredByStencilZero(
      rect, kMsaa1x, 0, 0, 800, 808, 0, 10, 0, 10, 0));

  // Smaller sub-clear with stencil=0 and NO transfers preserves wider rect.
  UpdateStencilZeroRectOnDepthResolveClear(rect, /*had_clear_transfers=*/false,
                                           0, 0, 256, 256, /*stencil=*/0);
  assert(rect.valid);
  assert(rect.width == 800 && rect.height == 800);

  // Smaller sub-clear with clear_transfers_ non-empty shrinks to clear rect
  // (transfers wrote outside the clear rect).
  UpdateStencilZeroRectOnDepthResolveClear(rect, /*had_clear_transfers=*/true,
                                           0, 0, 256, 256, /*stencil=*/0);
  assert(rect.valid);
  assert(rect.width == 256 && rect.height == 256);
  assert(!rect.Covers(0, 0, 800, 800));
  assert(rect.Covers(0, 0, 256, 256));

  // Clear with non-zero stencil invalidates completely.
  UpdateStencilZeroRectOnDepthResolveClear(rect, /*had_clear_transfers=*/false,
                                           0, 0, 800, 800, /*stencil=*/1);
  assert(!rect.valid);

  // Draw stencil write analysis.
  // Stencil disabled -> cannot write non-zero stencil.
  assert(!DoesDrawPotentiallyWriteNonZeroStencil(
      false, false, kStencilIncrClamp, kStencilIncrClamp, kStencilIncrClamp,
      0xFF, 0xFF, kStencilIncrClamp, kStencilIncrClamp, kStencilIncrClamp, 0xFF,
      0xFF));
  // Stencil enabled, write_mask == 0 (read-only stencil test) -> cannot write.
  assert(!DoesDrawPotentiallyWriteNonZeroStencil(
      true, true, kStencilReplace, kStencilIncrClamp, kStencilInvert, 0xFF,
      0x00, kStencilReplace, kStencilIncrClamp, kStencilInvert, 0xFF, 0x00));
  // Stencil enabled, write_mask != 0, ops are only KEEP / ZERO / REPLACE(ref=0).
  assert(!DoesDrawPotentiallyWriteNonZeroStencil(
      true, false, kStencilKeep, kStencilZero, kStencilReplace, 0x00, 0xFF,
      kStencilIncrClamp, kStencilIncrClamp, kStencilIncrClamp, 0xFF, 0xFF));
  // Stencil enabled, REPLACE with (ref & write_mask) == 0 -> still zero.
  assert(!DoesDrawPotentiallyWriteNonZeroStencil(
      true, false, kStencilKeep, kStencilReplace, kStencilKeep, 0xF0, 0x0F,
      kStencilKeep, kStencilKeep, kStencilKeep, 0, 0));
  // Stencil enabled, REPLACE with (ref & write_mask) != 0 -> invalidates!
  assert(DoesDrawPotentiallyWriteNonZeroStencil(
      true, false, kStencilKeep, kStencilReplace, kStencilKeep, 0x01, 0x0F,
      kStencilKeep, kStencilKeep, kStencilKeep, 0, 0));
  // Stencil enabled, INCREMENT / INVERT / WRAP ops -> invalidates!
  assert(DoesDrawPotentiallyWriteNonZeroStencil(
      true, false, kStencilKeep, kStencilIncrClamp, kStencilKeep, 0x00, 0xFF,
      kStencilKeep, kStencilKeep, kStencilKeep, 0, 0));
  assert(DoesDrawPotentiallyWriteNonZeroStencil(
      true, false, kStencilKeep, kStencilKeep, kStencilInvert, 0x00, 0x01,
      kStencilKeep, kStencilKeep, kStencilKeep, 0, 0));
  assert(DoesDrawPotentiallyWriteNonZeroStencil(
      true, false, kStencilDecrWrap, kStencilKeep, kStencilKeep, 0x00, 0x01,
      kStencilKeep, kStencilKeep, kStencilKeep, 0, 0));
  assert(DoesDrawPotentiallyWriteNonZeroStencil(
      true, false, kStencilIncrWrap, kStencilKeep, kStencilKeep, 0x00, 0x01,
      kStencilKeep, kStencilKeep, kStencilKeep, 0, 0));
  // Backface enabled with non-zero back write while front is KEEP -> invalidates!
  assert(DoesDrawPotentiallyWriteNonZeroStencil(
      true, true, kStencilKeep, kStencilKeep, kStencilKeep, 0x00, 0xFF,
      kStencilKeep, kStencilReplace, kStencilKeep, 0x02, 0xFF));
  // Backface disabled -> back ops ignored.
  assert(!DoesDrawPotentiallyWriteNonZeroStencil(
      true, false, kStencilKeep, kStencilKeep, kStencilKeep, 0x00, 0xFF,
      kStencilKeep, kStencilReplace, kStencilKeep, 0x02, 0xFF));

  // RDR merge_tiling_bands: 1280x512 band 0 cleared to stencil=0, then
  // replicated across 1280x1024 host depth RT -> band 1 (rows 512..720) covered!
  UpdateStencilZeroRectOnDepthResolveClear(rect, false, 0, 0, 1280, 512, 0);
  UpdateStencilZeroRectOnBandReplication(rect, 1280, 1024, 512);
  assert(rect.valid);
  assert(rect.width == 1280 && rect.height == 1024);
  // Band 1 resolve: 1280x208 2xMSAA, band_tile_offset = 64 * 16 = 1024.
  assert(IsDepthResolveCoveredByStencilZero(
      rect, kMsaa2x, 0, 0, 1280, 208, 0, 16, 0, 16, 1024));
  // If band 0 wasn't fully covered before replication -> invalidate.
  UpdateStencilZeroRectOnDepthResolveClear(rect, true, 0, 0, 640, 512, 0);
  UpdateStencilZeroRectOnBandReplication(rect, 1280, 1024, 512);
  assert(!rect.valid);

  // Final fetch decision:
  // Default cvar false -> always fetch stencil unless probe bit is set.
  assert(ShouldFetchStencilInDepthResolve(false, false, true));
  assert(!ShouldFetchStencilInDepthResolve(true, false, false));
  // Cvar true -> skip stencil ONLY when covered_by_stencil_zero is true.
  assert(!ShouldFetchStencilInDepthResolve(false, true, true));
  assert(ShouldFetchStencilInDepthResolve(false, true, false));
}

void TestItem5ComputeWorkgroup64x1() {
  assert(IsFastEdramResolveCopyShaderIndex(0));
  assert(IsFastEdramResolveCopyShaderIndex(1));
  assert(IsFastEdramResolveCopyShaderIndex(2));
  assert(IsFastEdramResolveCopyShaderIndex(3));
  assert(!IsFastEdramResolveCopyShaderIndex(4));
  assert(!IsFastEdramResolveCopyShaderIndex(6));

  uint32_t gx = 0, gy = 0;
  // Default 8x8 dispatch for 1280x512 depth at 4px/thread -> thread_count =
  // 320x512 -> 40x64 groups.
  ComputeDirectHostResolveDispatchGroups(320, 512, /*use_64x1=*/false, gx, gy);
  assert(gx == 40 && gy == 64);

  // 64x1 dispatch for 1280x512 depth at 4px/thread -> thread_count = 320x512
  // -> 5x512 groups (exact multiple of 64!).
  ComputeDirectHostResolveDispatchGroups(320, 512, /*use_64x1=*/true, gx, gy);
  assert(gx == 5 && gy == 512);

  // Non-multiple width (e.g. 800/8 = 100 threads) rounds up X to 2 groups.
  ComputeDirectHostResolveDispatchGroups(100, 800, /*use_64x1=*/true, gx, gy);
  assert(gx == 2 && gy == 800);

  // Fast EDRAM resolve copy group counts (width=1280, height=512):
  // 32bpp 1x/2x (8px/thread): 8x8 -> 20x64; 64x1 -> 3x512.
  ComputeFastEdramResolveGroupCount(/*base_x_log2=*/6, /*base_y_log2=*/3,
                                    1280, 512, /*use_64x1=*/false, gx, gy);
  assert(gx == 20 && gy == 64);
  ComputeFastEdramResolveGroupCount(/*base_x_log2=*/6, /*base_y_log2=*/3,
                                    1280, 512, /*use_64x1=*/true, gx, gy);
  assert(gx == 3 && gy == 512);

  // 64bpp 1x/2x (4px/thread): 8x8 -> 40x64; 64x1 -> 5x512.
  ComputeFastEdramResolveGroupCount(/*base_x_log2=*/5, /*base_y_log2=*/3,
                                    1280, 512, /*use_64x1=*/false, gx, gy);
  assert(gx == 40 && gy == 64);
  ComputeFastEdramResolveGroupCount(/*base_x_log2=*/5, /*base_y_log2=*/3,
                                    1280, 512, /*use_64x1=*/true, gx, gy);
  assert(gx == 5 && gy == 512);
}

void TestSourceContracts(const std::string& cpp_root) {
  const std::string rtc_cc = ReadFileText(
      cpp_root + "/xenia/src/xenia/gpu/vulkan/vulkan_render_target_cache.cc");
  const std::string draw_util_cc =
      ReadFileText(cpp_root + "/xenia/src/xenia/gpu/draw_util.cc");
  const std::string inpass_shader = ReadFileText(
      cpp_root +
      "/xenia/src/xenia/gpu/shaders/resolve_host_color_inpass_entry.xesli");
  const std::string host_color_shader = ReadFileText(
      cpp_root + "/xenia/src/xenia/gpu/shaders/resolve_host_color.xesli");
  const std::string host_depth_entry = ReadFileText(
      cpp_root +
      "/xenia/src/xenia/gpu/shaders/resolve_host_depth_entry.xesli");
  const std::string fast32_1x2x = ReadFileText(
      cpp_root +
      "/xenia/src/xenia/gpu/shaders/resolve_fast_32bpp_1x2xmsaa.xesli");
  const std::string fast32_4x = ReadFileText(
      cpp_root +
      "/xenia/src/xenia/gpu/shaders/resolve_fast_32bpp_4xmsaa.xesli");
  const std::string fast64_1x2x = ReadFileText(
      cpp_root +
      "/xenia/src/xenia/gpu/shaders/resolve_fast_64bpp_1x2xmsaa.xesli");
  const std::string fast64_4x = ReadFileText(
      cpp_root +
      "/xenia/src/xenia/gpu/shaders/resolve_fast_64bpp_4xmsaa.xesli");

  // All 5 new GPU cvars must exist and default to false.
  assert(rtc_cc.find("DEFINE_bool(\n    vulkan_in_pass_resolve_float, false,") !=
         std::string::npos);
  assert(rtc_cc.find("DEFINE_bool(\n    vulkan_in_pass_resolve_msaa2x, false,") !=
         std::string::npos);
  assert(rtc_cc.find(
             "DEFINE_bool(\n    vulkan_direct_host_resolve_gamma_alias, false,") !=
         std::string::npos);
  assert(rtc_cc.find(
             "DEFINE_bool(\n    vulkan_skip_unused_stencil_resolve, false,") !=
         std::string::npos);
  assert(rtc_cc.find("DEFINE_bool(\n    vulkan_resolve_compute_64x1, false,") !=
         std::string::npos);

  // None of the 5 new cvars may have UPDATE_from_*.
  assert(rtc_cc.find("UPDATE_from_bool(vulkan_in_pass_resolve_float") ==
         std::string::npos);
  assert(rtc_cc.find("UPDATE_from_bool(vulkan_in_pass_resolve_msaa2x") ==
         std::string::npos);
  assert(rtc_cc.find("UPDATE_from_bool(vulkan_direct_host_resolve_gamma_alias") ==
         std::string::npos);
  assert(rtc_cc.find("UPDATE_from_bool(vulkan_skip_unused_stencil_resolve") ==
         std::string::npos);
  assert(rtc_cc.find("UPDATE_from_bool(vulkan_resolve_compute_64x1") ==
         std::string::npos);

  // Android runtime debug properties wired.
  assert(rtc_cc.find("debug.xendroid.in_pass_resolve_float") !=
         std::string::npos);
  assert(rtc_cc.find("debug.xendroid.in_pass_resolve_msaa2x") !=
         std::string::npos);
  assert(rtc_cc.find("debug.xendroid.resolve_gamma_alias") !=
         std::string::npos);
  assert(rtc_cc.find("debug.xendroid.skip_unused_stencil_resolve") !=
         std::string::npos);
  assert(rtc_cc.find("debug.xendroid.resolve_compute_64x1") !=
         std::string::npos);

  // Item 2 WAR self-dependency barrier after CmdVkDraw when local-read barrier
  // was emitted (2x MSAA / non-ROAA).
  assert(rtc_cc.find("VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,\n"
                     "        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,\n"
                     "        VK_DEPENDENCY_BY_REGION_BIT") !=
         std::string::npos);

  // Item 1 shader guard: skip exp_bias multiply when factor == 1.0f.
  assert(inpass_shader.find("resolve_info.dest_exp_bias_factor != 1.0f") !=
         std::string::npos);

  // Item 3 shader flag: PWL gamma encode from UNORM16.
  assert(host_color_shader.find(
             "XE_RESOLVE_HOST_COLOR_FLAG_SOURCE_GAMMA_UNORM16") !=
         std::string::npos);
  assert(host_color_shader.find("XeLinearToPWLGamma3") != std::string::npos);

  // Item 5 shader specialization IDs + draw_util synchronization.
  assert(host_depth_entry.find("local_size_x_id = 0, local_size_y_id = 1") !=
         std::string::npos);
  assert(fast32_1x2x.find("local_size_x_id = 0, local_size_y_id = 1") !=
         std::string::npos);
  assert(fast32_4x.find("local_size_x_id = 0, local_size_y_id = 1") !=
         std::string::npos);
  assert(fast64_1x2x.find("local_size_x_id = 0, local_size_y_id = 1") !=
         std::string::npos);
  assert(fast64_4x.find("local_size_x_id = 0, local_size_y_id = 1") !=
         std::string::npos);
  assert(draw_util_cc.find("GetResolveCopyShaderGroupCount") !=
         std::string::npos);
}

}  // namespace

int main(int argc, char** argv) {
  assert(argc >= 2 && "Pass emulator-core/src/main/cpp path");
  TestItem1InPassFloatResolve();
  TestItem2InPassMsaa2xAndSyncAndTileWalk();
  TestItem3DirectHostGammaAlias();
  TestItem4TrackedStencilSkip();
  TestItem5ComputeWorkgroup64x1();
  TestSourceContracts(argv[1]);
  std::printf("gpu_resolve_opts_test: PASS\n");
  return 0;
}
