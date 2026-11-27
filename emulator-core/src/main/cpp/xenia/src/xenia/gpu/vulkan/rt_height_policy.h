/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_VULKAN_RT_HEIGHT_POLICY_H_
#define XENIA_GPU_VULKAN_RT_HEIGHT_POLICY_H_

#include <algorithm>
#include <cstdint>

namespace xe {
namespace gpu {
namespace vulkan {

// Pure, header-only EDRAM row-on-demand render target height policy shared
// between VulkanRenderTargetCache and host C++20 unit tests.

inline constexpr uint32_t kRtOnDemandMinHeight = 128u;
inline constexpr uint32_t kRtMergedTilingBandsHeight = 720u;
inline constexpr uint32_t kRtEdramTileHeightSamples = 16u;

inline constexpr uint32_t ComputeRtTileHeightPixels(uint32_t msaa_samples) {
  // Xenos EDRAM tile is 80x16 samples: 1x MSAA -> 16 px, 2x/4x MSAA -> 8 px.
  const uint32_t msaa_y_log2 = msaa_samples >= 1u ? 1u : 0u;
  return kRtEdramTileHeightSamples >> msaa_y_log2;
}

inline constexpr uint32_t AlignUpRtHeight(uint32_t value, uint32_t alignment) {
  if (alignment <= 1u || value == 0u) {
    return value;
  }
  const uint64_t aligned =
      ((uint64_t(value) + uint64_t(alignment - 1u)) / uint64_t(alignment)) *
      uint64_t(alignment);
  return aligned > UINT32_MAX ? UINT32_MAX : uint32_t(aligned);
}

inline constexpr uint32_t ComputeInitialRenderTargetHeight(
    uint32_t max_guest_height, uint32_t requested_guest_height,
    uint32_t msaa_samples, bool merge_tiling_bands,
    bool rows_on_demand_enabled) {
  if (!max_guest_height) {
    return 0u;
  }
  if (!rows_on_demand_enabled) {
    return max_guest_height;
  }
  const uint32_t tile_h = ComputeRtTileHeightPixels(msaa_samples);
  const uint32_t floor_h =
      (merge_tiling_bands && msaa_samples >= 1u)
          ? (kRtMergedTilingBandsHeight > kRtOnDemandMinHeight
                 ? kRtMergedTilingBandsHeight
                 : kRtOnDemandMinHeight)
          : kRtOnDemandMinHeight;
  const uint32_t target =
      requested_guest_height > floor_h ? requested_guest_height : floor_h;
  const uint32_t aligned = AlignUpRtHeight(target, tile_h);
  return aligned < max_guest_height ? aligned : max_guest_height;
}

inline constexpr uint32_t ComputeGrownRenderTargetHeight(
    uint32_t current_guest_height, uint32_t needed_guest_height,
    uint32_t max_guest_height, uint32_t msaa_samples) {
  if (!max_guest_height) {
    return 0u;
  }
  const uint32_t clamped_needed = needed_guest_height < max_guest_height
                                      ? needed_guest_height
                                      : max_guest_height;
  if (clamped_needed <= current_guest_height) {
    return current_guest_height;
  }
  const uint32_t tile_h = ComputeRtTileHeightPixels(msaa_samples);
  // Geometric >= 50% growth caps total reallocations up to max_guest_height.
  const uint64_t geometric =
      uint64_t(current_guest_height) + uint64_t(current_guest_height / 2u);
  const uint64_t base_target =
      uint64_t(clamped_needed) > geometric ? uint64_t(clamped_needed)
                                           : geometric;
  const uint64_t aligned =
      ((base_target + uint64_t(tile_h - 1u)) / uint64_t(tile_h)) *
      uint64_t(tile_h);
  return aligned < uint64_t(max_guest_height) ? uint32_t(aligned)
                                              : max_guest_height;
}

// Vulkan VUID requires every attachment's image height >= framebuffer.height.
// Taking the minimum allocated guest height across all bound attachments
// guarantees compliance while remaining >= height_used (ensured prior to pass).
inline uint32_t ComputeFramebufferGuestHeight(
    uint32_t fallback_guest_height, const uint32_t* attachment_guest_heights,
    uint32_t attachment_count) {
  if (!attachment_guest_heights || !attachment_count) {
    return fallback_guest_height;
  }
  uint32_t min_height = fallback_guest_height;
  bool any_nonzero = false;
  for (uint32_t i = 0; i < attachment_count; ++i) {
    const uint32_t h = attachment_guest_heights[i];
    if (!h) {
      continue;
    }
    if (!any_nonzero) {
      min_height =
          fallback_guest_height ? std::min(fallback_guest_height, h) : h;
      any_nonzero = true;
    } else {
      min_height = std::min(min_height, h);
    }
  }
  return any_nonzero ? min_height : fallback_guest_height;
}

// Returns true if a cached framebuffer key references the given render target
// EDRAM base in any active attachment slot matching its depth/color kind.
inline constexpr bool FramebufferReferencesRenderTargetBase(
    uint32_t depth_and_color_used_mask, uint32_t depth_base_tiles,
    const uint32_t color_base_tiles[4], bool rt_is_depth,
    uint32_t rt_base_tiles) {
  if (rt_is_depth) {
    return (depth_and_color_used_mask & (1u << 0)) != 0u &&
           depth_base_tiles == rt_base_tiles;
  }
  for (uint32_t i = 0; i < 4u; ++i) {
    if ((depth_and_color_used_mask & (1u << (1u + i))) != 0u &&
        color_base_tiles[i] == rt_base_tiles) {
      return true;
    }
  }
  return false;
}

}  // namespace vulkan
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_VULKAN_RT_HEIGHT_POLICY_H_
