/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_VULKAN_DEFERRED_RESOLVE_CLEAR_H_
#define XENIA_GPU_VULKAN_DEFERRED_RESOLVE_CLEAR_H_

#include <cstdint>

namespace xe {
namespace gpu {
namespace vulkan {

struct DeferredClearRect {
  int32_t x = 0;
  int32_t y = 0;
  uint32_t width = 0;
  uint32_t height = 0;
};

union DeferredClearColor {
  float float32[4];
  int32_t int32[4];
  uint32_t uint32[4];
};

struct DeferredClearDepthStencil {
  float depth = 0.0f;
  uint32_t stencil = 0;
};

union DeferredClearValue {
  DeferredClearColor color;
  DeferredClearDepthStencil depth_stencil;
};

enum class DeferredClearAspect : uint32_t {
  kNone = 0,
  kColor = 1u << 0,
  kDepth = 1u << 1,
  kStencil = 1u << 2,
  kDepthStencil = kDepth | kStencil,
};

inline constexpr DeferredClearAspect operator|(DeferredClearAspect a,
                                             DeferredClearAspect b) {
  return static_cast<DeferredClearAspect>(static_cast<uint32_t>(a) |
                                          static_cast<uint32_t>(b));
}

inline constexpr DeferredClearAspect operator&(DeferredClearAspect a,
                                             DeferredClearAspect b) {
  return static_cast<DeferredClearAspect>(static_cast<uint32_t>(a) &
                                          static_cast<uint32_t>(b));
}

inline constexpr bool AnyDeferredClearAspect(DeferredClearAspect aspect) {
  return static_cast<uint32_t>(aspect) != 0;
}

enum class DeferredClearFoldMode : uint8_t {
  kNone = 0,
  kLoadOpClear = 1,
  kCmdClearAttachments = 2,
};

struct PendingResolveClear {
  uint64_t rt_key = 0;
  uint32_t base_tiles = 0;
  uint32_t tile_count = 0;
  DeferredClearRect rect{};
  DeferredClearValue value{};
  DeferredClearAspect aspect = DeferredClearAspect::kNone;
  bool active = false;
};

inline bool CanDeferResolveClear(bool enabled, bool is_host_rt_path,
                                 bool has_separate_transfer_view,
                                 bool has_clear_transfers,
                                 bool copied_in_pass) {
  return enabled && is_host_rt_path && !has_separate_transfer_view &&
         !has_clear_transfers && !copied_in_pass;
}

inline bool DeferredClearCoversExtent(const DeferredClearRect& rect,
                                      uint32_t extent_width,
                                      uint32_t extent_height) {
  if (!extent_width || !extent_height || !rect.width || !rect.height) {
    return false;
  }
  const int64_t x1 = int64_t(rect.x) + int64_t(rect.width);
  const int64_t y1 = int64_t(rect.y) + int64_t(rect.height);
  return rect.x <= 0 && rect.y <= 0 && x1 >= int64_t(extent_width) &&
         y1 >= int64_t(extent_height);
}

inline bool DeferredClearRectFitsExtent(const DeferredClearRect& rect,
                                        uint32_t extent_width,
                                        uint32_t extent_height) {
  if (!rect.width || !rect.height || rect.x < 0 || rect.y < 0) {
    return false;
  }
  const int64_t x1 = int64_t(rect.x) + int64_t(rect.width);
  const int64_t y1 = int64_t(rect.y) + int64_t(rect.height);
  return x1 <= int64_t(extent_width) && y1 <= int64_t(extent_height);
}

inline DeferredClearFoldMode SelectDeferredClearFoldMode(
    bool use_dynamic_rendering, bool already_in_render_pass, bool full_rt_rect,
    bool mali_normalize_mixed_load_ops, uint32_t used_mask,
    uint32_t& dont_care_mask_inout, uint32_t load_op_clear_mask) {
  if (already_in_render_pass || !use_dynamic_rendering || !full_rt_rect) {
    return DeferredClearFoldMode::kCmdClearAttachments;
  }
  // A slot cleared via LOAD_OP_CLEAR cannot also be marked DONT_CARE.
  dont_care_mask_inout &= ~load_op_clear_mask;
  if (mali_normalize_mixed_load_ops) {
    // On Mali < r38p1, mixing DONT_CARE with CLEAR/LOAD across attachments in
    // the same pass can corrupt tile state; clear dont_care if any used slot
    // remains non-DONT_CARE.
    const uint32_t dont_care_used = dont_care_mask_inout & used_mask;
    if (dont_care_used != 0 && dont_care_used != used_mask) {
      dont_care_mask_inout = 0;
    }
  }
  return DeferredClearFoldMode::kLoadOpClear;
}

inline void ExpandDrawnExtentForClear(uint32_t& drawn_width,
                                      uint32_t& drawn_height,
                                      const DeferredClearRect& rect) {
  const uint32_t x0 = rect.x > 0 ? uint32_t(rect.x) : 0u;
  const uint32_t y0 = rect.y > 0 ? uint32_t(rect.y) : 0u;
  if (rect.width > 0) {
    const uint64_t right = uint64_t(x0) + rect.width;
    const uint32_t clamped_right =
        right > UINT32_MAX ? UINT32_MAX : uint32_t(right);
    if (clamped_right > drawn_width) {
      drawn_width = clamped_right;
    }
  }
  if (rect.height > 0) {
    const uint64_t bottom = uint64_t(y0) + rect.height;
    const uint32_t clamped_bottom =
        bottom > UINT32_MAX ? UINT32_MAX : uint32_t(bottom);
    if (clamped_bottom > drawn_height) {
      drawn_height = clamped_bottom;
    }
  }
}

inline bool DeferredClearRectCovers(const DeferredClearRect& outer,
                                    const DeferredClearRect& inner) {
  if (!inner.width || !inner.height) {
    return true;
  }
  if (!outer.width || !outer.height) {
    return false;
  }
  const int64_t outer_x0 = outer.x;
  const int64_t outer_y0 = outer.y;
  const int64_t outer_x1 = outer_x0 + int64_t(outer.width);
  const int64_t outer_y1 = outer_y0 + int64_t(outer.height);
  const int64_t inner_x0 = inner.x;
  const int64_t inner_y0 = inner.y;
  const int64_t inner_x1 = inner_x0 + int64_t(inner.width);
  const int64_t inner_y1 = inner_y0 + int64_t(inner.height);
  return outer_x0 <= inner_x0 && outer_y0 <= inner_y0 &&
         outer_x1 >= inner_x1 && outer_y1 >= inner_y1;
}

inline bool DeferredClearRectsOverlap(const DeferredClearRect& a,
                                      const DeferredClearRect& b) {
  if (!a.width || !a.height || !b.width || !b.height) {
    return false;
  }
  const int64_t a_x0 = a.x;
  const int64_t a_y0 = a.y;
  const int64_t a_x1 = a_x0 + int64_t(a.width);
  const int64_t a_y1 = a_y0 + int64_t(a.height);
  const int64_t b_x0 = b.x;
  const int64_t b_y0 = b.y;
  const int64_t b_x1 = b_x0 + int64_t(b.width);
  const int64_t b_y1 = b_y0 + int64_t(b.height);
  return a_x0 < b_x1 && a_x1 > b_x0 && a_y0 < b_y1 && a_y1 > b_y0;
}

inline bool DeferredClearTileRangesOverlap(uint32_t base_a, uint32_t count_a,
                                           uint32_t base_b, uint32_t count_b) {
  constexpr uint32_t kEdramTiles = 2048u;
  if (!count_a || !count_b) {
    return false;
  }
  if (count_a >= kEdramTiles || count_b >= kEdramTiles) {
    return true;
  }
  const uint32_t a0 = base_a & (kEdramTiles - 1u);
  const uint32_t b0 = base_b & (kEdramTiles - 1u);
  const uint32_t dab = (b0 + kEdramTiles - a0) & (kEdramTiles - 1u);
  const uint32_t dba = (a0 + kEdramTiles - b0) & (kEdramTiles - 1u);
  return dab < count_a || dba < count_b;
}

#ifdef VK_VERSION_1_0

inline VkRect2D ToVkRect2D(const DeferredClearRect& rect) {
  VkRect2D vk_rect{};
  vk_rect.offset.x = rect.x;
  vk_rect.offset.y = rect.y;
  vk_rect.extent.width = rect.width;
  vk_rect.extent.height = rect.height;
  return vk_rect;
}

inline VkImageAspectFlags ToVkImageAspectFlags(DeferredClearAspect aspect) {
  VkImageAspectFlags flags = 0;
  if (AnyDeferredClearAspect(aspect & DeferredClearAspect::kColor)) {
    flags |= VK_IMAGE_ASPECT_COLOR_BIT;
  }
  if (AnyDeferredClearAspect(aspect & DeferredClearAspect::kDepth)) {
    flags |= VK_IMAGE_ASPECT_DEPTH_BIT;
  }
  if (AnyDeferredClearAspect(aspect & DeferredClearAspect::kStencil)) {
    flags |= VK_IMAGE_ASPECT_STENCIL_BIT;
  }
  return flags;
}

inline DeferredClearAspect FromVkImageAspectFlags(VkImageAspectFlags flags) {
  DeferredClearAspect aspect = DeferredClearAspect::kNone;
  if (flags & VK_IMAGE_ASPECT_COLOR_BIT) {
    aspect = aspect | DeferredClearAspect::kColor;
  }
  if (flags & VK_IMAGE_ASPECT_DEPTH_BIT) {
    aspect = aspect | DeferredClearAspect::kDepth;
  }
  if (flags & VK_IMAGE_ASPECT_STENCIL_BIT) {
    aspect = aspect | DeferredClearAspect::kStencil;
  }
  return aspect;
}

inline VkClearValue ToVkClearValue(const DeferredClearValue& value,
                                   DeferredClearAspect aspect) {
  VkClearValue vk_value{};
  if (AnyDeferredClearAspect(aspect & DeferredClearAspect::kColor)) {
    for (uint32_t i = 0; i < 4; ++i) {
      vk_value.color.uint32[i] = value.color.uint32[i];
    }
  } else {
    vk_value.depthStencil.depth = value.depth_stencil.depth;
    vk_value.depthStencil.stencil = value.depth_stencil.stencil;
  }
  return vk_value;
}

inline DeferredClearValue FromVkClearValue(const VkClearValue& vk_value,
                                           DeferredClearAspect aspect) {
  DeferredClearValue value{};
  if (AnyDeferredClearAspect(aspect & DeferredClearAspect::kColor)) {
    for (uint32_t i = 0; i < 4; ++i) {
      value.color.uint32[i] = vk_value.color.uint32[i];
    }
  } else {
    value.depth_stencil.depth = vk_value.depthStencil.depth;
    value.depth_stencil.stencil = vk_value.depthStencil.stencil;
  }
  return value;
}

#endif  // VK_VERSION_1_0

}  // namespace vulkan
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_VULKAN_DEFERRED_RESOLVE_CLEAR_H_
