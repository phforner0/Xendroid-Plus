/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_UI_VULKAN_VULKAN_PRESENT_DIVISOR_H_
#define XENIA_UI_VULKAN_VULKAN_PRESENT_DIVISOR_H_

#include <algorithm>
#include <cstdint>

namespace xe {
namespace ui {
namespace vulkan {

struct PresentExtent {
  uint32_t width;
  uint32_t height;
};

inline PresentExtent ComputeAndroidPresentExtent(
    uint32_t panel_w, uint32_t panel_h, uint32_t min_w, uint32_t min_h,
    uint32_t max_w, uint32_t max_h, int32_t scale_x, int32_t scale_y,
    int32_t divisor_setting, uint32_t* chosen_divisor_out = nullptr) {
  const uint32_t sx = uint32_t(std::clamp(scale_x, int32_t(1), int32_t(3)));
  const uint32_t sy = uint32_t(std::clamp(scale_y, int32_t(1), int32_t(3)));

  const uint32_t guest_w = 1280u * sx;
  const uint32_t guest_h = 720u * sy;

  // Compare orientation-independently so preTransform 90/270 swaps are handled
  // symmetrically.
  const uint32_t panel_long = std::max(panel_w, panel_h);
  const uint32_t panel_short = std::min(panel_w, panel_h);
  const uint32_t guest_long = std::max(guest_w, guest_h);
  const uint32_t guest_short = std::min(guest_w, guest_h);

  uint32_t d = 1;
  if (divisor_setting == 0) {
    for (uint32_t cand = 4; cand >= 2; --cand) {
      if (panel_long / cand >= guest_long &&
          panel_short / cand >= guest_short) {
        d = cand;
        break;
      }
    }
  } else if (divisor_setting >= 2) {
    d = uint32_t(std::min(divisor_setting, int32_t(4)));
  } else {
    d = 1;
  }

  uint32_t out_w = std::max(1u, panel_w / d);
  uint32_t out_h = std::max(1u, panel_h / d);

  if (min_w <= max_w) {
    out_w = std::clamp(out_w, min_w, max_w);
  } else {
    out_w = max_w;
  }
  if (min_h <= max_h) {
    out_h = std::clamp(out_h, min_h, max_h);
  } else {
    out_h = max_h;
  }

  if (chosen_divisor_out) {
    *chosen_divisor_out = (out_w == panel_w && out_h == panel_h) ? 1u : d;
  }
  return PresentExtent{out_w, out_h};
}

}  // namespace vulkan
}  // namespace ui
}  // namespace xe

#endif  // XENIA_UI_VULKAN_VULKAN_PRESENT_DIVISOR_H_
