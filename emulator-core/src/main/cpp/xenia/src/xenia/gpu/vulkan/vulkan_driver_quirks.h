/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_VULKAN_VULKAN_DRIVER_QUIRKS_H_
#define XENIA_GPU_VULKAN_VULKAN_DRIVER_QUIRKS_H_

#include <cstdint>
#include <cstring>

namespace xe {
namespace gpu {
namespace vulkan {

// Numerical VkDriverId / PCI vendor constants kept header-only so both the
// Vulkan backend and dependency-free C++20 host tests share one source of
// truth (#81).
inline constexpr uint32_t kVkDriverIdMesaRadv = 3;
inline constexpr uint32_t kVkDriverIdArmProprietary = 9;
inline constexpr uint32_t kVkDriverIdMesaTurnip = 18;
inline constexpr uint32_t kVkDriverIdMesaPanvk = 20;
inline constexpr uint32_t kVkDriverIdSamsungProprietary = 21;

inline constexpr uint32_t kVendorIdArm = 0x13B5;
inline constexpr uint32_t kVendorIdSamsung = 0x144D;

// Samsung proprietary Xclipse driver detection. Open-source Mesa RADV running on
// Xclipse keeps Samsung's vendorID (0x144D) but reports VK_DRIVER_ID_MESA_RADV
// and supports standard BCn paths natively, so it must never be forced through
// the Samsung proprietary BCn->ASTC/EAC transcode (#81).
inline bool IsSamsungXclipseProprietary(uint32_t driver_id, uint32_t vendor_id,
                                        const char* device_name) {
  if (driver_id == kVkDriverIdSamsungProprietary) {
    return true;
  }
  if (driver_id != 0) {
    return false;
  }
  return vendor_id == kVendorIdSamsung ||
         (device_name != nullptr && std::strstr(device_name, "Xclipse") != nullptr);
}

// Non-Turnip mobile tiler drivers (proprietary Mali/Xclipse and PanVK) that
// benefit from keeping 4x MSAA render targets on 2x2 1x AFBC surfaces. Excludes
// Turnip and Mesa RADV (#81).
inline bool IsNonTurnipMobileTiler(uint32_t driver_id, uint32_t vendor_id) {
  if (driver_id == kVkDriverIdMesaTurnip || driver_id == kVkDriverIdMesaRadv) {
    return false;
  }
  return driver_id == kVkDriverIdArmProprietary ||
         driver_id == kVkDriverIdMesaPanvk || vendor_id == kVendorIdArm ||
         driver_id == kVkDriverIdSamsungProprietary ||
         (driver_id == 0 && vendor_id == kVendorIdSamsung);
}

}  // namespace vulkan
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_VULKAN_VULKAN_DRIVER_QUIRKS_H_
