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
#include <string>

namespace xe {
namespace gpu {
namespace vulkan {

// Numerical VkDriverId / PCI vendor constants kept header-only so both the
// Vulkan backend and dependency-free C++20 host tests share one source of
// truth (#81).
inline constexpr uint32_t kVkDriverIdMesaRadv = 3;
inline constexpr uint32_t kVkDriverIdQualcommProprietary = 8;
inline constexpr uint32_t kVkDriverIdArmProprietary = 9;
inline constexpr uint32_t kVkDriverIdMesaTurnip = 18;
inline constexpr uint32_t kVkDriverIdMesaPanvk = 20;
inline constexpr uint32_t kVkDriverIdSamsungProprietary = 21;

inline constexpr uint32_t kVendorIdArm = 0x13B5;
inline constexpr uint32_t kVendorIdSamsung = 0x144D;
inline constexpr uint32_t kVendorIdQualcomm = 0x5143;

inline constexpr char AsciiToLower(char c) {
  return (c >= 'A' && c <= 'Z') ? char(c - 'A' + 'a') : c;
}

inline bool ContainsCaseInsensitive(const char* haystack, const char* needle) {
  if (!haystack || !needle || needle[0] == '\0') {
    return false;
  }
  const size_t needle_len = std::strlen(needle);
  for (const char* p = haystack; *p != '\0'; ++p) {
    size_t i = 0;
    while (i < needle_len && p[i] != '\0' &&
           AsciiToLower(p[i]) == AsciiToLower(needle[i])) {
      ++i;
    }
    if (i == needle_len) {
      return true;
    }
  }
  return false;
}

// Parses Arm Mali proprietary driver version strings ("r34p0-01eac0",
// "r40p0", "r54p1-00rel0") into (major, minor). Returns false and zeroes
// outputs if no r<M>p<N> token is present.
inline bool ParseArmMaliDriverVersion(const char* version_source,
                                      uint32_t& major_out,
                                      uint32_t& minor_out) {
  major_out = 0u;
  minor_out = 0u;
  if (!version_source) {
    return false;
  }
  for (const char* p = version_source; *p != '\0'; ++p) {
    if (*p != 'r' || p[1] < '0' || p[1] > '9') {
      continue;
    }
    const char* q = p + 1;
    uint32_t major = 0u;
    while (*q >= '0' && *q <= '9') {
      if (major < 100000u) {
        major = major * 10u + uint32_t(*q - '0');
      }
      ++q;
    }
    if (*q != 'p' || q[1] < '0' || q[1] > '9') {
      continue;
    }
    ++q;
    uint32_t minor = 0u;
    while (*q >= '0' && *q <= '9') {
      if (minor < 100000u) {
        minor = minor * 10u + uint32_t(*q - '0');
      }
      ++q;
    }
    major_out = major;
    minor_out = minor;
    return true;
  }
  return false;
}

struct DriverClassification {
  bool is_panvk = false;
  bool is_malisx2 = false;
  bool is_arm_proprietary = false;
  bool is_armsx2_turnip = false;
  bool is_qualcomm_proprietary = false;
  uint32_t mali_driver_major = 0u;
  uint32_t mali_driver_minor = 0u;
};

// Pure driver identity classification distinguishing Mesa PanVK / malisx2
// (even when a kbase wrapper reports driverID == 0), Arm Mali proprietary
// blobs, armsx2-turnip custom builds, and Qualcomm proprietary blobs.
inline DriverClassification ClassifyVulkanDriver(
    uint32_t vendor_id, uint32_t driver_id, const char* driver_name,
    const char* driver_info, const char* mali_driver_override = nullptr) {
  DriverClassification out;
  out.is_panvk = driver_id == kVkDriverIdMesaPanvk ||
                 ContainsCaseInsensitive(driver_name, "panvk") ||
                 ContainsCaseInsensitive(driver_info, "malisx2") ||
                 ContainsCaseInsensitive(driver_name, "malisx2");
  out.is_malisx2 =
      out.is_panvk && (ContainsCaseInsensitive(driver_info, "malisx2") ||
                       ContainsCaseInsensitive(driver_name, "malisx2") ||
                       ContainsCaseInsensitive(driver_info, "kbase"));
  out.is_arm_proprietary =
      !out.is_panvk && (driver_id == kVkDriverIdArmProprietary ||
                        (driver_id == 0u && vendor_id == kVendorIdArm));
  out.is_armsx2_turnip =
      driver_id == kVkDriverIdMesaTurnip &&
      (ContainsCaseInsensitive(driver_info, "armsx2") ||
       ContainsCaseInsensitive(driver_name, "armsx2"));
  out.is_qualcomm_proprietary =
      driver_id == kVkDriverIdQualcommProprietary ||
      (driver_id == 0u && vendor_id == kVendorIdQualcomm);

  if (mali_driver_override && mali_driver_override[0] != '\0') {
    ParseArmMaliDriverVersion(mali_driver_override, out.mali_driver_major,
                              out.mali_driver_minor);
    out.is_arm_proprietary = true;
    out.is_panvk = false;
    out.is_malisx2 = false;
  } else if (out.is_arm_proprietary) {
    ParseArmMaliDriverVersion(driver_info, out.mali_driver_major,
                              out.mali_driver_minor);
  }
  return out;
}

struct MaliDriverQuirks {
  bool cap_spirv_13 = false;
  bool rewrite_scalar_select = false;
  bool allow_dynamic_rendering = true;
  bool allow_extended_dynamic_state = true;
  bool normalize_mixed_load_ops = false;
  bool allow_push_descriptor = true;
  bool enforce_host_coherent_readback = false;
};

// Centralized Arm Mali proprietary errata matrix (r34p0 / r40p0 / r47p0 /
// r54p1) vs open-source Mesa PanVK / malisx2 (which disables all proprietary
// Mali workarounds and permits VK_KHR_push_descriptor).
inline MaliDriverQuirks ComputeMaliQuirks(bool is_arm_proprietary,
                                          bool is_panvk, uint32_t major,
                                          uint32_t minor,
                                          bool is_job_manager) {
  MaliDriverQuirks q;
  if (is_panvk || !is_arm_proprietary) {
    return q;
  }
  const uint32_t rp = major * 100u + minor;
  const bool unknown_version = (major == 0u);
  q.cap_spirv_13 = unknown_version || major < 47u;
  q.rewrite_scalar_select = unknown_version || major < 47u;
  q.allow_dynamic_rendering = !unknown_version && major >= 52u;
  q.allow_extended_dynamic_state =
      !unknown_version && rp >= 4401u && major != 44u &&
      !(is_job_manager && major >= 46u && major <= 50u);
  q.normalize_mixed_load_ops = unknown_version || rp < 3801u;
  q.allow_push_descriptor = false;
  q.enforce_host_coherent_readback = true;
  return q;
}

// Strict '0'|'1' parser for Android debug.xendroid.* boolean properties.
inline bool ParseStrictBinaryBoolProp(const char* value, bool& parsed_out) {
  if (!value) {
    return false;
  }
  if (value[0] == '0' && value[1] == '\0') {
    parsed_out = false;
    return true;
  }
  if (value[0] == '1' && value[1] == '\0') {
    parsed_out = true;
    return true;
  }
  return false;
}

// Whitelist for opt-in armsx2-turnip TU_AUTOTUNE_ALGO diagnostic override.
inline bool IsValidTurnipAutotuneAlgo(const char* value) {
  if (!value || value[0] == '\0') {
    return false;
  }
  return std::strcmp(value, "gmem") == 0 ||
         std::strcmp(value, "sysmem") == 0 ||
         std::strcmp(value, "auto") == 0;
}

// Appends a comma-separated feature token to FD_DEV_FEATURES while preserving
// existing flags (such as enable_tp_ubwc_flag_hint=1) and avoiding duplicates.
inline std::string AppendFdDevFeature(const std::string& current,
                                      const char* feature) {
  if (!feature || feature[0] == '\0') {
    return current;
  }
  if (current.empty() || current == "none") {
    return std::string(feature);
  }
  if (current.find(feature) != std::string::npos) {
    return current;
  }
  return current + "," + feature;
}

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
