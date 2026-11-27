/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_BASE_PLATFORM_ARM64_H_
#define XENIA_BASE_PLATFORM_ARM64_H_
#include <cstdint>

namespace xe {
namespace arm64 {
enum A64FeatureFlags : uint64_t {
  kA64EmitLSE = 1 << 0,
  kA64FZFlushesInputs = 1 << 1,
  kA64EmitLRCPC = 1 << 2,
};

// Linux/Android AT_HWCAP bit 26 (HWCAP_LRCPC) for ARMv8.3-A FEAT_LRCPC.
inline constexpr uint64_t kArm64HwcapLrcpcBit = 1ULL << 26;

inline constexpr uint64_t DetectLrcpcFeatureFlag(uint64_t hwcap,
                                                 int64_t extension_mask) {
  const uint64_t lrcpc_flag = static_cast<uint64_t>(kA64EmitLRCPC);
  if ((static_cast<uint64_t>(extension_mask) & lrcpc_flag) != lrcpc_flag) {
    return 0;
  }
  return (hwcap & kArm64HwcapLrcpcBit) != 0 ? lrcpc_flag : uint64_t{0};
}

XE_NOALIAS
uint64_t GetFeatureFlags();
XE_COLD
void InitFeatureFlags();

}  // namespace arm64
}  // namespace xe

#endif  // XENIA_BASE_PLATFORM_ARM64_H_
