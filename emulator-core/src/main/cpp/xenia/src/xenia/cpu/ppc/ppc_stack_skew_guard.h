/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_CPU_PPC_PPC_STACK_SKEW_GUARD_H_
#define XENIA_CPU_PPC_PPC_STACK_SKEW_GUARD_H_

#include <cstdint>

namespace xe {
namespace cpu {
namespace ppc {

inline constexpr uint32_t kGuestStackRangeBegin = 0x70000000u;
inline constexpr uint32_t kGuestStackRangeEnd = 0x7F000000u;

constexpr bool IsSafeStackDispAccess(uint32_t ra, int64_t simm_offset,
                                     bool is_indexed) {
  return !is_indexed && ra == 1u && simm_offset >= -32768 &&
         simm_offset <= 32767;
}

constexpr bool IsGuestStackPointerInRange(uint32_t r1) {
  return r1 >= kGuestStackRangeBegin && r1 < kGuestStackRangeEnd;
}

constexpr uint32_t ApplyPhysicalSkewIfNeeded(uint32_t addr,
                                             uint32_t alloc_granularity,
                                             bool no_physical_skew) {
  if (!no_physical_skew && alloc_granularity > 0x1000u &&
      addr >= 0xE0000000u) {
    return addr + 0x1000u;
  }
  return addr;
}

}  // namespace ppc
}  // namespace cpu
}  // namespace xe

#endif  // XENIA_CPU_PPC_PPC_STACK_SKEW_GUARD_H_
