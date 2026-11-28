/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_KERNEL_XBOXKRNL_CRITICAL_SECTION_SPIN_H_
#define XENIA_KERNEL_XBOXKRNL_CRITICAL_SECTION_SPIN_H_

#include <cstdint>

#if defined(_MSC_VER)
#if defined(_M_ARM64)
#include <intrin.h>
#elif defined(_M_X64)
#include <immintrin.h>
#endif
#endif

namespace xe {
namespace kernel {
namespace xboxkrnl {

constexpr uint32_t kDefaultCriticalSectionMaxSpin = 64;

inline uint32_t ComputeCriticalSectionSpinLimit(uint32_t guest_spin_count,
                                                uint32_t max_spin_cap) {
  if (max_spin_cap > 0 && guest_spin_count > max_spin_cap) {
    return max_spin_cap;
  }
  return guest_spin_count;
}

inline void PauseCriticalSectionSpin() {
#if defined(__aarch64__) || defined(_M_ARM64)
#if defined(_MSC_VER)
  __isb(_ARM64_BARRIER_SY);
#else
  __asm__ __volatile__("isb sy" ::: "memory");
#endif
#elif defined(__x86_64__) || defined(_M_X64)
#if defined(_MSC_VER)
  _mm_pause();
#else
  __builtin_ia32_pause();
#endif
#endif
}

}  // namespace xboxkrnl
}  // namespace kernel
}  // namespace xe

#endif  // XENIA_KERNEL_XBOXKRNL_CRITICAL_SECTION_SPIN_H_
