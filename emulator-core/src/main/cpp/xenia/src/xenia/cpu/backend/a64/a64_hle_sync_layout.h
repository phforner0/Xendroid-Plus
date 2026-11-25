/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_CPU_BACKEND_A64_A64_HLE_SYNC_LAYOUT_H_
#define XENIA_CPU_BACKEND_A64_A64_HLE_SYNC_LAYOUT_H_

#include <cstdint>
#include <string_view>

namespace xe {
namespace cpu {
namespace backend {
namespace a64 {

enum class InlineHleSyncKind : uint8_t {
  kNone = 0,
  kKeRaiseIrqlToDpcLevel,
  kKfLowerIrql,
  kRtlEnterCriticalSection,
  kRtlLeaveCriticalSection,
  kRtlTryEnterCriticalSection,
  kKeAcquireSpinLockAtRaisedIrql,
  kKeReleaseSpinLockFromRaisedIrql,
  kKfAcquireSpinLock,
  kKfReleaseSpinLock,
};

// Verified by static_assert in xboxkrnl_rtl.cc and xboxkrnl_threading.cc.
constexpr uint32_t kKpcrCurrentIrqlOffset = 0x18;
constexpr uint32_t kKpcrCurrentThreadOffset = 0x100;
constexpr uint32_t kRtlCsLockCountOffset = 0x10;
constexpr uint32_t kRtlCsRecursionCountOffset = 0x14;
constexpr uint32_t kRtlCsOwningThreadOffset = 0x18;
constexpr uint32_t kSpinLockOwnerOffset = 0x0;
constexpr uint32_t kGuestPhysicalShiftBoundary = 0xE0000000u;

inline InlineHleSyncKind ClassifyInlineHleSync(std::string_view export_name) {
  if (export_name == "KeRaiseIrqlToDpcLevel") {
    return InlineHleSyncKind::kKeRaiseIrqlToDpcLevel;
  }
  if (export_name == "KfLowerIrql") {
    return InlineHleSyncKind::kKfLowerIrql;
  }
  if (export_name == "RtlEnterCriticalSection") {
    return InlineHleSyncKind::kRtlEnterCriticalSection;
  }
  if (export_name == "RtlLeaveCriticalSection") {
    return InlineHleSyncKind::kRtlLeaveCriticalSection;
  }
  if (export_name == "RtlTryEnterCriticalSection") {
    return InlineHleSyncKind::kRtlTryEnterCriticalSection;
  }
  if (export_name == "KeAcquireSpinLockAtRaisedIrql") {
    return InlineHleSyncKind::kKeAcquireSpinLockAtRaisedIrql;
  }
  if (export_name == "KeReleaseSpinLockFromRaisedIrql") {
    return InlineHleSyncKind::kKeReleaseSpinLockFromRaisedIrql;
  }
  if (export_name == "KfAcquireSpinLock") {
    return InlineHleSyncKind::kKfAcquireSpinLock;
  }
  if (export_name == "KfReleaseSpinLock") {
    return InlineHleSyncKind::kKfReleaseSpinLock;
  }
  return InlineHleSyncKind::kNone;
}

inline bool ShouldEmitInlineHleSync(bool cvar_enabled, bool hle_census_active,
                                    uint32_t export_tags,
                                    uint32_t log_tag_mask,
                                    bool has_wake_trampoline,
                                    InlineHleSyncKind kind) {
  if (!cvar_enabled || hle_census_active || (export_tags & log_tag_mask) != 0 ||
      kind == InlineHleSyncKind::kNone) {
    return false;
  }
  if (kind == InlineHleSyncKind::kRtlLeaveCriticalSection &&
      !has_wake_trampoline) {
    return false;
  }
  return true;
}

}  // namespace a64
}  // namespace backend
}  // namespace cpu
}  // namespace xe

#endif  // XENIA_CPU_BACKEND_A64_A64_HLE_SYNC_LAYOUT_H_
