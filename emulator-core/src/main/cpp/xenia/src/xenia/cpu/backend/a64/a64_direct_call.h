/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_CPU_BACKEND_A64_A64_DIRECT_CALL_H_
#define XENIA_CPU_BACKEND_A64_A64_DIRECT_CALL_H_

#include <atomic>
#include <cstdint>

namespace xe {
namespace cpu {
namespace backend {
namespace a64 {

inline constexpr int64_t kA64BranchRel26MinOffset = -0x08000000LL;
inline constexpr int64_t kA64BranchRel26MaxOffset = 0x07FFFFFCLL;

// Checks whether a PC-relative A64 B/BL (imm26 * 4) from |call_site_rx| to
// |target_rx| is 4-byte aligned and within [-128 MiB, +128 MiB - 4].
// Uses unsigned magnitude subtraction to avoid signed overflow on high 64-bit
// virtual addresses.
inline bool IsA64Rel26InRange(uintptr_t call_site_rx, uintptr_t target_rx) {
  if (((call_site_rx | target_rx) & 3u) != 0) {
    return false;
  }
  if (target_rx >= call_site_rx) {
    return (target_rx - call_site_rx) <=
           static_cast<uintptr_t>(kA64BranchRel26MaxOffset);
  }
  return (call_site_rx - target_rx) <=
         static_cast<uintptr_t>(-kA64BranchRel26MinOffset);
}

// Safely computes the signed byte offset (|target_rx| - |call_site_rx|) when
// within rel26 range.
inline bool ComputeA64Rel26Offset(uintptr_t call_site_rx, uintptr_t target_rx,
                                  int64_t* out_offset) {
  if (!IsA64Rel26InRange(call_site_rx, target_rx)) {
    return false;
  }
  if (out_offset) {
    if (target_rx >= call_site_rx) {
      *out_offset = static_cast<int64_t>(target_rx - call_site_rx);
    } else {
      *out_offset = -static_cast<int64_t>(call_site_rx - target_rx);
    }
  }
  return true;
}

// Encodes an A64 unconditional branch: BL (link=true, 0x94000000) or B
// (link=false, 0x14000000) with a 4-byte aligned byte offset in rel26 range.
inline constexpr uint32_t EncodeA64BranchRel26(int64_t byte_offset, bool link) {
  const uint32_t imm26 =
      static_cast<uint32_t>(byte_offset >> 2) & 0x03FFFFFFu;
  return (link ? 0x94000000u : 0x14000000u) | imm26;
}

// Decodes an A64 B or BL instruction. Returns false for any other opcode.
inline bool DecodeA64BranchRel26(uint32_t insn, int64_t* out_offset,
                                 bool* out_link) {
  if ((insn & 0x7C000000u) != 0x14000000u) {
    return false;
  }
  const uint32_t raw26 = insn & 0x03FFFFFFu;
  const int32_t imm26 = (raw26 & 0x02000000u) != 0
                            ? static_cast<int32_t>(raw26 | 0xFC000000u)
                            : static_cast<int32_t>(raw26);
  if (out_offset) {
    *out_offset = static_cast<int64_t>(imm26) * 4;
  }
  if (out_link) {
    *out_link = (insn & 0x80000000u) != 0;
  }
  return true;
}

// Atomically patches a single 32-bit A64 instruction word in a 4-byte aligned
// RW code-cache slot using std::atomic_ref<uint32_t> (avoids -Watomic-alignment
// under -Werror).
inline bool PatchA64InstructionAtomic(uint32_t* rw_slot, uint32_t new_insn) {
  if (!rw_slot || (reinterpret_cast<uintptr_t>(rw_slot) & 3u) != 0) {
    return false;
  }
  std::atomic_ref<uint32_t>(*rw_slot).store(new_insn,
                                            std::memory_order_release);
  return true;
}

}  // namespace a64
}  // namespace backend
}  // namespace cpu
}  // namespace xe

#endif  // XENIA_CPU_BACKEND_A64_A64_DIRECT_CALL_H_
