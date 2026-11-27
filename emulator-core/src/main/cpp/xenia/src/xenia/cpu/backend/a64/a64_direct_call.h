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
#include <cstddef>
#include <cstdint>

#include "xenia/base/memory.h"
#include "xenia/base/platform.h"
#include "xenia/base/platform_arm64.h"
#include "xenia/cpu/hir/opcodes.h"
#include "xenia/cpu/hir/value.h"

namespace xe {
namespace cpu {
namespace backend {
namespace a64 {

using xe::memory::ComputeCodeCacheNextCommitMark;
using xe::memory::TranslateCodeCacheWriteSlot;

inline constexpr uint32_t kA64NopInsn = 0xD503201Fu;
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

// Encodes MOVZ Wd, #imm16 (sf=0, opc=10, hw=00).
inline constexpr uint32_t EncodeA64MovzW(uint32_t rd, uint16_t imm16) {
  return 0x52800000u | (static_cast<uint32_t>(imm16) << 5) | (rd & 31u);
}

// Encodes MOVK Wd, #imm16, LSL #16 (sf=0, opc=11, hw=01).
inline constexpr uint32_t EncodeA64MovkW16(uint32_t rd, uint16_t imm16) {
  return 0x72A00000u | (static_cast<uint32_t>(imm16) << 5) | (rd & 31u);
}

inline bool DecodeA64MovzW(uint32_t insn, uint32_t* out_rd,
                           uint16_t* out_imm16) {
  if ((insn & 0xFFE00000u) != 0x52800000u) {
    return false;
  }
  if (out_rd) {
    *out_rd = insn & 31u;
  }
  if (out_imm16) {
    *out_imm16 = static_cast<uint16_t>((insn >> 5) & 0xFFFFu);
  }
  return true;
}

inline bool DecodeA64MovkW16(uint32_t insn, uint32_t* out_rd,
                             uint16_t* out_imm16) {
  if ((insn & 0xFFE00000u) != 0x72A00000u) {
    return false;
  }
  if (out_rd) {
    *out_rd = insn & 31u;
  }
  if (out_imm16) {
    *out_imm16 = static_cast<uint16_t>((insn >> 5) & 0xFFFFu);
  }
  return true;
}

// 1-slot monomorphic PIC state machine for CallIndirect (bctrl).
// Strict monotonic transitions: kUninitialized -> kPatched -> kFallback,
// or kUninitialized -> kFallback. Payload words (cmp_lo, cmp_hi, call_op) are
// written at most once while gate == `b pic_miss_stub` (unreachable by other
// threads), then published by atomically rewriting gate to NOP (0xD503201F).
// Invalidation rewrites ONLY gate to `b pic_fallback_lookup` (CMODX-safe).
enum class IndirectCallPicState : uint8_t {
  kUninitialized = 0,
  kPatched = 1,
  kFallback = 2,
};

struct IndirectCallPicSite {
  uintptr_t gate_rx = 0;
  uintptr_t cmp_lo_rx = 0;
  uintptr_t cmp_hi_rx = 0;
  uintptr_t call_op_rx = 0;
  uintptr_t fallback_rx = 0;
  uint32_t cached_guest_target = 0;
  bool is_tail = false;
  IndirectCallPicState state = IndirectCallPicState::kUninitialized;
};

struct IndirectCallPicPatchWords {
  uint32_t cmp_lo_insn = 0;
  uint32_t cmp_hi_insn = 0;
  uint32_t call_op_insn = 0;
  uint32_t gate_nop_insn = kA64NopInsn;
  size_t payload_flush_bytes = 0;
};

inline bool ComputeIndirectCallPicPatchWords(
    const IndirectCallPicSite& site, uint32_t guest_target_addr,
    uintptr_t target_rx, IndirectCallPicPatchWords* out_words) {
  if (!out_words || site.state != IndirectCallPicState::kUninitialized) {
    return false;
  }
  if (site.cmp_lo_rx < site.gate_rx + 4u || site.call_op_rx < site.cmp_hi_rx) {
    return false;
  }
  int64_t call_diff = 0;
  if (!ComputeA64Rel26Offset(site.call_op_rx, target_rx, &call_diff)) {
    return false;
  }
  out_words->cmp_lo_insn =
      EncodeA64MovzW(14u, static_cast<uint16_t>(guest_target_addr & 0xFFFFu));
  out_words->cmp_hi_insn =
      EncodeA64MovkW16(14u, static_cast<uint16_t>(guest_target_addr >> 16));
  out_words->call_op_insn = EncodeA64BranchRel26(call_diff, !site.is_tail);
  out_words->gate_nop_insn = kA64NopInsn;
  out_words->payload_flush_bytes =
      static_cast<size_t>((site.call_op_rx + sizeof(uint32_t)) - site.cmp_lo_rx);
  return true;
}

inline bool ComputeIndirectCallPicFallbackGate(
    const IndirectCallPicSite& site, uint32_t* out_gate_branch_insn) {
  if (!out_gate_branch_insn) {
    return false;
  }
  int64_t fb_diff = 0;
  if (!ComputeA64Rel26Offset(site.gate_rx, site.fallback_rx, &fb_diff)) {
    return false;
  }
  *out_gate_branch_insn = EncodeA64BranchRel26(fb_diff, false);
  return true;
}

// Hot function entry alignment: 64 B for functions >= 64 B when enabled,
// preserving 16 B alignment for tiny stubs or when disabled.
inline constexpr size_t ComputeFunctionEntryAlignment(bool align_hot_code,
                                                      size_t code_size) {
  return (align_hot_code && code_size >= 64u) ? 64u : 16u;
}

// Loop header alignment padding (bytes, multiple of 4):
// - If ((entry_mod_4k + current_offset) & 0xFFF) >= 0xFC0: nudge to the next
//   4 KB page boundary (strictly <= 64 B of NOPs, also 32 B aligned).
// - Otherwise: align to 32 B (<= 28 B of NOPs).
// Never performs blind 4 KB alignment (never emits up to 4092 B of NOPs).
inline constexpr size_t ComputeLoopHeaderAlignmentPaddingBytes(
    size_t current_offset, size_t entry_mod_4k) {
  if ((current_offset & 3u) != 0) {
    return 0;
  }
  const size_t page_off = ((entry_mod_4k & 0xFFFu) + current_offset) & 0xFFFu;
  if (page_off >= 0xFC0u) {
    return 0x1000u - page_off;
  }
  return (32u - (page_off & 31u)) & 31u;
}

// FEAT_LRCPC TSO load eligibility: scalar LOAD / LOAD_OFFSET only.
// V128, LVL, LVR, and RESERVED_LOAD always retain ldr + dmb(ISHLD).
inline constexpr bool IsScalarLrcpcLoadType(hir::TypeName type) {
  return type == hir::INT8_TYPE || type == hir::INT16_TYPE ||
         type == hir::INT32_TYPE || type == hir::INT64_TYPE ||
         type == hir::FLOAT32_TYPE || type == hir::FLOAT64_TYPE;
}

inline constexpr bool ShouldUseLrcpcForTsoLoad(bool guest_memory_tso,
                                               bool lrcpc_tso_enabled,
                                               uint64_t feature_flags,
                                               hir::Opcode opcode,
                                               hir::TypeName dest_type) {
  if (!guest_memory_tso || !lrcpc_tso_enabled) {
    return false;
  }
  if ((feature_flags & xe::arm64::kA64EmitLRCPC) == 0) {
    return false;
  }
  if (opcode != hir::OPCODE_LOAD && opcode != hir::OPCODE_LOAD_OFFSET) {
    return false;
  }
  return IsScalarLrcpcLoadType(dest_type);
}

}  // namespace a64
}  // namespace backend
}  // namespace cpu
}  // namespace xe

#endif  // XENIA_CPU_BACKEND_A64_A64_DIRECT_CALL_H_
