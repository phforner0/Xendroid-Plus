/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_CPU_PPC_PPC_JUMP_TABLE_H_
#define XENIA_CPU_PPC_PPC_JUMP_TABLE_H_

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace xe {
namespace cpu {
namespace ppc {

enum class JumpTableKind : uint8_t {
  kNone = 0,
  kAbsolute,
  kComputed,
  kByteOffset,
  kShortOffset,
};

struct JumpTableInfo {
  uint32_t bctr_address = 0;
  JumpTableKind kind = JumpTableKind::kNone;
  uint32_t table_address = 0;
  uint32_t base_address = 0;
  uint32_t count = 0;
  std::vector<uint32_t> targets;
};

using GuestReadU8Fn = bool (*)(void* ctx, uint32_t addr, uint8_t* out);
using GuestReadU16BEFn = bool (*)(void* ctx, uint32_t addr, uint16_t* out);
using GuestReadU32BEFn = bool (*)(void* ctx, uint32_t addr, uint32_t* out);
using CodeTargetValidFn = bool (*)(void* ctx, uint32_t target_addr);

struct JumpTableMemoryReader {
  void* ctx;
  GuestReadU8Fn read_u8;
  GuestReadU16BEFn read_u16;
  GuestReadU32BEFn read_u32;
  CodeTargetValidFn is_valid_code_target;
};

inline constexpr uint32_t kMaxJumpTableBacktrackInsns = 32;
inline constexpr uint32_t kMaxJumpTableEntries = 4096;
inline constexpr uint32_t kMaxJumpTableFunctionSpan = 0x100000u;

namespace detail {

constexpr uint32_t kInsnBctr = 0x4E800420u;

constexpr uint32_t PrimaryOpcode(uint32_t insn) { return (insn >> 26) & 0x3Fu; }
constexpr uint32_t FieldRT(uint32_t insn) { return (insn >> 21) & 0x1Fu; }
constexpr uint32_t FieldRA(uint32_t insn) { return (insn >> 16) & 0x1Fu; }
constexpr uint32_t FieldRB(uint32_t insn) { return (insn >> 11) & 0x1Fu; }
constexpr uint32_t FieldBO(uint32_t insn) { return (insn >> 21) & 0x1Fu; }
constexpr uint32_t FieldBI(uint32_t insn) { return (insn >> 16) & 0x1Fu; }
constexpr uint32_t FieldSH(uint32_t insn) { return (insn >> 11) & 0x1Fu; }
constexpr uint32_t FieldMB(uint32_t insn) { return (insn >> 6) & 0x1Fu; }
constexpr uint32_t FieldME(uint32_t insn) { return (insn >> 1) & 0x1Fu; }
constexpr uint32_t FieldXO10(uint32_t insn) { return (insn >> 1) & 0x3FFu; }
constexpr uint32_t FieldXO9(uint32_t insn) { return (insn >> 1) & 0x1FFu; }

constexpr int32_t SignExtend16(uint32_t v) {
  return static_cast<int32_t>(static_cast<int16_t>(v & 0xFFFFu));
}

constexpr bool IsMtctr(uint32_t insn, uint32_t* out_rs) {
  if ((insn & 0xFC1FFFFFu) == 0x7C0903A6u) {
    *out_rs = FieldRT(insn);
    return true;
  }
  return false;
}

constexpr bool IsHardControlBarrier(uint32_t insn) {
  if (insn == 0u) {
    return true;
  }
  const uint32_t op = PrimaryOpcode(insn);
  if (op == 17u) {  // sc
    return true;
  }
  if (op == 18u && (insn & 1u) != 0u) {  // bl / bla
    return true;
  }
  if (op == 16u && (insn & 1u) != 0u) {  // bcl / bcla
    return true;
  }
  if (op == 19u) {
    const uint32_t xo = FieldXO10(insn);
    if (xo == 16u || xo == 528u) {  // bclr / bclrl / bcctr / bcctrl
      return true;
    }
  }
  return false;
}

constexpr bool IsUnconditionalBranch(uint32_t insn) {
  return PrimaryOpcode(insn) == 18u && (insn & 1u) == 0u;
}

constexpr bool IsConditionalBranch(uint32_t insn) {
  return PrimaryOpcode(insn) == 16u && (insn & 1u) == 0u;
}

constexpr bool IsBranchSkippedByPrecedingBc(const uint32_t* window_insns,
                                            size_t start_idx,
                                            size_t branch_idx) {
  if (branch_idx == 0 || branch_idx <= start_idx) {
    return false;
  }
  const uint32_t prev = window_insns[branch_idx - 1];
  if (!IsConditionalBranch(prev)) {
    return false;
  }
  if ((prev & 2u) != 0u) {  // AA == 1
    return false;
  }
  const int32_t disp_bytes = static_cast<int16_t>(prev & 0xFFFCu);
  return disp_bytes == 8;
}

constexpr bool WritesCRField(uint32_t insn, uint32_t cr_field) {
  const uint32_t op = PrimaryOpcode(insn);
  if (op == 10u || op == 11u) {
    return ((insn >> 23) & 7u) == cr_field;
  }
  if (op == 13u || op == 28u || op == 29u) {
    return cr_field == 0u;
  }
  if (op == 20u || op == 21u || op == 23u || op == 30u) {
    return cr_field == 0u && (insn & 1u) != 0u;
  }
  if (op == 19u) {
    const uint32_t xo = FieldXO10(insn);
    if (xo == 0u || xo == 33u || xo == 129u || xo == 193u || xo == 225u ||
        xo == 257u || xo == 289u || xo == 417u || xo == 449u) {
      return (FieldRT(insn) >> 2) == cr_field;
    }
    return false;
  }
  if (op == 31u) {
    const uint32_t xo10 = FieldXO10(insn);
    if (xo10 == 0u || xo10 == 32u || xo10 == 512u) {
      return ((insn >> 23) & 7u) == cr_field;
    }
    if (xo10 == 144u) {  // mtcrf / mtocrf
      const uint32_t fxm = (insn >> 12) & 0xFFu;
      return (fxm & (0x80u >> cr_field)) != 0u;
    }
    return cr_field == 0u && (insn & 1u) != 0u;
  }
  return false;
}

constexpr bool WritesGpr(uint32_t insn, uint32_t reg) {
  const uint32_t op = PrimaryOpcode(insn);
  switch (op) {
    case 10u:  // cmpli
    case 11u:  // cmpi
    case 16u:  // bcx
    case 17u:  // sc
    case 18u:  // bx
    case 19u:  // bclrx / bcctrx / cr*
      return false;
    case 7u:   // mulli
    case 8u:   // subfic
    case 12u:  // addic
    case 13u:  // addic.
    case 14u:  // addi
    case 15u:  // addis
      return FieldRT(insn) == reg;
    case 20u:  // rlwimi
    case 21u:  // rlwinm
    case 23u:  // rlwnm
    case 24u:  // ori
    case 25u:  // oris
    case 26u:  // xori
    case 27u:  // xoris
    case 28u:  // andi.
    case 29u:  // andis.
    case 30u:  // rld*
      return FieldRA(insn) == reg;
    case 32u:  // lwz
    case 34u:  // lbz
    case 40u:  // lhz
    case 42u:  // lha
      return FieldRT(insn) == reg;
    case 33u:  // lwzu
    case 35u:  // lbzu
    case 41u:  // lhzu
    case 43u:  // lhau
      return FieldRT(insn) == reg || FieldRA(insn) == reg;
    case 36u:  // stw
    case 38u:  // stb
    case 44u:  // sth
    case 47u:  // stmw
    case 48u:  // lfs
    case 50u:  // lfd
    case 52u:  // stfs
    case 54u:  // stfd
      return false;
    case 37u:  // stwu
    case 39u:  // stbu
    case 45u:  // sthu
    case 49u:  // lfsu
    case 51u:  // lfdu
    case 53u:  // stfsu
    case 55u:  // stfdu
      return FieldRA(insn) == reg;
    case 46u:  // lmw
      return reg >= FieldRT(insn);
    case 58u:  // ld / ldu / lwa
      return FieldRT(insn) == reg ||
             ((insn & 3u) == 1u && FieldRA(insn) == reg);
    case 62u:  // std / stdu
      return (insn & 3u) == 1u && FieldRA(insn) == reg;
    case 31u: {
      const uint32_t xo10 = FieldXO10(insn);
      switch (xo10) {
        case 0u:     // cmp
        case 32u:    // cmpl
        case 4u:     // tw
        case 68u:    // td
        case 144u:   // mtcrf
        case 467u:   // mtspr
        case 598u:   // sync
        case 854u:   // eieio
        case 54u:    // dcbst
        case 86u:    // dcbf
        case 246u:   // dcbtst
        case 278u:   // dcbt
        case 982u:   // icbi
        case 1014u:  // dcbz
        case 149u:   // stdx
        case 150u:   // stwcx.
        case 151u:   // stwx
        case 214u:   // stdcx.
        case 215u:   // stbx
        case 407u:   // sthx
        case 660u:   // stdbrx
        case 662u:   // stwbrx
        case 918u:   // sthbrx
        case 535u:   // lfsx
        case 599u:   // lfdx
        case 663u:   // stfsx
        case 727u:   // stfdx
        case 983u:   // stfiwx
        case 7u:     // lvebx
        case 39u:    // lvehx
        case 71u:    // lvewx
        case 103u:   // lvx
        case 135u:   // stvebx
        case 167u:   // stvehx
        case 199u:   // stvewx
        case 231u:   // stvx
        case 359u:   // lvxl
        case 487u:   // stvxl
        case 519u:   // lvlx
        case 551u:   // lvrx
        case 775u:   // lvlxl
        case 807u:   // lvrxl
        case 391u:   // stvlx
        case 423u:   // stvrx
        case 647u:   // stvlxl
        case 679u:   // stvrxl
          return false;
        case 181u:  // stdux
        case 183u:  // stwux
        case 247u:  // stbux
        case 439u:  // sthux
        case 567u:  // lfsux
        case 631u:  // lfdux
        case 695u:  // stfsux
        case 759u:  // stfdux
          return FieldRA(insn) == reg;
        case 19u:   // mfcr
        case 20u:   // lwarx
        case 21u:   // ldx
        case 23u:   // lwzx
        case 83u:   // mfmsr
        case 84u:   // ldarx
        case 87u:   // lbzx
        case 279u:  // lhzx
        case 339u:  // mfspr
        case 341u:  // lwax
        case 343u:  // lhax
        case 532u:  // ldbrx
        case 534u:  // lwbrx
        case 790u:  // lhbrx
          return FieldRT(insn) == reg;
        case 53u:   // ldux
        case 55u:   // lwzux
        case 119u:  // lbzux
        case 311u:  // lhzux
        case 373u:  // lwaux
        case 375u:  // lhaux
          return FieldRT(insn) == reg || FieldRA(insn) == reg;
        case 24u:   // slw
        case 26u:   // cntlzw
        case 27u:   // sld
        case 28u:   // and
        case 58u:   // cntlzd
        case 60u:   // andc
        case 124u:  // nor
        case 284u:  // eqv
        case 316u:  // xor
        case 412u:  // orc
        case 444u:  // or
        case 476u:  // nand
        case 536u:  // srw
        case 539u:  // srd
        case 792u:  // sraw
        case 794u:  // srad
        case 824u:  // srawi
        case 922u:  // extsh
        case 954u:  // extsb
        case 986u:  // extsw
          return FieldRA(insn) == reg;
        default:
          break;
      }
      if ((xo10 >> 1) == 413u) {  // sradi
        return FieldRA(insn) == reg;
      }
      const uint32_t xo9 = FieldXO9(insn);
      switch (xo9) {
        case 8u:    // subfc
        case 9u:    // mulhdu
        case 10u:   // addc
        case 11u:   // mulhwu
        case 40u:   // subf
        case 73u:   // mulhd
        case 75u:   // mulhw
        case 104u:  // neg
        case 136u:  // subfe
        case 138u:  // adde
        case 200u:  // subfze
        case 202u:  // addze
        case 232u:  // subfme
        case 233u:  // mulld
        case 234u:  // addme
        case 235u:  // mullw
        case 266u:  // add
        case 457u:  // divdu
        case 459u:  // divwu
        case 489u:  // divd
        case 491u:  // divw
          return FieldRT(insn) == reg;
        default:
          return true;
      }
    }
    default:
      return true;
  }
}

inline bool ResolveConst32(const uint32_t* window_insns, size_t start_idx,
                           int from_idx, uint32_t reg, uint32_t* out_val,
                           int depth = 0) {
  if (depth > 8) {
    return false;
  }
  for (int i = from_idx; i >= static_cast<int>(start_idx); --i) {
    const uint32_t insn = window_insns[i];
    if (IsHardControlBarrier(insn)) {
      return false;
    }
    if (IsUnconditionalBranch(insn) &&
        !IsBranchSkippedByPrecedingBc(window_insns, start_idx,
                                      static_cast<size_t>(i))) {
      return false;
    }
    if (!WritesGpr(insn, reg)) {
      continue;
    }
    const uint32_t op = PrimaryOpcode(insn);
    if (op == 15u && FieldRT(insn) == reg) {  // addis / lis
      const uint32_t ra = FieldRA(insn);
      const uint32_t hi = (insn & 0xFFFFu) << 16;
      if (ra == 0u) {
        *out_val = hi;
        return true;
      }
      uint32_t base = 0;
      if (!ResolveConst32(window_insns, start_idx, i - 1, ra, &base,
                          depth + 1)) {
        return false;
      }
      *out_val = base + hi;
      return true;
    }
    if (op == 14u && FieldRT(insn) == reg) {  // addi / li
      const uint32_t ra = FieldRA(insn);
      const uint32_t simm = static_cast<uint32_t>(SignExtend16(insn));
      if (ra == 0u) {
        *out_val = simm;
        return true;
      }
      uint32_t base = 0;
      if (!ResolveConst32(window_insns, start_idx, i - 1, ra, &base,
                          depth + 1)) {
        return false;
      }
      *out_val = base + simm;
      return true;
    }
    if (op == 24u && FieldRA(insn) == reg) {  // ori
      const uint32_t rs = FieldRT(insn);
      uint32_t base = 0;
      if (!ResolveConst32(window_insns, start_idx, i - 1, rs, &base,
                          depth + 1)) {
        return false;
      }
      *out_val = base | (insn & 0xFFFFu);
      return true;
    }
    if (op == 25u && FieldRA(insn) == reg) {  // oris
      const uint32_t rs = FieldRT(insn);
      uint32_t base = 0;
      if (!ResolveConst32(window_insns, start_idx, i - 1, rs, &base,
                          depth + 1)) {
        return false;
      }
      *out_val = base | ((insn & 0xFFFFu) << 16);
      return true;
    }
    if (op == 31u && FieldXO10(insn) == 444u && FieldRA(insn) == reg &&
        FieldRT(insn) == FieldRB(insn)) {  // mr
      return ResolveConst32(window_insns, start_idx, i - 1, FieldRT(insn),
                            out_val, depth + 1);
    }
    return false;
  }
  return false;
}

inline bool FindLastGprDef(const uint32_t* window_insns, size_t start_idx,
                           int from_idx, uint32_t reg, int* out_def_idx) {
  for (int i = from_idx; i >= static_cast<int>(start_idx); --i) {
    const uint32_t insn = window_insns[i];
    if (IsHardControlBarrier(insn) || IsUnconditionalBranch(insn) ||
        IsConditionalBranch(insn)) {
      return false;
    }
    if (WritesGpr(insn, reg)) {
      *out_def_idx = i;
      return true;
    }
  }
  return false;
}

inline bool MatchLeftShiftRlwinm(uint32_t insn, uint32_t expected_dest,
                                 uint32_t* out_src, uint32_t* out_shift,
                                 uint32_t* out_mb, uint32_t* out_me) {
  if (PrimaryOpcode(insn) != 21u || FieldRA(insn) != expected_dest) {
    return false;
  }
  const uint32_t sh = FieldSH(insn);
  const uint32_t mb = FieldMB(insn);
  const uint32_t me = FieldME(insn);
  if (sh == 0u || me != 31u - sh || mb > me) {
    return false;
  }
  *out_src = FieldRT(insn);
  *out_shift = sh;
  *out_mb = mb;
  *out_me = me;
  return true;
}

inline bool CheckRlwinmWidthForMaxIndex(uint32_t mb, uint32_t me,
                                        uint32_t count) {
  const uint32_t width = me - mb + 1u;
  if (width >= 32u) {
    return true;
  }
  return (count - 1u) < (1ull << width);
}

inline bool FindBoundsCheck(const uint32_t* window_insns, size_t start_idx,
                            int idx_use_pos, size_t bctr_idx, uint32_t idx_reg,
                            uint32_t* out_count) {
  int bc_idx = -1;
  int uncond_b_idx = -1;
  uint32_t expected_cr = 0;
  bool is_gt_or_le = false;

  for (int i = idx_use_pos - 1; i >= static_cast<int>(start_idx); --i) {
    const uint32_t insn = window_insns[i];
    if (IsHardControlBarrier(insn) || WritesGpr(insn, idx_reg)) {
      return false;
    }
    if (IsUnconditionalBranch(insn)) {
      if (uncond_b_idx != -1) {
        return false;
      }
      uncond_b_idx = i;
      continue;
    }
    if (IsConditionalBranch(insn)) {
      const uint32_t bo = FieldBO(insn);
      if ((bo & 0x14u) != 0x04u) {
        return false;
      }
      const bool expect_true = (bo & 0x08u) != 0u;
      const uint32_t bi = FieldBI(insn);
      const uint32_t cr_field = bi >> 2;
      const uint32_t cr_bit = bi & 3u;
      const bool is_aa = (insn & 2u) != 0u;
      const int32_t disp_bytes = static_cast<int16_t>(insn & 0xFFFCu);

      if (uncond_b_idx != -1) {
        // Must be ble (expect_true=false, cr_bit=1) or blt
        // (expect_true=true, cr_bit=0) jumping past uncond_b_idx to dispatch.
        const bool is_ble = !expect_true && cr_bit == 1u;
        const bool is_blt = expect_true && cr_bit == 0u;
        if ((!is_ble && !is_blt) || is_aa || disp_bytes <= 0) {
          return false;
        }
        const int target_idx = i + (disp_bytes / 4);
        if (target_idx <= uncond_b_idx || target_idx > idx_use_pos) {
          return false;
        }
        bc_idx = i;
        expected_cr = cr_field;
        is_gt_or_le = is_ble;
        break;
      } else {
        // Must be bgt (expect_true=true, cr_bit=1) or bge
        // (expect_true=false, cr_bit=0) jumping out of the dispatch body.
        const bool is_bgt = expect_true && cr_bit == 1u;
        const bool is_bge = !expect_true && cr_bit == 0u;
        if (!is_bgt && !is_bge) {
          return false;
        }
        if (!is_aa) {
          const int target_idx = i + (disp_bytes / 4);
          if (target_idx > i && target_idx <= static_cast<int>(bctr_idx)) {
            return false;
          }
        }
        bc_idx = i;
        expected_cr = cr_field;
        is_gt_or_le = is_bgt;
        break;
      }
    }
  }

  if (bc_idx < 0) {
    return false;
  }

  for (int j = bc_idx - 1; j >= static_cast<int>(start_idx); --j) {
    const uint32_t insn = window_insns[j];
    if (IsHardControlBarrier(insn) || IsUnconditionalBranch(insn) ||
        IsConditionalBranch(insn) || WritesGpr(insn, idx_reg)) {
      return false;
    }
    if (PrimaryOpcode(insn) == 10u && ((insn >> 21) & 3u) == 0u &&
        ((insn >> 23) & 7u) == expected_cr && FieldRA(insn) == idx_reg) {
      const uint32_t imm = insn & 0xFFFFu;
      *out_count = is_gt_or_le ? (imm + 1u) : imm;
      return true;
    }
    if (WritesCRField(insn, expected_cr)) {
      return false;
    }
  }
  return false;
}

}  // namespace detail

inline bool TryRecoverJumpTable(uint32_t func_start, uint32_t bctr_address,
                                const uint32_t* window_insns,
                                size_t window_count,
                                const JumpTableMemoryReader& reader,
                                JumpTableInfo* out_table) {
  if (!out_table || !window_insns || window_count == 0 ||
      !reader.is_valid_code_target) {
    return false;
  }
  if ((func_start & 3u) != 0u || (bctr_address & 3u) != 0u ||
      bctr_address < func_start) {
    return false;
  }
  const size_t bctr_idx = window_count - 1;
  if (window_insns[bctr_idx] != detail::kInsnBctr) {
    return false;
  }
  const size_t start_idx =
      window_count > kMaxJumpTableBacktrackInsns
          ? (window_count - kMaxJumpTableBacktrackInsns)
          : 0;

  // Step 1: Locate mtctr rCtr before bctr.
  uint32_t ctr_reg = 0;
  int mtctr_idx = -1;
  for (int i = static_cast<int>(bctr_idx) - 1;
       i >= static_cast<int>(start_idx); --i) {
    const uint32_t insn = window_insns[i];
    if (detail::IsHardControlBarrier(insn) ||
        detail::IsUnconditionalBranch(insn) ||
        detail::IsConditionalBranch(insn)) {
      return false;
    }
    if (detail::IsMtctr(insn, &ctr_reg)) {
      mtctr_idx = i;
      break;
    }
  }
  if (mtctr_idx < 0) {
    return false;
  }

  // Step 2: Locate definition of ctr_reg.
  int ctr_def_idx = -1;
  if (!detail::FindLastGprDef(window_insns, start_idx, mtctr_idx - 1, ctr_reg,
                              &ctr_def_idx)) {
    return false;
  }
  const uint32_t ctr_def_insn = window_insns[ctr_def_idx];
  const uint32_t ctr_op = detail::PrimaryOpcode(ctr_def_insn);
  if (ctr_op != 31u) {
    return false;
  }

  JumpTableInfo candidate;
  candidate.bctr_address = bctr_address;

  uint32_t idx_reg = 0;
  int idx_use_pos = -1;
  uint32_t shift = 0;
  uint32_t rlwinm_mb = 0;
  uint32_t rlwinm_me = 31;
  bool has_idx_rlwinm = false;
  bool is_signed_short = false;
  bool matched = false;

  const uint32_t xo10 = detail::FieldXO10(ctr_def_insn);
  const uint32_t xo9 = detail::FieldXO9(ctr_def_insn);

  if (xo10 == 23u && detail::FieldRT(ctr_def_insn) == ctr_reg) {
    // Pattern 1: lwzx rCtr, rTable, rScaled (kAbsolute)
    const uint32_t ra = detail::FieldRA(ctr_def_insn);
    const uint32_t rb = detail::FieldRB(ctr_def_insn);
    const std::pair<uint32_t, uint32_t> pairs[2] = {{ra, rb}, {rb, ra}};
    for (int p = 0; p < 2; ++p) {
      const uint32_t tbl_reg = pairs[p].first;
      const uint32_t scl_reg = pairs[p].second;
      if (tbl_reg == 0u) {
        continue;
      }
      uint32_t tbl_addr = 0;
      if (!detail::ResolveConst32(window_insns, start_idx, ctr_def_idx - 1,
                                  tbl_reg, &tbl_addr)) {
        continue;
      }
      int scl_def_idx = -1;
      if (!detail::FindLastGprDef(window_insns, start_idx, ctr_def_idx - 1,
                                  scl_reg, &scl_def_idx)) {
        continue;
      }
      uint32_t src_reg = 0;
      uint32_t sh = 0;
      uint32_t mb = 0;
      uint32_t me = 0;
      if (!detail::MatchLeftShiftRlwinm(window_insns[scl_def_idx], scl_reg,
                                        &src_reg, &sh, &mb, &me) ||
          sh != 2u) {
        continue;
      }
      candidate.kind = JumpTableKind::kAbsolute;
      candidate.table_address = tbl_addr;
      idx_reg = src_reg;
      idx_use_pos = scl_def_idx;
      rlwinm_mb = mb;
      rlwinm_me = me;
      has_idx_rlwinm = true;
      matched = true;
      break;
    }
  } else if (xo9 == 266u && detail::FieldRT(ctr_def_insn) == ctr_reg) {
    // Pattern 2/3/4: add rCtr, rBase, rOff
    const uint32_t ra = detail::FieldRA(ctr_def_insn);
    const uint32_t rb = detail::FieldRB(ctr_def_insn);
    const std::pair<uint32_t, uint32_t> pairs[2] = {{ra, rb}, {rb, ra}};
    for (int p = 0; p < 2 && !matched; ++p) {
      const uint32_t base_reg = pairs[p].first;
      const uint32_t off_reg = pairs[p].second;
      uint32_t base_addr = 0;
      if (!detail::ResolveConst32(window_insns, start_idx, ctr_def_idx - 1,
                                  base_reg, &base_addr)) {
        continue;
      }
      int off_def_idx = -1;
      if (!detail::FindLastGprDef(window_insns, start_idx, ctr_def_idx - 1,
                                  off_reg, &off_def_idx)) {
        continue;
      }
      const uint32_t off_insn = window_insns[off_def_idx];
      uint32_t load_reg = off_reg;
      int load_idx = off_def_idx;
      uint32_t post_shift = 0;
      uint32_t post_mb = 0;
      uint32_t post_me = 31;
      bool has_post_rlwinm = false;

      if (detail::MatchLeftShiftRlwinm(off_insn, off_reg, &load_reg,
                                       &post_shift, &post_mb, &post_me)) {
        has_post_rlwinm = true;
        int prev_def = -1;
        if (detail::FindLastGprDef(window_insns, start_idx, off_def_idx - 1,
                                   load_reg, &prev_def)) {
          const uint32_t prev_insn = window_insns[prev_def];
          const uint32_t prev_xo = detail::FieldXO10(prev_insn);
          if (detail::PrimaryOpcode(prev_insn) == 31u &&
              detail::FieldRT(prev_insn) == load_reg &&
              (prev_xo == 87u || prev_xo == 279u || prev_xo == 343u)) {
            load_idx = prev_def;
          } else {
            load_idx = -1;
          }
        } else {
          load_idx = -1;
        }
      }

      if (load_idx >= 0) {
        const uint32_t load_insn = window_insns[load_idx];
        const uint32_t load_xo = detail::FieldXO10(load_insn);
        if (detail::PrimaryOpcode(load_insn) == 31u &&
            detail::FieldRT(load_insn) == load_reg) {
          const uint32_t lra = detail::FieldRA(load_insn);
          const uint32_t lrb = detail::FieldRB(load_insn);
          const std::pair<uint32_t, uint32_t> lpairs[2] = {{lra, lrb},
                                                           {lrb, lra}};
          if (load_xo == 87u) {  // lbzx -> kByteOffset
            if (!has_post_rlwinm || (post_mb == 0u && post_shift <= 24u)) {
              for (int lp = 0; lp < 2; ++lp) {
                const uint32_t tbl_reg = lpairs[lp].first;
                const uint32_t ix_reg = lpairs[lp].second;
                if (tbl_reg == 0u) {
                  continue;
                }
                uint32_t tbl_addr = 0;
                if (!detail::ResolveConst32(window_insns, start_idx,
                                            load_idx - 1, tbl_reg, &tbl_addr)) {
                  continue;
                }
                candidate.kind = JumpTableKind::kByteOffset;
                candidate.table_address = tbl_addr;
                candidate.base_address = base_addr;
                shift = post_shift;
                idx_reg = ix_reg;
                idx_use_pos = load_idx;
                has_idx_rlwinm = false;
                matched = true;
                break;
              }
            }
          } else if (load_xo == 279u || load_xo == 343u) {
            // lhzx / lhax -> kShortOffset
            if (!has_post_rlwinm || (post_mb == 0u && post_shift <= 24u)) {
              for (int lp = 0; lp < 2; ++lp) {
                const uint32_t tbl_reg = lpairs[lp].first;
                const uint32_t scl_reg = lpairs[lp].second;
                if (tbl_reg == 0u) {
                  continue;
                }
                uint32_t tbl_addr = 0;
                if (!detail::ResolveConst32(window_insns, start_idx,
                                            load_idx - 1, tbl_reg, &tbl_addr)) {
                  continue;
                }
                int scl_def_idx = -1;
                if (!detail::FindLastGprDef(window_insns, start_idx,
                                            load_idx - 1, scl_reg,
                                            &scl_def_idx)) {
                  continue;
                }
                uint32_t ix_reg = 0;
                uint32_t sh = 0;
                uint32_t mb = 0;
                uint32_t me = 0;
                if (!detail::MatchLeftShiftRlwinm(window_insns[scl_def_idx],
                                                  scl_reg, &ix_reg, &sh, &mb,
                                                  &me) ||
                    sh != 1u) {
                  continue;
                }
                candidate.kind = JumpTableKind::kShortOffset;
                candidate.table_address = tbl_addr;
                candidate.base_address = base_addr;
                shift = post_shift;
                is_signed_short = (load_xo == 343u);
                idx_reg = ix_reg;
                idx_use_pos = scl_def_idx;
                rlwinm_mb = mb;
                rlwinm_me = me;
                has_idx_rlwinm = true;
                matched = true;
                break;
              }
            }
          }
        }
      }

      if (!matched && has_post_rlwinm && load_idx < 0) {
        // Fallback to kComputed: rOff = rlwinm rIdx, shift
        candidate.kind = JumpTableKind::kComputed;
        candidate.base_address = base_addr;
        shift = post_shift;
        idx_reg = load_reg;
        idx_use_pos = off_def_idx;
        rlwinm_mb = post_mb;
        rlwinm_me = post_me;
        has_idx_rlwinm = true;
        matched = true;
      }
    }
  }

  if (!matched || idx_use_pos <= static_cast<int>(start_idx)) {
    return false;
  }

  // Step 3: Validate cmplwi + bc bounds check on idx_reg.
  uint32_t count = 0;
  if (!detail::FindBoundsCheck(window_insns, start_idx, idx_use_pos, bctr_idx,
                               idx_reg, &count)) {
    return false;
  }
  if (count == 0u || count > kMaxJumpTableEntries) {
    return false;
  }
  if (has_idx_rlwinm &&
      !detail::CheckRlwinmWidthForMaxIndex(rlwinm_mb, rlwinm_me, count)) {
    return false;
  }

  // Step 4: Decode and validate 100% of targets before mutating out_table.
  candidate.count = count;
  candidate.targets.reserve(count);
  for (uint32_t i = 0; i < count; ++i) {
    uint32_t target = 0;
    switch (candidate.kind) {
      case JumpTableKind::kAbsolute: {
        if (!reader.read_u32) {
          return false;
        }
        const uint32_t offset = i * 4u;
        if (candidate.table_address > UINT32_MAX - offset) {
          return false;
        }
        if (!reader.read_u32(reader.ctx, candidate.table_address + offset,
                             &target)) {
          return false;
        }
        break;
      }
      case JumpTableKind::kComputed: {
        if (shift >= 32u) {
          return false;
        }
        target = candidate.base_address + (i << shift);
        break;
      }
      case JumpTableKind::kByteOffset: {
        if (!reader.read_u8) {
          return false;
        }
        if (candidate.table_address > UINT32_MAX - i) {
          return false;
        }
        uint8_t byte_off = 0;
        if (!reader.read_u8(reader.ctx, candidate.table_address + i,
                            &byte_off)) {
          return false;
        }
        target = candidate.base_address +
                 (static_cast<uint32_t>(byte_off) << shift);
        break;
      }
      case JumpTableKind::kShortOffset: {
        if (!reader.read_u16) {
          return false;
        }
        const uint32_t offset = i * 2u;
        if (candidate.table_address > UINT32_MAX - offset) {
          return false;
        }
        uint16_t raw = 0;
        if (!reader.read_u16(reader.ctx, candidate.table_address + offset,
                             &raw)) {
          return false;
        }
        if (is_signed_short) {
          const int32_t signed_val = static_cast<int16_t>(raw);
          const uint32_t disp = static_cast<uint32_t>(signed_val) << shift;
          target = candidate.base_address + disp;
        } else {
          target = candidate.base_address +
                   (static_cast<uint32_t>(raw) << shift);
        }
        break;
      }
      default:
        return false;
    }

    if ((target & 3u) != 0u || target < func_start ||
        (target - func_start) >= kMaxJumpTableFunctionSpan ||
        !reader.is_valid_code_target(reader.ctx, target)) {
      return false;
    }
    candidate.targets.push_back(target);
  }

  *out_table = std::move(candidate);
  return true;
}

}  // namespace ppc
}  // namespace cpu
}  // namespace xe

#endif  // XENIA_CPU_PPC_PPC_JUMP_TABLE_H_
