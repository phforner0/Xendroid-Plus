#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef XE_DISABLE_ANDROID_DLOPEN
#define XE_DISABLE_ANDROID_DLOPEN 1
#endif

#include "xenia/cpu/backend/a64/a64_direct_call.h"
#include "xenia/cpu/hir/opcodes.h"
#include "xenia/cpu/ppc/ppc_hle_memory.h"
#include "xenia/cpu/ppc/ppc_jump_table.h"
#include "xenia/cpu/ppc/ppc_stack_skew_guard.h"

namespace {

#define WAVE3_CHECK(cond)                                                  \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "WAVE3_OPT check failed (%s:%d): %s\n",         \
                   __FILE__, __LINE__, #cond);                             \
      std::abort();                                                        \
    }                                                                      \
  } while (false)

using xe::cpu::backend::a64::DecodeA64BranchRel26;
using xe::cpu::backend::a64::EncodeA64BranchRel26;
using xe::cpu::backend::a64::IsA64Rel26InRange;
using xe::cpu::backend::a64::kA64BranchRel26MaxOffset;
using xe::cpu::backend::a64::kA64BranchRel26MinOffset;
using xe::cpu::backend::a64::PatchA64InstructionAtomic;
using xe::cpu::hir::LOAD_STORE_BYTE_SWAP;
using xe::cpu::hir::LOAD_STORE_NO_PHYSICAL_SKEW;
using xe::cpu::ppc::ApplyPhysicalSkewIfNeeded;
using xe::cpu::ppc::CheckGuestRange32NoWrap;
using xe::cpu::ppc::ComputePpcFunctionSignature;
using xe::cpu::ppc::CrossesSkew16KbBoundary;
using xe::cpu::ppc::HleMemoryOp;
using xe::cpu::ppc::IsGuestStackPointerInRange;
using xe::cpu::ppc::IsSafeStackDispAccess;
using xe::cpu::ppc::JumpTableInfo;
using xe::cpu::ppc::JumpTableKind;
using xe::cpu::ppc::JumpTableMemoryReader;
using xe::cpu::ppc::kGuestStackRangeBegin;
using xe::cpu::ppc::kGuestStackRangeEnd;
using xe::cpu::ppc::kMaxJumpTableBacktrackInsns;
using xe::cpu::ppc::kMaxJumpTableEntries;
using xe::cpu::ppc::MaskPpcRelocationBits;
using xe::cpu::ppc::MatchHleMemorySignature;
using xe::cpu::ppc::ParseHleOverrideEntry;
using xe::cpu::ppc::PhysicalRangesOverlap;
using xe::cpu::ppc::TryRecoverJumpTable;

std::string ReadSourceText(const std::filesystem::path& path) {
  std::ifstream in(path);
  WAVE3_CHECK(in.good());
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

// --- Pillar 1: Direct A64 Calls & Atomic Back-Patching ---

void TestDirectCallRangeAndEncoding() {
  static_assert(kA64BranchRel26MinOffset == -0x08000000LL);
  static_assert(kA64BranchRel26MaxOffset == 0x07FFFFFCLL);

  const uintptr_t site = 0x10000000ULL;
  WAVE3_CHECK(IsA64Rel26InRange(site, site));
  WAVE3_CHECK(IsA64Rel26InRange(site, site - 0x08000000ULL));
  WAVE3_CHECK(!IsA64Rel26InRange(site, site - 0x08000004ULL));
  WAVE3_CHECK(IsA64Rel26InRange(site, site + 0x07FFFFFCULL));
  WAVE3_CHECK(!IsA64Rel26InRange(site, site + 0x08000000ULL));

  // Unaligned call site or target must be rejected.
  WAVE3_CHECK(!IsA64Rel26InRange(site + 2, site + 0x100));
  WAVE3_CHECK(!IsA64Rel26InRange(site, site + 0x102));
  WAVE3_CHECK(!IsA64Rel26InRange(site, site + 0x103));

  // High 64-bit addresses (Android relocated high code cache VA).
  const uintptr_t high_site = 0x8002000000ULL;
  WAVE3_CHECK(IsA64Rel26InRange(high_site, high_site + 0x07FFFFFCULL));
  WAVE3_CHECK(IsA64Rel26InRange(high_site, high_site - 0x08000000ULL));
  WAVE3_CHECK(!IsA64Rel26InRange(high_site, high_site + 0x08000000ULL));
  WAVE3_CHECK(!IsA64Rel26InRange(high_site, high_site - 0x08000004ULL));

  // Exact A64 encodings for bl (link=true) and b (link=false).
  WAVE3_CHECK(EncodeA64BranchRel26(0x100, true) == 0x94000040u);
  WAVE3_CHECK(EncodeA64BranchRel26(-4, true) == 0x97FFFFFFu);
  WAVE3_CHECK(EncodeA64BranchRel26(-4, false) == 0x17FFFFFFu);
  WAVE3_CHECK(EncodeA64BranchRel26(0x07FFFFFCLL, false) == 0x15FFFFFFu);
  WAVE3_CHECK(EncodeA64BranchRel26(-0x08000000LL, true) == 0x96000000u);

  // Decode round-trips for bl / b and rejects non-branch opcodes.
  int64_t decoded_offset = 0;
  bool decoded_link = false;
  WAVE3_CHECK(DecodeA64BranchRel26(0x94000040u, &decoded_offset, &decoded_link));
  WAVE3_CHECK(decoded_offset == 0x100 && decoded_link == true);
  WAVE3_CHECK(DecodeA64BranchRel26(0x17FFFFFFu, &decoded_offset, &decoded_link));
  WAVE3_CHECK(decoded_offset == -4 && decoded_link == false);
  WAVE3_CHECK(DecodeA64BranchRel26(0x15FFFFFFu, &decoded_offset, &decoded_link));
  WAVE3_CHECK(decoded_offset == 0x07FFFFFCLL && decoded_link == false);
  WAVE3_CHECK(DecodeA64BranchRel26(0x96000000u, &decoded_offset, &decoded_link));
  WAVE3_CHECK(decoded_offset == -0x08000000LL && decoded_link == true);

  WAVE3_CHECK(!DecodeA64BranchRel26(0xD503201Fu, &decoded_offset, &decoded_link));
  WAVE3_CHECK(!DecodeA64BranchRel26(0xD4200000u, &decoded_offset, &decoded_link));

  // Atomic 32-bit instruction patch + alignment enforcement.
  alignas(8) uint8_t buf[16] = {};
  auto* aligned_slot = reinterpret_cast<uint32_t*>(buf);
  *aligned_slot = 0xD4200000u;
  WAVE3_CHECK(PatchA64InstructionAtomic(aligned_slot, 0x94000040u));
  WAVE3_CHECK(std::atomic_ref<uint32_t>(*aligned_slot)
                  .load(std::memory_order_acquire) == 0x94000040u);

  WAVE3_CHECK(!PatchA64InstructionAtomic(nullptr, 0x94000040u));
  auto* unaligned_slot = reinterpret_cast<uint32_t*>(buf + 2);
  WAVE3_CHECK(!PatchA64InstructionAtomic(unaligned_slot, 0x17FFFFFFu));
  // Original aligned slot must not be corrupted by rejected unaligned patch.
  WAVE3_CHECK(*aligned_slot == 0x94000040u);
}

// --- Pillar 2: HLE Memory Signature Matching & Range/Skew/Overlap Checks ---

void TestHleMemorySignatureAndValidation() {
  // MaskPpcRelocationBits: bl (opcode 18, LK=1, AA=0) masks LI bits [25:2].
  const uint32_t bl_fwd = 0x48001235u;   // bl +0x1234
  const uint32_t bl_back = 0x4BFFFC01u;  // bl -0x400
  WAVE3_CHECK(MaskPpcRelocationBits(bl_fwd) == 0x48000001u);
  WAVE3_CHECK(MaskPpcRelocationBits(bl_back) == 0x48000001u);

  // Non-call branches, bl with AA=1, and ordinary ALU/mem insns stay exact.
  WAVE3_CHECK(MaskPpcRelocationBits(0x48000010u) == 0x48000010u);  // b +0x10
  WAVE3_CHECK(MaskPpcRelocationBits(0x48001237u) == 0x48001237u);  // bla
  WAVE3_CHECK(MaskPpcRelocationBits(0x80640000u) == 0x80640000u);  // lwz
  WAVE3_CHECK(MaskPpcRelocationBits(0x90640000u) == 0x90640000u);  // stw
  WAVE3_CHECK(MaskPpcRelocationBits(0x4200FFF8u) == 0x4200FFF8u);  // bdnz

  // ComputePpcFunctionSignature: relocation-invariant & rejects invalid inputs.
  const std::array<uint32_t, 6> fn_a = {
      0x7C0802A6u, 0x48001235u, 0x7CA903A6u,
      0x88040000u, 0x98030000u, 0x4E800020u,
  };
  const std::array<uint32_t, 6> fn_b = {
      0x7C0802A6u, 0x4BFFFC01u, 0x7CA903A6u,
      0x88040000u, 0x98030000u, 0x4E800020u,
  };
  const std::array<uint32_t, 6> fn_diff = {
      0x7C0802A6u, 0x4BFFFC01u, 0x7CA903A6u,
      0x88040004u, 0x98030000u, 0x4E800020u,
  };
  const uint64_t sig_a = ComputePpcFunctionSignature(fn_a.data(), fn_a.size());
  const uint64_t sig_b = ComputePpcFunctionSignature(fn_b.data(), fn_b.size());
  const uint64_t sig_diff =
      ComputePpcFunctionSignature(fn_diff.data(), fn_diff.size());
  WAVE3_CHECK(sig_a != 0u);
  WAVE3_CHECK(sig_a == sig_b);
  WAVE3_CHECK(sig_diff != sig_a);

  // < 4 words or zero-word sentinel returns 0.
  WAVE3_CHECK(ComputePpcFunctionSignature(fn_a.data(), 3) == 0u);
  const std::array<uint32_t, 5> fn_zero = {0x7C0802A6u, 0x00000000u,
                                           0x7CA903A6u, 0x88040000u,
                                           0x4E800020u};
  WAVE3_CHECK(ComputePpcFunctionSignature(fn_zero.data(), fn_zero.size()) == 0u);

  // ParseHleOverrideEntry: comma/semicolon lists, exact addr+hash matching.
  const std::string overrides =
      "82100000:memcpy:0123456789ABCDEF;"
      "82100100:memset:fedcba9876543210,"
      "82100200:memmove:0011223344556677";
  WAVE3_CHECK(ParseHleOverrideEntry(overrides, 0x82100000u,
                                    0x0123456789ABCDEFULL) ==
              HleMemoryOp::kMemcpy);
  WAVE3_CHECK(ParseHleOverrideEntry(overrides, 0x82100100u,
                                    0xFEDCBA9876543210ULL) ==
              HleMemoryOp::kMemset);
  WAVE3_CHECK(ParseHleOverrideEntry(overrides, 0x82100200u,
                                    0x0011223344556677ULL) ==
              HleMemoryOp::kMemmove);
  WAVE3_CHECK(ParseHleOverrideEntry(overrides, 0x82100000u,
                                    0x0123456789ABCDE0ULL) ==
              HleMemoryOp::kNone);
  WAVE3_CHECK(ParseHleOverrideEntry(overrides, 0x82100004u,
                                    0x0123456789ABCDEFULL) ==
              HleMemoryOp::kNone);

  // MatchHleMemorySignature exercises all 5 canonical CRT/XDK signatures.
  {
    const std::array<uint32_t, 7> memset7 = {
        0x28050000u, 0x4D820020u, 0x7CA903A6u, 0x38C3FFFFu,
        0x9C860001u, 0x4200FFFCu, 0x4E800020u,
    };
    uint64_t sig = 0;
    WAVE3_CHECK(MatchHleMemorySignature(0x82001000u, memset7.data(),
                                        memset7.size(), "", &sig) ==
                HleMemoryOp::kMemset);
    WAVE3_CHECK(sig == 0x54242DB4826F145BULL);

    // Non-blr termination must not match canonical entry.
    auto bad_blr = memset7;
    bad_blr.back() = 0x60000000u;
    WAVE3_CHECK(MatchHleMemorySignature(0x82001000u, bad_blr.data(),
                                        bad_blr.size(), "") ==
                HleMemoryOp::kNone);
  }
  {
    const std::array<uint32_t, 8> memset8 = {
        0x28050000u, 0x4D820020u, 0x7CA903A6u, 0x38C3FFFFu,
        0x5484063Eu, 0x9C860001u, 0x4200FFFCu, 0x4E800020u,
    };
    uint64_t sig = 0;
    WAVE3_CHECK(MatchHleMemorySignature(0x82001100u, memset8.data(),
                                        memset8.size(), "", &sig) ==
                HleMemoryOp::kMemset);
    WAVE3_CHECK(sig == 0x69CF984463B81CBAULL);
  }
  {
    const std::array<uint32_t, 9> memcpy9a = {
        0x28050000u, 0x4D820020u, 0x7CA903A6u, 0x38C3FFFFu, 0x3884FFFFu,
        0x8C040001u, 0x9C060001u, 0x4200FFF8u, 0x4E800020u,
    };
    uint64_t sig = 0;
    WAVE3_CHECK(MatchHleMemorySignature(0x82001200u, memcpy9a.data(),
                                        memcpy9a.size(), "", &sig) ==
                HleMemoryOp::kMemcpy);
    WAVE3_CHECK(sig == 0x012460B3136CA583ULL);
  }
  {
    const std::array<uint32_t, 9> memcpy9b = {
        0x28050000u, 0x4D820020u, 0x7CA903A6u, 0x38C00000u, 0x7C0430AEu,
        0x7C0331AEu, 0x38C60001u, 0x4200FFF4u, 0x4E800020u,
    };
    uint64_t sig = 0;
    WAVE3_CHECK(MatchHleMemorySignature(0x82001300u, memcpy9b.data(),
                                        memcpy9b.size(), "", &sig) ==
                HleMemoryOp::kMemcpy);
    WAVE3_CHECK(sig == 0xAAEF509FA1FA0939ULL);
  }
  {
    const std::array<uint32_t, 19> memmove19 = {
        0x28050000u, 0x4D820020u, 0x7C032040u, 0x4D820020u, 0x41810020u,
        0x7CA903A6u, 0x38C3FFFFu, 0x3884FFFFu, 0x8C040001u, 0x9C060001u,
        0x4200FFF8u, 0x4E800020u, 0x7CA903A6u, 0x7CC32A14u, 0x7C842A14u,
        0x8C04FFFFu, 0x9C06FFFFu, 0x4200FFF8u, 0x4E800020u,
    };
    uint64_t sig = 0;
    WAVE3_CHECK(MatchHleMemorySignature(0x82001400u, memmove19.data(),
                                        memmove19.size(), "", &sig) ==
                HleMemoryOp::kMemmove);
    WAVE3_CHECK(sig == 0x99597BF7D25DF86EULL);

    // Extra word changes word count and must reject.
    std::array<uint32_t, 20> memmove20 = {};
    std::copy(memmove19.begin(), memmove19.end(), memmove20.begin());
    memmove20[19] = 0x4E800020u;
    WAVE3_CHECK(MatchHleMemorySignature(0x82001400u, memmove20.data(),
                                        memmove20.size(), "") ==
                HleMemoryOp::kNone);
  }

  // MatchHleMemorySignature exercises override + out_hash propagation.
  char hex_buf[48];
  std::snprintf(hex_buf, sizeof(hex_buf), "82200000:memcpy:%016llx",
                static_cast<unsigned long long>(sig_a));
  uint64_t out_hash = 0;
  WAVE3_CHECK(MatchHleMemorySignature(0x82200000u, fn_b.data(), fn_b.size(),
                                      hex_buf, &out_hash) ==
              HleMemoryOp::kMemcpy);
  WAVE3_CHECK(out_hash == sig_a);

  // CheckGuestRange32NoWrap.
  WAVE3_CHECK(CheckGuestRange32NoWrap(0xFFFFFFFFu, 1u));
  WAVE3_CHECK(!CheckGuestRange32NoWrap(0xFFFFFFFFu, 2u));
  WAVE3_CHECK(!CheckGuestRange32NoWrap(0x1000u, 0u));
  WAVE3_CHECK(CheckGuestRange32NoWrap(0x80000000u, 0x1000u));
  WAVE3_CHECK(CheckGuestRange32NoWrap(0xF0000000u, 0x10000000u));
  WAVE3_CHECK(!CheckGuestRange32NoWrap(0xF0000000u, 0x10000001u));

  // CrossesSkew16KbBoundary.
  WAVE3_CHECK(CrossesSkew16KbBoundary(0xDFFFF000u, 0x2000u, 16384u));
  WAVE3_CHECK(!CrossesSkew16KbBoundary(0xDFFFF000u, 0x1000u, 16384u));
  WAVE3_CHECK(!CrossesSkew16KbBoundary(0xE0000000u, 0x8000u, 16384u));
  WAVE3_CHECK(!CrossesSkew16KbBoundary(0xDFFFF000u, 0x2000u, 4096u));

  // PhysicalRangesOverlap (catches virtual alias collisions e.g. 0xA0010000 vs 0xE0010000).
  WAVE3_CHECK(PhysicalRangesOverlap(0x00010000u, 0x00010000u, 64u));
  WAVE3_CHECK(PhysicalRangesOverlap(0x00010000u, 0x00010020u, 64u));
  WAVE3_CHECK(PhysicalRangesOverlap(0x00010020u, 0x00010000u, 64u));
  WAVE3_CHECK(!PhysicalRangesOverlap(0x00010000u, 0x00010040u, 64u));
  WAVE3_CHECK(!PhysicalRangesOverlap(0x00010040u, 0x00010000u, 64u));
  WAVE3_CHECK(!PhysicalRangesOverlap(0x00010000u, 0x00010000u, 0u));
}

// --- Pillar 3: Static Jump Table Recovery in PPCScanner ---

struct MockGuestMemory {
  std::vector<std::pair<uint32_t, std::vector<uint8_t>>> blobs;
  uint32_t text_start = 0x82000000u;
  uint32_t text_end = 0x82100000u;

  void PutU32BE(uint32_t addr, uint32_t val) {
    std::vector<uint8_t> bytes = {
        static_cast<uint8_t>((val >> 24) & 0xFFu),
        static_cast<uint8_t>((val >> 16) & 0xFFu),
        static_cast<uint8_t>((val >> 8) & 0xFFu),
        static_cast<uint8_t>(val & 0xFFu),
    };
    blobs.emplace_back(addr, std::move(bytes));
  }

  void PutU16BE(uint32_t addr, uint16_t val) {
    std::vector<uint8_t> bytes = {
        static_cast<uint8_t>((val >> 8) & 0xFFu),
        static_cast<uint8_t>(val & 0xFFu),
    };
    blobs.emplace_back(addr, std::move(bytes));
  }

  void PutU8(uint32_t addr, uint8_t val) {
    blobs.emplace_back(addr, std::vector<uint8_t>{val});
  }

  bool ReadByte(uint32_t addr, uint8_t* out) const {
    for (const auto& [base, data] : blobs) {
      if (addr >= base && addr < base + data.size()) {
        *out = data[addr - base];
        return true;
      }
    }
    return false;
  }

  static bool ReadU8(void* ctx, uint32_t addr, uint8_t* out) {
    return static_cast<const MockGuestMemory*>(ctx)->ReadByte(addr, out);
  }
  static bool ReadU16BE(void* ctx, uint32_t addr, uint16_t* out) {
    uint8_t b0 = 0, b1 = 0;
    const auto* m = static_cast<const MockGuestMemory*>(ctx);
    if (!m->ReadByte(addr, &b0) || !m->ReadByte(addr + 1, &b1)) {
      return false;
    }
    *out = static_cast<uint16_t>((uint16_t(b0) << 8) | b1);
    return true;
  }
  static bool ReadU32BE(void* ctx, uint32_t addr, uint32_t* out) {
    uint8_t b0 = 0, b1 = 0, b2 = 0, b3 = 0;
    const auto* m = static_cast<const MockGuestMemory*>(ctx);
    if (!m->ReadByte(addr, &b0) || !m->ReadByte(addr + 1, &b1) ||
        !m->ReadByte(addr + 2, &b2) || !m->ReadByte(addr + 3, &b3)) {
      return false;
    }
    *out = (uint32_t(b0) << 24) | (uint32_t(b1) << 16) |
           (uint32_t(b2) << 8) | uint32_t(b3);
    return true;
  }
  static bool IsValidCode(void* ctx, uint32_t addr) {
    const auto* m = static_cast<const MockGuestMemory*>(ctx);
    return addr >= m->text_start && addr < m->text_end;
  }

  JumpTableMemoryReader Reader() {
    return JumpTableMemoryReader{this, &ReadU8, &ReadU16BE, &ReadU32BE,
                                 &IsValidCode};
  }
};

constexpr uint32_t PpcCmplwi(uint32_t cr, uint32_t ra, uint16_t uimm) {
  return (10u << 26) | ((cr & 7u) << 23) | ((ra & 31u) << 16) | uimm;
}
constexpr uint32_t PpcBgt(uint32_t cr, int16_t byte_disp) {
  return (16u << 26) | (12u << 21) | (((cr & 7u) * 4u + 1u) << 16) |
         (static_cast<uint32_t>(byte_disp) & 0xFFFCu);
}
constexpr uint32_t PpcBle(uint32_t cr, int16_t byte_disp) {
  return (16u << 26) | (4u << 21) | (((cr & 7u) * 4u + 1u) << 16) |
         (static_cast<uint32_t>(byte_disp) & 0xFFFCu);
}
constexpr uint32_t PpcB(int32_t byte_disp) {
  return (18u << 26) | (static_cast<uint32_t>(byte_disp) & 0x03FFFFFCu);
}
constexpr uint32_t PpcBl(int32_t byte_disp) {
  return PpcB(byte_disp) | 1u;
}
constexpr uint32_t PpcRlwinm(uint32_t ra, uint32_t rs, uint32_t sh, uint32_t mb,
                             uint32_t me) {
  return (21u << 26) | ((rs & 31u) << 21) | ((ra & 31u) << 16) |
         ((sh & 31u) << 11) | ((mb & 31u) << 6) | ((me & 31u) << 1);
}
constexpr uint32_t PpcLis(uint32_t rd, uint16_t hi16) {
  return (15u << 26) | ((rd & 31u) << 21) | hi16;
}
constexpr uint32_t PpcAddi(uint32_t rd, uint32_t ra, int16_t simm) {
  return (14u << 26) | ((rd & 31u) << 21) | ((ra & 31u) << 16) |
         (static_cast<uint32_t>(simm) & 0xFFFFu);
}
constexpr uint32_t PpcOri(uint32_t ra, uint32_t rs, uint16_t uimm) {
  return (24u << 26) | ((rs & 31u) << 21) | ((ra & 31u) << 16) | uimm;
}
constexpr uint32_t PpcLisAddiHi(uint32_t addr) {
  return ((addr + 0x8000u) >> 16) & 0xFFFFu;
}
constexpr int16_t PpcLisAddiLo(uint32_t addr) {
  return static_cast<int16_t>(addr & 0xFFFFu);
}
constexpr uint32_t PpcLwzx(uint32_t rd, uint32_t ra, uint32_t rb) {
  return (31u << 26) | ((rd & 31u) << 21) | ((ra & 31u) << 16) |
         ((rb & 31u) << 11) | (23u << 1);
}
constexpr uint32_t PpcLbzx(uint32_t rd, uint32_t ra, uint32_t rb) {
  return (31u << 26) | ((rd & 31u) << 21) | ((ra & 31u) << 16) |
         ((rb & 31u) << 11) | (87u << 1);
}
constexpr uint32_t PpcLhzx(uint32_t rd, uint32_t ra, uint32_t rb) {
  return (31u << 26) | ((rd & 31u) << 21) | ((ra & 31u) << 16) |
         ((rb & 31u) << 11) | (279u << 1);
}
constexpr uint32_t PpcLhax(uint32_t rd, uint32_t ra, uint32_t rb) {
  return (31u << 26) | ((rd & 31u) << 21) | ((ra & 31u) << 16) |
         ((rb & 31u) << 11) | (343u << 1);
}
constexpr uint32_t PpcAdd(uint32_t rd, uint32_t ra, uint32_t rb) {
  return (31u << 26) | ((rd & 31u) << 21) | ((ra & 31u) << 16) |
         ((rb & 31u) << 11) | (266u << 1);
}
constexpr uint32_t PpcMtctr(uint32_t rs) {
  return 0x7C0903A6u | ((rs & 31u) << 21);
}
constexpr uint32_t kPpcNop = 0x60000000u;
constexpr uint32_t kPpcBctr = 0x4E800420u;

void TestJumpTableRecoveryPatterns() {
  static_assert(kMaxJumpTableBacktrackInsns == 32);
  static_assert(kMaxJumpTableEntries == 4096);

  const uint32_t func_start = 0x82010000u;

  // Pattern 1: kAbsolute (lwzx of 32-bit target addresses).
  {
    MockGuestMemory mem;
    const uint32_t table_addr = 0x82008020u;
    const std::array<uint32_t, 4> expected = {0x82010040u, 0x82010058u,
                                              0x82010070u, 0x82010088u};
    for (uint32_t i = 0; i < 4; ++i) {
      mem.PutU32BE(table_addr + i * 4, expected[i]);
    }
    const std::array<uint32_t, 8> insns = {
        PpcCmplwi(6, 3, 3),
        PpcBgt(6, 0x80),
        PpcRlwinm(11, 3, 2, 0, 29),
        PpcLis(10, PpcLisAddiHi(table_addr)),
        PpcAddi(10, 10, PpcLisAddiLo(table_addr)),
        PpcLwzx(11, 10, 11),
        PpcMtctr(11),
        kPpcBctr,
    };
    JumpTableInfo info;
    WAVE3_CHECK(TryRecoverJumpTable(func_start, func_start + 0x20, insns.data(),
                                    insns.size(), mem.Reader(), &info));
    WAVE3_CHECK(info.kind == JumpTableKind::kAbsolute);
    WAVE3_CHECK(info.count == 4u && info.targets.size() == 4u);
    for (size_t i = 0; i < 4; ++i) {
      WAVE3_CHECK(info.targets[i] == expected[i]);
    }
  }

  // Pattern 2: kComputed (lis + ori + scaled index + add).
  {
    MockGuestMemory mem;
    const uint32_t base_addr = 0x82010100u;
    const std::array<uint32_t, 7> insns = {
        PpcCmplwi(0, 3, 2),
        PpcBgt(0, 0x40),
        PpcRlwinm(11, 3, 3, 0, 28),
        PpcLis(10, static_cast<uint16_t>(base_addr >> 16)),
        PpcOri(10, 10, static_cast<uint16_t>(base_addr & 0xFFFFu)),
        PpcAdd(11, 10, 11),
        PpcMtctr(11),
    };
    std::vector<uint32_t> with_bctr(insns.begin(), insns.end());
    with_bctr.push_back(kPpcBctr);

    JumpTableInfo info;
    WAVE3_CHECK(TryRecoverJumpTable(func_start, func_start + 0x40,
                                    with_bctr.data(), with_bctr.size(),
                                    mem.Reader(), &info));
    WAVE3_CHECK(info.kind == JumpTableKind::kComputed);
    WAVE3_CHECK(info.count == 3u && info.targets.size() == 3u);
    WAVE3_CHECK(info.targets[0] == base_addr + 0u);
    WAVE3_CHECK(info.targets[1] == base_addr + 8u);
    WAVE3_CHECK(info.targets[2] == base_addr + 16u);
  }

  // Pattern 3: kByteOffset (lbzx + rlwinm << 2 + base).
  {
    MockGuestMemory mem;
    const uint32_t table_addr = 0x82008800u;
    const uint32_t base_addr = 0x82010200u;
    const std::array<uint8_t, 4> offs = {0u, 3u, 6u, 11u};
    for (uint32_t i = 0; i < 4; ++i) {
      mem.PutU8(table_addr + i, offs[i]);
    }
    const std::array<uint32_t, 10> insns = {
        PpcCmplwi(0, 3, 3),
        PpcBgt(0, 0x40),
        PpcLis(9, PpcLisAddiHi(table_addr)),
        PpcAddi(9, 9, PpcLisAddiLo(table_addr)),
        PpcLbzx(11, 9, 3),
        PpcRlwinm(11, 11, 2, 0, 29),
        PpcLis(10, PpcLisAddiHi(base_addr)),
        PpcAddi(10, 10, PpcLisAddiLo(base_addr)),
        PpcAdd(11, 10, 11),
        PpcMtctr(11),
    };
    std::vector<uint32_t> with_bctr(insns.begin(), insns.end());
    with_bctr.push_back(kPpcBctr);

    JumpTableInfo info;
    WAVE3_CHECK(TryRecoverJumpTable(func_start, func_start + 0x50,
                                    with_bctr.data(), with_bctr.size(),
                                    mem.Reader(), &info));
    WAVE3_CHECK(info.kind == JumpTableKind::kByteOffset);
    WAVE3_CHECK(info.count == 4u && info.targets.size() == 4u);
    for (size_t i = 0; i < 4; ++i) {
      WAVE3_CHECK(info.targets[i] == base_addr + uint32_t(offs[i]) * 4u);
    }
  }

  // Pattern 4a: kShortOffset unsigned (ble dispatch + b default + lhzx + rlwinm << 2).
  {
    MockGuestMemory mem;
    const uint32_t table_addr = 0x82009000u;
    const uint32_t base_addr = 0x82010300u;
    const std::array<uint16_t, 4> offs = {0u, 4u, 9u, 20u};
    for (uint32_t i = 0; i < 4; ++i) {
      mem.PutU16BE(table_addr + i * 2, offs[i]);
    }
    const std::array<uint32_t, 12> insns = {
        PpcCmplwi(0, 3, 3),
        PpcBle(0, 8),
        PpcB(0x40),
        PpcRlwinm(11, 3, 1, 0, 30),
        PpcLis(9, PpcLisAddiHi(table_addr)),
        PpcAddi(9, 9, PpcLisAddiLo(table_addr)),
        PpcLhzx(11, 9, 11),
        PpcRlwinm(11, 11, 2, 0, 29),
        PpcLis(10, PpcLisAddiHi(base_addr)),
        PpcAddi(10, 10, PpcLisAddiLo(base_addr)),
        PpcAdd(11, 10, 11),
        PpcMtctr(11),
    };
    std::vector<uint32_t> with_bctr(insns.begin(), insns.end());
    with_bctr.push_back(kPpcBctr);

    JumpTableInfo info;
    WAVE3_CHECK(TryRecoverJumpTable(func_start, func_start + 0x60,
                                    with_bctr.data(), with_bctr.size(),
                                    mem.Reader(), &info));
    WAVE3_CHECK(info.kind == JumpTableKind::kShortOffset);
    WAVE3_CHECK(info.count == 4u && info.targets.size() == 4u);
    for (size_t i = 0; i < 4; ++i) {
      WAVE3_CHECK(info.targets[i] == base_addr + uint32_t(offs[i]) * 4u);
    }
  }

  // Pattern 4b: kShortOffset signed negative (lhax direct byte offset).
  {
    MockGuestMemory mem;
    const uint32_t table_addr = 0x82009200u;
    const uint32_t base_addr = 0x82010400u;
    const std::array<int16_t, 3> signed_offs = {-0x40, -0x20, 0x10};
    for (uint32_t i = 0; i < 3; ++i) {
      mem.PutU16BE(table_addr + i * 2, static_cast<uint16_t>(signed_offs[i]));
    }
    const std::array<uint32_t, 10> insns = {
        PpcCmplwi(0, 3, 2),
        PpcBgt(0, 0x40),
        PpcRlwinm(11, 3, 1, 0, 30),
        PpcLis(9, PpcLisAddiHi(table_addr)),
        PpcAddi(9, 9, PpcLisAddiLo(table_addr)),
        PpcLhax(11, 9, 11),
        PpcLis(10, PpcLisAddiHi(base_addr)),
        PpcAddi(10, 10, PpcLisAddiLo(base_addr)),
        PpcAdd(11, 10, 11),
        PpcMtctr(11),
    };
    std::vector<uint32_t> with_bctr(insns.begin(), insns.end());
    with_bctr.push_back(kPpcBctr);

    JumpTableInfo info;
    WAVE3_CHECK(TryRecoverJumpTable(func_start, func_start + 0x70,
                                    with_bctr.data(), with_bctr.size(),
                                    mem.Reader(), &info));
    WAVE3_CHECK(info.kind == JumpTableKind::kShortOffset);
    WAVE3_CHECK(info.count == 3u && info.targets.size() == 3u);
    for (size_t i = 0; i < 3; ++i) {
      const uint32_t expected =
          static_cast<uint32_t>(int32_t(base_addr) + signed_offs[i]);
      WAVE3_CHECK(info.targets[i] == expected);
    }
  }

  // Negative case 1: cmplwi > 32 instructions before bctr (window capped at 32).
  {
    MockGuestMemory mem;
    const uint32_t table_addr = 0x82008020u;
    for (uint32_t i = 0; i < 4; ++i) {
      mem.PutU32BE(table_addr + i * 4, 0x82010040u + i * 4);
    }
    std::vector<uint32_t> insns;
    insns.push_back(PpcCmplwi(0, 3, 3));
    insns.push_back(PpcBgt(0, 0x80));
    for (int i = 0; i < 27; ++i) {
      insns.push_back(kPpcNop);
    }
    insns.push_back(PpcRlwinm(11, 3, 2, 0, 29));
    insns.push_back(PpcLis(10, PpcLisAddiHi(table_addr)));
    insns.push_back(PpcAddi(10, 10, PpcLisAddiLo(table_addr)));
    insns.push_back(PpcLwzx(11, 10, 11));
    insns.push_back(PpcMtctr(11));
    insns.push_back(kPpcBctr);
    // Slice last 32 instructions (excludes cmplwi + bgt).
    const uint32_t* tail = insns.data() + (insns.size() - kMaxJumpTableBacktrackInsns);
    JumpTableInfo info;
    WAVE3_CHECK(!TryRecoverJumpTable(func_start, func_start + 0x100, tail,
                                     kMaxJumpTableBacktrackInsns, mem.Reader(),
                                     &info));
  }

  // Negative case 2: count > kMaxJumpTableEntries (4096).
  {
    MockGuestMemory mem;
    const uint32_t table_addr = 0x82008020u;
    const std::array<uint32_t, 8> insns = {
        PpcCmplwi(0, 3, 4096),  // count = 4097 > 4096
        PpcBgt(0, 0x80),
        PpcRlwinm(11, 3, 2, 0, 29),
        PpcLis(10, PpcLisAddiHi(table_addr)),
        PpcAddi(10, 10, PpcLisAddiLo(table_addr)),
        PpcLwzx(11, 10, 11),
        PpcMtctr(11),
        kPpcBctr,
    };
    JumpTableInfo info;
    WAVE3_CHECK(!TryRecoverJumpTable(func_start, func_start + 0x20, insns.data(),
                                     insns.size(), mem.Reader(), &info));
  }

  // Negative case 3: unaligned target (target & 3 != 0).
  {
    MockGuestMemory mem;
    const uint32_t table_addr = 0x82008020u;
    mem.PutU32BE(table_addr + 0, 0x82010040u);
    mem.PutU32BE(table_addr + 4, 0x82010042u);  // unaligned!
    const std::array<uint32_t, 8> insns = {
        PpcCmplwi(0, 3, 1),
        PpcBgt(0, 0x80),
        PpcRlwinm(11, 3, 2, 0, 29),
        PpcLis(10, PpcLisAddiHi(table_addr)),
        PpcAddi(10, 10, PpcLisAddiLo(table_addr)),
        PpcLwzx(11, 10, 11),
        PpcMtctr(11),
        kPpcBctr,
    };
    JumpTableInfo info;
    WAVE3_CHECK(!TryRecoverJumpTable(func_start, func_start + 0x20, insns.data(),
                                     insns.size(), mem.Reader(), &info));
  }

  // Negative case 4: target outside .text or < func_start.
  {
    MockGuestMemory mem;
    const uint32_t table_addr = 0x82008020u;
    mem.PutU32BE(table_addr + 0, 0x82010040u);
    mem.PutU32BE(table_addr + 4, 0x82200000u);  // outside [text_start, text_end)
    const std::array<uint32_t, 8> insns = {
        PpcCmplwi(0, 3, 1),
        PpcBgt(0, 0x80),
        PpcRlwinm(11, 3, 2, 0, 29),
        PpcLis(10, PpcLisAddiHi(table_addr)),
        PpcAddi(10, 10, PpcLisAddiLo(table_addr)),
        PpcLwzx(11, 10, 11),
        PpcMtctr(11),
        kPpcBctr,
    };
    JumpTableInfo info;
    WAVE3_CHECK(!TryRecoverJumpTable(func_start, func_start + 0x20, insns.data(),
                                     insns.size(), mem.Reader(), &info));

    // Also test target < func_start.
    mem.blobs.clear();
    mem.PutU32BE(table_addr + 0, 0x82010040u);
    mem.PutU32BE(table_addr + 4, 0x8200FFC0u);  // in .text, but < func_start
    WAVE3_CHECK(!TryRecoverJumpTable(func_start, func_start + 0x20, insns.data(),
                                     insns.size(), mem.Reader(), &info));
  }

  // Negative case 5: table read failure (MMIO / unmapped).
  {
    MockGuestMemory mem;
    const uint32_t table_addr = 0x82008020u;
    mem.PutU32BE(table_addr + 0, 0x82010040u);
    // Entry 1 missing -> read_u32 returns false.
    const std::array<uint32_t, 8> insns = {
        PpcCmplwi(0, 3, 1),
        PpcBgt(0, 0x80),
        PpcRlwinm(11, 3, 2, 0, 29),
        PpcLis(10, PpcLisAddiHi(table_addr)),
        PpcAddi(10, 10, PpcLisAddiLo(table_addr)),
        PpcLwzx(11, 10, 11),
        PpcMtctr(11),
        kPpcBctr,
    };
    JumpTableInfo info;
    WAVE3_CHECK(!TryRecoverJumpTable(func_start, func_start + 0x20, insns.data(),
                                     insns.size(), mem.Reader(), &info));
  }

  // Negative case 6: intervening bl between cmplwi and mtctr.
  {
    MockGuestMemory mem;
    const uint32_t table_addr = 0x82008020u;
    mem.PutU32BE(table_addr + 0, 0x82010040u);
    mem.PutU32BE(table_addr + 4, 0x82010050u);
    const std::array<uint32_t, 9> insns = {
        PpcCmplwi(0, 3, 1),
        PpcBgt(0, 0x80),
        PpcBl(0x100),  // clobbers volatile regs / CTR!
        PpcRlwinm(11, 3, 2, 0, 29),
        PpcLis(10, PpcLisAddiHi(table_addr)),
        PpcAddi(10, 10, PpcLisAddiLo(table_addr)),
        PpcLwzx(11, 10, 11),
        PpcMtctr(11),
        kPpcBctr,
    };
    JumpTableInfo info;
    WAVE3_CHECK(!TryRecoverJumpTable(func_start, func_start + 0x24, insns.data(),
                                     insns.size(), mem.Reader(), &info));
  }

  // Negative case 7: post-load rlwinm with post_mb > 0 must be rejected.
  {
    MockGuestMemory mem;
    const uint32_t table_addr = 0x82008800u;
    const uint32_t base_addr = 0x82010200u;
    for (uint32_t i = 0; i < 4; ++i) {
      mem.PutU8(table_addr + i, static_cast<uint8_t>(i * 4u));
    }
    const std::array<uint32_t, 11> insns = {
        PpcCmplwi(0, 3, 3),
        PpcBgt(0, 0x40),
        PpcLis(9, PpcLisAddiHi(table_addr)),
        PpcAddi(9, 9, PpcLisAddiLo(table_addr)),
        PpcLbzx(11, 9, 3),
        PpcRlwinm(11, 11, 2, 25, 29),  // post_mb = 25 > 0 -> masks high bits!
        PpcLis(10, PpcLisAddiHi(base_addr)),
        PpcAddi(10, 10, PpcLisAddiLo(base_addr)),
        PpcAdd(11, 10, 11),
        PpcMtctr(11),
        kPpcBctr,
    };
    JumpTableInfo info;
    WAVE3_CHECK(!TryRecoverJumpTable(func_start, func_start + 0x50, insns.data(),
                                     insns.size(), mem.Reader(), &info));
  }
  {
    MockGuestMemory mem;
    const uint32_t table_addr = 0x82009200u;
    const uint32_t base_addr = 0x82010400u;
    for (uint32_t i = 0; i < 3; ++i) {
      mem.PutU16BE(table_addr + i * 2, static_cast<uint16_t>(0x10u));
    }
    const std::array<uint32_t, 12> insns = {
        PpcCmplwi(0, 3, 2),
        PpcBgt(0, 0x40),
        PpcRlwinm(11, 3, 1, 0, 30),
        PpcLis(9, PpcLisAddiHi(table_addr)),
        PpcAddi(9, 9, PpcLisAddiLo(table_addr)),
        PpcLhax(11, 9, 11),
        PpcRlwinm(11, 11, 2, 16, 29),  // post_mb = 16 > 0 -> clears sign bits!
        PpcLis(10, PpcLisAddiHi(base_addr)),
        PpcAddi(10, 10, PpcLisAddiLo(base_addr)),
        PpcAdd(11, 10, 11),
        PpcMtctr(11),
        kPpcBctr,
    };
    JumpTableInfo info;
    WAVE3_CHECK(!TryRecoverJumpTable(func_start, func_start + 0x70, insns.data(),
                                     insns.size(), mem.Reader(), &info));
  }
}

// --- Pillar 4: Stack r1 + simm16 Skew Fixup Guard & HIR Flag ---

void TestStackSkewGuardAndHirFlags() {
  static_assert(kGuestStackRangeBegin == 0x70000000u);
  static_assert(kGuestStackRangeEnd == 0x7F000000u);

  static_assert(IsSafeStackDispAccess(1, -0x80, false));
  static_assert(IsSafeStackDispAccess(1, 0x7FFF, false));
  static_assert(IsSafeStackDispAccess(1, -32768, false));
  static_assert(!IsSafeStackDispAccess(0, 0x10, false));
  static_assert(!IsSafeStackDispAccess(3, 0x10, false));
  static_assert(!IsSafeStackDispAccess(1, 0, true));
  static_assert(!IsSafeStackDispAccess(1, 32768, false));
  static_assert(!IsSafeStackDispAccess(1, -32769, false));

  static_assert(IsGuestStackPointerInRange(0x70000000u));
  static_assert(IsGuestStackPointerInRange(0x7EFFFFFFu));
  static_assert(!IsGuestStackPointerInRange(0x7F000000u));
  static_assert(!IsGuestStackPointerInRange(0xE0001000u));
  static_assert(!IsGuestStackPointerInRange(0x6FFFFFFCu));

  WAVE3_CHECK(ApplyPhysicalSkewIfNeeded(0xE0001000u, 16384u, false) ==
              0xE0002000u);
  WAVE3_CHECK(ApplyPhysicalSkewIfNeeded(0xE0001000u, 16384u, true) ==
              0xE0001000u);
  WAVE3_CHECK(ApplyPhysicalSkewIfNeeded(0x70001000u, 16384u, false) ==
              0x70001000u);
  WAVE3_CHECK(ApplyPhysicalSkewIfNeeded(0xE0001000u, 4096u, false) ==
              0xE0001000u);

  // HIR flag orthogonality across MemorySequenceCombinationPass XOR toggle.
  static_assert(LOAD_STORE_NO_PHYSICAL_SKEW != 0);
  static_assert((LOAD_STORE_NO_PHYSICAL_SKEW & LOAD_STORE_BYTE_SWAP) == 0);
  uint32_t flags = LOAD_STORE_NO_PHYSICAL_SKEW | LOAD_STORE_BYTE_SWAP;
  flags ^= LOAD_STORE_BYTE_SWAP;
  WAVE3_CHECK((flags & LOAD_STORE_NO_PHYSICAL_SKEW) != 0);
  WAVE3_CHECK((flags & LOAD_STORE_BYTE_SWAP) == 0);
  flags ^= LOAD_STORE_BYTE_SWAP;
  WAVE3_CHECK((flags & LOAD_STORE_NO_PHYSICAL_SKEW) != 0);
  WAVE3_CHECK((flags & LOAD_STORE_BYTE_SWAP) != 0);
}

// --- Contract Source Checks: Wave 3 CVars are TOML-only ---

void TestTomlOnlyCvarContracts(const std::filesystem::path& cpp_dir) {
  const std::filesystem::path repo_root =
      cpp_dir.parent_path().parent_path().parent_path().parent_path();
  const std::filesystem::path schema_path =
      repo_root / "app/src/main/java/xendroid/compose/settings/SettingsSchema.kt";
  const std::filesystem::path catalog_path =
      repo_root / "app/src/main/java/xendroid/compose/settings/SettingCatalog.kt";
  WAVE3_CHECK(std::filesystem::exists(schema_path));
  WAVE3_CHECK(std::filesystem::exists(catalog_path));

  const std::string schema_kt = ReadSourceText(schema_path);
  const std::string catalog_kt = ReadSourceText(catalog_path);

  const std::string a64_backend_cc =
      ReadSourceText(cpp_dir / "xenia/src/xenia/cpu/backend/a64/a64_backend.cc");
  const std::string ppc_hle_cc =
      ReadSourceText(cpp_dir / "xenia/src/xenia/cpu/ppc/ppc_hle_memory.cc");
  const std::string ppc_scanner_cc =
      ReadSourceText(cpp_dir / "xenia/src/xenia/cpu/ppc/ppc_scanner.cc");
  const std::string ppc_mem_cc =
      ReadSourceText(cpp_dir / "xenia/src/xenia/cpu/ppc/ppc_emit_memory.cc");

  const std::array<std::pair<const char*, const std::string*>, 6> cvars = {{
      {"a64_direct_calls", &a64_backend_cc},
      {"ppc_hle_memory_ops", &ppc_hle_cc},
      {"ppc_hle_memory_functions", &ppc_hle_cc},
      {"ppc_hle_memory_min_bytes", &ppc_hle_cc},
      {"ppc_scanner_jump_tables", &ppc_scanner_cc},
      {"a64_skip_stack_skew_fixup", &ppc_mem_cc},
  }};

  for (const auto& [name, src] : cvars) {
    WAVE3_CHECK(src->find(name) != std::string::npos);
    WAVE3_CHECK(schema_kt.find(name) == std::string::npos);
    WAVE3_CHECK(catalog_kt.find(name) == std::string::npos);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "Usage: %s <emulator-core/src/main/cpp>\n", argv[0]);
    return 2;
  }
  const std::filesystem::path cpp_dir(argv[1]);

  TestDirectCallRangeAndEncoding();
  TestHleMemorySignatureAndValidation();
  TestJumpTableRecoveryPatterns();
  TestStackSkewGuardAndHirFlags();
  TestTomlOnlyCvarContracts(cpp_dir);

  std::puts("wave3_jit_hle_logic_test: PASS");
  return 0;
}
