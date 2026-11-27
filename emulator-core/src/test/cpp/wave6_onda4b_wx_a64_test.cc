// Wave 6 (Onda 4B) W^X Dual-Mapping & A64 JIT Refinement TDD tests.
// Validates pure-logic invariants and source-level safety contracts for:
//   1. W^X Dual-Mapping: live SELinux probe (memfd_create -> ASharedMemory),
//      strict '0'|'1' prop parsing, WriteSlotForExecuteAddress commit bounds,
//      PlaceData execute VA, EnsureCommitted (no POSIX mprotect churn),
//      guest_trampoline dual-mapping + FlushCodeRange, Breakpoint RW+ic ivau.
//   2. 1-slot Monomorphic PIC in CallIndirect: 2-stage 1-word atomic CMODX
//      (kUninitialized -> kPatched -> kFallback), movz/movk/bl/b encodings,
//      +-128 MiB rel26 bounds, RevertDirectCallSitesInRange safe rollback.
//   3. Hot Code Alignment: 64 B function entry alignment (>= 64 B), 32 B loop
//      header alignment + 4 KB anti-page-split nudge (strictly <= 64 B NOPs,
//      never blind 4092 B alignment).
//   4. FEAT_LRCPC in a64_guest_memory_tso: AT_HWCAP bit 15 detection honoring
//      a64_extension_mask, scalar LOAD/LOAD_OFFSET selection, exclusion of
//      V128/LVL/LVR/RESERVED_LOAD, preserving -march=armv8-a.
//   5. 16 KB Skew < 0xA0000000 analysis in a64_stack_skew_analysis.h:
//      constant/lis/SDA r13 positive cases, >= 0x9FF00000 / underflow /
//      offset overflow / mutated r13 / post-call negative cases.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "xenia/base/memory.h"
#include "xenia/base/platform.h"
#include "xenia/base/platform_arm64.h"
#include "xenia/cpu/backend/a64/a64_direct_call.h"
#include "xenia/cpu/backend/a64/a64_stack_skew_analysis.h"
#include "xenia/cpu/hir/opcodes.h"

namespace {

#define WAVE6_4B_CHECK(cond)                                                 \
  do {                                                                       \
    if (!(cond)) {                                                           \
      std::fprintf(stderr, "WAVE6_4B check failed (%s:%d): %s\n", __FILE__,  \
                   __LINE__, #cond);                                         \
      std::abort();                                                          \
    }                                                                        \
  } while (false)

namespace fs = std::filesystem;
namespace a64 = xe::cpu::backend::a64;
namespace hir = xe::cpu::hir;

std::string NormalizeNewlines(std::string s) {
  s.erase(std::remove(s.begin(), s.end(), '\r'), s.end());
  return s;
}

std::string ReadSourceText(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  WAVE6_4B_CHECK(in.good());
  std::ostringstream ss;
  ss << in.rdbuf();
  return NormalizeNewlines(ss.str());
}

std::string ExtractFunctionBody(const std::string& src,
                                const std::string& anchor) {
  const size_t start = src.find(anchor);
  WAVE6_4B_CHECK(start != std::string::npos);
  const size_t open = src.find('{', start);
  WAVE6_4B_CHECK(open != std::string::npos);
  int depth = 0;
  for (size_t i = open; i < src.size(); ++i) {
    if (src[i] == '{') {
      ++depth;
    } else if (src[i] == '}') {
      --depth;
      if (depth == 0) {
        return src.substr(open, i - open + 1);
      }
    }
  }
  WAVE6_4B_CHECK(false && "Unmatched brace while extracting function body");
  return {};
}

// ---------------------------------------------------------------------------
// 1. W^X Dual-Mapping pure logic: prop parsing, probe decision, write-slot
//    translation, PlaceData VA mapping, and EnsureCommitted boundaries.
// ---------------------------------------------------------------------------
void TestWxDualMappingContracts() {
  using xe::memory::DualMappingWxBackend;
  using xe::memory::kDualMappingWxProbeSentinel;
  using xe::memory::ParseStrictBoolProp;
  using xe::memory::ShouldActivateWxDualMapping;

  // Sentinel must be ARM64 NOP (0xD503201F).
  WAVE6_4B_CHECK(kDualMappingWxProbeSentinel == 0xD503201Fu);

  // Strict '0' | '1' parser must accept ONLY exact "0" or "1".
  bool val = true;
  WAVE6_4B_CHECK(ParseStrictBoolProp("0", &val) && val == false);
  WAVE6_4B_CHECK(ParseStrictBoolProp("1", &val) && val == true);

  val = false;
  for (const char* bad :
       {"", "true", "false", "2", "01", "10", " 1", "1 ", "yes", "on"}) {
    WAVE6_4B_CHECK(!ParseStrictBoolProp(bad, &val));
    WAVE6_4B_CHECK(val == false);
  }
  WAVE6_4B_CHECK(!ParseStrictBoolProp(nullptr, &val));
  WAVE6_4B_CHECK(!ParseStrictBoolProp("1", nullptr));

  // W^X activation respects both request flag and live probe result.
  WAVE6_4B_CHECK(!ShouldActivateWxDualMapping(false, DualMappingWxBackend::kMemfd));
  WAVE6_4B_CHECK(!ShouldActivateWxDualMapping(false, DualMappingWxBackend::kAshmem));
  WAVE6_4B_CHECK(ShouldActivateWxDualMapping(true, DualMappingWxBackend::kMemfd));
  WAVE6_4B_CHECK(ShouldActivateWxDualMapping(true, DualMappingWxBackend::kAshmem));
  WAVE6_4B_CHECK(ShouldActivateWxDualMapping(true, DualMappingWxBackend::kPosixShm));
  // SELinux block -> clean RWX fallback.
  WAVE6_4B_CHECK(!ShouldActivateWxDualMapping(true, DualMappingWxBackend::kUnsupported));

  // Dual-mapping WriteSlotForExecuteAddress translation & commit boundaries.
  constexpr size_t kTotalSize = 0x0FFFFFFFu;
  constexpr size_t kCommitMark = 16u * 1024u * 1024u;
  alignas(64) std::array<uint32_t, 64> rx_backing{};
  alignas(64) std::array<uint32_t, 64> rw_backing{};
  const uintptr_t exec_base = reinterpret_cast<uintptr_t>(rx_backing.data());
  const uintptr_t write_base = reinterpret_cast<uintptr_t>(rw_backing.data());

  // Inside committed region, 4-byte aligned -> maps to corresponding RW slot.
  uint32_t* slot0 = a64::TranslateCodeCacheWriteSlot(
      exec_base, exec_base, write_base, kTotalSize, kCommitMark);
  WAVE6_4B_CHECK(slot0 == rw_backing.data());
  uint32_t* slot4 = a64::TranslateCodeCacheWriteSlot(
      exec_base + 16, exec_base, write_base, kTotalSize, kCommitMark);
  WAVE6_4B_CHECK(slot4 == rw_backing.data() + 4);

  // Exact commit boundary: last 4-byte word within commit_mark succeeds;
  // word starting at commit_mark fails.
  const uintptr_t synthetic_exec = 0x2A0000000ull;
  const uintptr_t synthetic_write = 0x2B0000000ull;
  uint32_t* last_ok = a64::TranslateCodeCacheWriteSlot(
      synthetic_exec + kCommitMark - 4, synthetic_exec, synthetic_write,
      kTotalSize, kCommitMark);
  WAVE6_4B_CHECK(reinterpret_cast<uintptr_t>(last_ok) ==
                 synthetic_write + kCommitMark - 4);
  WAVE6_4B_CHECK(a64::TranslateCodeCacheWriteSlot(
                     synthetic_exec + kCommitMark, synthetic_exec,
                     synthetic_write, kTotalSize, kCommitMark) == nullptr);

  // Unaligned or out-of-range execute address rejected.
  WAVE6_4B_CHECK(a64::TranslateCodeCacheWriteSlot(
                     synthetic_exec + 2, synthetic_exec, synthetic_write,
                     kTotalSize, kCommitMark) == nullptr);
  WAVE6_4B_CHECK(a64::TranslateCodeCacheWriteSlot(
                     synthetic_exec - 4, synthetic_exec, synthetic_write,
                     kTotalSize, kCommitMark) == nullptr);
  WAVE6_4B_CHECK(a64::TranslateCodeCacheWriteSlot(
                     synthetic_exec, synthetic_exec, 0, kTotalSize,
                     kCommitMark) == nullptr);
  // Zero commit_mark rejects everything.
  WAVE6_4B_CHECK(a64::TranslateCodeCacheWriteSlot(
                     synthetic_exec, synthetic_exec, synthetic_write,
                     kTotalSize, 0) == nullptr);

  // EnsureCommitted next-mark math: 16 MiB chunks, clamped to kTotalSize.
  constexpr size_t kChunk = 16u * 1024u * 1024u;
  WAVE6_4B_CHECK(a64::ComputeCodeCacheNextCommitMark(0, 0, kChunk, kTotalSize) == 0);
  WAVE6_4B_CHECK(a64::ComputeCodeCacheNextCommitMark(0, 64, kChunk, kTotalSize) ==
                 kChunk);
  WAVE6_4B_CHECK(a64::ComputeCodeCacheNextCommitMark(kChunk, kChunk, kChunk,
                                                     kTotalSize) == kChunk);
  WAVE6_4B_CHECK(a64::ComputeCodeCacheNextCommitMark(kChunk, kChunk + 4, kChunk,
                                                     kTotalSize) == 2 * kChunk);
  // Large single placement (> 16 MiB) jumps directly over high_mark.
  WAVE6_4B_CHECK(a64::ComputeCodeCacheNextCommitMark(0, 3 * kChunk + 16, kChunk,
                                                     kTotalSize) == 4 * kChunk);
  // Near end of 0x0FFFFFFF code cache clamps cleanly to kTotalSize.
  WAVE6_4B_CHECK(a64::ComputeCodeCacheNextCommitMark(
                     15 * kChunk, kTotalSize - 16, kChunk, kTotalSize) ==
                 kTotalSize);
}

// ---------------------------------------------------------------------------
// 2. 1-slot Monomorphic PIC in CallIndirect: 2-stage 1-word atomic CMODX
// ---------------------------------------------------------------------------
void TestMonomorphicPicStateMachine() {
  using a64::ComputeIndirectCallPicFallbackGate;
  using a64::ComputeIndirectCallPicPatchWords;
  using a64::DecodeA64BranchRel26;
  using a64::DecodeA64MovkW16;
  using a64::DecodeA64MovzW;
  using a64::EncodeA64BranchRel26;
  using a64::EncodeA64MovkW16;
  using a64::EncodeA64MovzW;
  using a64::IndirectCallPicPatchWords;
  using a64::IndirectCallPicSite;
  using a64::IndirectCallPicState;
  using a64::kA64NopInsn;
  using a64::PatchA64InstructionAtomic;

  // Verify MOVZ / MOVK W14 encodings round-trip across corner values.
  for (uint16_t imm : {uint16_t(0x0000), uint16_t(0x1234), uint16_t(0x8210),
                       uint16_t(0xFFFF)}) {
    const uint32_t movz = EncodeA64MovzW(14, imm);
    const uint32_t movk = EncodeA64MovkW16(14, imm);
    uint32_t rd = 99;
    uint16_t dec = 0;
    WAVE6_4B_CHECK(DecodeA64MovzW(movz, &rd, &dec) && rd == 14 && dec == imm);
    WAVE6_4B_CHECK(DecodeA64MovkW16(movk, &rd, &dec) && rd == 14 && dec == imm);
    WAVE6_4B_CHECK(!DecodeA64MovzW(movk, &rd, &dec));
    WAVE6_4B_CHECK(!DecodeA64MovkW16(movz, &rd, &dec));
  }
  WAVE6_4B_CHECK(kA64NopInsn == 0xD503201Fu);

  // Simulate an inline 6-instruction PIC site + tail fallback stub in a buffer.
  // Layout: [0]=gate, [1]=cmp_lo, [2]=cmp_hi, [3]=cmp, [4]=b.ne, [5]=call_op,
  //         [16]=fallback_lookup
  alignas(64) std::array<uint32_t, 32> rw_code{};
  const uintptr_t base_rx = 0x2A0010000ull;
  const uintptr_t gate_rx = base_rx + 0;
  const uintptr_t cmp_lo_rx = base_rx + 4;
  const uintptr_t cmp_hi_rx = base_rx + 8;
  const uintptr_t call_op_rx = base_rx + 20;
  const uintptr_t fallback_rx = base_rx + 64;

  const uint32_t initial_gate = EncodeA64BranchRel26(128, false);
  rw_code[0] = initial_gate;
  rw_code[1] = EncodeA64MovzW(14, 0);
  rw_code[2] = EncodeA64MovkW16(14, 0);
  rw_code[5] = EncodeA64BranchRel26(static_cast<int64_t>(fallback_rx - call_op_rx),
                                    true);

  IndirectCallPicSite site{};
  site.gate_rx = gate_rx;
  site.cmp_lo_rx = cmp_lo_rx;
  site.cmp_hi_rx = cmp_hi_rx;
  site.call_op_rx = call_op_rx;
  site.fallback_rx = fallback_rx;
  site.is_tail = false;
  site.state = IndirectCallPicState::kUninitialized;

  // Stage 1 + Stage 2: kUninitialized -> kPatched for target in +-128 MiB.
  const uint32_t guest_target = 0x8234ABCDu;
  const uintptr_t target_rx = base_rx + 0x00400000ull;  // +4 MiB
  IndirectCallPicPatchWords words{};
  WAVE6_4B_CHECK(
      ComputeIndirectCallPicPatchWords(site, guest_target, target_rx, &words));
  WAVE6_4B_CHECK(words.gate_nop_insn == kA64NopInsn);
  WAVE6_4B_CHECK(words.payload_flush_bytes == 20u);

  // Write payload words first (while gate is still b pic_miss_stub).
  rw_code[1] = words.cmp_lo_insn;
  rw_code[2] = words.cmp_hi_insn;
  rw_code[5] = words.call_op_insn;
  // Publish gate atomically with NOP.
  WAVE6_4B_CHECK(PatchA64InstructionAtomic(&rw_code[0], words.gate_nop_insn));
  site.state = IndirectCallPicState::kPatched;
  site.cached_guest_target = guest_target;

  // Verify decoded instructions match guest_target and BL target_rx.
  uint32_t rd = 0;
  uint16_t lo = 0, hi = 0;
  WAVE6_4B_CHECK(DecodeA64MovzW(rw_code[1], &rd, &lo) && rd == 14 &&
                 lo == 0xABCDu);
  WAVE6_4B_CHECK(DecodeA64MovkW16(rw_code[2], &rd, &hi) && rd == 14 &&
                 hi == 0x8234u);
  int64_t call_off = 0;
  bool is_bl = false;
  WAVE6_4B_CHECK(DecodeA64BranchRel26(rw_code[5], &call_off, &is_bl));
  WAVE6_4B_CHECK(is_bl == true);
  WAVE6_4B_CHECK(static_cast<uintptr_t>(static_cast<int64_t>(call_op_rx) +
                                        call_off) == target_rx);
  WAVE6_4B_CHECK(rw_code[0] == kA64NopInsn);

  // Concurrent/second patch attempt on already-kPatched site must be rejected.
  WAVE6_4B_CHECK(!ComputeIndirectCallPicPatchWords(
      site, 0x82990000u, base_rx + 0x1000ull, &words));

  // RevertDirectCallSitesInRange: kPatched -> kFallback patches ONLY gate to
  // `b pic_fallback_lookup` and leaves cmp_lo/cmp_hi/call_op completely intact.
  const uint32_t saved_lo = rw_code[1];
  const uint32_t saved_hi = rw_code[2];
  const uint32_t saved_call = rw_code[5];
  uint32_t fallback_gate = 0;
  WAVE6_4B_CHECK(ComputeIndirectCallPicFallbackGate(site, &fallback_gate));
  WAVE6_4B_CHECK(PatchA64InstructionAtomic(&rw_code[0], fallback_gate));
  site.state = IndirectCallPicState::kFallback;

  int64_t fb_off = 0;
  bool fb_link = true;
  WAVE6_4B_CHECK(DecodeA64BranchRel26(rw_code[0], &fb_off, &fb_link));
  WAVE6_4B_CHECK(!fb_link);
  WAVE6_4B_CHECK(static_cast<uintptr_t>(static_cast<int64_t>(gate_rx) + fb_off) ==
                 fallback_rx);
  // Payload instructions MUST NOT be modified during revert (threads in flight).
  WAVE6_4B_CHECK(rw_code[1] == saved_lo);
  WAVE6_4B_CHECK(rw_code[2] == saved_hi);
  WAVE6_4B_CHECK(rw_code[5] == saved_call);
  // Cannot transition back from kFallback -> kPatched.
  WAVE6_4B_CHECK(
      !ComputeIndirectCallPicPatchWords(site, guest_target, target_rx, &words));

  // Out-of-range +-128 MiB target fails ComputeIndirectCallPicPatchWords cleanly
  // so caller transitions directly to kFallback.
  IndirectCallPicSite far_site = site;
  far_site.state = IndirectCallPicState::kUninitialized;
  WAVE6_4B_CHECK(!ComputeIndirectCallPicPatchWords(
      far_site, 0x82550000u, call_op_rx + 0x08000000ull, &words));
  WAVE6_4B_CHECK(!ComputeIndirectCallPicPatchWords(
      far_site, 0x82550000u, call_op_rx - 0x08000004ull, &words));
  // Boundary inside range succeeds (max positive +0x07FFFFFC, max negative -0x08000000).
  WAVE6_4B_CHECK(ComputeIndirectCallPicPatchWords(
      far_site, 0x82550000u, call_op_rx + 0x07FFFFFCull, &words));
  WAVE6_4B_CHECK(ComputeIndirectCallPicPatchWords(
      far_site, 0x82550000u, call_op_rx - 0x08000000ull, &words));

  // Tail-call PIC emits B (link=false) instead of BL.
  IndirectCallPicSite tail_site = site;
  tail_site.state = IndirectCallPicState::kUninitialized;
  tail_site.is_tail = true;
  WAVE6_4B_CHECK(ComputeIndirectCallPicPatchWords(tail_site, guest_target,
                                                  target_rx, &words));
  WAVE6_4B_CHECK(DecodeA64BranchRel26(words.call_op_insn, &call_off, &is_bl));
  WAVE6_4B_CHECK(!is_bl);
}

// ---------------------------------------------------------------------------
// 3. Hot Code Alignment (64 B entries + 32 B / anti-page-split 4 KB loops)
// ---------------------------------------------------------------------------
void TestHotCodeAndLoopAlignment() {
  using a64::ComputeFunctionEntryAlignment;
  using a64::ComputeLoopHeaderAlignmentPaddingBytes;

  // Entry alignment policy.
  WAVE6_4B_CHECK(ComputeFunctionEntryAlignment(false, 16) == 16u);
  WAVE6_4B_CHECK(ComputeFunctionEntryAlignment(false, 256) == 16u);
  WAVE6_4B_CHECK(ComputeFunctionEntryAlignment(true, 16) == 16u);
  WAVE6_4B_CHECK(ComputeFunctionEntryAlignment(true, 60) == 16u);
  WAVE6_4B_CHECK(ComputeFunctionEntryAlignment(true, 64) == 64u);
  WAVE6_4B_CHECK(ComputeFunctionEntryAlignment(true, 4096) == 64u);

  // Exhaustive verification over every 64 B aligned entry_mod_4k and every
  // 4-byte aligned function offset in [0, 4096):
  size_t max_pad_seen = 0;
  size_t nudged_count = 0;
  for (size_t entry_mod = 0; entry_mod < 4096; entry_mod += 64) {
    for (size_t off = 0; off < 4096; off += 4) {
      const size_t pad = ComputeLoopHeaderAlignmentPaddingBytes(off, entry_mod);
      WAVE6_4B_CHECK((pad & 3u) == 0);
      // NEVER blind 4 KB alignment (4092 B); strictly capped at 64 B!
      WAVE6_4B_CHECK(pad <= 64u);
      max_pad_seen = std::max(max_pad_seen, pad);

      const size_t final_pos = entry_mod + off + pad;
      // Always 32-byte aligned in final RX VA.
      WAVE6_4B_CHECK((final_pos & 31u) == 0);

      const size_t start_page_off = (entry_mod + off) & 0xFFFu;
      if (start_page_off >= 0xFC0u) {
        // Anti-page-split nudge lands squarely on the next 4 KB boundary.
        WAVE6_4B_CHECK((final_pos & 0xFFFu) == 0);
        ++nudged_count;
      } else {
        // Normal 32 B alignment uses at most 28 B of NOPs.
        WAVE6_4B_CHECK(pad <= 28u);
      }
    }
  }
  WAVE6_4B_CHECK(max_pad_seen == 64u);
  WAVE6_4B_CHECK(nudged_count > 0);
}

// ---------------------------------------------------------------------------
// 4. FEAT_LRCPC HWCAP bit 15 & TSO load eligibility
// ---------------------------------------------------------------------------
void TestLrcpcFeatureAndTsoSelection() {
  using xe::arm64::DetectLrcpcFeatureFlag;
  using xe::arm64::kA64EmitLRCPC;
  using xe::arm64::kA64EmitLSE;
  using xe::arm64::kA64FZFlushesInputs;
  using xe::arm64::kArm64HwcapLrcpcBit;

  WAVE6_4B_CHECK(kA64EmitLRCPC == (1ULL << 2));
  WAVE6_4B_CHECK(kArm64HwcapLrcpcBit == (1ULL << 15));

  // HWCAP bit 15 (HWCAP_LRCPC) + extension mask interactions.
  WAVE6_4B_CHECK(DetectLrcpcFeatureFlag(kArm64HwcapLrcpcBit, -1LL) ==
                 kA64EmitLRCPC);
  WAVE6_4B_CHECK(DetectLrcpcFeatureFlag(0ULL, -1LL) == 0ULL);
  WAVE6_4B_CHECK(DetectLrcpcFeatureFlag(kArm64HwcapLrcpcBit, 0LL) == 0ULL);
  WAVE6_4B_CHECK(DetectLrcpcFeatureFlag(
                     kArm64HwcapLrcpcBit,
                     static_cast<int64_t>(kA64EmitLSE | kA64FZFlushesInputs)) ==
                 0ULL);
  WAVE6_4B_CHECK(DetectLrcpcFeatureFlag(
                     kArm64HwcapLrcpcBit, static_cast<int64_t>(kA64EmitLRCPC)) ==
                 kA64EmitLRCPC);
  WAVE6_4B_CHECK(DetectLrcpcFeatureFlag(1ULL << 14, -1LL) == 0ULL);
  WAVE6_4B_CHECK(DetectLrcpcFeatureFlag(1ULL << 26, -1LL) == 0ULL);

  // Scalar LOAD and LOAD_OFFSET eligible when TSO + cvar + HWCAP all active.
  for (hir::Opcode op : {hir::OPCODE_LOAD, hir::OPCODE_LOAD_OFFSET}) {
    for (hir::TypeName ty :
         {hir::INT8_TYPE, hir::INT16_TYPE, hir::INT32_TYPE, hir::INT64_TYPE,
          hir::FLOAT32_TYPE, hir::FLOAT64_TYPE}) {
      WAVE6_4B_CHECK(
          a64::ShouldUseLrcpcForTsoLoad(true, true, kA64EmitLRCPC, op, ty));
      WAVE6_4B_CHECK(
          !a64::ShouldUseLrcpcForTsoLoad(false, true, kA64EmitLRCPC, op, ty));
      WAVE6_4B_CHECK(
          !a64::ShouldUseLrcpcForTsoLoad(true, false, kA64EmitLRCPC, op, ty));
      WAVE6_4B_CHECK(!a64::ShouldUseLrcpcForTsoLoad(true, true, 0ULL, op, ty));
    }
    // V128 must stay on ldr + dmb(ISHLD).
    WAVE6_4B_CHECK(!a64::ShouldUseLrcpcForTsoLoad(true, true, kA64EmitLRCPC, op,
                                                  hir::VEC128_TYPE));
  }

  // LVL, LVR, RESERVED_LOAD must NEVER use LRCPC (keep ldr + dmb(ISHLD)).
  for (hir::Opcode non_lrcpc_op :
       {hir::OPCODE_LVL, hir::OPCODE_LVR, hir::OPCODE_RESERVED_LOAD}) {
    WAVE6_4B_CHECK(!a64::ShouldUseLrcpcForTsoLoad(
        true, true, kA64EmitLRCPC, non_lrcpc_op, hir::INT32_TYPE));
    WAVE6_4B_CHECK(!a64::ShouldUseLrcpcForTsoLoad(
        true, true, kA64EmitLRCPC, non_lrcpc_op, hir::INT64_TYPE));
    WAVE6_4B_CHECK(!a64::ShouldUseLrcpcForTsoLoad(
        true, true, kA64EmitLRCPC, non_lrcpc_op, hir::VEC128_TYPE));
  }
}

// ---------------------------------------------------------------------------
// 5. 16 KB Skew < 0xA0000000 analysis in a64_stack_skew_analysis.h
// ---------------------------------------------------------------------------
struct SyntheticHirGraph {
  std::vector<hir::Block> blocks;
  std::vector<hir::Instr> instrs;
  std::vector<hir::Value> values;
  std::vector<hir::Label> labels;
  std::vector<hir::OpcodeInfo> op_infos;

  SyntheticHirGraph() {
    blocks.reserve(16);
    instrs.reserve(128);
    values.reserve(128);
    labels.reserve(16);
    op_infos.reserve(128);
  }

  hir::Block* AddBlock() {
    blocks.emplace_back();
    hir::Block* b = &blocks.back();
    std::memset(b, 0, sizeof(*b));
    if (blocks.size() > 1) {
      hir::Block* prev = &blocks[blocks.size() - 2];
      prev->next = b;
      b->prev = prev;
    }
    return b;
  }

  hir::Label* AddLabel(hir::Block* b) {
    labels.emplace_back();
    hir::Label* l = &labels.back();
    std::memset(l, 0, sizeof(*l));
    l->block = b;
    return l;
  }

  hir::Value* MakeConstI64(int64_t v) {
    values.emplace_back();
    hir::Value* val = &values.back();
    std::memset(val, 0, sizeof(*val));
    val->set_constant(v);
    return val;
  }

  hir::Value* MakeConstU64(uint64_t v) {
    values.emplace_back();
    hir::Value* val = &values.back();
    std::memset(val, 0, sizeof(*val));
    val->set_constant(v);
    return val;
  }

  hir::Value* MakeReg(hir::TypeName ty) {
    values.emplace_back();
    hir::Value* val = &values.back();
    std::memset(val, 0, sizeof(*val));
    val->type = ty;
    return val;
  }

  hir::Instr* AppendInstr(hir::Block* b, hir::Opcode op, hir::Value* dest,
                          uint16_t flags = 0) {
    op_infos.push_back(hir::OpcodeInfo{op, 0, 0});
    instrs.emplace_back();
    hir::Instr* i = &instrs.back();
    std::memset(i, 0, sizeof(*i));
    i->block = b;
    i->opcode = &op_infos.back();
    i->flags = flags;
    i->dest = dest;
    if (dest) {
      dest->def = i;
    }
    if (!b->instr_head) {
      b->instr_head = i;
      b->instr_tail = i;
    } else {
      i->prev = b->instr_tail;
      b->instr_tail->next = i;
      b->instr_tail = i;
    }
    return i;
  }
};

void TestLowAddrSkewAnalysis() {
  using a64::kLowAddrConstantBaseCeiling;
  using a64::kLowAddrPhysicalCeiling;
  using a64::kLowAddrSdaCeiling;
  using a64::kR13ContextOffset;
  using a64::LowAddrSkewSummary;
  using a64::MarkLowAddrNoPhysicalSkewBlocks;

  WAVE6_4B_CHECK(kLowAddrPhysicalCeiling == 0xA0000000u);
  WAVE6_4B_CHECK(kLowAddrConstantBaseCeiling == 0x9FF00000u);
  WAVE6_4B_CHECK(kLowAddrSdaCeiling == 0x90000000u);

  // Positive 1: PPC `lis` sign-extended constant (0xFFFFFFFF82100000) + OR 0x1234
  // + LOAD_OFFSET 0x10 -> proven < 0xA0000000, no r13 guard needed.
  {
    SyntheticHirGraph g;
    hir::Block* b = g.AddBlock();
    hir::Value* lis_val = g.MakeConstU64(0xFFFFFFFF82100000ull);
    hir::Value* or_res = g.MakeReg(hir::INT64_TYPE);
    hir::Instr* or_i = g.AppendInstr(b, hir::OPCODE_OR, or_res);
    or_i->src1.value = lis_val;
    or_i->src2.value = g.MakeConstU64(0x1234u);

    hir::Value* loaded = g.MakeReg(hir::INT32_TYPE);
    hir::Instr* ld = g.AppendInstr(b, hir::OPCODE_LOAD_OFFSET, loaded);
    ld->src1.value = or_res;
    ld->src2.value = g.MakeConstI64(0x10);

    LowAddrSkewSummary summary{};
    MarkLowAddrNoPhysicalSkewBlocks(&g.blocks[0], &summary);
    WAVE6_4B_CHECK((ld->flags & hir::LOAD_STORE_NO_PHYSICAL_SKEW) != 0);
    WAVE6_4B_CHECK(summary.marked_const_ops == 1);
    WAVE6_4B_CHECK(summary.marked_r13_ops == 0);
    WAVE6_4B_CHECK(!summary.r13_requires_negative_headroom);
  }

  // Positive 2: Constant 0x82000000 + ADD +0x4000 -> proven < 0xA0000000.
  {
    SyntheticHirGraph g;
    hir::Block* b = g.AddBlock();
    hir::Value* sum = g.MakeReg(hir::INT64_TYPE);
    hir::Instr* add_i = g.AppendInstr(b, hir::OPCODE_ADD, sum);
    add_i->src1.value = g.MakeConstU64(0x82000000u);
    add_i->src2.value = g.MakeConstI64(0x4000);

    hir::Instr* st = g.AppendInstr(b, hir::OPCODE_STORE, nullptr);
    st->src1.value = sum;
    st->src2.value = g.MakeConstU64(42);

    LowAddrSkewSummary summary{};
    MarkLowAddrNoPhysicalSkewBlocks(&g.blocks[0], &summary);
    WAVE6_4B_CHECK((st->flags & hir::LOAD_STORE_NO_PHYSICAL_SKEW) != 0);
    WAVE6_4B_CHECK(summary.marked_const_ops == 1);
    WAVE6_4B_CHECK(summary.marked_r13_ops == 0);
  }

  // Positive 3: r13 (SDA/KPCR) with positive offset -> marked, no low-bound
  // headroom required.
  {
    SyntheticHirGraph g;
    hir::Block* b = g.AddBlock();
    hir::Value* r13 = g.MakeReg(hir::INT64_TYPE);
    hir::Instr* lctx = g.AppendInstr(b, hir::OPCODE_LOAD_CONTEXT, r13);
    lctx->src1.offset = kR13ContextOffset;

    hir::Value* loaded = g.MakeReg(hir::INT32_TYPE);
    hir::Instr* ld = g.AppendInstr(b, hir::OPCODE_LOAD_OFFSET, loaded);
    ld->src1.value = r13;
    ld->src2.value = g.MakeConstI64(0x100);

    LowAddrSkewSummary summary{};
    MarkLowAddrNoPhysicalSkewBlocks(&g.blocks[0], &summary);
    WAVE6_4B_CHECK((ld->flags & hir::LOAD_STORE_NO_PHYSICAL_SKEW) != 0);
    WAVE6_4B_CHECK(summary.marked_const_ops == 0);
    WAVE6_4B_CHECK(summary.marked_r13_ops == 1);
    WAVE6_4B_CHECK(!summary.r13_requires_negative_headroom);
  }

  // Positive 4: r13 (SDA) with negative signed-16 offset (-0x4000) -> marked,
  // sets r13_requires_negative_headroom so prolog enforces r13 >= 1 MiB.
  {
    SyntheticHirGraph g;
    hir::Block* b = g.AddBlock();
    hir::Value* r13 = g.MakeReg(hir::INT64_TYPE);
    hir::Instr* lctx = g.AppendInstr(b, hir::OPCODE_LOAD_CONTEXT, r13);
    lctx->src1.offset = kR13ContextOffset;

    hir::Value* loaded = g.MakeReg(hir::INT32_TYPE);
    hir::Instr* ld = g.AppendInstr(b, hir::OPCODE_LOAD_OFFSET, loaded);
    ld->src1.value = r13;
    ld->src2.value = g.MakeConstI64(-0x4000);

    LowAddrSkewSummary summary{};
    MarkLowAddrNoPhysicalSkewBlocks(&g.blocks[0], &summary);
    WAVE6_4B_CHECK((ld->flags & hir::LOAD_STORE_NO_PHYSICAL_SKEW) != 0);
    WAVE6_4B_CHECK(summary.marked_r13_ops == 1);
    WAVE6_4B_CHECK(summary.r13_requires_negative_headroom);
  }

  // Negative 1: Constant base >= 0x9FF00000 (0x9FF00000, 0xA0000000, 0xE0000000)
  // MUST NOT be marked!
  for (uint64_t bad_addr :
       {0x9FF00000ull, 0xA0000000ull, 0xE0000000ull, 0xFFFFFFFFE0001000ull}) {
    SyntheticHirGraph g;
    hir::Block* b = g.AddBlock();
    hir::Value* loaded = g.MakeReg(hir::INT32_TYPE);
    hir::Instr* ld = g.AppendInstr(b, hir::OPCODE_LOAD, loaded);
    ld->src1.value = g.MakeConstU64(bad_addr);

    LowAddrSkewSummary summary{};
    MarkLowAddrNoPhysicalSkewBlocks(&g.blocks[0], &summary);
    WAVE6_4B_CHECK((ld->flags & hir::LOAD_STORE_NO_PHYSICAL_SKEW) == 0);
    WAVE6_4B_CHECK(summary.marked_const_ops == 0);
  }

  // Negative 2: Constant < 0x9FF00000 whose offset overflows +-1 MiB or crosses
  // >= 0xA0000000 or underflows below 0 MUST NOT be marked!
  {
    SyntheticHirGraph g;
    hir::Block* b = g.AddBlock();
    // Base 0x9FE80000 + offset +0x200000 (> 1 MiB AND >= 0xA0000000)
    hir::Value* loaded1 = g.MakeReg(hir::INT32_TYPE);
    hir::Instr* ld1 = g.AppendInstr(b, hir::OPCODE_LOAD_OFFSET, loaded1);
    ld1->src1.value = g.MakeConstU64(0x9FE80000u);
    ld1->src2.value = g.MakeConstI64(0x200000);

    // Base 0x9FE80000 + ADD +0x100000 + LOAD_OFFSET +0x80000 = 0xA0000000!
    hir::Value* sum2 = g.MakeReg(hir::INT64_TYPE);
    hir::Instr* add2 = g.AppendInstr(b, hir::OPCODE_ADD, sum2);
    add2->src1.value = g.MakeConstU64(0x9FE80000u);
    add2->src2.value = g.MakeConstI64(0x100000);
    hir::Value* loaded2 = g.MakeReg(hir::INT32_TYPE);
    hir::Instr* ld2 = g.AppendInstr(b, hir::OPCODE_LOAD_OFFSET, loaded2);
    ld2->src1.value = sum2;
    ld2->src2.value = g.MakeConstI64(0x80000);

    // Base 0x00000010 + offset -0x100 -> wraps past 0 into 0xFFFFFF10 (>= 0xE0000000)!
    hir::Value* loaded3 = g.MakeReg(hir::INT32_TYPE);
    hir::Instr* ld3 = g.AppendInstr(b, hir::OPCODE_LOAD_OFFSET, loaded3);
    ld3->src1.value = g.MakeConstU64(0x10u);
    ld3->src2.value = g.MakeConstI64(-0x100);

    LowAddrSkewSummary summary{};
    MarkLowAddrNoPhysicalSkewBlocks(&g.blocks[0], &summary);
    WAVE6_4B_CHECK((ld1->flags & hir::LOAD_STORE_NO_PHYSICAL_SKEW) == 0);
    WAVE6_4B_CHECK((ld2->flags & hir::LOAD_STORE_NO_PHYSICAL_SKEW) == 0);
    WAVE6_4B_CHECK((ld3->flags & hir::LOAD_STORE_NO_PHYSICAL_SKEW) == 0);
    WAVE6_4B_CHECK(summary.marked_const_ops == 0);
  }

  // Negative 3: r13 offset exceeding +-1 MiB MUST NOT be marked.
  {
    SyntheticHirGraph g;
    hir::Block* b = g.AddBlock();
    hir::Value* r13 = g.MakeReg(hir::INT64_TYPE);
    hir::Instr* lctx = g.AppendInstr(b, hir::OPCODE_LOAD_CONTEXT, r13);
    lctx->src1.offset = kR13ContextOffset;

    hir::Value* loaded = g.MakeReg(hir::INT32_TYPE);
    hir::Instr* ld = g.AppendInstr(b, hir::OPCODE_LOAD_OFFSET, loaded);
    ld->src1.value = r13;
    ld->src2.value = g.MakeConstI64(0x100004);  // > 1 MiB

    LowAddrSkewSummary summary{};
    MarkLowAddrNoPhysicalSkewBlocks(&g.blocks[0], &summary);
    WAVE6_4B_CHECK((ld->flags & hir::LOAD_STORE_NO_PHYSICAL_SKEW) == 0);
    WAVE6_4B_CHECK(summary.marked_r13_ops == 0);
  }

  // Negative 4: r13 after CALL or unsafe STORE_CONTEXT(r13) MUST NOT be marked.
  {
    SyntheticHirGraph g;
    hir::Block* b = g.AddBlock();
    g.AppendInstr(b, hir::OPCODE_CALL, nullptr);

    hir::Value* r13 = g.MakeReg(hir::INT64_TYPE);
    hir::Instr* lctx = g.AppendInstr(b, hir::OPCODE_LOAD_CONTEXT, r13);
    lctx->src1.offset = kR13ContextOffset;

    hir::Value* loaded = g.MakeReg(hir::INT32_TYPE);
    hir::Instr* ld = g.AppendInstr(b, hir::OPCODE_LOAD_OFFSET, loaded);
    ld->src1.value = r13;
    ld->src2.value = g.MakeConstI64(0x10);

    LowAddrSkewSummary summary{};
    MarkLowAddrNoPhysicalSkewBlocks(&g.blocks[0], &summary);
    WAVE6_4B_CHECK((ld->flags & hir::LOAD_STORE_NO_PHYSICAL_SKEW) == 0);
    WAVE6_4B_CHECK(summary.marked_r13_ops == 0);
  }

  // Negative 5: r13 mutated inside a loop back-edge MUST NOT be marked.
  {
    SyntheticHirGraph g;
    hir::Block* b0 = g.AddBlock();
    hir::Block* b1 = g.AddBlock();
    hir::Label* loop_lbl = g.AddLabel(b1);
    hir::Instr* br0 = g.AppendInstr(b0, hir::OPCODE_BRANCH, nullptr);
    br0->src1.label = loop_lbl;

    hir::Value* r13 = g.MakeReg(hir::INT64_TYPE);
    hir::Instr* lctx = g.AppendInstr(b1, hir::OPCODE_LOAD_CONTEXT, r13);
    lctx->src1.offset = kR13ContextOffset;
    hir::Value* next_r13 = g.MakeReg(hir::INT64_TYPE);
    hir::Instr* add_i = g.AppendInstr(b1, hir::OPCODE_ADD, next_r13);
    add_i->src1.value = r13;
    add_i->src2.value = g.MakeConstI64(0x100);
    hir::Instr* sctx = g.AppendInstr(b1, hir::OPCODE_STORE_CONTEXT, nullptr);
    sctx->src1.offset = kR13ContextOffset;
    sctx->src2.value = next_r13;

    hir::Value* loaded = g.MakeReg(hir::INT32_TYPE);
    hir::Instr* ld = g.AppendInstr(b1, hir::OPCODE_LOAD_OFFSET, loaded);
    ld->src1.value = next_r13;
    ld->src2.value = g.MakeConstI64(0x10);

    hir::Instr* back_br = g.AppendInstr(b1, hir::OPCODE_BRANCH_TRUE, nullptr);
    back_br->src1.value = g.MakeConstU64(1);
    back_br->src2.label = loop_lbl;

    LowAddrSkewSummary summary{};
    MarkLowAddrNoPhysicalSkewBlocks(&g.blocks[0], &summary);
    WAVE6_4B_CHECK((ld->flags & hir::LOAD_STORE_NO_PHYSICAL_SKEW) == 0);
    WAVE6_4B_CHECK(summary.marked_r13_ops == 0);
  }
}

// ---------------------------------------------------------------------------
// 6. Source-level safety contracts for all 4B files
// ---------------------------------------------------------------------------
void TestSourceContracts(const fs::path& cpp_root) {
  const fs::path xenia = cpp_root / "xenia/src/xenia";

  const std::string mem_cc = ReadSourceText(xenia / "base/memory.cc");
  const std::string mem_posix = ReadSourceText(xenia / "base/memory_posix.cc");
  const std::string plat_h = ReadSourceText(xenia / "base/platform_arm64.h");
  const std::string plat_cc = ReadSourceText(xenia / "base/platform_arm64.cc");
  const std::string cc_base =
      ReadSourceText(xenia / "cpu/backend/code_cache_base.h");
  const std::string a64_be_h =
      ReadSourceText(xenia / "cpu/backend/a64/a64_backend.h");
  const std::string a64_be_cc =
      ReadSourceText(xenia / "cpu/backend/a64/a64_backend.cc");
  const std::string a64_cc_h =
      ReadSourceText(xenia / "cpu/backend/a64/a64_code_cache.h");
  const std::string a64_cc_cc =
      ReadSourceText(xenia / "cpu/backend/a64/a64_code_cache.cc");
  const std::string a64_em_cc =
      ReadSourceText(xenia / "cpu/backend/a64/a64_emitter.cc");
  const std::string a64_seq_mem =
      ReadSourceText(xenia / "cpu/backend/a64/a64_seq_memory.cc");
  const std::string mmio_handler =
      ReadSourceText(xenia / "cpu/mmio_handler.cc");
  const std::string cmake = ReadSourceText(cpp_root / "CMakeLists.txt");

  // 6a. All 5 new CVars MUST default to false, ZERO new UPDATE_from_*,
  //     and strict '0'|'1' live properties via __system_property_get.
  WAVE6_4B_CHECK(mem_cc.find("a64_wx_dual_mapping, false,") !=
                 std::string::npos);
  WAVE6_4B_CHECK(mem_cc.find("debug.xendroid.jit_wx") != std::string::npos);
  WAVE6_4B_CHECK(mem_cc.find("ParseStrictBoolProp(") != std::string::npos);
  WAVE6_4B_CHECK(mem_cc.find("UPDATE_from_") == std::string::npos);

  WAVE6_4B_CHECK(a64_em_cc.find("a64_indirect_call_pic, false,") !=
                     std::string::npos ||
                 a64_be_cc.find("a64_indirect_call_pic, false,") !=
                     std::string::npos);
  WAVE6_4B_CHECK(a64_em_cc.find("a64_align_hot_code, false,") !=
                     std::string::npos ||
                 a64_cc_cc.find("a64_align_hot_code, false,") !=
                     std::string::npos);
  WAVE6_4B_CHECK(a64_em_cc.find("a64_lrcpc_tso, false,") != std::string::npos);
  WAVE6_4B_CHECK(a64_em_cc.find("a64_skip_low_addr_skew_fixup, false,") !=
                 std::string::npos);
  for (const char* prop : {"debug.xendroid.a64_indirect_call_pic",
                           "debug.xendroid.a64_align_hot_code",
                           "debug.xendroid.a64_lrcpc_tso",
                           "debug.xendroid.a64_low_addr_skew"}) {
    WAVE6_4B_CHECK(a64_be_cc.find(prop) != std::string::npos);
  }
  WAVE6_4B_CHECK(a64_em_cc.find("UPDATE_from_") == std::string::npos);
  WAVE6_4B_CHECK(a64_cc_cc.find("UPDATE_from_") == std::string::npos);

  // 6b. W^X live probe + memfd_create + ASharedMemory fallback + sentinel.
  WAVE6_4B_CHECK(mem_posix.find("ProbeDualMappingWxSupport()") !=
                 std::string::npos);
  WAVE6_4B_CHECK(mem_posix.find("__NR_memfd_create") != std::string::npos);
  WAVE6_4B_CHECK(mem_posix.find("ASharedMemory_create") != std::string::npos);
  WAVE6_4B_CHECK(mem_posix.find("kDualMappingWxProbeSentinel") !=
                 std::string::npos);
  WAVE6_4B_CHECK(mem_posix.find("std::once_flag") != std::string::npos);
  // Regression (#121 P1): CreateFileMappingHandle MUST require
  // !IsWritableExecutableMemoryPreferred() before probing/using memfd_create
  // so default a64_wx_dual_mapping=false stays 100% on ASharedMemory_create.
  const std::string create_fmh = ExtractFunctionBody(
      mem_posix, "FileMappingHandle CreateFileMappingHandle(");
  {
    size_t pos = 0;
    size_t guarded_memfd_checks = 0;
    while ((pos = create_fmh.find("ProbeDualMappingWxSupport()", pos)) !=
           std::string::npos) {
      const size_t window_start = (pos > 96) ? (pos - 96) : 0;
      const std::string prefix =
          create_fmh.substr(window_start, pos - window_start);
      WAVE6_4B_CHECK(prefix.find("!IsWritableExecutableMemoryPreferred()") !=
                     std::string::npos);
      ++guarded_memfd_checks;
      pos += 27;
    }
    WAVE6_4B_CHECK(guarded_memfd_checks == 2);
  }

  // 6c. W^X 4 blockers + EnsureCommitted.
  // Blocker 1: guest_trampoline dual-mapping + FlushCodeRange.
  WAVE6_4B_CHECK(a64_be_h.find("guest_trampoline_execute_memory_") !=
                 std::string::npos);
  WAVE6_4B_CHECK(a64_be_h.find("guest_trampoline_write_memory_") !=
                 std::string::npos);
  const std::string create_tramp =
      ExtractFunctionBody(a64_be_cc, "uint32_t A64Backend::CreateGuestTrampoline(");
  WAVE6_4B_CHECK(create_tramp.find("FlushCodeRange(write_pos, exec_pos,") !=
                 std::string::npos);
  WAVE6_4B_CHECK(create_tramp.find("AddIndirection64(indirection_guest_addr,") !=
                 std::string::npos);
  WAVE6_4B_CHECK(create_tramp.find("exec_pos)") != std::string::npos);

  // Blocker 2: Breakpoints write via WriteSlotForExecuteAddress + FlushPatchedInstruction.
  const std::string install_bp1 =
      ExtractFunctionBody(a64_be_cc, "void A64Backend::InstallBreakpoint(Breakpoint* breakpoint) {");
  const std::string install_bp2 =
      ExtractFunctionBody(a64_be_cc, "void A64Backend::InstallBreakpoint(Breakpoint* breakpoint, Function* fn) {");
  const std::string uninstall_bp =
      ExtractFunctionBody(a64_be_cc, "void A64Backend::UninstallBreakpoint(");
  for (const std::string& body : {install_bp1, install_bp2, uninstall_bp}) {
    WAVE6_4B_CHECK(body.find("WriteSlotForExecuteAddress(") != std::string::npos);
    WAVE6_4B_CHECK(body.find("PatchA64InstructionAtomic(") != std::string::npos);
    WAVE6_4B_CHECK(body.find("FlushPatchedInstruction(") != std::string::npos);
    WAVE6_4B_CHECK(body.find("xe::store<uint32_t>(ptr") == std::string::npos);
  }

  // Blocker 3: PlaceData returns execute VA (not write VA).
  const std::string place_data =
      ExtractFunctionBody(cc_base, "uint32_t PlaceData(");
  WAVE6_4B_CHECK(place_data.find("data_execute_address") != std::string::npos);
  WAVE6_4B_CHECK(place_data.find("return uint32_t(uintptr_t(data_execute_address))") !=
                 std::string::npos);

  // Blocker 4 (bonus): EnsureCommitted skips redundant POSIX mprotect.
  const std::string ensure_comm =
      ExtractFunctionBody(cc_base, "void EnsureCommitted(");
  WAVE6_4B_CHECK(ensure_comm.find("#if XE_PLATFORM_WIN32") != std::string::npos);
  WAVE6_4B_CHECK(ensure_comm.find("ComputeCodeCacheNextCommitMark(") !=
                 std::string::npos);

  // 6d. 1-slot Monomorphic PIC in CallIndirect + atomic Revert.
  WAVE6_4B_CHECK(a64_be_h.find("patch_indirect_call_pic_thunk_") !=
                 std::string::npos);
  WAVE6_4B_CHECK(a64_cc_h.find("RegisterIndirectCallPicSite(") !=
                 std::string::npos);
  WAVE6_4B_CHECK(a64_cc_h.find("TryPatchIndirectCallPicSite(") !=
                 std::string::npos);
  const std::string revert_body =
      ExtractFunctionBody(a64_cc_cc, "void A64CodeCache::RevertDirectCallSitesInRange(");
  WAVE6_4B_CHECK(revert_body.find("patched_pic_sites_") != std::string::npos);
  WAVE6_4B_CHECK(revert_body.find("ComputeIndirectCallPicFallbackGate(") !=
                 std::string::npos);

  // 6e. FEAT_LRCPC & -march=armv8-a preserved.
  WAVE6_4B_CHECK(cmake.find("-march=armv8-a") != std::string::npos);
  WAVE6_4B_CHECK(plat_h.find("kA64EmitLRCPC") != std::string::npos);
  WAVE6_4B_CHECK(plat_cc.find("getauxval(AT_HWCAP)") != std::string::npos);
  WAVE6_4B_CHECK(plat_cc.find("DetectLrcpcFeatureFlag(") != std::string::npos);
  WAVE6_4B_CHECK(a64_seq_mem.find("ldaprb(") != std::string::npos);
  WAVE6_4B_CHECK(a64_seq_mem.find("ldaprh(") != std::string::npos);
  WAVE6_4B_CHECK(a64_seq_mem.find("ldapr(") != std::string::npos);

  // Regression (#121 P0): MMIOHandler::TryDecodeLoadStore on ARM64 MUST decode
  // 32-bit LDAPR Wt, [Xn|SP] (0xB8BFC000 / mask 0xFFFFFC00) before
  // kArm64LoadStoreAnyFMask, preserving WZR -> constant 0.
  {
    const std::string try_decode = ExtractFunctionBody(
        mmio_handler, "bool MMIOHandler::TryDecodeLoadStore(");
    const size_t ldapr_pos = try_decode.find("0xB8BFC000u");
    const size_t any_mask_pos = try_decode.find("kArm64LoadStoreAnyFMask");
    WAVE6_4B_CHECK(ldapr_pos != std::string::npos);
    WAVE6_4B_CHECK(any_mask_pos != std::string::npos);
    WAVE6_4B_CHECK(ldapr_pos < any_mask_pos);
    WAVE6_4B_CHECK(try_decode.find("0xFFFFFC00u") != std::string::npos);
    WAVE6_4B_CHECK(try_decode.find("DecodedLoadStore::kArm64ValueRegZero") !=
                   std::string::npos);
    // Bit-level encoding verification for LDAPR Wt, [Xn|SP]:
    auto is_ldapr_w = [](uint32_t insn) {
      return (insn & 0xFFFFFC00u) == 0xB8BFC000u;
    };
    const uint32_t ldapr_w5_x16 = 0xB8BFC000u | (16u << 5) | 5u;
    const uint32_t ldapr_wzr_x16 = 0xB8BFC000u | (16u << 5) | 31u;
    const uint32_t ldapr_w0_sp = 0xB8BFC000u | (31u << 5) | 0u;
    const uint32_t ldaprb_w5_x16 = 0x38BFC000u | (16u << 5) | 5u;
    const uint32_t ldaprh_w5_x16 = 0x78BFC000u | (16u << 5) | 5u;
    const uint32_t ldapr_x5_x16 = 0xF8BFC000u | (16u << 5) | 5u;
    const uint32_t ldr_w5_x16 = 0xB9400000u | (16u << 5) | 5u;
    WAVE6_4B_CHECK(is_ldapr_w(ldapr_w5_x16));
    WAVE6_4B_CHECK((ldapr_w5_x16 & 31u) == 5u && ((ldapr_w5_x16 >> 5) & 31u) == 16u);
    WAVE6_4B_CHECK(is_ldapr_w(ldapr_wzr_x16));
    WAVE6_4B_CHECK((ldapr_wzr_x16 & 31u) == 31u);
    WAVE6_4B_CHECK(is_ldapr_w(ldapr_w0_sp));
    WAVE6_4B_CHECK(!is_ldapr_w(ldaprb_w5_x16));
    WAVE6_4B_CHECK(!is_ldapr_w(ldaprh_w5_x16));
    WAVE6_4B_CHECK(!is_ldapr_w(ldapr_x5_x16));
    WAVE6_4B_CHECK(!is_ldapr_w(ldr_w5_x16));
  }

  // 6f. Forbidden items MUST NOT be introduced.
  WAVE6_4B_CHECK(a64_em_cc.find("TryInlineGprlrSaverest") == std::string::npos);
  WAVE6_4B_CHECK(a64_be_cc.find("TryInlineGprlrSaverest") == std::string::npos);
}

}  // namespace

int main(int argc, char** argv) {
  TestWxDualMappingContracts();
  TestMonomorphicPicStateMachine();
  TestHotCodeAndLoopAlignment();
  TestLrcpcFeatureAndTsoSelection();
  TestLowAddrSkewAnalysis();
  if (argc >= 2 && argv[1] && argv[1][0] != '\0') {
    TestSourceContracts(fs::path(argv[1]));
  }
  std::printf("wave6_onda4b_wx_a64_test: PASS\n");
  return 0;
}
