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

#ifndef XE_DISABLE_ANDROID_DLOPEN
#define XE_DISABLE_ANDROID_DLOPEN 1
#endif

#include "xenia/base/xxhash.h"
#include "xenia/cpu/compiler/passes/cross_block_context_analysis.h"
#include "xenia/cpu/jit_profile_file.h"

namespace {

#define WAVE4_CHECK(cond)                                                  \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "WAVE4_LOGIC check failed (%s:%d): %s\n",       \
                   __FILE__, __LINE__, #cond);                             \
      std::abort();                                                        \
    }                                                                      \
  } while (false)

namespace fs = std::filesystem;
namespace jpf = xe::cpu::jit_profile_file;
namespace xba = xe::cpu::compiler::passes::cross_block;

std::string NormalizeNewlines(std::string s) {
  s.erase(std::remove(s.begin(), s.end(), '\r'), s.end());
  return s;
}

std::string ReadSourceText(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  WAVE4_CHECK(in.good());
  std::ostringstream ss;
  ss << in.rdbuf();
  return NormalizeNewlines(ss.str());
}

std::vector<uint8_t> ReadAllBytes(const fs::path& path) {
  std::vector<uint8_t> bytes(static_cast<size_t>(fs::file_size(path)));
  FILE* f = std::fopen(path.string().c_str(), "rb");
  WAVE4_CHECK(f != nullptr);
  if (!bytes.empty()) {
    WAVE4_CHECK(std::fread(bytes.data(), 1, bytes.size(), f) == bytes.size());
  }
  std::fclose(f);
  return bytes;
}

void WriteAllBytes(const fs::path& path, const std::vector<uint8_t>& bytes) {
  FILE* f = std::fopen(path.string().c_str(), "wb");
  WAVE4_CHECK(f != nullptr);
  if (!bytes.empty()) {
    WAVE4_CHECK(std::fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size());
  }
  std::fclose(f);
}

// =============================================================================
// Pillar 3: .jprof Binary Format, Integrity, Mismatch & Atomic Write
// =============================================================================

void TestJprofFileFormat(const fs::path& scratch_root) {
  static_assert(sizeof(jpf::Header) == jpf::kHeaderSize);
  static_assert(jpf::kHeaderSize == 48);
  static_assert(jpf::kFormatVersion == 1);

  fs::remove_all(scratch_root);
  fs::create_directories(scratch_root);

  const fs::path jprof_path = scratch_root / "4D530919.jprof";
  const uint32_t kTitleId = 0x4D530919u;
  const uint64_t kModuleHash = 0x1122334455667788ULL;

  // Missing file -> kMissing.
  {
    auto res = jpf::Load(jprof_path, kTitleId, kModuleHash);
    WAVE4_CHECK(res.status == jpf::LoadStatus::kMissing);
    WAVE4_CHECK(res.addresses.empty());
  }

  // Round-trip save/load: filters 0 and unaligned addresses, sorts & deduplicates.
  {
    const std::vector<uint32_t> raw_addrs = {
        0x82010200u, 0x00000000u, 0x82010010u, 0x82010012u,
        0x82010200u, 0x82010004u, 0x82010011u, 0x82010010u,
    };
    WAVE4_CHECK(
        jpf::WriteAtomically(jprof_path, kTitleId, kModuleHash, raw_addrs));
    WAVE4_CHECK(!fs::exists(fs::path(jprof_path.string() + ".tmp")));

    auto res = jpf::Load(jprof_path, kTitleId, kModuleHash);
    WAVE4_CHECK(res.status == jpf::LoadStatus::kLoaded);
    WAVE4_CHECK(res.title_id == kTitleId);
    WAVE4_CHECK(res.module_hash == kModuleHash);
    const std::vector<uint32_t> expected = {
        0x82010004u, 0x82010010u, 0x82010200u,
    };
    WAVE4_CHECK(res.addresses == expected);

    // Header magic/version/checksum layout verification.
    const auto raw_file = ReadAllBytes(jprof_path);
    WAVE4_CHECK(raw_file.size() == jpf::kHeaderSize + expected.size() * 4);
    jpf::Header hdr = {};
    std::memcpy(&hdr, raw_file.data(), sizeof(hdr));
    WAVE4_CHECK(std::memcmp(hdr.magic, "XDJPROF\0", 8) == 0);
    WAVE4_CHECK(hdr.header_size == 48u);
    WAVE4_CHECK(hdr.version == 1u);
    WAVE4_CHECK(hdr.title_id == kTitleId);
    WAVE4_CHECK(hdr.entry_count == expected.size());
    WAVE4_CHECK(hdr.module_hash == kModuleHash);
    WAVE4_CHECK(hdr.payload_size == expected.size() * 4);
    WAVE4_CHECK(hdr.payload_hash ==
                XXH3_64bits(raw_file.data() + jpf::kHeaderSize,
                            hdr.payload_size));
  }

  // Empty address list is still valid (0-byte payload).
  {
    const fs::path empty_path = scratch_root / "EMPTY.jprof";
    WAVE4_CHECK(jpf::WriteAtomically(empty_path, kTitleId, kModuleHash, {}));
    auto res = jpf::Load(empty_path, kTitleId, kModuleHash);
    WAVE4_CHECK(res.status == jpf::LoadStatus::kLoaded);
    WAVE4_CHECK(res.addresses.empty());
  }

  // Corrupt magic, version, header_size, checksum, truncated payload.
  {
    const auto valid_bytes = ReadAllBytes(jprof_path);

    // Bad magic.
    auto bad = valid_bytes;
    bad[0] = 'Z';
    WriteAllBytes(jprof_path, bad);
    WAVE4_CHECK(jpf::Load(jprof_path, kTitleId, kModuleHash).status ==
                jpf::LoadStatus::kCorrupt);

    // Bad version.
    bad = valid_bytes;
    uint32_t bad_ver = 99u;
    std::memcpy(bad.data() + 12, &bad_ver, 4);
    WriteAllBytes(jprof_path, bad);
    WAVE4_CHECK(jpf::Load(jprof_path, kTitleId, kModuleHash).status ==
                jpf::LoadStatus::kCorrupt);

    // Bad header_size.
    bad = valid_bytes;
    uint32_t bad_hsz = 32u;
    std::memcpy(bad.data() + 8, &bad_hsz, 4);
    WriteAllBytes(jprof_path, bad);
    WAVE4_CHECK(jpf::Load(jprof_path, kTitleId, kModuleHash).status ==
                jpf::LoadStatus::kCorrupt);

    // Flipped bit in payload (XXH3_64bits mismatch).
    bad = valid_bytes;
    bad[jpf::kHeaderSize + 2] ^= 0x40u;
    WriteAllBytes(jprof_path, bad);
    WAVE4_CHECK(jpf::Load(jprof_path, kTitleId, kModuleHash).status ==
                jpf::LoadStatus::kCorrupt);

    // Truncated payload.
    bad = valid_bytes;
    bad.resize(bad.size() - 3);
    WriteAllBytes(jprof_path, bad);
    WAVE4_CHECK(jpf::Load(jprof_path, kTitleId, kModuleHash).status ==
                jpf::LoadStatus::kCorrupt);

    // Truncated header.
    bad = valid_bytes;
    bad.resize(24);
    WriteAllBytes(jprof_path, bad);
    WAVE4_CHECK(jpf::Load(jprof_path, kTitleId, kModuleHash).status ==
                jpf::LoadStatus::kCorrupt);

    // Restore valid bytes.
    WriteAllBytes(jprof_path, valid_bytes);
    WAVE4_CHECK(jpf::Load(jprof_path, kTitleId, kModuleHash).status ==
                jpf::LoadStatus::kLoaded);
  }

  // Oversized file > max_bytes rejected BEFORE reading payload.
  {
    WAVE4_CHECK(jpf::Load(jprof_path, kTitleId, kModuleHash,
                          jpf::kHeaderSize + 4)
                    .status == jpf::LoadStatus::kTooLarge);
  }

  // Title ID and non-zero module_hash mismatch checks.
  {
    WAVE4_CHECK(jpf::Load(jprof_path, kTitleId + 1u, kModuleHash).status ==
                jpf::LoadStatus::kMismatch);
    WAVE4_CHECK(jpf::Load(jprof_path, kTitleId, kModuleHash ^ 1ULL).status ==
                jpf::LoadStatus::kMismatch);
    // expected_module_hash == 0 allows loading regardless of stored module_hash.
    WAVE4_CHECK(jpf::Load(jprof_path, kTitleId, 0ULL).status ==
                jpf::LoadStatus::kLoaded);
  }

  // Atomic write preserves prior .jprof if .tmp creation/write fails.
  {
    const auto before = ReadAllBytes(jprof_path);
    const fs::path tmp_blocker = fs::path(jprof_path.string() + ".tmp");
    fs::create_directories(tmp_blocker);  // fopen("wb") on directory fails
    WAVE4_CHECK(!jpf::WriteAtomically(jprof_path, kTitleId, kModuleHash,
                                      {0x82099990u}));
    WAVE4_CHECK(ReadAllBytes(jprof_path) == before);
    fs::remove_all(tmp_blocker);
  }

  // max_entries cap honored by WriteAtomically.
  {
    const fs::path capped_path = scratch_root / "CAPPED.jprof";
    std::vector<uint32_t> many = {0x82001000u, 0x82001004u, 0x82001008u,
                                  0x8200100Cu, 0x82001010u};
    WAVE4_CHECK(
        jpf::WriteAtomically(capped_path, kTitleId, kModuleHash, many, 3u));
    auto res = jpf::Load(capped_path, kTitleId, kModuleHash);
    WAVE4_CHECK(res.status == jpf::LoadStatus::kLoaded);
    WAVE4_CHECK(res.addresses.size() == 3u);
  }

  fs::remove_all(scratch_root);
}

// =============================================================================
// Pillar 2: 229-Slot Bitset, Dual Aliasing, Extended-Block Liveness & Loops
// =============================================================================

void TestContextBitsetAndDualAliasing() {
  static_assert(xba::kTotalSlots == 229);
  static_assert(xba::kCrSlotStart == 0);
  static_assert(xba::kGprSlotStart == 32);
  static_assert(xba::kFprSlotStart == 64);
  static_assert(xba::kCtrSlot == 96);
  static_assert(xba::kLrSlot == 97);
  static_assert(xba::kXerCaSlot == 98);
  static_assert(xba::kXerOvSlot == 99);
  static_assert(xba::kXerSoSlot == 100);
  static_assert(xba::kVecSlotStart == 101);

  const xba::ContextLayout layout;

  // All-tracked masks.
  {
    const auto cr_only = xba::MakeAllTrackedMask(false, false);
    WAVE4_CHECK(cr_only.Test(0) && cr_only.Test(31));
    WAVE4_CHECK(!cr_only.Test(xba::kGprSlotStart));
    WAVE4_CHECK(!cr_only.Test(xba::kVecSlotStart));

    const auto no_vec = xba::MakeAllTrackedMask(true, false);
    WAVE4_CHECK(no_vec.Test(0) && no_vec.Test(31));
    WAVE4_CHECK(no_vec.Test(xba::kGprSlotStart + 31));
    WAVE4_CHECK(no_vec.Test(xba::kFprSlotStart + 31));
    WAVE4_CHECK(no_vec.Test(xba::kCtrSlot) && no_vec.Test(xba::kLrSlot));
    WAVE4_CHECK(no_vec.Test(xba::kXerCaSlot) && no_vec.Test(xba::kXerOvSlot) &&
                no_vec.Test(xba::kXerSoSlot));
    WAVE4_CHECK(!no_vec.Test(xba::kVecSlotStart));
    WAVE4_CHECK(!no_vec.Test(xba::kVecSlotStart + 127));

    const auto with_vec = xba::MakeAllTrackedMask(true, true);
    WAVE4_CHECK(with_vec.Test(xba::kVecSlotStart));
    WAVE4_CHECK(with_vec.Test(xba::kVecSlotStart + 127));
  }

  // cr0.value (4B) touches AND kills all 4 cr0 byte slots; cr0_eq (1B) kills only byte 2.
  {
    const auto cr0_full = xba::ComputeContextAccessMasks(
        layout, layout.cr_offset, 4u, true, false);
    WAVE4_CHECK(cr0_full.covered_bytes == 4u);
    for (size_t b = 0; b < 4; ++b) {
      WAVE4_CHECK(cr0_full.touch_mask.Test(b));
      WAVE4_CHECK(cr0_full.kill_mask.Test(b));
    }
    WAVE4_CHECK(!cr0_full.touch_mask.Test(4));

    const auto cr0_eq = xba::ComputeContextAccessMasks(
        layout, layout.cr_offset + 2u, 1u, true, false);
    WAVE4_CHECK(cr0_eq.covered_bytes == 1u);
    WAVE4_CHECK(!cr0_eq.touch_mask.Test(0) && cr0_eq.touch_mask.Test(2));
    WAVE4_CHECK(!cr0_eq.kill_mask.Test(0) && cr0_eq.kill_mask.Test(2));
  }

  // Partial 4B write to r[5] (8B) touches r[5] but does NOT kill r[5]!
  {
    const uint32_t r5_off = layout.r_offset + 5u * 8u;
    const auto r5_partial =
        xba::ComputeContextAccessMasks(layout, r5_off, 4u, true, false);
    WAVE4_CHECK(r5_partial.covered_bytes == 4u);
    WAVE4_CHECK(r5_partial.touch_mask.Test(xba::kGprSlotStart + 5));
    WAVE4_CHECK(!r5_partial.kill_mask.Test(xba::kGprSlotStart + 5));

    const auto r5_full =
        xba::ComputeContextAccessMasks(layout, r5_off, 8u, true, false);
    WAVE4_CHECK(r5_full.covered_bytes == 8u);
    WAVE4_CHECK(r5_full.touch_mask.Test(xba::kGprSlotStart + 5));
    WAVE4_CHECK(r5_full.kill_mask.Test(xba::kGprSlotStart + 5));
  }

  // xer_ca / xer_ov / xer_so byte isolation.
  {
    const auto ca = xba::ComputeContextAccessMasks(
        layout, layout.xer_ca_offset, 1u, true, false);
    const auto ov = xba::ComputeContextAccessMasks(
        layout, layout.xer_ov_offset, 1u, true, false);
    const auto so = xba::ComputeContextAccessMasks(
        layout, layout.xer_so_offset, 1u, true, false);
    WAVE4_CHECK(ca.covered_bytes == 1u && ca.kill_mask.Test(xba::kXerCaSlot));
    WAVE4_CHECK(!ca.touch_mask.Test(xba::kXerOvSlot) &&
                !ca.touch_mask.Test(xba::kXerSoSlot));
    WAVE4_CHECK(ov.covered_bytes == 1u && ov.kill_mask.Test(xba::kXerOvSlot));
    WAVE4_CHECK(!ov.touch_mask.Test(xba::kXerCaSlot) &&
                !ov.touch_mask.Test(xba::kXerSoSlot));
    WAVE4_CHECK(so.covered_bytes == 1u && so.kill_mask.Test(xba::kXerSoSlot));
  }

  // v[0..127] gated off when !track_vec128 (covered_bytes == 0).
  {
    const uint32_t v7_off = layout.v_offset + 7u * 16u;
    const auto gated_off =
        xba::ComputeContextAccessMasks(layout, v7_off, 16u, true, false);
    WAVE4_CHECK(gated_off.covered_bytes == 0u);
    WAVE4_CHECK(!gated_off.touch_mask.Any());
    WAVE4_CHECK(!gated_off.kill_mask.Any());

    const auto gated_on =
        xba::ComputeContextAccessMasks(layout, v7_off, 16u, true, true);
    WAVE4_CHECK(gated_on.covered_bytes == 16u);
    WAVE4_CHECK(gated_on.touch_mask.Test(xba::kVecSlotStart + 7));
    WAVE4_CHECK(gated_on.kill_mask.Test(xba::kVecSlotStart + 7));
  }
}

void TestExtendedBlockBackwardLiveness() {
  const xba::ContextLayout layout;
  const uint32_t r5 = layout.r_offset + 5u * 8u;
  const uint32_t cr0 = layout.cr_offset;

  // 1. Mid-block BRANCH overwrites live with target_live; instructions after
  //    unconditional BRANCH in same extended block do NOT keep prior store live.
  {
    std::vector<xba::LivenessBlock> blocks(2);
    // Block 0: store r5; branch -> B1; [unreachable in extended block: load r5]; return
    blocks[0].instrs = {
        {xba::LivenessInstrKind::kStoreContext, r5, 8u},
        {xba::LivenessInstrKind::kBranch, 0, 0, 1u},
        {xba::LivenessInstrKind::kLoadContext, r5, 8u},
    };
    blocks[0].successors = {1u};
    // Block 1: store r5 (overwrites); exit (no successors -> all-live at exit)
    blocks[1].instrs = {
        {xba::LivenessInstrKind::kStoreContext, r5, 8u},
    };
    auto res = xba::SolveCrossBlockContextLiveness(blocks, layout, true, false);
    WAVE4_CHECK(res.dead_store[0][0] == true);
    WAVE4_CHECK(res.dead_store[1][0] == false);
  }

  // 2. Mid-block BRANCH_COND merges fall-through (post-branch load r5) with target_live.
  {
    std::vector<xba::LivenessBlock> blocks(2);
    blocks[0].instrs = {
        {xba::LivenessInstrKind::kStoreContext, r5, 8u},
        {xba::LivenessInstrKind::kBranchCond, 0, 0, 1u},
        {xba::LivenessInstrKind::kLoadContext, r5, 8u},
    };
    blocks[0].successors = {1u};
    blocks[1].instrs = {
        {xba::LivenessInstrKind::kStoreContext, r5, 8u},
    };
    auto res = xba::SolveCrossBlockContextLiveness(blocks, layout, true, false);
    WAVE4_CHECK(res.dead_store[0][0] == false);
  }

  // 3. CHECK_PREEMPT preserves PPCContext::r[13] (KPCR) and cr0..cr7, while
  //    remaining transparent to unrelated slots (e.g. r[5]).
  //    CONTEXT_BARRIER, VOLATILE, MEMORY make all live.
  {
    const uint32_t r13 = layout.r_offset + 13u * 8u;
    std::vector<xba::LivenessBlock> blocks(2);
    blocks[0].instrs = {
        {xba::LivenessInstrKind::kStoreContext, r5, 8u},
        {xba::LivenessInstrKind::kStoreContext, r13, 8u},
        {xba::LivenessInstrKind::kStoreContext, cr0, 4u},
        {xba::LivenessInstrKind::kCheckPreempt},
        {xba::LivenessInstrKind::kBranch, 0, 0, 1u},
    };
    blocks[0].successors = {1u};
    blocks[1].instrs = {
        {xba::LivenessInstrKind::kStoreContext, r5, 8u},
        {xba::LivenessInstrKind::kStoreContext, r13, 8u},
        {xba::LivenessInstrKind::kStoreContext, cr0, 4u},
    };
    auto res = xba::SolveCrossBlockContextLiveness(blocks, layout, true, false);
    WAVE4_CHECK(res.dead_store[0][0] == true);   // r5 dead across CHECK_PREEMPT
    WAVE4_CHECK(res.dead_store[0][1] == false);  // r13 kept live by CHECK_PREEMPT
    WAVE4_CHECK(res.dead_store[0][2] == false);  // cr0 kept live by CHECK_PREEMPT

    // Replace CHECK_PREEMPT with CONTEXT_BARRIER -> must keep r5 store alive.
    blocks[0].instrs[3].kind = xba::LivenessInstrKind::kContextBarrier;
    res = xba::SolveCrossBlockContextLiveness(blocks, layout, true, false);
    WAVE4_CHECK(res.dead_store[0][0] == false);

    // Replace with OPCODE_FLAG_VOLATILE -> must keep alive.
    blocks[0].instrs[3].kind = xba::LivenessInstrKind::kVolatile;
    res = xba::SolveCrossBlockContextLiveness(blocks, layout, true, false);
    WAVE4_CHECK(res.dead_store[0][0] == false);

    // Replace with OPCODE_FLAG_MEMORY -> must keep alive (including v[4] when tracked).
    const uint32_t v4 = layout.v_offset + 4u * 16u;
    blocks[0].instrs.resize(2);
    blocks[0].instrs[0] = {xba::LivenessInstrKind::kStoreContext, v4, 16u};
    blocks[0].instrs[1] = {xba::LivenessInstrKind::kMemory};
    blocks[1].instrs.resize(1);
    blocks[1].instrs[0] = {xba::LivenessInstrKind::kStoreContext, v4, 16u};
    res = xba::SolveCrossBlockContextLiveness(blocks, layout, true, true);
    WAVE4_CHECK(res.dead_store[0][0] == false);
  }

  // 4. Partial 4B write to r5 must NOT kill prior 8B store to r5, whereas
  //    4B write to cr0.value DOES kill prior 1B store to cr0_lt.
  {
    std::vector<xba::LivenessBlock> blocks(1);
    blocks[0].instrs = {
        {xba::LivenessInstrKind::kStoreContext, r5, 8u},
        {xba::LivenessInstrKind::kStoreContext, r5, 4u},
        {xba::LivenessInstrKind::kStoreContext, cr0 + 0u, 1u},
        {xba::LivenessInstrKind::kStoreContext, cr0, 4u},
    };
    auto res = xba::SolveCrossBlockContextLiveness(blocks, layout, true, false);
    WAVE4_CHECK(res.dead_store[0][0] == false);  // r5 8B kept alive by exit!
    WAVE4_CHECK(res.dead_store[0][2] == true);   // cr0_lt 1B killed by cr0.value 4B!
  }
}

void TestNaturalLoopDetectionAndBudgeter() {
  const xba::ContextLayout layout;
  const uint32_t r4 = layout.r_offset + 4u * 8u;
  const uint32_t r5 = layout.r_offset + 5u * 8u;
  const uint32_t r6 = layout.r_offset + 6u * 8u;
  const uint32_t r7 = layout.r_offset + 7u * 8u;
  const uint32_t r8 = layout.r_offset + 8u * 8u;
  const uint32_t v0 = layout.v_offset;

  auto make_hot_loop_body = [&](uint32_t reg_off) {
    return std::vector<xba::LoopInstr>{
        {xba::LoopInstrKind::kLoadContext, reg_off, 8u},
        {xba::LoopInstrKind::kStoreContext, reg_off, 8u},
    };
  };

  // 1. Accepts 1-block and 4-block single-entry loops; rejects 5-block loops.
  {
    // 1-block loop: B0(entry) -> B1(header/latch: B1->B1, fallthrough B2) -> B2(ret)
    std::vector<xba::LoopBlock> blocks(3);
    blocks[0].fallthrough_block = 1u;
    blocks[1].instrs = make_hot_loop_body(r4);
    blocks[1].instrs.push_back(
        {xba::LoopInstrKind::kBranchCond, 0, 0, 1u, false, false});
    blocks[1].fallthrough_block = 2u;
    blocks[2].instrs = {{xba::LoopInstrKind::kReturn, 0, 0, UINT32_MAX, false, true}};

    xba::LoopAnalysisOptions opts;
    auto plans = xba::SelectPromotableLoops(blocks, layout, opts);
    WAVE4_CHECK(plans.size() == 1u);
    WAVE4_CHECK(plans[0].header == 1u);
    WAVE4_CHECK(plans[0].blocks.size() == 1u);
    WAVE4_CHECK(plans[0].promoted.size() == 1u);
    WAVE4_CHECK(plans[0].promoted[0].offset == r4);

    // 4-block loop: B0 -> B1 -> B2 -> B3 -> B4 -> B1, exit to B5.
    std::vector<xba::LoopBlock> loop4(6);
    loop4[0].fallthrough_block = 1u;
    loop4[1].instrs = {{xba::LoopInstrKind::kLoadContext, r4, 8u}};
    loop4[1].fallthrough_block = 2u;
    loop4[2].fallthrough_block = 3u;
    loop4[3].fallthrough_block = 4u;
    loop4[4].instrs = {
        {xba::LoopInstrKind::kStoreContext, r4, 8u},
        {xba::LoopInstrKind::kBranchCond, 0, 0, 1u},
    };
    loop4[4].fallthrough_block = 5u;
    loop4[5].instrs = {{xba::LoopInstrKind::kReturn, 0, 0, UINT32_MAX, false, true}};
    plans = xba::SelectPromotableLoops(loop4, layout, opts);
    WAVE4_CHECK(plans.size() == 1u);
    WAVE4_CHECK(plans[0].blocks.size() == 4u);

    // 5-block loop -> rejected.
    std::vector<xba::LoopBlock> loop5(7);
    loop5[0].fallthrough_block = 1u;
    loop5[1].instrs = {{xba::LoopInstrKind::kLoadContext, r4, 8u}};
    loop5[1].fallthrough_block = 2u;
    loop5[2].fallthrough_block = 3u;
    loop5[3].fallthrough_block = 4u;
    loop5[4].fallthrough_block = 5u;
    loop5[5].instrs = {
        {xba::LoopInstrKind::kStoreContext, r4, 8u},
        {xba::LoopInstrKind::kBranchCond, 0, 0, 1u},
    };
    loop5[5].fallthrough_block = 6u;
    loop5[6].instrs = {{xba::LoopInstrKind::kReturn, 0, 0, UINT32_MAX, false, true}};
    plans = xba::SelectPromotableLoops(loop5, layout, opts);
    WAVE4_CHECK(plans.empty());
  }

  // 2. Rejects multi-entry irreducible loops.
  {
    // B0 branches conditionally to B1 or B2; B1 <-> B2 mutual cycle.
    std::vector<xba::LoopBlock> irred(4);
    irred[0].instrs = {{xba::LoopInstrKind::kBranchCond, 0, 0, 2u}};
    irred[0].fallthrough_block = 1u;
    irred[1].instrs = make_hot_loop_body(r4);
    irred[1].instrs.push_back({xba::LoopInstrKind::kBranchCond, 0, 0, 2u});
    irred[1].fallthrough_block = 3u;
    irred[2].instrs = make_hot_loop_body(r4);
    irred[2].instrs.push_back({xba::LoopInstrKind::kBranchCond, 0, 0, 1u});
    irred[2].fallthrough_block = 3u;
    irred[3].instrs = {{xba::LoopInstrKind::kReturn, 0, 0, UINT32_MAX, false, true}};
    auto plans = xba::SelectPromotableLoops(irred, layout, {});
    WAVE4_CHECK(plans.empty());
  }

  // 3. Rejects loops containing CALL, TRAP, CONTEXT_BARRIER, or VOLATILE.
  for (auto bad_kind : {xba::LoopInstrKind::kCall, xba::LoopInstrKind::kTrap,
                        xba::LoopInstrKind::kContextBarrier,
                        xba::LoopInstrKind::kVolatile}) {
    std::vector<xba::LoopBlock> blocks(3);
    blocks[0].fallthrough_block = 1u;
    blocks[1].instrs = {
        {xba::LoopInstrKind::kLoadContext, r4, 8u},
        {bad_kind},
        {xba::LoopInstrKind::kStoreContext, r4, 8u},
        {xba::LoopInstrKind::kBranchCond, 0, 0, 1u},
    };
    blocks[1].fallthrough_block = 2u;
    blocks[2].instrs = {{xba::LoopInstrKind::kReturn, 0, 0, UINT32_MAX, false, true}};
    WAVE4_CHECK(xba::SelectPromotableLoops(blocks, layout, {}).empty());
  }

  // 4. Nested loops: picks innermost loop.
  {
    // Outer loop {B1, B2, B3}; inner loop {B2} with higher-frequency back-edge B2->B2.
    std::vector<xba::LoopBlock> nested(5);
    nested[0].fallthrough_block = 1u;
    nested[1].instrs = {{xba::LoopInstrKind::kLoadContext, r4, 8u},
                        {xba::LoopInstrKind::kStoreContext, r4, 8u}};
    nested[1].fallthrough_block = 2u;
    nested[2].instrs = {
        {xba::LoopInstrKind::kLoadContext, r5, 8u},
        {xba::LoopInstrKind::kStoreContext, r5, 8u},
        {xba::LoopInstrKind::kBranchCond, 0, 0, 2u},
    };
    nested[2].fallthrough_block = 3u;
    nested[3].instrs = {{xba::LoopInstrKind::kBranchCond, 0, 0, 1u}};
    nested[3].fallthrough_block = 4u;
    nested[4].instrs = {{xba::LoopInstrKind::kReturn, 0, 0, UINT32_MAX, false, true}};

    auto plans = xba::SelectPromotableLoops(nested, layout, {});
    WAVE4_CHECK(plans.size() == 1u);
    WAVE4_CHECK(plans[0].header == 2u);
    WAVE4_CHECK(plans[0].blocks == std::vector<uint32_t>{2u});
  }

  // 5. Extended-block fall-through split: post-loop instructions after loop_branch
  //    (e.g., CALL after loop exit in the same block) do NOT disqualify the loop.
  {
    std::vector<xba::LoopBlock> ext(2);
    ext[0].fallthrough_block = 1u;
    ext[1].instrs = {
        {xba::LoopInstrKind::kLoadContext, r4, 8u},
        {xba::LoopInstrKind::kStoreContext, r4, 8u},
        {xba::LoopInstrKind::kBranchCond, 0, 0, 1u},  // index 2 = loop_end
        {xba::LoopInstrKind::kCall},                  // post-loop in same block!
        {xba::LoopInstrKind::kReturn, 0, 0, UINT32_MAX, false, true},
    };
    auto plans = xba::SelectPromotableLoops(ext, layout, {});
    WAVE4_CHECK(plans.size() == 1u);
    WAVE4_CHECK(plans[0].header == 1u);
    WAVE4_CHECK(plans[0].loop_end_instr_index[0] == 2u);
  }

  // 6. Register budgeter: <= 3 callee-saved GPRs when ClobbersVolatileGprs,
  //    <= 4 VECs, global cap <= 6.
  {
    std::vector<xba::LoopBlock> heavy(3);
    heavy[0].fallthrough_block = 1u;
    for (uint32_t k = 0; k < 5; ++k) {
      const uint32_t rk = layout.r_offset + (4u + k) * 8u;
      heavy[1].instrs.push_back({xba::LoopInstrKind::kLoadContext, rk, 8u});
      heavy[1].instrs.push_back({xba::LoopInstrKind::kStoreContext, rk, 8u});
    }
    for (uint32_t k = 0; k < 5; ++k) {
      const uint32_t vk = v0 + k * 16u;
      heavy[1].instrs.push_back({xba::LoopInstrKind::kLoadContext, vk, 16u});
      heavy[1].instrs.push_back({xba::LoopInstrKind::kStoreContext, vk, 16u});
    }
    heavy[1].instrs.push_back(
        {xba::LoopInstrKind::kCheckPreempt, 0, 0, UINT32_MAX, true, false});
    heavy[1].instrs.push_back({xba::LoopInstrKind::kBranchCond, 0, 0, 1u});
    heavy[1].fallthrough_block = 2u;
    heavy[2].instrs = {{xba::LoopInstrKind::kReturn, 0, 0, UINT32_MAX, false, true}};

    xba::LoopAnalysisOptions opts;
    opts.promote_vec128 = true;
    opts.max_regs = 6u;
    auto plans = xba::SelectPromotableLoops(heavy, layout, opts);
    WAVE4_CHECK(plans.size() == 1u);
    WAVE4_CHECK(plans[0].clobbers_volatile_gprs == true);
    WAVE4_CHECK(plans[0].promoted.size() <= 6u);

    uint32_t gpr_count = 0;
    uint32_t vec_count = 0;
    for (const auto& p : plans[0].promoted) {
      if (p.is_vec) {
        ++vec_count;
        WAVE4_CHECK(p.host_reg_index >= 24u && p.host_reg_index <= 27u);
      } else {
        ++gpr_count;
        // Must be callee-saved indices 0..6 (specifically x26..x28 -> 4..6).
        WAVE4_CHECK(p.host_reg_index >= 4u && p.host_reg_index <= 6u);
      }
    }
    WAVE4_CHECK(gpr_count <= 3u);
    WAVE4_CHECK(vec_count <= 4u);

    // Without clobber, up to 5 GPRs allowed when max_regs allows, still capped by global 6.
    heavy[1].instrs[20].clobbers_volatile_gprs = false;
    (void)r6;
    (void)r7;
    (void)r8;
    opts.promote_vec128 = false;
    plans = xba::SelectPromotableLoops(heavy, layout, opts);
    WAVE4_CHECK(plans.size() == 1u);
    WAVE4_CHECK(plans[0].promoted.size() <= 5u);
    for (const auto& p : plans[0].promoted) {
      WAVE4_CHECK(!p.is_vec);
    }
  }

  // 7. CHECK_PREEMPT must exclude PPCContext::r[13] (KPCR) from loop promotion
  //    while still allowing other hot GPRs (e.g. r[5]) in the same loop.
  {
    const uint32_t r13 = layout.r_offset + 13u * 8u;
    std::vector<xba::LoopBlock> blocks(3);
    blocks[0].fallthrough_block = 1u;
    blocks[1].instrs = {
        {xba::LoopInstrKind::kLoadContext, r13, 8u},
        {xba::LoopInstrKind::kLoadContext, r13, 8u},
        {xba::LoopInstrKind::kStoreContext, r13, 8u},
        {xba::LoopInstrKind::kLoadContext, r5, 8u},
        {xba::LoopInstrKind::kStoreContext, r5, 8u},
        {xba::LoopInstrKind::kCheckPreempt, 0, 0, UINT32_MAX, true, false},
        {xba::LoopInstrKind::kBranchCond, 0, 0, 1u},
    };
    blocks[1].fallthrough_block = 2u;
    blocks[2].instrs = {
        {xba::LoopInstrKind::kReturn, 0, 0, UINT32_MAX, false, true}};

    auto with_preempt = xba::SelectPromotableLoops(blocks, layout, {});
    WAVE4_CHECK(with_preempt.size() == 1u);
    bool saw_r5 = false;
    bool saw_r13 = false;
    for (const auto& p : with_preempt[0].promoted) {
      if (p.offset == r5) saw_r5 = true;
      if (p.offset == r13) saw_r13 = true;
    }
    WAVE4_CHECK(saw_r5);
    WAVE4_CHECK(!saw_r13);

    // Removing CHECK_PREEMPT allows r13 to be promoted again.
    blocks[1].instrs[5] = {xba::LoopInstrKind::kOther};
    auto without_preempt = xba::SelectPromotableLoops(blocks, layout, {});
    WAVE4_CHECK(without_preempt.size() == 1u);
    bool saw_r13_without = false;
    for (const auto& p : without_preempt[0].promoted) {
      if (p.offset == r13) saw_r13_without = true;
    }
    WAVE4_CHECK(saw_r13_without);
  }
}

// =============================================================================
// Pillar 1, 3, 4 Contract & Regression Source Checks
// =============================================================================

void TestWave4SourceContracts(const fs::path& cpp_dir) {
  const fs::path repo_root =
      cpp_dir.parent_path().parent_path().parent_path().parent_path();

  // Pillar 1: context_promotion_pass.cc, register_allocation_pass.cc, value.cc,
  //           plus Wave 3 follow-ups in processor.h / a64_backend.cc / a64_function.h.
  {
    const std::string ctx_promo = ReadSourceText(
        cpp_dir /
        "xenia/src/xenia/cpu/compiler/passes/context_promotion_pass.cc");
    WAVE4_CHECK(ctx_promo.find("OPCODE_CONTEXT_BARRIER_info") !=
                std::string::npos);
    // Regression check: must NOT exclude PPCContext::v in OPCODE_FLAG_MEMORY.
    WAVE4_CHECK(ctx_promo.find("v_begin") == std::string::npos);
    WAVE4_CHECK(ctx_promo.find("v_end") == std::string::npos);
    WAVE4_CHECK(ctx_promo.find("DEFINE_bool(\n    context_promote_vec128, true,") !=
                    std::string::npos ||
                ctx_promo.find("DEFINE_bool(context_promote_vec128, true,") !=
                    std::string::npos);

    const std::string reg_alloc = ReadSourceText(
        cpp_dir /
        "xenia/src/xenia/cpu/compiler/passes/register_allocation_pass.cc");
    WAVE4_CHECK(reg_alloc.find("OPCODE_CONTEXT_BARRIER_info") !=
                std::string::npos);

    const std::string hir_value =
        ReadSourceText(cpp_dir / "xenia/src/xenia/cpu/hir/value.cc");
    WAVE4_CHECK(hir_value.find("(index->constant.u8 & 0xF) ^ 3") !=
                std::string::npos);
    WAVE4_CHECK(hir_value.find("(index->constant.u8 & 0x7) ^ 1") !=
                std::string::npos);
    WAVE4_CHECK(hir_value.find("index->constant.u8 & 0x3") !=
                std::string::npos);

    const std::string proc_h =
        ReadSourceText(cpp_dir / "xenia/src/xenia/cpu/processor.h");
    WAVE4_CHECK(proc_h.find("BumpCodeSweepCount") != std::string::npos);
    WAVE4_CHECK(proc_h.find("SetJitProfileFilter") != std::string::npos);
    WAVE4_CHECK(proc_h.find("DrainJitProfileAddresses") != std::string::npos);

    const std::string a64_backend = ReadSourceText(
        cpp_dir / "xenia/src/xenia/cpu/backend/a64/a64_backend.cc");
    WAVE4_CHECK(a64_backend.find("BumpCodeSweepCount()") != std::string::npos);

    const std::string a64_fn_h = ReadSourceText(
        cpp_dir / "xenia/src/xenia/cpu/backend/a64/a64_function.h");
    WAVE4_CHECK(a64_fn_h.find("std::atomic<size_t> stack_size_") !=
                std::string::npos);
  }

  // Pillar 3: Emulator::Shutdown and Emulator::TerminateTitle call
  //           StopJitProfileSession(true) before teardown, and
  //           CompleteLaunchModule arms StartJitProfileSession before
  //           main_thread_->Resume().
  {
    const std::string emu_cc =
        ReadSourceText(cpp_dir / "xenia/src/xenia/emulator.cc");
    const size_t stop_pos = emu_cc.find("StopJitProfileSession(true)");
    const size_t ks_reset_pos = emu_cc.find("kernel_state_.reset()");
    const size_t proc_reset_pos = emu_cc.find("processor_.reset()");
    WAVE4_CHECK(stop_pos != std::string::npos);
    WAVE4_CHECK(ks_reset_pos != std::string::npos);
    WAVE4_CHECK(proc_reset_pos != std::string::npos);
    WAVE4_CHECK(stop_pos < ks_reset_pos);
    WAVE4_CHECK(stop_pos < proc_reset_pos);

    const size_t term_pos = emu_cc.find("X_STATUS Emulator::TerminateTitle()");
    WAVE4_CHECK(term_pos != std::string::npos);
    const size_t term_stop_pos = emu_cc.find("StopJitProfileSession(true);", term_pos);
    const size_t term_ks_pos = emu_cc.find("kernel_state_->TerminateTitle();", term_pos);
    WAVE4_CHECK(term_stop_pos != std::string::npos);
    WAVE4_CHECK(term_ks_pos != std::string::npos);
    WAVE4_CHECK(term_stop_pos < term_ks_pos);

    const size_t start_pos = emu_cc.find("StartJitProfileSession(module.get());");
    const size_t resume_pos = emu_cc.find("main_thread_->Resume();");
    WAVE4_CHECK(start_pos != std::string::npos);
    WAVE4_CHECK(resume_pos != std::string::npos);
    WAVE4_CHECK(start_pos < resume_pos);
  }

  // Pillar 4: CMakeLists.txt & build.gradle PGO opt-in flags (default OFF).
  {
    const std::string cmake_txt = ReadSourceText(cpp_dir / "CMakeLists.txt");
    WAVE4_CHECK(cmake_txt.find("XENDROID_PGO_MODE") != std::string::npos);
    WAVE4_CHECK(cmake_txt.find("XENDROID_PGO_PROFILE") != std::string::npos);
    WAVE4_CHECK(cmake_txt.find("\"OFF\"") != std::string::npos);
    WAVE4_CHECK(cmake_txt.find("-fprofile-generate=") != std::string::npos);
    WAVE4_CHECK(cmake_txt.find("-fprofile-use=") != std::string::npos);
    WAVE4_CHECK(cmake_txt.find("-Wno-profile-instr-unprofiled") !=
                std::string::npos);
    WAVE4_CHECK(cmake_txt.find("-Wno-profile-instr-out-of-date") !=
                std::string::npos);

    const std::string build_gradle = ReadSourceText(
        cpp_dir.parent_path().parent_path().parent_path() / "build.gradle");
    WAVE4_CHECK(build_gradle.find("xendroidPgoMode") != std::string::npos);
    WAVE4_CHECK(build_gradle.find("xendroidPgoProfile") != std::string::npos);
    WAVE4_CHECK(build_gradle.find("XENDROID_PGO_MODE") != std::string::npos);
    WAVE4_CHECK(build_gradle.find("XENDROID_PGO_PROFILE") != std::string::npos);
  }

  // TOML-only vs UI-exposed CVar contract checks.
  {
    const std::string schema_kt = ReadSourceText(
        repo_root / "app/src/main/java/xendroid/compose/settings/SettingsSchema.kt");
    const std::string catalog_kt = ReadSourceText(
        repo_root / "app/src/main/java/xendroid/compose/settings/SettingCatalog.kt");

    // UI-exposed: jit_warmup_jprof must appear in both schema & catalog.
    WAVE4_CHECK(schema_kt.find("jit_warmup_jprof") != std::string::npos);
    WAVE4_CHECK(catalog_kt.find("CPU|jit_warmup_jprof") != std::string::npos);

    // TOML-only internal CVars must NOT appear in UI schema or catalog.
    const std::array<const char*, 9> toml_only_cvars = {
        "dse_cross_block_context",
        "dse_cross_block_vec128",
        "a64_loop_context_promotion",
        "a64_loop_promote_max_regs",
        "a64_loop_promote_defer_stores_across_memory",
        "context_promote_vec128",
        "jit_warmup_jprof_threads",
        "jit_warmup_jprof_save_interval_ms",
        "jit_warmup_jprof_save_batch",
    };
    for (const char* name : toml_only_cvars) {
      WAVE4_CHECK(schema_kt.find(name) == std::string::npos);
      WAVE4_CHECK(catalog_kt.find(name) == std::string::npos);
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(
        stderr,
        "Usage: %s <emulator-core/src/main/cpp> <scratch_dir>\n", argv[0]);
    return 2;
  }
  const fs::path cpp_dir(argv[1]);
  const fs::path scratch_dir(argv[2]);

  TestJprofFileFormat(scratch_dir / "jprof");
  TestContextBitsetAndDualAliasing();
  TestExtendedBlockBackwardLiveness();
  TestNaturalLoopDetectionAndBudgeter();
  TestWave4SourceContracts(cpp_dir);

  std::puts("wave4_crossblock_jprof_logic_test: PASS");
  return 0;
}
