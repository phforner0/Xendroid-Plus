#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "xenia/gpu/pm4_ib_cache.h"
#include "xenia/gpu/pm4_noop_filter.h"

namespace {

#define PM4_IB_CHECK(cond)                                                  \
  do {                                                                      \
    if (!(cond)) {                                                          \
      std::fprintf(stderr, "GPU_PM4_IB_CACHE check failed (%s:%d): %s\n",   \
                   __FILE__, __LINE__, #cond);                              \
      std::abort();                                                         \
    }                                                                       \
  } while (false)

namespace fs = std::filesystem;
namespace nf = xe::gpu::pm4_noop;
namespace ib = xe::gpu::pm4_ib;

uint32_t Swap32(uint32_t v) { return __builtin_bswap32(v); }

uint32_t MakeType0(uint16_t base_reg, uint16_t count, bool one_reg = false) {
  return (0u << 30) | (uint32_t((count - 1u) & 0x3FFFu) << 16) |
         (one_reg ? (1u << 15) : 0u) | uint32_t(base_reg & 0x7FFFu);
}

uint32_t MakeType1(uint16_t reg1, uint16_t reg2) {
  return (1u << 30) | (uint32_t(reg2 & 0x7FFu) << 11) |
         uint32_t(reg1 & 0x7FFu);
}

uint32_t MakeType2() { return 2u << 30; }

uint32_t MakeType3(uint32_t op, uint16_t count, bool predicated = false) {
  return (3u << 30) | (uint32_t((count - 1u) & 0x3FFFu) << 16) |
         ((op & 0x7Fu) << 8) | (predicated ? 1u : 0u);
}

uint32_t MakeDrawInitiator(uint32_t prim_type, uint32_t source_select,
                           uint32_t index_size, uint32_t num_indices) {
  return (prim_type & 0x3Fu) | ((source_select & 0x3u) << 6) |
         ((index_size & 0x1u) << 11) | ((num_indices & 0xFFFFu) << 16);
}

std::string NormalizeNewlines(std::string s) {
  s.erase(std::remove(s.begin(), s.end(), '\r'), s.end());
  return s;
}

std::string ReadSourceText(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  PM4_IB_CHECK(in.good());
  std::ostringstream ss;
  ss << in.rdbuf();
  return NormalizeNewlines(ss.str());
}

void TestNoopFilter4WideAllLengthsAndAlignments() {
  constexpr uint32_t kRegFileSize = 0x5003u;
  std::vector<uint32_t> regs(kRegFileSize, 0u);
  std::vector<uint32_t> expected(kRegFileSize, 0u);

  // Test aligned and unaligned register starts (+0..+3), sizes 1..19, AND
  // unaligned byte source buffers (+0..+3 bytes) for both BigEndian and
  // HostEndian variants.
  for (uint32_t start_align = 0; start_align < 4u; ++start_align) {
    const uint32_t base_reg = 0x2010u + start_align;  // non-side-effect range
    for (uint32_t len = 1u; len <= 19u; ++len) {
      // Seed initial values.
      for (uint32_t i = 0; i < len; ++i) {
        const uint32_t v = 0x10000000u + (start_align << 16) + (len << 8) + i;
        regs[base_reg + i] = v;
        expected[base_reg + i] = v;
      }

      // Prepare big-endian source in an intentionally unaligned byte buffer.
      std::vector<uint8_t> raw_be(len * 4u + 8u, 0xEEu);
      const uint32_t byte_shift = (start_align + len) & 3u;
      uint8_t* be_dst = raw_be.data() + byte_shift;
      for (uint32_t i = 0; i < len; ++i) {
        const uint32_t swapped = Swap32(expected[base_reg + i]);
        std::memcpy(be_dst + i * 4u, &swapped, 4u);
      }

      // 1) Identical write -> ZERO changes, ZERO dirty callbacks.
      std::vector<uint32_t> dirty_regs;
      auto res = nf::FilterWriteRegisters4WideBigEndian(
          regs.data(), base_reg, reinterpret_cast<const uint32_t*>(be_dst), len,
          [&](uint32_t r) { dirty_regs.push_back(r); });
      PM4_IB_CHECK(!res.any_changed);
      PM4_IB_CHECK(!res.system_constants_dirty);
      PM4_IB_CHECK(res.changed_count == 0u);
      PM4_IB_CHECK(res.skipped_count == len);
      PM4_IB_CHECK(dirty_regs.empty());

      // 2) Mutate each single lane individually -> exact single dirty callback,
      //    bit-identical register_file_ update, including +0.0f vs -0.0f!
      for (uint32_t pos = 0; pos < len; ++pos) {
        const uint32_t new_val =
            (pos == 0u) ? 0x00000000u
                        : ((pos == 1u) ? 0x80000000u
                                       : (expected[base_reg + pos] ^ 0xA55A00FFu));
        // Force +0.0f -> -0.0f test when pos == 1.
        if (pos == 1u) {
          regs[base_reg + pos] = 0x00000000u;
          expected[base_reg + pos] = 0x00000000u;
        }
        expected[base_reg + pos] = new_val;
        const uint32_t swapped = Swap32(new_val);
        std::memcpy(be_dst + pos * 4u, &swapped, 4u);

        dirty_regs.clear();
        res = nf::FilterWriteRegisters4WideBigEndian(
            regs.data(), base_reg, reinterpret_cast<const uint32_t*>(be_dst),
            len, [&](uint32_t r) { dirty_regs.push_back(r); });
        PM4_IB_CHECK(res.any_changed);
        PM4_IB_CHECK(res.changed_count == 1u);
        PM4_IB_CHECK(res.skipped_count == len - 1u);
        PM4_IB_CHECK(res.first_changed_index == base_reg + pos);
        PM4_IB_CHECK(res.last_changed_index == base_reg + pos);
        PM4_IB_CHECK(dirty_regs.size() == 1u &&
                     dirty_regs[0] == base_reg + pos);
        for (uint32_t k = 0; k < len; ++k) {
          PM4_IB_CHECK(regs[base_reg + k] == expected[base_reg + k]);
        }

        // Immediate repeat -> ZERO dirtying again.
        dirty_regs.clear();
        res = nf::FilterWriteRegisters4WideBigEndian(
            regs.data(), base_reg, reinterpret_cast<const uint32_t*>(be_dst),
            len, [&](uint32_t r) { dirty_regs.push_back(r); });
        PM4_IB_CHECK(!res.any_changed);
        PM4_IB_CHECK(res.changed_count == 0u);
        PM4_IB_CHECK(res.skipped_count == len);
        PM4_IB_CHECK(dirty_regs.empty());
      }

      // 3) Mixed changes (every 3rd lane) via HostEndian path.
      std::vector<uint32_t> host_src(len);
      std::vector<uint32_t> expected_dirty;
      for (uint32_t i = 0; i < len; ++i) {
        if ((i % 3u) == 0u) {
          host_src[i] = expected[base_reg + i] ^ 0x0F0F0F0Fu;
          expected[base_reg + i] = host_src[i];
          expected_dirty.push_back(base_reg + i);
        } else {
          host_src[i] = expected[base_reg + i];
        }
      }
      dirty_regs.clear();
      res = nf::FilterWriteRegisters4WideHostEndian(
          regs.data(), base_reg, host_src.data(), len,
          [&](uint32_t r) { dirty_regs.push_back(r); });
      PM4_IB_CHECK(res.any_changed);
      PM4_IB_CHECK(res.changed_count == expected_dirty.size());
      PM4_IB_CHECK(res.skipped_count == len - expected_dirty.size());
      PM4_IB_CHECK(res.first_changed_index == expected_dirty.front());
      PM4_IB_CHECK(res.last_changed_index == expected_dirty.back());
      PM4_IB_CHECK(dirty_regs == expected_dirty);
      for (uint32_t k = 0; k < len; ++k) {
        PM4_IB_CHECK(regs[base_reg + k] == expected[base_reg + k]);
      }

      // 4) Non-mutating DispatchChangedRegisters4WideHostEndian MUST present
      //    the OLD register value inside the callback (critical for Vulkan
      //    WriteRegister fetch/float dirty tracking).
      expected_dirty.clear();
      for (uint32_t i = 0; i < len; ++i) {
        if ((i & 1u) != 0u) {
          host_src[i] = regs[base_reg + i] ^ 0x13572468u;
          expected_dirty.push_back(base_reg + i);
        } else {
          host_src[i] = regs[base_reg + i];
        }
      }
      dirty_regs.clear();
      res = nf::DispatchChangedRegisters4WideHostEndian(
          regs.data(), base_reg, host_src.data(), len,
          [&](uint32_t reg_idx, uint32_t new_val) {
            // Caller MUST observe old value != new_val before updating!
            PM4_IB_CHECK(regs[reg_idx] != new_val);
            regs[reg_idx] = new_val;
            dirty_regs.push_back(reg_idx);
          });
      PM4_IB_CHECK(res.changed_count == expected_dirty.size());
      PM4_IB_CHECK(res.skipped_count == len - expected_dirty.size());
      PM4_IB_CHECK(dirty_regs == expected_dirty);
      for (uint32_t k = 0; k < len; ++k) {
        PM4_IB_CHECK(regs[base_reg + k] == host_src[k]);
      }
    }
  }
}

void TestNoopFilterMandatorySideEffectsAndSysConst() {
  // Exact classification boundaries for mandatory side-effect registers.
  PM4_IB_CHECK(!nf::HasMandatoryWriteSideEffect(nf::kRegScratchReg0 - 1u));
  for (uint32_t r = nf::kRegScratchReg0; r <= nf::kRegScratchReg7; ++r) {
    PM4_IB_CHECK(nf::HasMandatoryWriteSideEffect(r));
  }
  PM4_IB_CHECK(!nf::HasMandatoryWriteSideEffect(nf::kRegScratchReg7 + 1u));

  PM4_IB_CHECK(!nf::HasMandatoryWriteSideEffect(nf::kRegCoherStatusHost - 1u));
  PM4_IB_CHECK(nf::HasMandatoryWriteSideEffect(nf::kRegCoherStatusHost));
  PM4_IB_CHECK(!nf::HasMandatoryWriteSideEffect(nf::kRegCoherStatusHost + 1u));

  PM4_IB_CHECK(!nf::HasMandatoryWriteSideEffect(nf::kRegDcLutRwIndex - 1u));
  for (uint32_t r = nf::kRegDcLutRwIndex; r <= nf::kRegDcLut30Color; ++r) {
    PM4_IB_CHECK(nf::HasMandatoryWriteSideEffect(r));
  }
  PM4_IB_CHECK(!nf::HasMandatoryWriteSideEffect(nf::kRegDcLut30Color + 1u));

  std::vector<uint32_t> regs(0x5003u, 0u);

  // Pre-populate ranges spanning all 3 side-effect islands with non-zero values.
  auto test_island = [&](uint32_t start, uint32_t count,
                         uint32_t expected_specials) {
    std::vector<uint32_t> src_be(count);
    std::vector<uint32_t> src_he(count);
    for (uint32_t i = 0; i < count; ++i) {
      const uint32_t val = 0x20000000u + start + i;
      regs[start + i] = val;
      src_he[i] = val;
      src_be[i] = Swap32(val);
    }

    // Identical write across island (BigEndian store-in-place):
    // ALL mandatory side-effect regs MUST still invoke on_special_write even
    // though new == old, while surrounding unchanged regs do NOT dirty!
    std::vector<uint32_t> specials_fired;
    std::vector<uint32_t> dirtied;
    auto res = nf::FilterWritePossiblySpecialRegistersBigEndian(
        regs.data(), start, src_be.data(), count,
        [&](uint32_t r, uint32_t v) {
          PM4_IB_CHECK(v == src_he[r - start]);
          specials_fired.push_back(r);
          if (r == nf::kRegCoherStatusHost) {
            // Simulate CommandProcessor::HandleSpecialRegisterWrite setting bit 31.
            regs[r] |= 0x80000000u;
          }
        },
        [&](uint32_t r) { dirtied.push_back(r); });
    PM4_IB_CHECK(res.side_effect_count == expected_specials);
    PM4_IB_CHECK(specials_fired.size() == expected_specials);
    PM4_IB_CHECK(dirtied.empty());
    PM4_IB_CHECK(res.skipped_count == count - expected_specials);
    if (start <= nf::kRegCoherStatusHost &&
        nf::kRegCoherStatusHost < start + count) {
      PM4_IB_CHECK((regs[nf::kRegCoherStatusHost] & 0x80000000u) != 0u);
      // Restore value for dispatcher check below.
      regs[nf::kRegCoherStatusHost] = src_he[nf::kRegCoherStatusHost - start];
    }

    // Same invariant on DispatchChangedRegisters4WideHostEndian!
    std::vector<uint32_t> dispatched;
    auto dres = nf::DispatchChangedRegisters4WideHostEndian(
        regs.data(), start, src_he.data(), count,
        [&](uint32_t r, uint32_t v) {
          dispatched.push_back(r);
          regs[r] = v;
        });
    PM4_IB_CHECK(dres.side_effect_count == expected_specials);
    PM4_IB_CHECK(dispatched == specials_fired);
    PM4_IB_CHECK(dres.skipped_count == count - expected_specials);
  };

  test_island(nf::kRegScratchReg0 - 4u, 16u, 8u);
  test_island(nf::kRegCoherStatusHost - 5u, 11u, 1u);
  test_island(nf::kRegDcLutRwIndex - 4u, 12u, 4u);

  // Spanning ALL 3 islands in one single large write.
  test_island(nf::kRegScratchReg0 - 2u,
              (nf::kRegDcLut30Color - nf::kRegScratchReg0) + 6u, 13u);

  // System constant buffer dirtying: equal write -> NO dirty; changed write ->
  // dirty; changed write to guard-band (PA_CL_GB_*) -> stored bit-exact, NO
  // system constant dirty!
  constexpr uint32_t kPaClGbVertClipAdj = 0x2303u;
  const uint32_t ucp_start = nf::kRegPaClUcp0X;
  std::vector<uint32_t> ucp_be(8u);
  for (uint32_t i = 0; i < 8u; ++i) {
    regs[ucp_start + i] = 0x3F800000u + i;
    ucp_be[i] = Swap32(0x3F800000u + i);
  }
  auto res = nf::FilterWritePossiblySpecialRegistersBigEndian(
      regs.data(), ucp_start, ucp_be.data(), 8u,
      [](uint32_t, uint32_t) { PM4_IB_CHECK(false); });
  PM4_IB_CHECK(!res.any_changed);
  PM4_IB_CHECK(!res.system_constants_dirty);

  ucp_be[5] = Swap32(0x40400000u);
  res = nf::FilterWritePossiblySpecialRegistersBigEndian(
      regs.data(), ucp_start, ucp_be.data(), 8u,
      [](uint32_t, uint32_t) { PM4_IB_CHECK(false); });
  PM4_IB_CHECK(res.any_changed);
  PM4_IB_CHECK(res.system_constants_dirty);
  PM4_IB_CHECK(regs[ucp_start + 5u] == 0x40400000u);

  // Guard-band 4 registers (0x2303..0x2306): MUST be stored bit-exact for
  // PM4_REG_TO_MEM/COND_WRITE, without setting system_constants_dirty.
  uint32_t gb_be[4] = {Swap32(0x3F800000u), Swap32(0x40000000u),
                       Swap32(0x40400000u), Swap32(0x40800000u)};
  res = nf::FilterWriteRegisters4WideBigEndian(regs.data(), kPaClGbVertClipAdj,
                                               gb_be, 4u);
  PM4_IB_CHECK(res.any_changed);
  PM4_IB_CHECK(!res.system_constants_dirty);
  PM4_IB_CHECK(regs[0x2303u] == 0x3F800000u);
  PM4_IB_CHECK(regs[0x2304u] == 0x40000000u);
  PM4_IB_CHECK(regs[0x2305u] == 0x40400000u);
  PM4_IB_CHECK(regs[0x2306u] == 0x40800000u);
}

void TestIbCacheHysteresisEvictionAndLimits() {
  ib::Pm4IndirectBufferCache cache;
  constexpr uint32_t kGuestAddr = 0xA0100000u;
  constexpr uint32_t kPhysAddr = 0x00100000u;

  std::vector<uint32_t> pkt = {
      Swap32(MakeType0(0x2010u, 2u)),
      Swap32(0x11223344u),
      Swap32(0x55667788u),
  };

  // Visit 1: miss + hysteresis records first seen WITHOUT arming write-watch.
  PM4_IB_CHECK(cache.Lookup(kGuestAddr, uint32_t(pkt.size())) == nullptr);
  PM4_IB_CHECK(cache.ObserveMiss(kGuestAddr, pkt.data(), uint32_t(pkt.size())) ==
               ib::CandidateDecision::kFirstSeen);

  // Visit 2 (identical bytes): hysteresis confirms -> kReadyToCompile!
  PM4_IB_CHECK(cache.Lookup(kPhysAddr, uint32_t(pkt.size())) == nullptr);
  PM4_IB_CHECK(cache.ObserveMiss(kPhysAddr, pkt.data(), uint32_t(pkt.size())) ==
               ib::CandidateDecision::kReadyToCompile);

  ib::IbWatchSnapshot snap{};
  PM4_IB_CHECK(
      cache.BeginArmAndSnapshot(kGuestAddr, uint32_t(pkt.size()), snap));
  PM4_IB_CHECK(snap.phys_address == kPhysAddr);
  PM4_IB_CHECK(snap.byte_length == uint32_t(pkt.size() * 4u));
  PM4_IB_CHECK(snap.page_count == 1u);

  auto compiled = std::make_shared<ib::CompiledIndirectBuffer>();
  PM4_IB_CHECK(ib::DecodeIndirectBuffer(snap.phys_address, pkt.data(),
                                        uint32_t(pkt.size()), *compiled) ==
               ib::DecodeStatus::kOk);
  PM4_IB_CHECK(
      cache.CommitIfUnchanged(kGuestAddr, uint32_t(pkt.size()), snap, compiled));

  // Visit 3: HIT under both masked physical address and high-bit guest address!
  auto hit1 = cache.Lookup(kGuestAddr, uint32_t(pkt.size()));
  auto hit2 = cache.Lookup(kPhysAddr, uint32_t(pkt.size()));
  PM4_IB_CHECK(hit1 != nullptr && hit1.get() == compiled.get());
  PM4_IB_CHECK(hit2 != nullptr && hit2.get() == compiled.get());
  // Different dword_count -> miss.
  PM4_IB_CHECK(cache.Lookup(kPhysAddr, uint32_t(pkt.size()) + 1u) == nullptr);

  // Dynamic IB that mutates every visit: MUST NEVER reach kReadyToCompile!
  std::vector<uint32_t> dyn_pkt = pkt;
  for (uint32_t iter = 0; iter < 16u; ++iter) {
    dyn_pkt[1] = Swap32(0xDEAD0000u + iter);
    auto dec =
        cache.ObserveMiss(0x00150000u, dyn_pkt.data(), uint32_t(dyn_pkt.size()));
    PM4_IB_CHECK(dec == (iter == 0u ? ib::CandidateDecision::kFirstSeen
                                    : ib::CandidateDecision::kDynamicReset));
  }

  // Oversized IB (> 4096 dwords) or 0 dwords cleanly declined.
  std::vector<uint32_t> big_ib(ib::kIbCacheMaxDwords + 1u, Swap32(MakeType2()));
  PM4_IB_CHECK(cache.ObserveMiss(0x00200000u, big_ib.data(),
                                 uint32_t(big_ib.size())) ==
               ib::CandidateDecision::kSkipInvalidRange);
  PM4_IB_CHECK(!cache.BeginArmAndSnapshot(0x00200000u, uint32_t(big_ib.size()),
                                          snap));
  PM4_IB_CHECK(cache.ObserveMiss(0x00200000u, pkt.data(), 0u) ==
               ib::CandidateDecision::kSkipInvalidRange);

  // Capacity bound (1024 entries) + FIFO eviction.
  ib::Pm4IndirectBufferCache cap_cache;
  auto dummy_compiled = std::make_shared<ib::CompiledIndirectBuffer>();
  for (uint32_t i = 0; i < ib::kIbCacheMaxEntries + 12u; ++i) {
    const uint32_t addr = 0x01000000u + i * 64u;
    ib::IbWatchSnapshot s{};
    PM4_IB_CHECK(cap_cache.BeginArmAndSnapshot(addr, 4u, s));
    PM4_IB_CHECK(cap_cache.CommitIfUnchanged(addr, 4u, s, dummy_compiled));
  }
  PM4_IB_CHECK(cap_cache.entry_count() == ib::kIbCacheMaxEntries);
  PM4_IB_CHECK(cap_cache.stats().evictions == 12u);
  // Oldest 12 evicted; entry #12 and newest entry still cached.
  PM4_IB_CHECK(cap_cache.Lookup(0x01000000u, 4u) == nullptr);
  PM4_IB_CHECK(cap_cache.Lookup(0x01000000u + 11u * 64u, 4u) == nullptr);
  PM4_IB_CHECK(cap_cache.Lookup(0x01000000u + 12u * 64u, 4u) != nullptr);
  PM4_IB_CHECK(
      cap_cache.Lookup(0x01000000u + (ib::kIbCacheMaxEntries + 11u) * 64u, 4u) !=
      nullptr);
}

void TestIbCacheEpochClampCooldown64AndConcurrency() {
  ib::Pm4IndirectBufferCache cache;
  constexpr uint32_t kPhysAddr = 0x00120000u;
  std::vector<uint32_t> pkt = {Swap32(MakeType0(0x2020u, 1u)),
                               Swap32(0xCAFEBABEu)};
  auto compiled = std::make_shared<ib::CompiledIndirectBuffer>();
  PM4_IB_CHECK(ib::DecodeIndirectBuffer(kPhysAddr, pkt.data(),
                                        uint32_t(pkt.size()), *compiled) ==
               ib::DecodeStatus::kOk);

  // Hysteresis 2 hits -> arm & commit.
  PM4_IB_CHECK(cache.ObserveMiss(kPhysAddr, pkt.data(), 2u) ==
               ib::CandidateDecision::kFirstSeen);
  PM4_IB_CHECK(cache.ObserveMiss(kPhysAddr, pkt.data(), 2u) ==
               ib::CandidateDecision::kReadyToCompile);
  ib::IbWatchSnapshot snap{};
  PM4_IB_CHECK(cache.BeginArmAndSnapshot(kPhysAddr, 2u, snap));
  PM4_IB_CHECK(cache.CommitIfUnchanged(kPhysAddr, 2u, snap, compiled));
  PM4_IB_CHECK(cache.Lookup(kPhysAddr, 2u) != nullptr);

  // Far write in a DIFFERENT 4 MiB window -> unconstrained unwatch {0, UINT32_MAX}.
  auto far_unwatch =
      cache.OnMemoryInvalidatedForUnwatch(0x00800000u, 4096u, false);
  PM4_IB_CHECK(far_unwatch.first == 0u && far_unwatch.second == UINT32_MAX);
  PM4_IB_CHECK(cache.Lookup(kPhysAddr, 2u) != nullptr);

  // Adjacent page in the SAME 4 MiB window -> MUST clamp to (0x00121000, 4096)!
  auto near_unwatch =
      cache.OnMemoryInvalidatedForUnwatch(0x00121000u, 4096u, false);
  PM4_IB_CHECK(near_unwatch.first == 0x00121000u &&
               near_unwatch.second == 4096u);
  PM4_IB_CHECK(cache.Lookup(kPhysAddr, 2u) != nullptr);

  // Write to the cached IB's OWN page -> invalidates entry AND starts 64-frame
  // cooldown on that page!
  cache.OnMemoryInvalidated(kPhysAddr + 4u, 4u);
  PM4_IB_CHECK(cache.Lookup(kPhysAddr, 2u) == nullptr);
  PM4_IB_CHECK(cache.IsRangeInCooldown(kPhysAddr, 2u));

  // Verify ALL 64 cooldown frames (offsets 0..63) block re-arming even on
  // static bytes (both at same address and another offset on the same page!).
  for (uint64_t f = 0; f < ib::kIbCacheInvalidationCooldownFrames; ++f) {
    PM4_IB_CHECK(cache.IsRangeInCooldown(kPhysAddr, 2u));
    PM4_IB_CHECK(cache.ObserveMiss(kPhysAddr, pkt.data(), 2u) ==
                 ib::CandidateDecision::kInCooldown);
    PM4_IB_CHECK(cache.ObserveMiss(kPhysAddr + 0x80u, pkt.data(), 2u) ==
                 ib::CandidateDecision::kInCooldown);
    PM4_IB_CHECK(!cache.BeginArmAndSnapshot(kPhysAddr, 2u, snap));
    cache.AdvanceFrame();
  }

  // Frame +64: cooldown expired! Must again pass 2-hit hysteresis before arming.
  PM4_IB_CHECK(!cache.IsRangeInCooldown(kPhysAddr, 2u));
  PM4_IB_CHECK(cache.ObserveMiss(kPhysAddr, pkt.data(), 2u) ==
               ib::CandidateDecision::kFirstSeen);
  PM4_IB_CHECK(cache.ObserveMiss(kPhysAddr, pkt.data(), 2u) ==
               ib::CandidateDecision::kReadyToCompile);
  PM4_IB_CHECK(cache.BeginArmAndSnapshot(kPhysAddr, 2u, snap));

  // TOCTOU race: CPU writes BETWEEN BeginArmAndSnapshot and CommitIfUnchanged!
  cache.OnMemoryInvalidated(kPhysAddr, 4096u);
  PM4_IB_CHECK(!cache.CommitIfUnchanged(kPhysAddr, 2u, snap, compiled));
  PM4_IB_CHECK(cache.Lookup(kPhysAddr, 2u) == nullptr);
  PM4_IB_CHECK(cache.IsRangeInCooldown(kPhysAddr, 2u));

  // Multi-page IB spanning page boundary (0x00200F00 .. 0x00201700, 2 pages).
  constexpr uint32_t kSpanAddr = 0x00200F00u;
  constexpr uint32_t kSpanDwords = 512u;
  PM4_IB_CHECK(cache.BeginArmAndSnapshot(kSpanAddr, kSpanDwords, snap));
  PM4_IB_CHECK(snap.page_count == 2u);
  PM4_IB_CHECK(cache.CommitIfUnchanged(kSpanAddr, kSpanDwords, snap, compiled));
  PM4_IB_CHECK(cache.Lookup(kSpanAddr, kSpanDwords) != nullptr);
  // Invalidate ONLY the second page -> MUST invalidate spanning IB!
  cache.OnMemoryInvalidated(0x00201000u, 4096u);
  PM4_IB_CHECK(cache.Lookup(kSpanAddr, kSpanDwords) == nullptr);

  // Global invalidation (simulates live toggle 1 -> 0 or ClearCaches):
  // increments global_epoch_, clears entries_ and armed_pages_.
  const uint64_t epoch_before = cache.global_epoch();
  constexpr uint32_t kToggleAddr = 0x00600000u;
  PM4_IB_CHECK(cache.BeginArmAndSnapshot(kToggleAddr, 2u, snap));
  PM4_IB_CHECK(cache.CommitIfUnchanged(kToggleAddr, 2u, snap, compiled));
  PM4_IB_CHECK(cache.Lookup(kToggleAddr, 2u) != nullptr);
  cache.InvalidateAll();
  PM4_IB_CHECK(cache.global_epoch() == epoch_before + 1u);
  PM4_IB_CHECK(cache.entry_count() == 0u);
  PM4_IB_CHECK(cache.Lookup(kToggleAddr, 2u) == nullptr);
  // After InvalidateAll(), armed_pages_ is cleared so adjacent unwatch is NOT clamped.
  auto after_clear_unwatch =
      cache.OnMemoryInvalidatedForUnwatch(kToggleAddr + 4096u, 4096u, false);
  PM4_IB_CHECK(after_clear_unwatch.first == 0u &&
               after_clear_unwatch.second == UINT32_MAX);

  // Concurrent multi-threaded epoch stress test (writer thread vs GPU thread).
  std::atomic<bool> stop_writer{false};
  std::thread writer([&]() {
    uint32_t iter = 0;
    while (!stop_writer.load(std::memory_order_relaxed)) {
      cache.OnMemoryInvalidatedForUnwatch(
          0x00400000u + ((iter & 7u) * 4096u), 4096u, false);
      ++iter;
    }
  });
  for (int i = 0; i < 6000; ++i) {
    const uint32_t addr = 0x00400000u + (uint32_t(i & 7) * 4096u);
    if ((i & 63) == 0) {
      cache.AdvanceFrame();
    }
    if (cache.Lookup(addr, 2u) == nullptr) {
      ib::IbWatchSnapshot s{};
      if (cache.BeginArmAndSnapshot(addr, 2u, s)) {
        cache.CommitIfUnchanged(addr, 2u, s, compiled);
      }
    }
  }
  stop_writer.store(true, std::memory_order_relaxed);
  writer.join();
  // Final invalidation after writer stops MUST leave every entry invalid.
  cache.OnMemoryInvalidated(0x00400000u, 8u * 4096u);
  for (uint32_t p = 0; p < 8u; ++p) {
    PM4_IB_CHECK(cache.Lookup(0x00400000u + p * 4096u, 2u) == nullptr);
  }
}

// Mini CommandProcessor harness that runs compiled IB replay end-to-end to
// verify nested IBs, ib_call_depth_/ib_call_sites_, dynamic predication,
// re-entrant cache eviction, and bit-exact register/draw/shader execution.
struct FakeReplayCp {
  static constexpr uint32_t kMaxCallDepth = 16u;
  std::unordered_map<uint32_t, std::vector<uint32_t>> guest_mem;
  std::vector<uint32_t> regs = std::vector<uint32_t>(0x5003u, 0u);
  ib::Pm4IndirectBufferCache cache;
  uint64_t bin_select = 0xFFFFFFFFull;
  uint64_t bin_mask = 0xFFFFFFFFull;
  uint32_t ib_call_depth = 0;
  std::array<uint32_t, kMaxCallDepth> ib_call_sites{};
  uint32_t cur_ib_base = 0;
  uint32_t cur_ib_byte_offset = 0;

  struct CallSiteRecord {
    uint32_t depth;
    uint32_t call_site_addr;
    uint32_t child_ptr;
  };
  struct DrawRecord {
    bool is_draw_indx;
    uint32_t viz_cond;
    uint32_t initiator;
    uint32_t dma_base;
    uint32_t dma_size;
    uint32_t post_draw_guest_addr;
  };
  struct ImLoadRecord {
    uint32_t shader_type;
    uint32_t addr;
    uint32_t dwords;
  };
  std::vector<CallSiteRecord> recorded_calls;
  std::vector<DrawRecord> recorded_draws;
  std::vector<ImLoadRecord> recorded_im_loads;
  uint32_t bin_select_lo_callbacks = 0;
  uint32_t special_writes = 0;
  bool force_evict_on_child = false;

  uint32_t GuestReadPtrOffset() const {
    return cur_ib_base + cur_ib_byte_offset;
  }

  void WriteReg(uint32_t idx, uint32_t val) {
    regs[idx] = val;
    if (nf::HasMandatoryWriteSideEffect(idx)) {
      ++special_writes;
    }
  }

  void WriteRegsHostEndian(uint32_t base, const uint32_t* vals, uint32_t n) {
    nf::DispatchChangedRegisters4WideHostEndian(
        regs.data(), base, vals, n,
        [this](uint32_t r, uint32_t v) { WriteReg(r, v); });
  }

  void Replay(const ib::CompiledIndirectBuffer& compiled) {
    const uint32_t* swapped = compiled.swapped_dwords.data();
    for (const ib::DecodedIbOp& op : compiled.ops) {
      if (op.predicated && !(bin_select & bin_mask)) {
        continue;
      }
      switch (op.kind) {
        case ib::IbOpKind::kType0WriteRange: {
          const uint32_t* vals = swapped + op.data_offset;
          if (!op.one_reg_wr) {
            WriteRegsHostEndian(op.base_reg, vals, op.count);
          } else {
            for (uint32_t m = 0; m < op.count; ++m) {
              WriteReg(op.base_reg, vals[m]);
            }
          }
          break;
        }
        case ib::IbOpKind::kType1WritePair: {
          const uint32_t* vals = swapped + op.data_offset;
          WriteReg(op.base_reg, vals[0]);
          WriteReg(op.aux_reg, vals[1]);
          break;
        }
        case ib::IbOpKind::kSetConstant: {
          const uint32_t idx = op.arg0 & 0x7FFu;
          const uint32_t type = (op.arg0 >> 16) & 0xFFu;
          static constexpr uint32_t kBases[5] = {0x4000u, 0x4800u, 0x4900u,
                                                 0x4908u, 0x2000u};
          PM4_IB_CHECK(type < 5u);
          WriteRegsHostEndian(kBases[type] + idx, swapped + op.data_offset,
                              op.count);
          break;
        }
        case ib::IbOpKind::kSetConstant2:
          WriteRegsHostEndian(op.base_reg, swapped + op.data_offset, op.count);
          break;
        case ib::IbOpKind::kLoadAluConstant: {
          const uint32_t idx = op.arg1 & 0x7FFu;
          const uint32_t type = (op.arg1 >> 16) & 0xFFu;
          static constexpr uint32_t kBases[5] = {0x4000u, 0x4800u, 0x4900u,
                                                 0x4908u, 0x2000u};
          PM4_IB_CHECK(type < 5u);
          const auto& mem = guest_mem.at(op.arg0);
          for (uint32_t i = 0; i < op.arg2; ++i) {
            WriteReg(kBases[type] + idx + i, Swap32(mem[i]));
          }
          break;
        }
        case ib::IbOpKind::kImLoad:
          recorded_im_loads.push_back({op.arg0, op.arg1, op.arg2});
          break;
        case ib::IbOpKind::kDrawIndx:
        case ib::IbOpKind::kDrawIndx2:
          cur_ib_byte_offset = op.post_packet_byte_offset;
          recorded_draws.push_back({op.kind == ib::IbOpKind::kDrawIndx, op.arg0,
                                    op.arg1, op.arg2, op.arg3,
                                    GuestReadPtrOffset()});
          break;
        case ib::IbOpKind::kBinMaskLo:
          bin_mask = (bin_mask & 0xFFFFFFFF00000000ull) | op.arg0;
          break;
        case ib::IbOpKind::kBinMaskHi:
          bin_mask =
              (bin_mask & 0xFFFFFFFFull) | (static_cast<uint64_t>(op.arg0) << 32);
          break;
        case ib::IbOpKind::kBinSelectLo:
          bin_select = (bin_select & 0xFFFFFFFF00000000ull) | op.arg0;
          ++bin_select_lo_callbacks;
          break;
        case ib::IbOpKind::kBinSelectHi:
          bin_select = (bin_select & 0xFFFFFFFFull) |
                       (static_cast<uint64_t>(op.arg0) << 32);
          break;
        case ib::IbOpKind::kBinMask:
          bin_mask = (static_cast<uint64_t>(op.arg0) << 32) | op.arg1;
          break;
        case ib::IbOpKind::kBinSelect:
          bin_select = (static_cast<uint64_t>(op.arg0) << 32) | op.arg1;
          break;
        case ib::IbOpKind::kNestedIb:
          cur_ib_byte_offset = op.post_packet_byte_offset;
          ExecuteIb(op.arg0, op.arg1);
          break;
      }
    }
  }

  void ExecuteIb(uint32_t ptr, uint32_t count) {
    const uint32_t depth = ib_call_depth;
    if (depth < kMaxCallDepth) {
      ib_call_sites[depth] = GuestReadPtrOffset();
    }
    recorded_calls.push_back({depth, GuestReadPtrOffset(), ptr});
    ib_call_depth = depth + 1u;

    const uint32_t old_base = cur_ib_base;
    const uint32_t old_off = cur_ib_byte_offset;
    cur_ib_base = ptr;
    cur_ib_byte_offset = 0;

    if (force_evict_on_child && depth > 0) {
      // Simulate re-entrant cache eviction / clear during nested IB!
      cache.InvalidateAll();
    }

    const auto& buf = guest_mem.at(ptr);
    PM4_IB_CHECK(buf.size() >= count);

    if (auto hit = cache.Lookup(ptr, count)) {
      Replay(*hit);
    } else {
      auto dec = cache.ObserveMiss(ptr, buf.data(), count);
      if (dec == ib::CandidateDecision::kReadyToCompile) {
        PM4_IB_CHECK(ib::PrecheckCacheableIndirectBuffer(buf.data(), count) ==
                     ib::DecodeStatus::kOk);
        ib::IbWatchSnapshot snap{};
        PM4_IB_CHECK(cache.BeginArmAndSnapshot(ptr, count, snap));
        auto built = std::make_shared<ib::CompiledIndirectBuffer>();
        PM4_IB_CHECK(ib::DecodeIndirectBuffer(snap.phys_address, buf.data(),
                                              count, *built) ==
                     ib::DecodeStatus::kOk);
        PM4_IB_CHECK(cache.CommitIfUnchanged(ptr, count, snap, built));
        Replay(*built);
      } else {
        // Fallback: decode transient without caching (matches raw path).
        ib::CompiledIndirectBuffer temp;
        PM4_IB_CHECK(ib::DecodeIndirectBuffer(ptr, buf.data(), count, temp) ==
                     ib::DecodeStatus::kOk);
        Replay(temp);
      }
    }

    cur_ib_base = old_base;
    cur_ib_byte_offset = old_off;
    ib_call_depth = depth;
  }
};

void TestIbCacheNestedAndPredication() {
  FakeReplayCp cp;
  constexpr uint32_t kParentAddr = 0x00300000u;
  constexpr uint32_t kChildAddr = 0x00310000u;
  constexpr uint32_t kConstMemAddr = 0x00320000u;

  cp.guest_mem[kConstMemAddr] = {Swap32(0x3F800000u), Swap32(0x40000000u)};

  // Child IB: Type-0 write to 0x2050 + predicated DRAW_INDX_2 (auto-index).
  cp.guest_mem[kChildAddr] = {
      Swap32(MakeType0(0x2050u, 1u)),
      Swap32(0xAAAA0001u),
      Swap32(MakeType3(ib::opcode::kDrawIndx2, 1u, /*predicated=*/true)),
      Swap32(MakeDrawInitiator(4u, 2u, 0u, 36u)),
  };

  // Parent IB:
  //  - Type-0 reg write 0x2040 = 0x11111111
  //  - Type-1 pair write (0x100 = 0x2222, 0x101 = 0x3333)
  //  - Type-2 NOP + 0x00000000 + 0x0BADF00D + PM4_NOP
  //  - Predicated SET_CONSTANT (ALU c0 = 0x12345678)
  //  - Predicated IM_LOAD (VS @ 0x00400000, 64 dwords)
  //  - Predicated LOAD_ALU_CONSTANT (ALU c2..c3 from kConstMemAddr)
  //  - Non-predicated nested PM4_INDIRECT_BUFFER -> kChildAddr (4 dwords)
  //  - Predicated DRAW_INDX (DMA indexed draw)
  //  - Type-0 one_reg_wr=true writing 0x0578 (SCRATCH_REG0) twice!
  //  - Type-0 reg write 0x2041 = 0x99999999 (after nested IB!)
  cp.guest_mem[kParentAddr] = {
      Swap32(MakeType0(0x2040u, 1u)),
      Swap32(0x11111111u),
      Swap32(MakeType1(0x100u, 0x101u)),
      Swap32(0x2222u),
      Swap32(0x3333u),
      Swap32(MakeType2()),
      Swap32(0x00000000u),
      Swap32(0x0BADF00Du),
      Swap32(MakeType3(ib::opcode::kNop, 1u)),
      Swap32(0x0u),
      Swap32(MakeType3(ib::opcode::kSetConstant, 2u, /*predicated=*/true)),
      Swap32((0u << 16) | 0u),  // ALU[0]
      Swap32(0x12345678u),
      Swap32(MakeType3(ib::opcode::kImLoad, 2u, /*predicated=*/true)),
      Swap32(0x00400000u | 0u),  // Vertex shader
      Swap32(64u),
      Swap32(MakeType3(ib::opcode::kLoadAluConstant, 3u, /*predicated=*/true)),
      Swap32(kConstMemAddr),
      Swap32((0u << 16) | 2u),  // ALU[2]
      Swap32(2u),
      Swap32(MakeType3(ib::opcode::kIndirectBuffer, 2u, /*predicated=*/false)),
      Swap32(kChildAddr),
      Swap32(4u),
      Swap32(MakeType3(ib::opcode::kDrawIndx, 4u, /*predicated=*/true)),
      Swap32(0x00000105u),                       // viz_cond
      Swap32(MakeDrawInitiator(4u, 0u, 0u, 96u)),  // DMA 16-bit
      Swap32(0x00500000u),                       // dma_base
      Swap32(96u),                               // dma_size
      Swap32(MakeType0(nf::kRegScratchReg0, 2u, /*one_reg=*/true)),
      Swap32(0x77770001u),
      Swap32(0x77770001u),
      Swap32(MakeType0(0x2041u, 1u)),
      Swap32(0x99999999u),
  };
  const uint32_t parent_dwords = uint32_t(cp.guest_mem[kParentAddr].size());
  // Exact byte offset right past the nested INDIRECT_BUFFER packet (dword index 23).
  const uint32_t expected_child_call_site = kParentAddr + 23u * 4u;
  // Exact byte offset right past the parent DRAW_INDX packet (dword index 28).
  const uint32_t expected_parent_draw_addr = kParentAddr + 28u * 4u;
  // Exact byte offset right past the child DRAW_INDX_2 packet (dword index 4).
  const uint32_t expected_child_draw_addr = kChildAddr + 4u * 4u;

  // Run twice to pass 2-hit hysteresis and compile BOTH parent and child!
  cp.cur_ib_base = 0x00010000u;
  cp.cur_ib_byte_offset = 0x40u;
  cp.ExecuteIb(kParentAddr, parent_dwords);
  cp.ExecuteIb(kParentAddr, parent_dwords);

  // Verify parent was compiled with kNestedIb (NEVER inlined!).
  auto parent_entry = cp.cache.Lookup(kParentAddr, parent_dwords);
  auto child_entry = cp.cache.Lookup(kChildAddr, 4u);
  PM4_IB_CHECK(parent_entry != nullptr);
  PM4_IB_CHECK(child_entry != nullptr);
  PM4_IB_CHECK(parent_entry->has_nested_ib);
  PM4_IB_CHECK(!child_entry->has_nested_ib);
  uint32_t nested_op_count = 0;
  for (const auto& op : parent_entry->ops) {
    if (op.kind == ib::IbOpKind::kNestedIb) {
      ++nested_op_count;
      PM4_IB_CHECK(op.arg0 == kChildAddr && op.arg1 == 4u);
      PM4_IB_CHECK(op.post_packet_byte_offset == 23u * 4u);
    }
  }
  PM4_IB_CHECK(nested_op_count == 1u);

  // Run 3 (cache HIT on both parent & child) with predication PASSING.
  cp.recorded_calls.clear();
  cp.recorded_draws.clear();
  cp.recorded_im_loads.clear();
  cp.special_writes = 0;
  cp.bin_select = 0x00000002ull;
  cp.bin_mask = 0x00000002ull;
  cp.ExecuteIb(kParentAddr, parent_dwords);

  PM4_IB_CHECK(cp.ib_call_depth == 0u);
  PM4_IB_CHECK(cp.recorded_calls.size() == 2u);
  PM4_IB_CHECK(cp.recorded_calls[0].depth == 0u &&
               cp.recorded_calls[0].call_site_addr == 0x00010040u);
  PM4_IB_CHECK(cp.recorded_calls[1].depth == 1u &&
               cp.recorded_calls[1].call_site_addr == expected_child_call_site &&
               cp.recorded_calls[1].child_ptr == kChildAddr);
  PM4_IB_CHECK(cp.recorded_draws.size() == 2u);
  PM4_IB_CHECK(!cp.recorded_draws[0].is_draw_indx &&
               cp.recorded_draws[0].post_draw_guest_addr ==
                   expected_child_draw_addr);
  PM4_IB_CHECK(cp.recorded_draws[1].is_draw_indx &&
               cp.recorded_draws[1].viz_cond == 0x00000105u &&
               cp.recorded_draws[1].dma_base == 0x00500000u &&
               cp.recorded_draws[1].dma_size == 96u &&
               cp.recorded_draws[1].post_draw_guest_addr ==
                   expected_parent_draw_addr);
  PM4_IB_CHECK(cp.recorded_im_loads.size() == 1u &&
               cp.recorded_im_loads[0].addr == 0x00400000u &&
               cp.recorded_im_loads[0].dwords == 64u);
  PM4_IB_CHECK(cp.regs[0x2040u] == 0x11111111u);
  PM4_IB_CHECK(cp.regs[0x2041u] == 0x99999999u);
  PM4_IB_CHECK(cp.regs[0x2050u] == 0xAAAA0001u);
  PM4_IB_CHECK(cp.regs[0x100u] == 0x2222u && cp.regs[0x101u] == 0x3333u);
  PM4_IB_CHECK(cp.regs[0x4000u] == 0x12345678u);
  PM4_IB_CHECK(cp.regs[0x4002u] == 0x3F800000u &&
               cp.regs[0x4003u] == 0x40000000u);
  // one_reg_wr=true wrote SCRATCH_REG0 twice even with identical value!
  PM4_IB_CHECK(cp.special_writes == 2u);

  // Run 4 (SAME cached IBs!) with predication FAILING ((bin_select & bin_mask) == 0):
  // Predicated SET_CONSTANT, IM_LOAD, LOAD_ALU_CONSTANT, DRAW_INDX, DRAW_INDX_2
  // MUST ALL BE SKIPPED, while non-predicated ops still run!
  cp.recorded_draws.clear();
  cp.recorded_im_loads.clear();
  cp.regs[0x4000u] = 0u;
  cp.regs[0x4002u] = 0u;
  cp.bin_select = 0x00000001ull;
  cp.bin_mask = 0x00000002ull;
  cp.ExecuteIb(kParentAddr, parent_dwords);
  PM4_IB_CHECK(cp.recorded_draws.empty());
  PM4_IB_CHECK(cp.recorded_im_loads.empty());
  PM4_IB_CHECK(cp.regs[0x4000u] == 0u);
  PM4_IB_CHECK(cp.regs[0x4002u] == 0u);
  PM4_IB_CHECK(cp.regs[0x2050u] == 0xAAAA0001u);  // non-predicated child ran!

  // Invalidate ONLY child IB page -> mutate child value -> re-run parent!
  // Parent MUST stay a cache HIT, while child executes the NEW mutated value!
  const uint64_t hits_before = cp.cache.stats().hits;
  cp.guest_mem[kChildAddr][1] = Swap32(0xBBBB9999u);
  cp.cache.OnMemoryInvalidated(kChildAddr, 4096u);
  cp.bin_select = 0xFFFFFFFFull;
  cp.bin_mask = 0xFFFFFFFFull;
  cp.ExecuteIb(kParentAddr, parent_dwords);
  PM4_IB_CHECK(cp.cache.stats().hits == hits_before + 1u);  // parent HIT!
  PM4_IB_CHECK(cp.regs[0x2050u] == 0xBBBB9999u);            // child updated!

  // Re-entrant cache invalidation/eviction inside nested child IB:
  // Parent's shared_ptr MUST keep compiled parent alive through post-child ops!
  cp.regs[0x2041u] = 0u;
  cp.force_evict_on_child = true;
  cp.ExecuteIb(kParentAddr, parent_dwords);
  PM4_IB_CHECK(cp.regs[0x2041u] == 0x99999999u);
  cp.force_evict_on_child = false;

  // Mid-IB bin_mask / bin_select dynamic flip inside a single cached IB.
  constexpr uint32_t kBinFlipAddr = 0x00330000u;
  cp.guest_mem[kBinFlipAddr] = {
      Swap32(MakeType3(ib::opcode::kSetBinSelectLo, 1u)),
      Swap32(0x00000001u),
      Swap32(MakeType3(ib::opcode::kSetBinMaskLo, 1u)),
      Swap32(0x00000002u),
      Swap32(MakeType3(ib::opcode::kDrawIndx2, 1u, /*predicated=*/true)),
      Swap32(MakeDrawInitiator(4u, 2u, 0u, 3u)),  // skipped!
      Swap32(MakeType3(ib::opcode::kSetBinMaskLo, 1u)),
      Swap32(0x00000001u),
      Swap32(MakeType3(ib::opcode::kDrawIndx2, 1u, /*predicated=*/true)),
      Swap32(MakeDrawInitiator(4u, 2u, 0u, 6u)),  // executed!
  };
  const uint32_t flip_dwords = uint32_t(cp.guest_mem[kBinFlipAddr].size());
  cp.ExecuteIb(kBinFlipAddr, flip_dwords);
  cp.ExecuteIb(kBinFlipAddr, flip_dwords);
  cp.recorded_draws.clear();
  cp.bin_select_lo_callbacks = 0;
  cp.ExecuteIb(kBinFlipAddr, flip_dwords);
  PM4_IB_CHECK(cp.bin_select_lo_callbacks == 1u);
  PM4_IB_CHECK(cp.recorded_draws.size() == 1u);
  PM4_IB_CHECK(((cp.recorded_draws[0].initiator >> 16) & 0xFFFFu) == 6u);
}

void TestIbCacheMandatoryBailoutOpcodes() {
  const std::vector<uint32_t> bailout_ops = {
      ib::opcode::kWaitRegMem,      ib::opcode::kCondWrite,
      ib::opcode::kRegRmw,          ib::opcode::kRegToMem,
      ib::opcode::kMemWrite,        ib::opcode::kInterrupt,
      ib::opcode::kXeSwap,          ib::opcode::kMeInit,
      ib::opcode::kEventWrite,      ib::opcode::kEventWriteShd,
      ib::opcode::kEventWriteCfl,   ib::opcode::kEventWriteExt,
      ib::opcode::kEventWriteZpd,   ib::opcode::kVizQuery,
      ib::opcode::kImLoadImmediate, ib::opcode::kWaitForIdle,
      ib::opcode::kIndirectBufferPfd,
      0x7Fu,  // unknown opcode
  };

  for (uint32_t op : bailout_ops) {
    for (bool pred : {false, true}) {
      PM4_IB_CHECK(op == 0x7Fu || ib::IsMandatoryBailoutType3Opcode(op));
      std::vector<uint32_t> pkt = {
          Swap32(MakeType0(0x2010u, 1u)),
          Swap32(0x12345678u),
          Swap32(MakeType3(op, 1u, pred)),
          Swap32(0u),
      };
      PM4_IB_CHECK(ib::PrecheckCacheableIndirectBuffer(
                       pkt.data(), uint32_t(pkt.size())) ==
                   ib::DecodeStatus::kBailout);
      ib::CompiledIndirectBuffer out;
      PM4_IB_CHECK(ib::DecodeIndirectBuffer(0x00500000u, pkt.data(),
                                            uint32_t(pkt.size()), out) ==
                   ib::DecodeStatus::kBailout);
    }
  }

  // DRAW_INDX with SourceSelect::kImmediate (1) -> bailout.
  std::vector<uint32_t> imm_draw = {
      Swap32(MakeType3(ib::opcode::kDrawIndx2, 2u)),
      Swap32(MakeDrawInitiator(4u, 1u, 0u, 3u)),
      Swap32(0u),
  };
  PM4_IB_CHECK(ib::PrecheckCacheableIndirectBuffer(
                   imm_draw.data(), uint32_t(imm_draw.size())) ==
               ib::DecodeStatus::kBailout);

  // Uninitialized memory sentinel 0xCDCDCDCD -> malformed.
  std::vector<uint32_t> uninit = {Swap32(0xCDCDCDCDu)};
  PM4_IB_CHECK(ib::PrecheckCacheableIndirectBuffer(uninit.data(), 1u) ==
               ib::DecodeStatus::kMalformed);

  // Truncated Type-0 / Type-1 / Type-3 -> malformed.
  std::vector<uint32_t> trunc_t0 = {Swap32(MakeType0(0x2010u, 4u)), Swap32(1u)};
  PM4_IB_CHECK(ib::PrecheckCacheableIndirectBuffer(trunc_t0.data(), 2u) ==
               ib::DecodeStatus::kMalformed);
  std::vector<uint32_t> trunc_t1 = {Swap32(MakeType1(0x10u, 0x11u)), Swap32(1u)};
  PM4_IB_CHECK(ib::PrecheckCacheableIndirectBuffer(trunc_t1.data(), 2u) ==
               ib::DecodeStatus::kMalformed);
  std::vector<uint32_t> trunc_t3 = {
      Swap32(MakeType3(ib::opcode::kSetConstant, 5u)), Swap32(0u)};
  PM4_IB_CHECK(ib::PrecheckCacheableIndirectBuffer(trunc_t3.data(), 2u) ==
               ib::DecodeStatus::kMalformed);

  // Negative candidate caching: marking bailout IB non-cacheable prevents
  // future arming attempts while content hash matches.
  ib::Pm4IndirectBufferCache cache;
  std::vector<uint32_t> wait_pkt = {
      Swap32(MakeType3(ib::opcode::kWaitRegMem, 5u)),
      Swap32(0u), Swap32(0u), Swap32(0u), Swap32(0u), Swap32(0u)};
  PM4_IB_CHECK(cache.ObserveMiss(0x00700000u, wait_pkt.data(), 6u) ==
               ib::CandidateDecision::kFirstSeen);
  PM4_IB_CHECK(cache.ObserveMiss(0x00700000u, wait_pkt.data(), 6u) ==
               ib::CandidateDecision::kReadyToCompile);
  cache.MarkCandidateNonCacheable(0x00700000u, 6u);
  PM4_IB_CHECK(cache.ObserveMiss(0x00700000u, wait_pkt.data(), 6u) ==
               ib::CandidateDecision::kNonCacheable);
}

void TestSourceInvariants(const fs::path& cpp_root) {
  const fs::path xenia_src = cpp_root / "xenia" / "src" / "xenia";
  const fs::path gpu_dir = xenia_src / "gpu";
  const fs::path ui_vk_dir = xenia_src / "ui" / "vulkan";

  const std::string noop_h = ReadSourceText(gpu_dir / "pm4_noop_filter.h");
  const std::string ib_h = ReadSourceText(gpu_dir / "pm4_ib_cache.h");
  const std::string cp_h = ReadSourceText(gpu_dir / "command_processor.h");
  const std::string cp_cc = ReadSourceText(gpu_dir / "command_processor.cc");
  const std::string pm4_decl =
      ReadSourceText(gpu_dir / "pm4_command_processor_declare.h");
  const std::string pm4_impl =
      ReadSourceText(gpu_dir / "pm4_command_processor_implement.h");
  const std::string zerofg_inc =
      ReadSourceText(ui_vk_dir / "zerofg_device_handoff.inc");

  // 1) AArch64 NEON 4-wide intrinsics present in pm4_noop_filter.h.
  PM4_IB_CHECK(noop_h.find("vld1q_u32") != std::string::npos);
  PM4_IB_CHECK(noop_h.find("vrev32q_u8") != std::string::npos);
  PM4_IB_CHECK(noop_h.find("vceqq_u32") != std::string::npos);
  PM4_IB_CHECK(noop_h.find("vminvq_u32") != std::string::npos);
  PM4_IB_CHECK(noop_h.find("vst1q_u32") != std::string::npos);

  // 2) Both new cvars default to false and MUST NOT use UPDATE_from_*.
  PM4_IB_CHECK(cp_cc.find("vulkan_pm4_noop_reg_filter_4wide, false") !=
               std::string::npos);
  PM4_IB_CHECK(cp_cc.find("pm4_indirect_buffer_cache, false") !=
               std::string::npos);
  PM4_IB_CHECK(cp_cc.find("UPDATE_from_bool(vulkan_pm4_noop_reg_filter_4wide") ==
               std::string::npos);
  PM4_IB_CHECK(cp_cc.find("UPDATE_from_bool(pm4_indirect_buffer_cache") ==
               std::string::npos);

  // 3) Android live debug properties wired in command_processor.cc.
  PM4_IB_CHECK(cp_cc.find("debug.xendroid.pm4_noop_reg_filter") !=
               std::string::npos);
  PM4_IB_CHECK(cp_cc.find("debug.xendroid.pm4_ib_cache") != std::string::npos);

  // 4) Write-watch registration, unregistration, ClearCaches, InvalidateGpuMemory,
  //    Restore, and arming BEFORE decode.
  PM4_IB_CHECK(cp_h.find("WriteRegistersHostEndian") != std::string::npos);
  PM4_IB_CHECK(cp_cc.find("RegisterPhysicalMemoryInvalidationCallback") !=
               std::string::npos);
  PM4_IB_CHECK(cp_cc.find("UnregisterPhysicalMemoryInvalidationCallback") !=
               std::string::npos);
  PM4_IB_CHECK(cp_cc.find("DispatchChangedRegisters4WideHostEndian") !=
               std::string::npos);
  PM4_IB_CHECK(pm4_decl.find("TryExecuteCachedIndirectBuffer") !=
               std::string::npos);
  PM4_IB_CHECK(pm4_impl.find("!trace_writer_.is_open()") != std::string::npos);
  PM4_IB_CHECK(pm4_impl.find("!bin_trace_.frames_left") != std::string::npos);
  PM4_IB_CHECK(pm4_impl.find("!cvars::disassemble_pm4") != std::string::npos);
  PM4_IB_CHECK(pm4_impl.find("InvalidatePm4IbCache()") != std::string::npos);
  PM4_IB_CHECK(pm4_impl.find("OnPm4IbCacheEndFrame()") != std::string::npos);

  const size_t arm_pos =
      pm4_impl.find("memory_->EnablePhysicalMemoryAccessCallbacks(");
  const size_t decode_pos = pm4_impl.find("DecodeIndirectBuffer(");
  const size_t commit_pos = pm4_impl.find("pm4_ib_cache_.CommitIfUnchanged(");
  PM4_IB_CHECK(arm_pos != std::string::npos &&
               decode_pos != std::string::npos &&
               commit_pos != std::string::npos);
  PM4_IB_CHECK(arm_pos < decode_pos && decode_pos < commit_pos);

  // 5) IB cache 4 MiB unwatch clamp present.
  PM4_IB_CHECK(ib_h.find("kIbCacheMaxUnwatchExcess") != std::string::npos);
  PM4_IB_CHECK(ib_h.find("OnMemoryInvalidatedForUnwatch") != std::string::npos);

  // 6) ZeroFG ReleaseLease hardening: HandoffExport runs OUTSIDE mutex_.
  const size_t rel_pos =
      zerofg_inc.find("bool ZeroFGDeviceHandoff::ReleaseLease(");
  const size_t acq_pos = zerofg_inc.find("bool ZeroFGDeviceHandoff::Acquire(");
  PM4_IB_CHECK(rel_pos != std::string::npos && acq_pos != std::string::npos &&
               rel_pos < acq_pos);
  const std::string rel_body = zerofg_inc.substr(rel_pos, acq_pos - rel_pos);
  const size_t first_lock = rel_body.find("std::lock_guard<std::mutex>");
  const size_t export_call = rel_body.find("HandoffExport(b_, b_release");
  const size_t second_lock =
      rel_body.find("std::lock_guard<std::mutex>", export_call);
  PM4_IB_CHECK(first_lock != std::string::npos &&
               export_call != std::string::npos &&
               second_lock != std::string::npos);
  PM4_IB_CHECK(first_lock < export_call && export_call < second_lock);

  // 7) VulkanCommandProcessor post-merge 3B hooks:
  //    - #include "xenia/gpu/pm4_noop_filter.h"
  //    - 4-wide no-op filter in WritePossiblySpecialRegistersFromMem and
  //      WriteRegistersFromMem (`regular`) gated by
  //      cvars::vulkan_pm4_noop_reg_filter_4wide
  //    - CommandProcessor::InvalidateGpuMemory() inside
  //      VulkanCommandProcessor::InvalidateGpuMemory()
  //    - OnPm4IbMemoryInvalidated(base_ptr, length) inside
  //      VulkanCommandProcessor::TracePlaybackWroteMemory()
  const std::string vk_cp_cc =
      ReadSourceText(gpu_dir / "vulkan" / "vulkan_command_processor.cc");
  PM4_IB_CHECK(vk_cp_cc.find("#include \"xenia/gpu/pm4_noop_filter.h\"") !=
               std::string::npos);

  const size_t vk_inv_pos =
      vk_cp_cc.find("void VulkanCommandProcessor::InvalidateGpuMemory()");
  const size_t vk_trace_pos =
      vk_cp_cc.find("void VulkanCommandProcessor::TracePlaybackWroteMemory(");
  const size_t vk_init_sh_pos =
      vk_cp_cc.find("void VulkanCommandProcessor::InitializeShaderStorage(");
  PM4_IB_CHECK(vk_inv_pos != std::string::npos &&
               vk_trace_pos != std::string::npos &&
               vk_init_sh_pos != std::string::npos &&
               vk_inv_pos < vk_trace_pos && vk_trace_pos < vk_init_sh_pos);
  const std::string vk_inv_body =
      vk_cp_cc.substr(vk_inv_pos, vk_trace_pos - vk_inv_pos);
  PM4_IB_CHECK(vk_inv_body.find("CommandProcessor::InvalidateGpuMemory();") !=
               std::string::npos);
  const std::string vk_trace_body =
      vk_cp_cc.substr(vk_trace_pos, vk_init_sh_pos - vk_trace_pos);
  PM4_IB_CHECK(
      vk_trace_body.find("OnPm4IbMemoryInvalidated(base_ptr, length);") !=
      std::string::npos);

  const size_t vk_spec_pos = vk_cp_cc.find(
      "void VulkanCommandProcessor::WritePossiblySpecialRegistersFromMem(");
  const size_t vk_mem_pos =
      vk_cp_cc.find("void VulkanCommandProcessor::WriteRegistersFromMem(");
  const size_t vk_ring_pos = vk_cp_cc.find(
      "void VulkanCommandProcessor::WriteRegisterRangeFromRing(");
  PM4_IB_CHECK(vk_spec_pos != std::string::npos &&
               vk_mem_pos != std::string::npos &&
               vk_ring_pos != std::string::npos &&
               vk_spec_pos < vk_mem_pos && vk_mem_pos < vk_ring_pos);
  const std::string vk_spec_body =
      vk_cp_cc.substr(vk_spec_pos, vk_mem_pos - vk_spec_pos);
  PM4_IB_CHECK(vk_spec_body.find("cvars::vulkan_pm4_noop_reg_filter_4wide") !=
               std::string::npos);
  PM4_IB_CHECK(
      vk_spec_body.find("pm4_noop::FilterWritePossiblySpecialRegistersBigEndian(") !=
      std::string::npos);
  const std::string vk_mem_body =
      vk_cp_cc.substr(vk_mem_pos, vk_ring_pos - vk_mem_pos);
  PM4_IB_CHECK(vk_mem_body.find("cvars::vulkan_pm4_noop_reg_filter_4wide") !=
               std::string::npos);
  PM4_IB_CHECK(
      vk_mem_body.find("pm4_noop::FilterWriteRegisters4WideBigEndian(") !=
      std::string::npos);
}

}  // namespace

int main(int argc, char** argv) {
  TestNoopFilter4WideAllLengthsAndAlignments();
  TestNoopFilterMandatorySideEffectsAndSysConst();
  TestIbCacheHysteresisEvictionAndLimits();
  TestIbCacheEpochClampCooldown64AndConcurrency();
  TestIbCacheNestedAndPredication();
  TestIbCacheMandatoryBailoutOpcodes();
  if (argc >= 2) {
    TestSourceInvariants(fs::path(argv[1]));
  } else {
    std::fprintf(stderr, "Usage: gpu_pm4_ib_cache_test <cpp_root>\n");
    return 1;
  }
  std::printf("gpu_pm4_ib_cache_test: PASS\n");
  return 0;
}
