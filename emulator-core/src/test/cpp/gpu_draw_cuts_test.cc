#include <algorithm>
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
#include <thread>
#include <vector>

#define XENIA_BASE_CLOCK_PURE_LOGIC_ONLY
#include "xenia/base/clock.h"

#define XENIA_GPU_DRAW_CUTS_PURE_LOGIC_ONLY
#include "xenia/gpu/vulkan/vulkan_command_processor.h"

namespace {

#define DRAW_CUTS_CHECK(cond)                                               \
  do {                                                                      \
    if (!(cond)) {                                                          \
      std::fprintf(stderr, "GPU_DRAW_CUTS check failed (%s:%d): %s\n",      \
                   __FILE__, __LINE__, #cond);                              \
      std::abort();                                                         \
    }                                                                       \
  } while (false)

namespace fs = std::filesystem;
namespace dc = xe::gpu::draw_cuts;

uint32_t Swap32(uint32_t v) { return __builtin_bswap32(v); }

std::string NormalizeNewlines(std::string s) {
  s.erase(std::remove(s.begin(), s.end(), '\r'), s.end());
  return s;
}

std::string ReadSourceText(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  DRAW_CUTS_CHECK(in.good());
  std::ostringstream ss;
  ss << in.rdbuf();
  return NormalizeNewlines(ss.str());
}

void TestFloatConstantValueCompare() {
  constexpr uint32_t kFloatBase = 0x4000;
  std::vector<uint32_t> reg_file(512 * 4, 0u);
  uint64_t vertex_map[4] = {0, 0, 0, 0};
  uint64_t pixel_map[4] = {0, 0, 0, 0};
  // Vertex uses c0 and c3; pixel uses c256 (pixel index 0) and c260 (index 4).
  vertex_map[0] = (UINT64_C(1) << 0) | (UINT64_C(1) << 3);
  pixel_map[0] = (UINT64_C(1) << 0) | (UINT64_C(1) << 4);

  // Initial write of 4 vec4s (c0..c3) with all zeros: matches reg_file -> no dirty!
  std::vector<uint32_t> src(16, 0u);
  auto res = dc::CopySwapAndTrackFloatConstants(
      reg_file.data(), src.data(), kFloatBase, kFloatBase, 16, vertex_map,
      pixel_map, /*check_vertex=*/true, /*check_pixel=*/true);
  DRAW_CUTS_CHECK(!res.any_changed);
  DRAW_CUTS_CHECK(!res.vertex_dirty);
  DRAW_CUTS_CHECK(!res.pixel_dirty);

  // Change ONLY unused c1 (dwords 4..7): reg_file updated, vertex_dirty stays false!
  src[4] = Swap32(0x3F800000u);  // 1.0f
  src[7] = Swap32(0x40000000u);  // 2.0f
  res = dc::CopySwapAndTrackFloatConstants(
      reg_file.data(), src.data(), kFloatBase, kFloatBase, 16, vertex_map,
      pixel_map, true, true);
  DRAW_CUTS_CHECK(res.any_changed);
  DRAW_CUTS_CHECK(!res.vertex_dirty);
  DRAW_CUTS_CHECK(!res.pixel_dirty);
  DRAW_CUTS_CHECK(reg_file[4] == 0x3F800000u);
  DRAW_CUTS_CHECK(reg_file[7] == 0x40000000u);

  // Re-writing the exact same bytes again: zero change!
  res = dc::CopySwapAndTrackFloatConstants(
      reg_file.data(), src.data(), kFloatBase, kFloatBase, 16, vertex_map,
      pixel_map, true, true);
  DRAW_CUTS_CHECK(!res.any_changed);
  DRAW_CUTS_CHECK(!res.vertex_dirty);
  DRAW_CUTS_CHECK(!res.pixel_dirty);

  // Bitwise exactness: +0.0f (0x00000000) -> -0.0f (0x80000000) on used c0 MUST dirty!
  src[0] = Swap32(0x80000000u);
  res = dc::CopySwapAndTrackFloatConstants(
      reg_file.data(), src.data(), kFloatBase, kFloatBase, 16, vertex_map,
      pixel_map, true, true);
  DRAW_CUTS_CHECK(res.any_changed);
  DRAW_CUTS_CHECK(res.vertex_dirty);
  DRAW_CUTS_CHECK(!res.pixel_dirty);
  DRAW_CUTS_CHECK(reg_file[0] == 0x80000000u);

  // Cross-bank write spanning VS c255 (unused) and PS c256 (used, pixel index 0):
  // Only pixel_dirty should become true!
  std::vector<uint32_t> cross_src(8, 0u);
  cross_src[0] = Swap32(0x11223344u);  // c255.x (unused VS)
  cross_src[5] = Swap32(0x55667788u);  // c256.y (used PS)
  const uint32_t cross_start = kFloatBase + 255 * 4;
  res = dc::CopySwapAndTrackFloatConstants(
      reg_file.data() + 255 * 4, cross_src.data(), cross_start, kFloatBase, 8,
      vertex_map, pixel_map, true, true);
  DRAW_CUTS_CHECK(res.any_changed);
  DRAW_CUTS_CHECK(!res.vertex_dirty);
  DRAW_CUTS_CHECK(res.pixel_dirty);
  DRAW_CUTS_CHECK(reg_file[255 * 4 + 0] == 0x11223344u);
  DRAW_CUTS_CHECK(reg_file[256 * 4 + 1] == 0x55667788u);

  // Unaligned head + vec4 middle + unaligned tail:
  // start at c2.z (offset 10), write 7 dwords -> c2.zw (unused), c3.xyzw (used!), c4.x (unused).
  std::vector<uint32_t> unaligned_src(7, 0u);
  unaligned_src[0] = Swap32(0xAABBCCDDu);  // c2.z (unused)
  unaligned_src[6] = Swap32(0x01020304u);  // c4.x (unused)
  res = dc::CopySwapAndTrackFloatConstants(
      reg_file.data() + 10, unaligned_src.data(), kFloatBase + 10, kFloatBase,
      7, vertex_map, pixel_map, true, true);
  DRAW_CUTS_CHECK(res.any_changed);
  DRAW_CUTS_CHECK(!res.vertex_dirty);
  DRAW_CUTS_CHECK(reg_file[10] == 0xAABBCCDDu);
  DRAW_CUTS_CHECK(reg_file[16] == 0x01020304u);

  // Now modify c3.w (unaligned_src[5], inside used vec4 c3):
  unaligned_src[5] = Swap32(0xDEADBEEFu);
  res = dc::CopySwapAndTrackFloatConstants(
      reg_file.data() + 10, unaligned_src.data(), kFloatBase + 10, kFloatBase,
      7, vertex_map, pixel_map, true, true);
  DRAW_CUTS_CHECK(res.any_changed);
  DRAW_CUTS_CHECK(res.vertex_dirty);
  DRAW_CUTS_CHECK(reg_file[15] == 0xDEADBEEFu);

  // With check_vertex=false (!frame_open_): writes still happen, dirty flags stay false.
  unaligned_src[5] = Swap32(0xCAFEBABEu);
  res = dc::CopySwapAndTrackFloatConstants(
      reg_file.data() + 10, unaligned_src.data(), kFloatBase + 10, kFloatBase,
      7, vertex_map, pixel_map, false, false);
  DRAW_CUTS_CHECK(res.any_changed);
  DRAW_CUTS_CHECK(!res.vertex_dirty);
  DRAW_CUTS_CHECK(!res.pixel_dirty);
  DRAW_CUTS_CHECK(reg_file[15] == 0xCAFEBABEu);
}

void TestSystemConstantsDirtyMask() {
  using dc::kSysConstGroupAll;
  using dc::kSysConstGroupColorExpBias;
  using dc::kSysConstGroupFlags;
  using dc::kSysConstGroupFsi;
  using dc::kSysConstGroupIndex;
  using dc::kSysConstGroupPoint;
  using dc::kSysConstGroupTess;
  using dc::kSysConstGroupTextures;
  using dc::kSysConstGroupUcp;

  // Unrelated registers produce zero dirty bits.
  DRAW_CUTS_CHECK(dc::ClassifySysConstRegisterDirtyGroups(0x0578) == 0u);
  DRAW_CUTS_CHECK(dc::ClassifySysConstRegisterDirtyGroups(0x4000) == 0u);
  DRAW_CUTS_CHECK(dc::ClassifySysConstRegisterDirtyGroups(0x1925) == 0u);

  // Individual register classification checks.
  DRAW_CUTS_CHECK((dc::ClassifySysConstRegisterDirtyGroups(0x2388) &
                   kSysConstGroupUcp) != 0);
  DRAW_CUTS_CHECK((dc::ClassifySysConstRegisterDirtyGroups(0x2388) &
                   ~kSysConstGroupUcp) == 0);
  DRAW_CUTS_CHECK((dc::ClassifySysConstRegisterDirtyGroups(0x2204) &
                   kSysConstGroupUcp) != 0);
  DRAW_CUTS_CHECK((dc::ClassifySysConstRegisterDirtyGroups(0x2286) &
                   kSysConstGroupTess) != 0);
  DRAW_CUTS_CHECK((dc::ClassifySysConstRegisterDirtyGroups(0x2287) &
                   kSysConstGroupTess) != 0);
  DRAW_CUTS_CHECK((dc::ClassifySysConstRegisterDirtyGroups(0x2102) &
                   (kSysConstGroupIndex | kSysConstGroupTess)) ==
                  (kSysConstGroupIndex | kSysConstGroupTess));
  DRAW_CUTS_CHECK((dc::ClassifySysConstRegisterDirtyGroups(0x2280) &
                   kSysConstGroupPoint) != 0);
  DRAW_CUTS_CHECK((dc::ClassifySysConstRegisterDirtyGroups(0x2281) &
                   kSysConstGroupPoint) != 0);
  DRAW_CUTS_CHECK((dc::ClassifySysConstRegisterDirtyGroups(0x2001) &
                   (kSysConstGroupFlags | kSysConstGroupColorExpBias |
                    kSysConstGroupFsi)) ==
                  (kSysConstGroupFlags | kSysConstGroupColorExpBias |
                   kSysConstGroupFsi));
  DRAW_CUTS_CHECK((dc::ClassifySysConstRegisterDirtyGroups(0x2284) &
                   kSysConstGroupFlags) != 0);
  DRAW_CUTS_CHECK((dc::ClassifySysConstRegisterDirtyGroups(0x4800) &
                   kSysConstGroupTextures) != 0);

  // Range classification checks.
  DRAW_CUTS_CHECK(dc::ClassifySysConstRangeDirtyGroups(0x0000, 0x2000) == 0u);
  DRAW_CUTS_CHECK(dc::ClassifySysConstRangeDirtyGroups(0x2388, 24) ==
                  kSysConstGroupUcp);
  DRAW_CUTS_CHECK((dc::ClassifySysConstRangeDirtyGroups(0x2000, 6) &
                   (kSysConstGroupFlags | kSysConstGroupColorExpBias |
                    kSysConstGroupFsi)) ==
                  (kSysConstGroupFlags | kSysConstGroupColorExpBias |
                   kSysConstGroupFsi));
  DRAW_CUTS_CHECK((dc::ClassifySysConstRangeDirtyGroups(0x2100, 3) &
                   (kSysConstGroupTess | kSysConstGroupIndex)) ==
                  (kSysConstGroupTess | kSysConstGroupIndex));

  // Dirty tracker lifecycle.
  dc::SystemConstantsDirtyTracker tracker;
  DRAW_CUTS_CHECK(tracker.dirty_mask() == kSysConstGroupAll);
  tracker.ClearAll();
  DRAW_CUTS_CHECK(tracker.dirty_mask() == 0u);
  tracker.NoteRegisterWrite(0x2390);
  DRAW_CUTS_CHECK(tracker.dirty_mask() == kSysConstGroupUcp);
  tracker.NoteRegisterRangeWrite(0x2280, 2);
  DRAW_CUTS_CHECK(tracker.dirty_mask() ==
                  (kSysConstGroupUcp | kSysConstGroupPoint));
  tracker.InvalidateAll();
  DRAW_CUTS_CHECK(tracker.dirty_mask() == kSysConstGroupAll);
}

void TestShaderLoadWatchCacheEpoch() {
  dc::ShaderLoadWatchCache<int> cache;
  int shader_a = 101;
  int shader_b = 202;
  int shader_c = 303;

  // Miss on cold address (with high guest bits 0xA0000000 masked to physical).
  constexpr uint32_t kGuestAddr = 0xA0123000u;
  constexpr uint32_t kPhysAddr = 0x00123000u;
  constexpr uint32_t kDwords = 512;  // 2048 bytes (1 page)
  DRAW_CUTS_CHECK(cache.Lookup(0, kGuestAddr, kDwords) == nullptr);

  // Arm + commit sequence.
  dc::ShaderWatchSnapshot snap{};
  DRAW_CUTS_CHECK(cache.BeginArmAndSnapshot(kGuestAddr, kDwords, snap));
  DRAW_CUTS_CHECK(snap.phys_address == kPhysAddr);
  DRAW_CUTS_CHECK(snap.byte_length == kDwords * 4);
  DRAW_CUTS_CHECK(snap.page_count == 1);
  DRAW_CUTS_CHECK(cache.CommitIfUnchanged(0, kGuestAddr, kDwords, snap,
                                          &shader_a));

  // Hit on same (shader_type, addr, dwords); miss on different shader_type or dwords.
  DRAW_CUTS_CHECK(cache.Lookup(0, kGuestAddr, kDwords) == &shader_a);
  DRAW_CUTS_CHECK(cache.Lookup(0, kPhysAddr, kDwords) == &shader_a);
  DRAW_CUTS_CHECK(cache.Lookup(1, kGuestAddr, kDwords) == nullptr);
  DRAW_CUTS_CHECK(cache.Lookup(0, kGuestAddr, kDwords + 4) == nullptr);

  // Write to an UNRELATED page in a different 4 MiB window does NOT invalidate
  // shader_a and returns an unconstrained unwatch range {0, UINT32_MAX}.
  auto far_unwatch =
      cache.OnMemoryInvalidatedForUnwatch(0x00500000u, 4096, false);
  DRAW_CUTS_CHECK(far_unwatch.first == 0u && far_unwatch.second == UINT32_MAX);
  DRAW_CUTS_CHECK(cache.Lookup(0, kGuestAddr, kDwords) == &shader_a);

  // Write to an ADJACENT page in the SAME 4 MiB window (0x00124000) does NOT
  // invalidate shader_a AND MUST clamp the unwatch range to (0x00124000, 4096)
  // so Memory does not silently unprotect shader_a's page (0x00123000)!
  auto near_unwatch =
      cache.OnMemoryInvalidatedForUnwatch(0x00124000u, 4096, false);
  DRAW_CUTS_CHECK(near_unwatch.first == 0x00124000u &&
                  near_unwatch.second == 4096u);
  DRAW_CUTS_CHECK(cache.Lookup(0, kGuestAddr, kDwords) == &shader_a);

  // Write to the SAME physical page invalidates shader_a immediately!
  cache.OnMemoryInvalidated(kPhysAddr + 128, 64);
  DRAW_CUTS_CHECK(cache.Lookup(0, kGuestAddr, kDwords) == nullptr);

  // TOCTOU race test: write happens BETWEEN BeginArmAndSnapshot and CommitIfUnchanged!
  // Commit MUST fail so stale hash is never cached.
  DRAW_CUTS_CHECK(cache.BeginArmAndSnapshot(kGuestAddr, kDwords, snap));
  cache.OnMemoryInvalidated(kPhysAddr, 4096);
  DRAW_CUTS_CHECK(
      !cache.CommitIfUnchanged(0, kGuestAddr, kDwords, snap, &shader_b));
  DRAW_CUTS_CHECK(cache.Lookup(0, kGuestAddr, kDwords) == nullptr);

  // Multi-page shader (spans 2 pages: 0x00200F00 .. 0x00201700).
  constexpr uint32_t kSpanAddr = 0x00200F00u;
  constexpr uint32_t kSpanDwords = 512;  // 2048 bytes -> crosses 0x201000 boundary
  DRAW_CUTS_CHECK(cache.BeginArmAndSnapshot(kSpanAddr, kSpanDwords, snap));
  DRAW_CUTS_CHECK(snap.page_count == 2);
  DRAW_CUTS_CHECK(
      cache.CommitIfUnchanged(1, kSpanAddr, kSpanDwords, snap, &shader_c));
  DRAW_CUTS_CHECK(cache.Lookup(1, kSpanAddr, kSpanDwords) == &shader_c);
  // Invalidate ONLY the second page -> MUST invalidate the spanning shader!
  cache.OnMemoryInvalidated(0x00201000u, 4096);
  DRAW_CUTS_CHECK(cache.Lookup(1, kSpanAddr, kSpanDwords) == nullptr);

  // Global invalidation clears all cached entries.
  DRAW_CUTS_CHECK(cache.BeginArmAndSnapshot(kGuestAddr, kDwords, snap));
  DRAW_CUTS_CHECK(
      cache.CommitIfUnchanged(0, kGuestAddr, kDwords, snap, &shader_b));
  DRAW_CUTS_CHECK(cache.Lookup(0, kGuestAddr, kDwords) == &shader_b);
  cache.InvalidateAll();
  DRAW_CUTS_CHECK(cache.Lookup(0, kGuestAddr, kDwords) == nullptr);

  // Oversized shader (> max tracked pages) gracefully declines caching.
  DRAW_CUTS_CHECK(!cache.BeginArmAndSnapshot(0x00300000u, 16384, snap));

  // Concurrent stress test: Guest CPU thread invalidates while GPU thread looks up/commits.
  std::atomic<bool> stop_writer{false};
  std::thread writer([&]() {
    uint32_t iter = 0;
    while (!stop_writer.load(std::memory_order_relaxed)) {
      cache.OnMemoryInvalidated(0x00400000u + ((iter & 7u) * 4096u), 4096u);
      ++iter;
    }
  });
  for (int i = 0; i < 4000; ++i) {
    const uint32_t addr = 0x00400000u + (uint32_t(i & 7) * 4096u);
    if (cache.Lookup(0, addr, 64) == nullptr) {
      dc::ShaderWatchSnapshot s{};
      if (cache.BeginArmAndSnapshot(addr, 64, s)) {
        cache.CommitIfUnchanged(0, addr, 64, s, &shader_a);
      }
    }
  }
  stop_writer.store(true, std::memory_order_relaxed);
  writer.join();
  // Final invalidation after writer stops MUST leave entry invalid.
  cache.OnMemoryInvalidated(0x00400000u, 8u * 4096u);
  for (uint32_t p = 0; p < 8; ++p) {
    DRAW_CUTS_CHECK(cache.Lookup(0, 0x00400000u + p * 4096u, 64) == nullptr);
  }
}

void TestPushDescriptorDriverGateAndClamp() {
  constexpr uint32_t kVendorArm = 0x13B5;
  constexpr uint32_t kVendorQualcomm = 0x5143;
  constexpr uint32_t kVendorAmd = 0x1002;
  constexpr uint32_t kDriverQualcommProp = 8;
  constexpr uint32_t kDriverArmProp = 9;
  constexpr uint32_t kDriverMesaTurnip = 18;
  constexpr uint32_t kDriverMesaPanvk = 20;
  constexpr uint32_t kDriverMesaRadv = 3;

  // Cvar off -> always disabled.
  auto off = dc::EvaluatePushDescriptorPolicy(
      /*cvar_enabled=*/false, /*ext_enabled=*/true, /*fn_loaded=*/true,
      kVendorQualcomm, kDriverMesaTurnip, 32, 5);
  DRAW_CUTS_CHECK(!off.enabled);

  // Strict blacklist: Qualcomm proprietary (driverID == 8) & legacy Qualcomm (driverID == 0).
  auto qcom_prop = dc::EvaluatePushDescriptorPolicy(
      true, true, true, kVendorQualcomm, kDriverQualcommProp, 32, 5);
  DRAW_CUTS_CHECK(!qcom_prop.enabled);
  auto qcom_legacy = dc::EvaluatePushDescriptorPolicy(
      true, true, true, kVendorQualcomm, 0, 32, 5);
  DRAW_CUTS_CHECK(!qcom_legacy.enabled);

  // Strict blacklist: Arm proprietary blob (driverID == 9 or driverID == 0 with Arm vendor).
  auto arm_prop = dc::EvaluatePushDescriptorPolicy(
      true, true, true, kVendorArm, kDriverArmProp, 32, 5);
  DRAW_CUTS_CHECK(!arm_prop.enabled);
  auto arm_legacy = dc::EvaluatePushDescriptorPolicy(
      true, true, true, kVendorArm, 0, 32, 5);
  DRAW_CUTS_CHECK(!arm_legacy.enabled);

  // Allowed: Mesa Turnip (driverID == 18) with Eden clamp on huge maxPushDescriptors!
  auto turnip = dc::EvaluatePushDescriptorPolicy(
      true, true, true, kVendorQualcomm, kDriverMesaTurnip, 1024, 5);
  DRAW_CUTS_CHECK(turnip.enabled);
  DRAW_CUTS_CHECK(turnip.clamped_max_push_descriptors ==
                  dc::kMaxPushDescriptorsClamp);

  // Allowed: Mesa PanVK / malisx2 (vendorID == 0x13B5, driverID == 20).
  auto panvk = dc::EvaluatePushDescriptorPolicy(
      true, true, true, kVendorArm, kDriverMesaPanvk, 24, 5);
  DRAW_CUTS_CHECK(panvk.enabled);
  DRAW_CUTS_CHECK(panvk.clamped_max_push_descriptors == 24);

  // Allowed: Desktop Mesa RADV.
  auto radv = dc::EvaluatePushDescriptorPolicy(
      true, true, true, kVendorAmd, kDriverMesaRadv, 32, 5);
  DRAW_CUTS_CHECK(radv.enabled);
  DRAW_CUTS_CHECK(radv.clamped_max_push_descriptors == 32);

  // Too-small reported maxPushDescriptors (< required bindings) -> disabled!
  auto too_small = dc::EvaluatePushDescriptorPolicy(
      true, true, true, kVendorQualcomm, kDriverMesaTurnip, 4, 5);
  DRAW_CUTS_CHECK(!too_small.enabled);

  // Missing function pointer -> disabled!
  auto no_fn = dc::EvaluatePushDescriptorPolicy(
      true, true, false, kVendorQualcomm, kDriverMesaTurnip, 32, 5);
  DRAW_CUTS_CHECK(!no_fn.enabled);
}

void TestSmallReadbackAsyncRing() {
  // Size threshold check (<= 4 KiB).
  DRAW_CUTS_CHECK(!dc::IsSmallReadbackRingEligible(0));
  DRAW_CUTS_CHECK(dc::IsSmallReadbackRingEligible(4));
  DRAW_CUTS_CHECK(dc::IsSmallReadbackRingEligible(4096));
  DRAW_CUTS_CHECK(!dc::IsSmallReadbackRingEligible(4097));

  dc::SmallReadbackRingState ring;
  DRAW_CUTS_CHECK(ring.empty());
  DRAW_CUTS_CHECK(!ring.full());

  // Enqueue two resolves to the SAME exposure address across submissions 1 and 2.
  uint32_t slot0 = UINT32_MAX;
  uint32_t slot1 = UINT32_MAX;
  DRAW_CUTS_CHECK(ring.TryEnqueue(1, 0x01000000u, 64, slot0));
  DRAW_CUTS_CHECK(ring.TryEnqueue(2, 0x01000000u, 64, slot1));
  DRAW_CUTS_CHECK(slot0 != slot1);
  DRAW_CUTS_CHECK(ring.pending_count() == 2);
  DRAW_CUTS_CHECK(ring.OldestSubmission() == 1);

  // Nothing drains before submission 1 completes.
  std::vector<uint64_t> drained_subs;
  ring.DrainCompleted(0, [&](const dc::SmallReadbackRingSlot& s) {
    drained_subs.push_back(s.submission);
  });
  DRAW_CUTS_CHECK(drained_subs.empty());
  DRAW_CUTS_CHECK(ring.pending_count() == 2);

  // Submission 1 completes -> drains slot 0 ONLY (FIFO).
  ring.DrainCompleted(1, [&](const dc::SmallReadbackRingSlot& s) {
    DRAW_CUTS_CHECK(s.slot_index == slot0);
    DRAW_CUTS_CHECK(s.guest_address == 0x01000000u);
    DRAW_CUTS_CHECK(s.length == 64);
    drained_subs.push_back(s.submission);
  });
  DRAW_CUTS_CHECK(drained_subs.size() == 1 && drained_subs[0] == 1);
  DRAW_CUTS_CHECK(ring.pending_count() == 1);
  DRAW_CUTS_CHECK(ring.OldestSubmission() == 2);

  // Submission 2 completes -> drains slot 1.
  ring.DrainCompleted(2, [&](const dc::SmallReadbackRingSlot& s) {
    DRAW_CUTS_CHECK(s.slot_index == slot1);
    drained_subs.push_back(s.submission);
  });
  DRAW_CUTS_CHECK(drained_subs.size() == 2 && drained_subs[1] == 2);
  DRAW_CUTS_CHECK(ring.empty());

  // Fill entire ring (kSmallReadbackRingSlotCount entries) -> backpressure on next enqueue.
  for (uint32_t i = 0; i < dc::kSmallReadbackRingSlotCount; ++i) {
    uint32_t slot = UINT32_MAX;
    DRAW_CUTS_CHECK(ring.TryEnqueue(10 + i, 0x02000000u + i * 64, 32, slot));
  }
  DRAW_CUTS_CHECK(ring.full());
  uint32_t overflow_slot = UINT32_MAX;
  DRAW_CUTS_CHECK(!ring.TryEnqueue(999, 0x03000000u, 32, overflow_slot));
  DRAW_CUTS_CHECK(ring.OldestSubmission() == 10);

  // Drain first 10 slots and wrap around past capacity.
  uint32_t drained_count = 0;
  ring.DrainCompleted(19, [&](const dc::SmallReadbackRingSlot&) {
    ++drained_count;
  });
  DRAW_CUTS_CHECK(drained_count == 10);
  DRAW_CUTS_CHECK(ring.pending_count() == dc::kSmallReadbackRingSlotCount - 10);
  for (uint32_t i = 0; i < 10; ++i) {
    uint32_t slot = UINT32_MAX;
    DRAW_CUTS_CHECK(ring.TryEnqueue(200 + i, 0x04000000u, 16, slot));
  }
  DRAW_CUTS_CHECK(ring.full());
  ring.Reset();
  DRAW_CUTS_CHECK(ring.empty());
}

void TestGuestTimeHoldRefCount() {
  xe::GuestTimeHoldRefCounter counter;
  DRAW_CUTS_CHECK(counter.count() == 0);
  DRAW_CUTS_CHECK(!counter.is_held());

  // Underflow guard (e.g., Emulator::Shutdown calling HoldGuestTime(false) when unheld).
  DRAW_CUTS_CHECK(!counter.Release());
  DRAW_CUTS_CHECK(counter.count() == 0);
  DRAW_CUTS_CHECK(!counter.is_held());

  // Emulator::Pause acquires (0 -> 1).
  DRAW_CUTS_CHECK(counter.Acquire());
  DRAW_CUTS_CHECK(counter.count() == 1);
  DRAW_CUTS_CHECK(counter.is_held());

  // Pipeline compilation wait acquires nested (1 -> 2).
  DRAW_CUTS_CHECK(!counter.Acquire());
  DRAW_CUTS_CHECK(counter.count() == 2);
  DRAW_CUTS_CHECK(counter.is_held());

  // Pipeline compilation wait finishes (2 -> 1): MUST NOT unpause emulator!
  DRAW_CUTS_CHECK(!counter.Release());
  DRAW_CUTS_CHECK(counter.count() == 1);
  DRAW_CUTS_CHECK(counter.is_held());

  // Emulator::Resume releases (1 -> 0): NOW resumes!
  DRAW_CUTS_CHECK(counter.Release());
  DRAW_CUTS_CHECK(counter.count() == 0);
  DRAW_CUTS_CHECK(!counter.is_held());

  // Extra release stays clamped at 0.
  DRAW_CUTS_CHECK(!counter.Release());
  DRAW_CUTS_CHECK(counter.count() == 0);

  // Concurrent multi-threaded refcount stress while emulator holds outer pause!
  DRAW_CUTS_CHECK(counter.Acquire());
  std::vector<std::thread> workers;
  std::atomic<int> unexpected_resume{0};
  for (int t = 0; t < 4; ++t) {
    workers.emplace_back([&]() {
      for (int i = 0; i < 5000; ++i) {
        counter.Acquire();
        if (!counter.is_held()) {
          unexpected_resume.fetch_add(1, std::memory_order_relaxed);
        }
        if (counter.Release()) {
          unexpected_resume.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }
  for (auto& th : workers) {
    th.join();
  }
  DRAW_CUTS_CHECK(unexpected_resume.load() == 0);
  DRAW_CUTS_CHECK(counter.count() == 1);
  DRAW_CUTS_CHECK(counter.is_held());
  DRAW_CUTS_CHECK(counter.Release());
  DRAW_CUTS_CHECK(counter.count() == 0);
  DRAW_CUTS_CHECK(!counter.is_held());
}

void TestSourceInvariants(const fs::path& cpp_root) {
  const fs::path xenia_src = cpp_root / "xenia" / "src" / "xenia";
  const fs::path vk_dir = xenia_src / "gpu" / "vulkan";
  const fs::path ui_vk_dir = xenia_src / "ui" / "vulkan";

  const std::string cp_h =
      ReadSourceText(vk_dir / "vulkan_command_processor.h");
  const std::string cp_cc =
      ReadSourceText(vk_dir / "vulkan_command_processor.cc");
  const std::string pcache_h =
      ReadSourceText(vk_dir / "vulkan_pipeline_cache.h");
  const std::string pcache_cc =
      ReadSourceText(vk_dir / "vulkan_pipeline_cache.cc");
  const std::string dcb_h =
      ReadSourceText(vk_dir / "deferred_command_buffer.h");
  const std::string dcb_cc =
      ReadSourceText(vk_dir / "deferred_command_buffer.cc");
  const std::string pm4_h =
      ReadSourceText(xenia_src / "gpu" / "pm4_command_processor_implement.h");
  const std::string dev_h = ReadSourceText(ui_vk_dir / "vulkan_device.h");
  const std::string dev_cc = ReadSourceText(ui_vk_dir / "vulkan_device.cc");
  const std::string push_inc = ReadSourceText(
      ui_vk_dir / "functions" / "device_khr_push_descriptor.inc");
  const std::string clock_h = ReadSourceText(xenia_src / "base" / "clock.h");
  const std::string clock_cc = ReadSourceText(xenia_src / "base" / "clock.cc");

  // All new GPU cvars MUST default to false and MUST NOT use UPDATE_from_*.
  DRAW_CUTS_CHECK(
      cp_cc.find("vulkan_skip_redundant_float_constant_writes, false") !=
      std::string::npos);
  DRAW_CUTS_CHECK(cp_cc.find("vulkan_memoize_system_constants, false") !=
                  std::string::npos);
  DRAW_CUTS_CHECK(cp_cc.find("vulkan_use_push_descriptors, false") !=
                  std::string::npos);
  DRAW_CUTS_CHECK(cp_cc.find("vulkan_small_readback_async_ring, false") !=
                  std::string::npos);
  DRAW_CUTS_CHECK(pcache_cc.find("vulkan_shader_load_watch_cache, false") !=
                  std::string::npos);
  DRAW_CUTS_CHECK(
      pcache_cc.find("vulkan_hold_guest_time_on_pipeline_wait, false") !=
      std::string::npos);

  DRAW_CUTS_CHECK(
      cp_cc.find("UPDATE_from_bool(vulkan_skip_redundant_float_constant_writes") ==
      std::string::npos);
  DRAW_CUTS_CHECK(
      cp_cc.find("UPDATE_from_bool(vulkan_memoize_system_constants") ==
      std::string::npos);
  DRAW_CUTS_CHECK(cp_cc.find("UPDATE_from_bool(vulkan_use_push_descriptors") ==
                  std::string::npos);
  DRAW_CUTS_CHECK(
      cp_cc.find("UPDATE_from_bool(vulkan_small_readback_async_ring") ==
      std::string::npos);
  DRAW_CUTS_CHECK(
      pcache_cc.find("UPDATE_from_bool(vulkan_shader_load_watch_cache") ==
      std::string::npos);
  DRAW_CUTS_CHECK(
      pcache_cc.find(
          "UPDATE_from_bool(vulkan_hold_guest_time_on_pipeline_wait") ==
      std::string::npos);

  // Android debug property overrides for ABBA validation.
  DRAW_CUTS_CHECK(cp_cc.find("debug.xendroid.memoize_sysconst") !=
                  std::string::npos);
  DRAW_CUTS_CHECK(cp_cc.find("debug.xendroid.shader_load_watch_cache") !=
                  std::string::npos);
  DRAW_CUTS_CHECK(cp_cc.find("debug.xendroid.small_readback_ring") !=
                  std::string::npos);

  // Feature 1: WriteShaderConstantsFromMem value compare.
  DRAW_CUTS_CHECK(cp_cc.find("CopySwapAndTrackFloatConstants(") !=
                  std::string::npos);

  // Feature 2: System constants dirty-mask memoization.
  DRAW_CUTS_CHECK(cp_h.find("SystemConstantsDirtyTracker") !=
                  std::string::npos);
  DRAW_CUTS_CHECK(cp_cc.find("sysconst_dirty_tracker_") != std::string::npos);

  // Feature 3: LoadShader watch-cache via PM4_IM_LOAD.
  DRAW_CUTS_CHECK(pm4_h.find("LoadShaderFromGuestAddress") !=
                  std::string::npos);
  DRAW_CUTS_CHECK(pcache_h.find("LoadShaderFromGuestAddress") !=
                  std::string::npos);
  DRAW_CUTS_CHECK(pcache_cc.find("EnablePhysicalMemoryAccessCallbacks") !=
                  std::string::npos);
  DRAW_CUTS_CHECK(
      pcache_cc.find("RegisterPhysicalMemoryInvalidationCallback") !=
      std::string::npos);

  // Feature 4: VK_KHR_push_descriptor extension + blacklist + clamp + DCB.
  DRAW_CUTS_CHECK(push_inc.find("vkCmdPushDescriptorSetKHR") !=
                  std::string::npos);
  DRAW_CUTS_CHECK(dev_h.find("ext_KHR_push_descriptor") != std::string::npos);
  DRAW_CUTS_CHECK(dev_h.find("maxPushDescriptors") != std::string::npos);
  DRAW_CUTS_CHECK(dev_cc.find("device_khr_push_descriptor.inc") !=
                  std::string::npos);
  DRAW_CUTS_CHECK(dev_cc.find("VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES_KHR") !=
                  std::string::npos);
  DRAW_CUTS_CHECK(dcb_h.find("CmdVkPushDescriptorSetKHR") != std::string::npos);
  DRAW_CUTS_CHECK(dcb_cc.find("vkCmdPushDescriptorSetKHR") !=
                  std::string::npos);
  DRAW_CUTS_CHECK(cp_cc.find("EvaluatePushDescriptorPolicy(") !=
                  std::string::npos);
  DRAW_CUTS_CHECK(
      cp_cc.find("VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR") !=
      std::string::npos);

  // Feature 5: Small readback async ring.
  DRAW_CUTS_CHECK(cp_cc.find("DrainCompletedSmallReadbacks(") !=
                  std::string::npos);
  DRAW_CUTS_CHECK(cp_cc.find("TryEnqueueSmallReadback(") != std::string::npos);

  // Feature 6: Clock::HoldGuestTime refcounted + pipeline waits.
  DRAW_CUTS_CHECK(clock_h.find("GuestTimeHoldRefCounter") != std::string::npos);
  DRAW_CUTS_CHECK(clock_h.find("ScopedGuestTimeHold") != std::string::npos);
  DRAW_CUTS_CHECK(clock_cc.find("guest_time_hold_counter_") !=
                  std::string::npos);
  DRAW_CUTS_CHECK(pcache_cc.find("Clock::ScopedGuestTimeHold") !=
                  std::string::npos);
}

struct FakeShader {};
struct FakeCpWithWatchHook {
  FakeShader watch_shader;
  FakeShader fallback_shader;
  uint32_t last_guest_addr = 0;
  FakeShader* LoadShaderFromGuestAddress(uint32_t /*st*/, uint32_t guest_addr,
                                         const uint32_t* /*host*/,
                                         uint32_t /*dwords*/) {
    last_guest_addr = guest_addr;
    return &watch_shader;
  }
  FakeShader* LoadShader(uint32_t /*st*/, const uint32_t* /*host*/,
                         uint32_t /*dwords*/) {
    return &fallback_shader;
  }
  FakeShader* Dispatch(uint32_t addr, const uint32_t* host_ptr,
                       uint32_t dwords) {
    return [&]<typename Cp>(Cp* cp) -> FakeShader* {
      if constexpr (requires(Cp& c, uint32_t st, uint32_t guest_addr,
                             const uint32_t* p, uint32_t dw) {
                      c.LoadShaderFromGuestAddress(st, guest_addr, p, dw);
                    }) {
        return cp->LoadShaderFromGuestAddress(0u, addr, host_ptr, dwords);
      } else {
        return cp->LoadShader(0u, host_ptr, dwords);
      }
    }(this);
  }
};

struct FakeCpWithoutWatchHook {
  FakeShader fallback_shader;
  FakeShader* LoadShader(uint32_t /*st*/, const uint32_t* /*host*/,
                         uint32_t /*dwords*/) {
    return &fallback_shader;
  }
  FakeShader* Dispatch(uint32_t addr, const uint32_t* host_ptr,
                       uint32_t dwords) {
    return [&]<typename Cp>(Cp* cp) -> FakeShader* {
      if constexpr (requires(Cp& c, uint32_t st, uint32_t guest_addr,
                             const uint32_t* p, uint32_t dw) {
                      c.LoadShaderFromGuestAddress(st, guest_addr, p, dw);
                    }) {
        return cp->LoadShaderFromGuestAddress(0u, addr, host_ptr, dwords);
      } else {
        return cp->LoadShader(0u, host_ptr, dwords);
      }
    }(this);
  }
};

void TestPm4LoadShaderHookDispatch() {
  uint32_t dummy = 0;
  FakeCpWithWatchHook with_hook;
  DRAW_CUTS_CHECK(with_hook.Dispatch(0x12340000u, &dummy, 16) ==
                  &with_hook.watch_shader);
  DRAW_CUTS_CHECK(with_hook.last_guest_addr == 0x12340000u);

  FakeCpWithoutWatchHook without_hook;
  DRAW_CUTS_CHECK(without_hook.Dispatch(0x12340000u, &dummy, 16) ==
                  &without_hook.fallback_shader);
}

}  // namespace

int main(int argc, char** argv) {
  TestFloatConstantValueCompare();
  TestSystemConstantsDirtyMask();
  TestShaderLoadWatchCacheEpoch();
  TestPushDescriptorDriverGateAndClamp();
  TestSmallReadbackAsyncRing();
  TestGuestTimeHoldRefCount();
  TestPm4LoadShaderHookDispatch();
  if (argc >= 2) {
    TestSourceInvariants(fs::path(argv[1]));
  } else {
    std::fprintf(stderr, "Usage: gpu_draw_cuts_test <cpp_root>\n");
    return 1;
  }
  std::printf("gpu_draw_cuts_test: PASS\n");
  return 0;
}
