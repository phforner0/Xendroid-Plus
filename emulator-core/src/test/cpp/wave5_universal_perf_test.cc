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

#include "xenia/cpu/backend/a64/a64_hle_sync_layout.h"
#include "xe_wave5_perf_logic.h"

namespace {

#define WAVE5_CHECK(cond)                                                  \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "WAVE5_LOGIC check failed (%s:%d): %s\n",       \
                   __FILE__, __LINE__, #cond);                             \
      std::abort();                                                        \
    }                                                                      \
  } while (false)

namespace fs = std::filesystem;
namespace a64sync = xe::cpu::backend::a64;

std::string NormalizeNewlines(std::string s) {
  s.erase(std::remove(s.begin(), s.end(), '\r'), s.end());
  return s;
}

std::string ReadSourceText(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  WAVE5_CHECK(in.good());
  std::ostringstream ss;
  ss << in.rdbuf();
  return NormalizeNewlines(ss.str());
}

void PackFloatConstantsScalarRef(uint8_t* dst, const uint32_t* regs_base,
                                 const uint64_t map[4]) {
  for (uint32_t i = 0; i < 4; ++i) {
    for (uint32_t bit = 0; bit < 64; ++bit) {
      if (map[i] & (UINT64_C(1) << bit)) {
        std::memcpy(dst, regs_base + ((i << 8) + (bit << 2)),
                    sizeof(float) * 4);
        dst += sizeof(float) * 4;
      }
    }
  }
}

void TestInlineHleSyncClassificationAndGating() {
  using K = a64sync::InlineHleSyncKind;
  WAVE5_CHECK(a64sync::ClassifyInlineHleSync("KeRaiseIrqlToDpcLevel") ==
              K::kKeRaiseIrqlToDpcLevel);
  WAVE5_CHECK(a64sync::ClassifyInlineHleSync("KfLowerIrql") == K::kKfLowerIrql);
  WAVE5_CHECK(a64sync::ClassifyInlineHleSync("RtlEnterCriticalSection") ==
              K::kRtlEnterCriticalSection);
  WAVE5_CHECK(a64sync::ClassifyInlineHleSync("RtlLeaveCriticalSection") ==
              K::kRtlLeaveCriticalSection);
  WAVE5_CHECK(a64sync::ClassifyInlineHleSync("RtlTryEnterCriticalSection") ==
              K::kRtlTryEnterCriticalSection);
  WAVE5_CHECK(a64sync::ClassifyInlineHleSync("KeAcquireSpinLockAtRaisedIrql") ==
              K::kKeAcquireSpinLockAtRaisedIrql);
  WAVE5_CHECK(
      a64sync::ClassifyInlineHleSync("KeReleaseSpinLockFromRaisedIrql") ==
      K::kKeReleaseSpinLockFromRaisedIrql);
  WAVE5_CHECK(a64sync::ClassifyInlineHleSync("KfAcquireSpinLock") ==
              K::kKfAcquireSpinLock);
  WAVE5_CHECK(a64sync::ClassifyInlineHleSync("KfReleaseSpinLock") ==
              K::kKfReleaseSpinLock);
  WAVE5_CHECK(a64sync::ClassifyInlineHleSync("KeWaitForSingleObject") ==
              K::kNone);
  WAVE5_CHECK(a64sync::ClassifyInlineHleSync("") == K::kNone);

  constexpr uint32_t kImplementedTag = 1u << 0;
  constexpr uint32_t kBlockingTag = 1u << 5;
  constexpr uint32_t kHighFreqTag = 1u << 3;
  constexpr uint32_t kLogTag = 1u << 30;
  constexpr uint32_t kLogResultTag = 1u << 31;

  // Real xboxkrnl DECLARE_XBOXKRNL_EXPORT* permanently ORs ExportTag::kLog.
  // High-frequency exports suppress logging at runtime unless
  // log_high_frequency_kernel_calls is enabled or ExportTag::kLogResult is set.
  const uint32_t kNormalSyncTags = kImplementedTag | kLogTag | kHighFreqTag;
  const uint32_t kBlockingSyncTags =
      kImplementedTag | kBlockingTag | kLogTag | kHighFreqTag;
  const uint32_t kAlwaysLoggedTags = kImplementedTag | kLogTag;

  WAVE5_CHECK(!a64sync::IsInlineHleSyncLoggingActive(
      kNormalSyncTags, kLogTag, kLogResultTag, kHighFreqTag, false));
  WAVE5_CHECK(!a64sync::IsInlineHleSyncLoggingActive(
      kBlockingSyncTags, kLogTag, kLogResultTag, kHighFreqTag, false));
  WAVE5_CHECK(a64sync::IsInlineHleSyncLoggingActive(
      kNormalSyncTags, kLogTag, kLogResultTag, kHighFreqTag, true));
  WAVE5_CHECK(a64sync::IsInlineHleSyncLoggingActive(
      kAlwaysLoggedTags, kLogTag, kLogResultTag, kHighFreqTag, false));
  WAVE5_CHECK(a64sync::IsInlineHleSyncLoggingActive(
      kNormalSyncTags | kLogResultTag, kLogTag, kLogResultTag, kHighFreqTag,
      false));

  const char* kTargetExports[] = {
      "KeRaiseIrqlToDpcLevel",
      "KfLowerIrql",
      "RtlEnterCriticalSection",
      "RtlLeaveCriticalSection",
      "RtlTryEnterCriticalSection",
      "KeAcquireSpinLockAtRaisedIrql",
      "KeReleaseSpinLockFromRaisedIrql",
      "KfAcquireSpinLock",
      "KfReleaseSpinLock",
  };
  for (const char* name : kTargetExports) {
    const K kind = a64sync::ClassifyInlineHleSync(name);
    WAVE5_CHECK(kind != K::kNone);
    const bool log_off = a64sync::IsInlineHleSyncLoggingActive(
        kNormalSyncTags, kLogTag, kLogResultTag, kHighFreqTag, false);
    WAVE5_CHECK(
        a64sync::ShouldEmitInlineHleSync(true, false, log_off, true, kind));
    const bool log_on = a64sync::IsInlineHleSyncLoggingActive(
        kNormalSyncTags, kLogTag, kLogResultTag, kHighFreqTag, true);
    WAVE5_CHECK(
        !a64sync::ShouldEmitInlineHleSync(true, false, log_on, true, kind));
  }

  WAVE5_CHECK(!a64sync::ShouldEmitInlineHleSync(
      false, false, false, true, K::kRtlEnterCriticalSection));
  WAVE5_CHECK(!a64sync::ShouldEmitInlineHleSync(
      true, true, false, true, K::kRtlEnterCriticalSection));
  WAVE5_CHECK(!a64sync::ShouldEmitInlineHleSync(
      true, false, true, true, K::kRtlEnterCriticalSection));
  WAVE5_CHECK(!a64sync::ShouldEmitInlineHleSync(
      true, false, false, false, K::kRtlLeaveCriticalSection));
  WAVE5_CHECK(a64sync::ShouldEmitInlineHleSync(
      true, false, false, false, K::kKeRaiseIrqlToDpcLevel));
}

void TestPackFloatConstantsByRuns() {
  std::vector<uint32_t> regs(1024, 0);
  for (uint32_t i = 0; i < 1024; ++i) {
    regs[i] = 0xA5000000u ^ (i * 0x01010101u);
  }
  const std::array<std::array<uint64_t, 4>, 6> cases = {{
      {0, 0, 0, 0},
      {1ULL, 0, 0, (1ULL << 63)},
      {0x0000FFFF0000FFFFULL, ~0ULL, 0x8000000000000001ULL, 0x0F0F0F0F0F0F0F0FULL},
      {~0ULL, ~0ULL, ~0ULL, ~0ULL},
      {0xAAAAAAAAAAAAAAAAULL, 0x5555555555555555ULL, 0x00FF00FF00FF00FFULL, 0xF00000000000000FULL},
      {0x123456789ABCDEF0ULL, 0x0FEDCBA987654321ULL, 0x0000000180000000ULL, 0x0000FFFFFFFF0000ULL},
  }};
  for (const auto& m : cases) {
    size_t count = 0;
    for (int i = 0; i < 4; ++i) {
      uint64_t x = m[i];
      while (x) {
        count += (x & 1u);
        x >>= 1;
      }
    }
    std::vector<uint8_t> actual(count * 16 + 32, 0xCD);
    std::vector<uint8_t> expected(count * 16 + 32, 0xCD);
    xendroid::PackFloatConstantsByRuns(actual.data(), regs.data(), m.data());
    PackFloatConstantsScalarRef(expected.data(), regs.data(), m.data());
    WAVE5_CHECK(actual == expected);
  }
}

void TestPixelSpecCacheState() {
  xendroid::PixelSpecCacheState st;
  const int shader_a = 1;
  const int shader_b = 2;
  WAVE5_CHECK(!xendroid::EvaluatePixelSpecCacheHit(st, &shader_a, 0x15, 0x3));

  st.shader = &shader_a;
  st.flags_key = 0x15;
  st.fetch_up_to_date = ~uint32_t(0);
  WAVE5_CHECK(xendroid::EvaluatePixelSpecCacheHit(st, &shader_a, 0x15, 0x5));
  WAVE5_CHECK(!xendroid::EvaluatePixelSpecCacheHit(st, &shader_b, 0x15, 0x5));
  WAVE5_CHECK(!xendroid::EvaluatePixelSpecCacheHit(st, &shader_a, 0x17, 0x5));

  // Invalidating slot 2 leaves slots 0 and 1 hit-eligible, slot 2 miss-eligible.
  xendroid::InvalidatePixelSpecCacheFetches(st, ~(UINT32_C(1) << 2));
  WAVE5_CHECK(xendroid::EvaluatePixelSpecCacheHit(st, &shader_a, 0x15, 0x3));
  WAVE5_CHECK(!xendroid::EvaluatePixelSpecCacheHit(st, &shader_a, 0x15, 0x5));

  // Simulated intra-frame RestoreRegisters(..., false) over fetch slots 1..2.
  const uint32_t kFetch00 = 0x4800;
  const uint32_t kFetch31_5 = 0x4800 + 32 * 6 - 1;
  st.fetch_up_to_date = ~uint32_t(0);
  const uint32_t dirty_mask = xendroid::ComputeRestoredFetchInvalidationMask(
      kFetch00 + 6 + 3, 6, kFetch00, kFetch31_5);
  WAVE5_CHECK(dirty_mask == ((UINT32_C(1) << 1) | (UINT32_C(1) << 2)));
  xendroid::InvalidatePixelSpecCacheFetches(st, ~dirty_mask);
  WAVE5_CHECK(xendroid::EvaluatePixelSpecCacheHit(st, &shader_a, 0x15, 0x1));
  WAVE5_CHECK(!xendroid::EvaluatePixelSpecCacheHit(st, &shader_a, 0x15, 0x2));
  WAVE5_CHECK(!xendroid::EvaluatePixelSpecCacheHit(st, &shader_a, 0x15, 0x4));
  WAVE5_CHECK(xendroid::ComputeRestoredFetchInvalidationMask(
                  0x2000, 16, kFetch00, kFetch31_5) == 0);

  // fragmentStoresAndAtomics is required whenever local-read attachments are on.
  WAVE5_CHECK(xendroid::ShouldEnableLocalReadAttachments(true, true, true, true, true));
  WAVE5_CHECK(!xendroid::ShouldEnableLocalReadAttachments(true, true, true, true, false));
  WAVE5_CHECK(!xendroid::ShouldEnableLocalReadAttachments(true, true, true, false, true));
}

void TestAudioAdaptiveQueueAndAAudioDrain() {
  xendroid::AudioAdaptiveState st;
  // Simulate heavy-boot title starting at depth=1 (never reaches topup_base=4 initially).
  // On the 4th block, filled latches true and the active depth<=1 stall bumps extra 0 -> 2.
  for (uint64_t t = 0; t < 4; ++t) {
    auto res = xendroid::StepAudioAdaptiveQueue(
        st, true, true, true, 1, 4, 11, 0, 5333, 100 + t * 10);
    WAVE5_CHECK(res.target_blocks <= 6);
  }
  WAVE5_CHECK(st.filled && st.extra == 2);

  // Sustained stall grows extra (2 -> 4 -> 6 -> 7 == deepest-topup_base) obeying 500 ms cooldown.
  uint64_t now_ms = 200;
  for (int k = 0; k < 4; ++k) {
    now_ms += 550;
    auto res = xendroid::StepAudioAdaptiveQueue(
        st, true, true, true, 1, 4, 11, 30000, 5333, now_ms);
    if (k < 3) {
      WAVE5_CHECK(res.grew);
    } else {
      WAVE5_CHECK(!res.grew);
    }
  }
  WAVE5_CHECK(st.extra == 7);
  WAVE5_CHECK(xendroid::ComputeAAudioHighDrainDepth(6, 4 + st.extra) == 13);

  // Calm window shrinks by 1 after 20 s.
  auto calm = xendroid::StepAudioAdaptiveQueue(
      st, true, true, true, 8, 4, 11, 200, 5333, now_ms + 20001);
  WAVE5_CHECK(calm.shrunk);
  WAVE5_CHECK(calm.target_blocks == 10);

  WAVE5_CHECK(
      xendroid::ShouldRunEmergencySecondAudioTopup(true, 1, 6, 4000, 5333));
  WAVE5_CHECK(
      !xendroid::ShouldRunEmergencySecondAudioTopup(false, 0, 6, 4000, 5333));
  WAVE5_CHECK(
      !xendroid::ShouldRunEmergencySecondAudioTopup(true, 2, 6, 4000, 5333));
  WAVE5_CHECK(
      !xendroid::ShouldRunEmergencySecondAudioTopup(true, 1, 6, 11000, 5333));

  // topup_base > deepest must stay clamped <= deepest.
  xendroid::AudioAdaptiveState clamp_st;
  auto clamp_res = xendroid::StepAudioAdaptiveQueue(
      clamp_st, true, true, true, 0, 8, 5, 0, 5333, 100);
  WAVE5_CHECK(clamp_res.target_blocks == 5);
}

void TestSourceContracts(const fs::path& cpp_root) {
  const std::string a64 =
      ReadSourceText(cpp_root / "xenia/src/xenia/cpu/backend/a64/a64_emitter.cc");
  WAVE5_CHECK(a64.find("DEFINE_bool(\n    a64_inline_hle_sync, true,") !=
              std::string::npos);
  WAVE5_CHECK(a64.find("TryEmitInlineHleSync(") != std::string::npos);
  WAVE5_CHECK(a64.find("IsInlineHleSyncLoggingActive(") != std::string::npos);
  WAVE5_CHECK(a64.find("hle_census_active()") != std::string::npos);
  WAVE5_CHECK(a64.find("log_high_frequency_kernel_calls()") !=
              std::string::npos);
  WAVE5_CHECK(a64.find("tst(w_addr, 3)") != std::string::npos);
  WAVE5_CHECK(a64.find("kGuestPhysicalShiftBoundary") != std::string::npos);
  WAVE5_CHECK(a64.find("ldaxr(") != std::string::npos);
  WAVE5_CHECK(a64.find("stlxr(") != std::string::npos);
  WAVE5_CHECK(a64.find("casal") == std::string::npos);

  const std::string rtl = ReadSourceText(
      cpp_root / "xenia/src/xenia/kernel/xboxkrnl/xboxkrnl_rtl.cc");
  WAVE5_CHECK(rtl.find("kRtlCsLockCountOffset") != std::string::npos);
  WAVE5_CHECK(rtl.find("xeRtlLeaveCriticalSectionWakeTrampoline") !=
              std::string::npos);
  WAVE5_CHECK(rtl.find("DECLARE_XBOXKRNL_EXPORT2(RtlEnterCriticalSection, kNone, kImplemented,\n                         kHighFrequency)") !=
              std::string::npos);
  WAVE5_CHECK(rtl.find("DECLARE_XBOXKRNL_EXPORT2(RtlTryEnterCriticalSection, kNone, kImplemented,\n                         kHighFrequency)") !=
              std::string::npos);
  WAVE5_CHECK(rtl.find("DECLARE_XBOXKRNL_EXPORT2(RtlLeaveCriticalSection, kNone, kImplemented,\n                         kHighFrequency)") !=
              std::string::npos);

  const std::string thr = ReadSourceText(
      cpp_root / "xenia/src/xenia/kernel/xboxkrnl/xboxkrnl_threading.cc");
  WAVE5_CHECK(thr.find("kKpcrCurrentIrqlOffset") != std::string::npos);
  WAVE5_CHECK(thr.find("kKpcrCurrentThreadOffset") != std::string::npos);
  for (const char* thr_exp :
       {"KfAcquireSpinLock", "KfReleaseSpinLock",
        "KeAcquireSpinLockAtRaisedIrql", "KeReleaseSpinLockFromRaisedIrql",
        "KeRaiseIrqlToDpcLevel", "KfLowerIrql"}) {
    const size_t pos = thr.find(std::string("DECLARE_XBOXKRNL_EXPORT") + "2(" +
                                thr_exp);
    const size_t pos3 = thr.find(std::string("DECLARE_XBOXKRNL_EXPORT") + "3(" +
                                 thr_exp);
    const size_t decl_pos = std::min(pos, pos3);
    WAVE5_CHECK(decl_pos != std::string::npos);
    const size_t semi = thr.find(';', decl_pos);
    WAVE5_CHECK(semi != std::string::npos);
    WAVE5_CHECK(thr.substr(decl_pos, semi - decl_pos).find("kHighFrequency") !=
                std::string::npos);
  }

  const std::string xmod = ReadSourceText(
      cpp_root / "xenia/src/xenia/kernel/xboxkrnl/xboxkrnl_module.cc");
  WAVE5_CHECK(xmod.find("set_log_high_frequency_kernel_calls_ptr(") !=
              std::string::npos);

  const std::string vcp = ReadSourceText(
      cpp_root / "xenia/src/xenia/gpu/vulkan/vulkan_command_processor.cc");
  WAVE5_CHECK(vcp.find("EvaluatePixelSpecCacheHit(") != std::string::npos);
  WAVE5_CHECK(vcp.find("PackFloatConstantsByRuns(") != std::string::npos);
  WAVE5_CHECK(vcp.find("texture_bindings_changed_vertex()") !=
              std::string::npos);
  WAVE5_CHECK(vcp.find("texture_usage_generation()") != std::string::npos);
  WAVE5_CHECK(vcp.find("void VulkanCommandProcessor::RestoreRegisters(") !=
              std::string::npos);
  WAVE5_CHECK(vcp.find("current_texture_set_3d_to_2d_vertex_") !=
              std::string::npos);

  const std::string vrtc = ReadSourceText(
      cpp_root / "xenia/src/xenia/gpu/vulkan/vulkan_render_target_cache.cc");
  WAVE5_CHECK(vrtc.find("vulkan_in_pass_resolve, true,") != std::string::npos);
  WAVE5_CHECK(vrtc.find("vulkan_in_pass_resolve_7e3, false,") !=
              std::string::npos);
  WAVE5_CHECK(vrtc.find("device_properties.fragmentStoresAndAtomics") !=
              std::string::npos);

  const std::string apu =
      ReadSourceText(cpp_root / "xenia/src/xenia/apu/audio_system.cc");
  WAVE5_CHECK(apu.find("DEFINE_uint32(apu_max_queued_frames, 12,") !=
              std::string::npos);
  WAVE5_CHECK(apu.find("StepAudioAdaptiveQueue(") != std::string::npos);
  WAVE5_CHECK(apu.find("ShouldRunEmergencySecondAudioTopup(") !=
              std::string::npos);
  WAVE5_CHECK(apu.find("driver->ReportsQueuedFrameCount(),") !=
              std::string::npos);
  WAVE5_CHECK(apu.find("adaptive_queues[i] = xendroid::AudioAdaptiveState{};") !=
              std::string::npos);

  const std::string aaudio =
      ReadSourceText(cpp_root / "xe_aaudio_audio_driver.cpp");
  WAVE5_CHECK(aaudio.find("ComputeAAudioHighDrainDepth(") != std::string::npos);
}

}  // namespace

int main(int argc, char** argv) {
  TestInlineHleSyncClassificationAndGating();
  TestPackFloatConstantsByRuns();
  TestPixelSpecCacheState();
  TestAudioAdaptiveQueueAndAAudioDrain();
  if (argc >= 2 && argv[1] && argv[1][0] != '\0') {
    TestSourceContracts(fs::path(argv[1]));
  }
  std::printf("wave5_universal_perf_test: PASS\n");
  return 0;
}
