#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#define XENIA_GPU_SHARED_MEMORY_PURE_LOGIC_ONLY
#include "xenia/gpu/shared_memory.h"

#define XENIA_GPU_VULKAN_SHARED_MEMORY_PURE_LOGIC_ONLY
#include "xenia/gpu/vulkan/vulkan_shared_memory.h"

namespace {

#define ZC_CHECK(cond)                                                     \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "GPU_ZEROCOPY check failed (%s:%d): %s\n",      \
                   __FILE__, __LINE__, #cond);                             \
      std::abort();                                                        \
    }                                                                      \
  } while (false)

namespace fs = std::filesystem;

std::string NormalizeNewlines(std::string s) {
  s.erase(std::remove(s.begin(), s.end(), '\r'), s.end());
  return s;
}

std::string ReadSourceText(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  ZC_CHECK(in.good());
  std::ostringstream ss;
  ss << in.rdbuf();
  return NormalizeNewlines(ss.str());
}

void TestZeroCopyPropertyParsing() {
  using xe::gpu::vulkan::ParseBinaryDebugProperty;
  bool val = false;
  ZC_CHECK(ParseBinaryDebugProperty("1", val) && val == true);
  ZC_CHECK(ParseBinaryDebugProperty("0", val) && val == false);
  // Must strictly accept only '0' or '1' (matching PollDebugPropertyOverride).
  ZC_CHECK(!ParseBinaryDebugProperty("", val));
  ZC_CHECK(!ParseBinaryDebugProperty("true", val));
  ZC_CHECK(!ParseBinaryDebugProperty("false", val));
  ZC_CHECK(!ParseBinaryDebugProperty("2", val));
  ZC_CHECK(!ParseBinaryDebugProperty("10", val));
  ZC_CHECK(!ParseBinaryDebugProperty("01", val));
  ZC_CHECK(!ParseBinaryDebugProperty(nullptr, val));
}

void TestZeroCopySubmissionHazardTracking() {
  using xe::gpu::vulkan::ZeroCopyPageSubmissionTracker;
  using xe::gpu::vulkan::PartitionZeroCopySpans;

  constexpr uint32_t kBufSizeLog2 = 20;  // 1 MB test buffer (256 x 4KB slots).
  ZeroCopyPageSubmissionTracker tracker;
  tracker.Initialize(kBufSizeLog2);

  // Never used by GPU (last_submission == 0) -> safe even at completed == 0.
  ZC_CHECK(tracker.IsByteRangeSafeForDirectWrite(0x0000, 0x4000,
                                                 /*completed_submission=*/0,
                                                 /*coherent=*/true,
                                                 /*atom_size=*/256));

  // Submission 5 uses pages [0x4000, 0x8000).
  tracker.MarkByteRangeUsed(0x4000, 0x4000, /*submission=*/5);

  // While completed_submission == 4 (submission 5 STILL IN FLIGHT):
  // Even if hoist were true, direct write to [0x4000, 0x8000) MUST be rejected!
  ZC_CHECK(!tracker.IsByteRangeSafeForDirectWrite(0x4000, 0x1000, 4, true, 256));
  ZC_CHECK(!tracker.IsByteRangeSafeForDirectWrite(0x7000, 0x1000, 4, true, 256));
  // Neighboring untouched pages remain safe.
  ZC_CHECK(tracker.IsByteRangeSafeForDirectWrite(0x0000, 0x4000, 4, true, 256));
  ZC_CHECK(tracker.IsByteRangeSafeForDirectWrite(0x8000, 0x4000, 4, true, 256));

  // Same-submission hazard: submission 6 (current) uses [0x8000, 0x9000),
  // completed_submission == 5 -> direct write in submission 6 MUST be rejected!
  tracker.MarkByteRangeUsed(0x8000, 0x1000, /*submission=*/6);
  ZC_CHECK(!tracker.IsByteRangeSafeForDirectWrite(0x8000, 0x1000, 5, true, 256));
  // Meanwhile submission 5 completed -> [0x4000, 0x8000) is safe again!
  ZC_CHECK(tracker.IsByteRangeSafeForDirectWrite(0x4000, 0x4000, 5, true, 256));

  // Monotonicity: stamping an older submission never regresses a newer stamp.
  tracker.MarkByteRangeUsed(0x8000, 0x1000, /*submission=*/2);
  ZC_CHECK(!tracker.IsByteRangeSafeForDirectWrite(0x8000, 0x1000, 5, true, 256));
  ZC_CHECK(tracker.IsByteRangeSafeForDirectWrite(0x8000, 0x1000, 6, true, 256));

  // Non-coherent atom alignment safety: if atom_size > unit_size (e.g. 8192B
  // atom vs 4096B unit), a 4KB unit cannot safely flush without touching an
  // adjacent 4KB subpage in the same non-coherent atom -> reject to staging.
  ZC_CHECK(!tracker.IsByteRangeSafeForDirectWrite(0x0000, 0x1000, 6,
                                                  /*coherent=*/false,
                                                  /*atom_size=*/8192));
  // When atom_size <= 4096 (normal hardware: 64..256B), 4KB unit is aligned.
  ZC_CHECK(tracker.IsByteRangeSafeForDirectWrite(0x0000, 0x1000, 6,
                                                 /*coherent=*/false,
                                                 /*atom_size=*/256));

  // Span partitioning: mixed safe and in-flight 4KB units in [0, 6)
  // Mark units 2 and 3 in-flight at submission 10, completed == 9.
  tracker.MarkByteRangeUsed(0x2000, 0x2000, /*submission=*/10);
  std::vector<std::pair<uint32_t, uint32_t>> zero_copy_spans;
  std::vector<std::pair<uint32_t, uint32_t>> staging_spans;
  PartitionZeroCopySpans(tracker, /*span_start=*/0, /*span_len=*/6,
                         /*unit_size_log2=*/12, /*completed_submission=*/9,
                         /*coherent=*/false, /*atom_size=*/256, zero_copy_spans,
                         staging_spans);
  // Expect zero-copy: [0, 2) and [4, 2); staging: [2, 2).
  ZC_CHECK(zero_copy_spans.size() == 2u);
  ZC_CHECK(zero_copy_spans[0] == std::make_pair(0u, 2u));
  ZC_CHECK(zero_copy_spans[1] == std::make_pair(4u, 2u));
  ZC_CHECK(staging_spans.size() == 1u);
  ZC_CHECK(staging_spans[0] == std::make_pair(2u, 2u));
}

void TestCoalesceAlignedFlushRanges() {
  using xe::gpu::vulkan::ByteRange;
  using xe::gpu::vulkan::CoalesceAlignedFlushRanges;

  constexpr uint64_t kBufSize = 512ULL * 1024 * 1024;

  // Empty input -> empty output.
  ZC_CHECK(CoalesceAlignedFlushRanges({}, 256, kBufSize).empty());

  // Single unaligned range aligned outward to atom_size=256.
  {
    const auto out =
        CoalesceAlignedFlushRanges({{300, 100}}, /*atom_size=*/256, kBufSize);
    ZC_CHECK(out.size() == 1u);
    ZC_CHECK(out[0].offset == 256u);
    ZC_CHECK(out[0].size == 256u);
  }

  // Two close ranges sharing or abutting an atom coalesce into one range;
  // a distant range stays separate (strictly written bytes, NEVER global).
  {
    const std::vector<ByteRange> in = {
        {0x4000, 0x1000},  // 16KB..20KB
        {0x5000, 0x1000},  // 20KB..24KB (adjacent -> coalesces with above!)
        {0x20000, 0x1000}, // 128KB..132KB (disjoint -> stays separate!)
        {0x80, 50},        // unsorted earlier range -> aligned [0, 256)
        {0x120, 100},      // adjacent atom -> aligned [256, 512), merges into [0, 512)
        {0x3000, 0},       // zero-length -> ignored
    };
    const auto out = CoalesceAlignedFlushRanges(in, /*atom_size=*/256, kBufSize);
    ZC_CHECK(out.size() == 3u);
    ZC_CHECK(out[0].offset == 0u && out[0].size == 512u);
    ZC_CHECK(out[1].offset == 0x4000u && out[1].size == 0x2000u);
    ZC_CHECK(out[2].offset == 0x20000u && out[2].size == 0x1000u);
    // Verify total flushed bytes << kBufSize (never global flush).
    uint64_t total = 0;
    for (const auto& r : out) {
      ZC_CHECK(r.offset % 256 == 0);
      ZC_CHECK(r.size % 256 == 0);
      ZC_CHECK(r.offset + r.size <= kBufSize);
      total += r.size;
    }
    ZC_CHECK(total == 512u + 0x2000u + 0x1000u);
  }

  // Clamp at buffer end and handle atom_size == 0 gracefully.
  {
    const auto clamped = CoalesceAlignedFlushRanges(
        {{kBufSize - 100, 500}}, /*atom_size=*/256, kBufSize);
    ZC_CHECK(clamped.size() == 1u);
    ZC_CHECK(clamped[0].offset == kBufSize - 256);
    ZC_CHECK(clamped[0].size == 256u);

    const auto zero_atom = CoalesceAlignedFlushRanges(
        {{123, 45}}, /*atom_size=*/0, kBufSize);
    ZC_CHECK(zero_atom.size() == 1u);
    ZC_CHECK(zero_atom[0].offset == 123u && zero_atom[0].size == 45u);
  }
}

void TestSubpage4KBTracker() {
  using xe::gpu::SubpageTracker;

  // 4 KB host page -> subpage tracking stays inactive (no-op).
  {
    SubpageTracker t4k;
    ZC_CHECK(!t4k.Initialize(/*buffer_size_log2=*/20, /*host_page_size_log2=*/12));
    ZC_CHECK(!t4k.active());
  }

  // 16 KB host page -> 4 logical 4 KB subpages per host page (4-bit mask 0x0F).
  SubpageTracker t;
  ZC_CHECK(t.Initialize(/*buffer_size_log2=*/20, /*host_page_size_log2=*/14));
  ZC_CHECK(t.active());
  ZC_CHECK(t.subpages_per_host_page() == 4u);
  ZC_CHECK(t.full_host_page_mask() == 0x0Fu);

  // Initially all subpages dirty.
  ZC_CHECK(!t.IsRangeValid(0x0000, 0x1000));
  ZC_CHECK(!t.IsHostPageFullyValid(0));

  // Validate ONLY subpage 0 (4 KB) of host page 0 (16 KB).
  t.MarkRangeValid(0x0000, 0x1000, /*written_by_gpu=*/false);
  ZC_CHECK(t.valid_mask(0) == 0x01u);
  ZC_CHECK(!t.IsHostPageFullyValid(0));
  ZC_CHECK(t.IsRangeValid(0x0000, 0x1000));
  ZC_CHECK(!t.IsRangeValid(0x1000, 0x1000));
  ZC_CHECK(!t.IsRangeValid(0x0000, 0x4000));

  // Requesting [0x0000, 0x1000) again -> 0 dirty subpages.
  std::vector<std::pair<uint32_t, uint32_t>> dirty;
  t.CollectDirtySubpageRanges(0x0000, 0x1000, dirty);
  ZC_CHECK(dirty.empty());

  // Requesting whole 16 KB page 0 -> only subpages [1, 3) (12 KB) dirty.
  t.CollectDirtySubpageRanges(0x0000, 0x4000, dirty);
  ZC_CHECK(dirty.size() == 1u);
  ZC_CHECK(dirty[0] == std::make_pair(1u, 3u));

  // Validate subpage 2 -> mask becomes 0b0101; dirty ranges in page 0 are
  // [1, 1) and [3, 1).
  t.MarkRangeValid(0x2000, 0x1000, false);
  ZC_CHECK(t.valid_mask(0) == 0x05u);
  t.CollectDirtySubpageRanges(0x0000, 0x4000, dirty);
  ZC_CHECK(dirty.size() == 2u);
  ZC_CHECK(dirty[0] == std::make_pair(1u, 1u));
  ZC_CHECK(dirty[1] == std::make_pair(3u, 1u));

  // Validate subpages 1 and 3 -> full 0x0F -> host page 0 is now fully valid.
  t.MarkRangeValid(0x1000, 0x1000, false);
  t.MarkRangeValid(0x3000, 0x1000, false);
  ZC_CHECK(t.valid_mask(0) == 0x0Fu);
  ZC_CHECK(t.IsHostPageFullyValid(0));
  ZC_CHECK(t.IsRangeValid(0x0000, 0x4000));

  // Store 4 KB upload hashes for all 4 subpages of host page 0.
  std::array<uint64_t, 4> guest_hashes = {101, 202, 303, 404};
  for (uint32_t sp = 0; sp < 4; ++sp) {
    t.SetSubpageUploadHash(sp, guest_hashes[sp]);
  }
  ZC_CHECK(t.hash_valid_mask(0) == 0x0Fu);

  // Simulate a 16 KB hardware writewatch fault on host page 0:
  // Guest CPU modified ONLY subpage 2 (hash changes 303 -> 999), while OS
  // faulted the whole 16 KB page.
  t.InvalidateHostPages(0, 0, /*mark_submission_invalidated=*/true);
  ZC_CHECK(t.valid_mask(0) == 0x00u);
  ZC_CHECK(t.hash_valid_mask(0) == 0x0Fu);  // Non-GPU hashes preserved!
  ZC_CHECK(t.AnySubpageRangeInvalidatedSinceSubmissionOpen(
      std::array<std::pair<uint32_t, uint32_t>, 1>{{{2, 1}}}.data(), 1));

  guest_hashes[2] = 999;

  // Collect dirty subpages for whole 16 KB page 0, then filter unchanged via
  // 4 KB hashes -> ONLY subpage 2 (4 KB) needs upload, subpages 0,1,3 are
  // restored valid with ZERO upload (4x reduction in upload amplification!).
  t.CollectDirtySubpageRanges(0x0000, 0x4000, dirty);
  ZC_CHECK(dirty.size() == 1u && dirty[0] == std::make_pair(0u, 4u));
  std::vector<std::pair<uint32_t, uint32_t>> filtered;
  std::vector<uint32_t> became_fully_valid;
  t.FilterUnchangedDirtySubpages(
      dirty,
      [&](uint32_t sp, uint64_t& out_hash) -> bool {
        if (sp >= 4) return false;
        out_hash = guest_hashes[sp];
        return true;
      },
      filtered, became_fully_valid);
  ZC_CHECK(filtered.size() == 1u);
  ZC_CHECK(filtered[0] == std::make_pair(2u, 1u));
  ZC_CHECK(t.valid_mask(0) == 0x0Bu);  // Subpages 0, 1, 3 restored valid!
  ZC_CHECK(became_fully_valid.empty());

  // Upload subpage 2 and store its new hash -> host page 0 fully valid again.
  t.MarkRangeValid(0x2000, 0x1000, false);
  t.SetSubpageUploadHash(2, 999);
  ZC_CHECK(t.IsHostPageFullyValid(0));

  // GPU write to subpage 1 only: clears subpage 1's CPU hash, marks only
  // subpage 1 GPU-written (subpage 0 remains non-GPU-written).
  t.MarkRangeValid(0x1000, 0x1000, /*written_by_gpu=*/true);
  ZC_CHECK(t.IsRangeGpuWritten(0x1000, 0x1000));
  ZC_CHECK(!t.IsRangeGpuWritten(0x0000, 0x1000));
  ZC_CHECK((t.hash_valid_mask(0) & (1u << 1)) == 0u);
  ZC_CHECK((t.hash_valid_mask(0) & (1u << 0)) != 0u);

  // SetPageValidFromGpuWritten keeps ONLY subpage 1 valid.
  t.OnGpuSubmissionOpened();
  ZC_CHECK(!t.AnySubpageRangeInvalidatedSinceSubmissionOpen(
      std::array<std::pair<uint32_t, uint32_t>, 1>{{{0, 4}}}.data(), 1));
  t.SetHostPageValidFromGpuWritten(0);
  ZC_CHECK(t.valid_mask(0) == 0x02u);
  ZC_CHECK(t.gpu_written_mask(0) == 0x02u);
  ZC_CHECK(!t.IsHostPageFullyValid(0));
  // Subpage 0 lost validity -> marked invalidated in submission, subpage 1 did not.
  ZC_CHECK(t.AnySubpageRangeInvalidatedSinceSubmissionOpen(
      std::array<std::pair<uint32_t, uint32_t>, 1>{{{0, 1}}}.data(), 1));
  ZC_CHECK(!t.AnySubpageRangeInvalidatedSinceSubmissionOpen(
      std::array<std::pair<uint32_t, uint32_t>, 1>{{{1, 1}}}.data(), 1));

  // Invalidating host page 0 clears GPU-written subpage 1's hash too.
  t.InvalidateHostPages(0, 0, true);
  ZC_CHECK(t.valid_mask(0) == 0x00u);
  ZC_CHECK(t.gpu_written_mask(0) == 0x00u);
  ZC_CHECK((t.hash_valid_mask(0) & (1u << 1)) == 0u);
}

void TestSourceContracts(const fs::path& cpp_root) {
  const fs::path gpu_dir = cpp_root / "xenia" / "src" / "xenia" / "gpu";
  const fs::path vk_dir = gpu_dir / "vulkan";

  // 1. VulkanSharedMemory zero-copy upload contracts.
  const std::string vk_shmem_h =
      ReadSourceText(vk_dir / "vulkan_shared_memory.h");
  const std::string vk_shmem_cc =
      ReadSourceText(vk_dir / "vulkan_shared_memory.cc");
  ZC_CHECK(vk_shmem_cc.find("vulkan_shared_memory_zero_copy_upload, false") !=
           std::string::npos);
  ZC_CHECK(vk_shmem_cc.find("UPDATE_from_bool(vulkan_shared_memory_zero_copy_upload") ==
           std::string::npos);
  ZC_CHECK(vk_shmem_cc.find("debug.xendroid.zero_copy_upload") !=
           std::string::npos);
  ZC_CHECK(vk_shmem_cc.find("vkFlushMappedMemoryRanges") != std::string::npos);
  ZC_CHECK(vk_shmem_cc.find("nonCoherentAtomSize") != std::string::npos);
  ZC_CHECK(vk_shmem_h.find("ZeroCopyPageSubmissionTracker") !=
           std::string::npos);
  // Ensure vkFlushMappedMemoryRanges never uses VK_WHOLE_SIZE.
  const size_t flush_pos = vk_shmem_cc.find("vkFlushMappedMemoryRanges");
  ZC_CHECK(flush_pos != std::string::npos);
  const std::string flush_window =
      vk_shmem_cc.substr(flush_pos > 800 ? flush_pos - 800 : 0, 1000);
  ZC_CHECK(flush_window.find("VK_WHOLE_SIZE") == std::string::npos);

  // 2. SharedMemory 4 KB subpage tracking contracts.
  const std::string shmem_h = ReadSourceText(gpu_dir / "shared_memory.h");
  const std::string shmem_cc = ReadSourceText(gpu_dir / "shared_memory.cc");
  ZC_CHECK(shmem_h.find("class SubpageTracker") != std::string::npos);
  ZC_CHECK(shmem_cc.find("shared_memory_subpage_tracking, false") !=
           std::string::npos);
  ZC_CHECK(shmem_cc.find("UPDATE_from_bool(shared_memory_subpage_tracking") ==
           std::string::npos);

  // 3. CommandProcessor GPU Commands thread nice cvar contracts (#55).
  const std::string cp_cc = ReadSourceText(gpu_dir / "command_processor.cc");
  ZC_CHECK(cp_cc.find("DEFINE_int32(\n    gpu_commands_thread_nice, 0,") !=
               std::string::npos ||
           cp_cc.find("gpu_commands_thread_nice, 0,") != std::string::npos);
  ZC_CHECK(cp_cc.find("UPDATE_from_int32(gpu_commands_thread_nice") ==
           std::string::npos);
  ZC_CHECK(cp_cc.find("setpriority(PRIO_PROCESS, 0,") != std::string::npos);
}

}  // namespace

int main(int argc, char** argv) {
  TestZeroCopyPropertyParsing();
  TestZeroCopySubmissionHazardTracking();
  TestCoalesceAlignedFlushRanges();
  TestSubpage4KBTracker();
  if (argc >= 2 && argv[1] && argv[1][0] != '\0') {
    TestSourceContracts(fs::path(argv[1]));
  } else {
    std::fprintf(stderr, "Usage: gpu_zerocopy_upload_test <cpp_root>\n");
    return 1;
  }
  std::printf("gpu_zerocopy_upload_test: PASS\n");
  return 0;
}
