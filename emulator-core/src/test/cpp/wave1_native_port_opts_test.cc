#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>

#include "xenia/kernel/xboxkrnl/critical_section_spin.h"
#include "xenia/ui/vulkan/vulkan_present_divisor.h"

namespace {

#define WAVE1_CHECK(cond)                                                 \
  do {                                                                    \
    if (!(cond)) {                                                        \
      std::fprintf(stderr, "WAVE1_OPT check failed (%s:%d): %s\n",        \
                   __FILE__, __LINE__, #cond);                            \
      std::abort();                                                       \
    }                                                                     \
  } while (false)

using xe::kernel::xboxkrnl::ComputeCriticalSectionSpinLimit;
using xe::kernel::xboxkrnl::kDefaultCriticalSectionMaxSpin;
using xe::ui::vulkan::ComputeAndroidPresentExtent;
using xe::ui::vulkan::PresentExtent;

std::string ReadSourceText(const std::filesystem::path& path) {
  std::ifstream in(path);
  WAVE1_CHECK(in.good());
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

void TestCriticalSectionSpinCap(const std::filesystem::path& cpp_dir) {
  WAVE1_CHECK(kDefaultCriticalSectionMaxSpin == 64u);
  WAVE1_CHECK(ComputeCriticalSectionSpinLimit(0u, 64u) == 0u);
  WAVE1_CHECK(ComputeCriticalSectionSpinLimit(50u, 64u) == 50u);
  WAVE1_CHECK(ComputeCriticalSectionSpinLimit(64u, 64u) == 64u);
  WAVE1_CHECK(ComputeCriticalSectionSpinLimit(65280u, 64u) == 64u);
  WAVE1_CHECK(ComputeCriticalSectionSpinLimit(65280u, 0u) == 65280u);
  WAVE1_CHECK(ComputeCriticalSectionSpinLimit(256u, 128u) == 128u);

  const std::string rtl_src = ReadSourceText(
      cpp_dir / "xenia/src/xenia/kernel/xboxkrnl/xboxkrnl_rtl.cc");
  const std::string atomic_src =
      ReadSourceText(cpp_dir / "xenia/src/xenia/base/atomic.h");
  WAVE1_CHECK(rtl_src.find("critical_section_max_spin") != std::string::npos);
  WAVE1_CHECK(rtl_src.find("ComputeCriticalSectionSpinLimit") !=
              std::string::npos);
  WAVE1_CHECK(rtl_src.find("xe::atomic_load_relaxed(&cs->lock_count)") !=
              std::string::npos);
  WAVE1_CHECK(atomic_src.find("atomic_load_relaxed") != std::string::npos);
  WAVE1_CHECK(atomic_src.find("__atomic_load_n(value, __ATOMIC_RELAXED)") !=
              std::string::npos);
  WAVE1_CHECK(rtl_src.find("PauseCriticalSectionSpin();") != std::string::npos);
}

void TestAndroidPresentDivisor(const std::filesystem::path& cpp_dir) {
  uint32_t chosen_divisor = 0;

  // Tab S10 Ultra 2960x1848 @ 1x auto -> 1480x924 (divisor 2).
  PresentExtent tab_1x = ComputeAndroidPresentExtent(
      2960, 1848, 1, 1, 16384, 16384, 1, 1, 0, &chosen_divisor);
  WAVE1_CHECK(tab_1x.width == 1480 && tab_1x.height == 924);
  WAVE1_CHECK(chosen_divisor == 2);

  // Smartphone QHD+ 3120x1440 @ 1x auto -> 1560x720 (divisor 2).
  PresentExtent qhd_1x = ComputeAndroidPresentExtent(
      3120, 1440, 1, 1, 16384, 16384, 1, 1, 0, &chosen_divisor);
  WAVE1_CHECK(qhd_1x.width == 1560 && qhd_1x.height == 720);
  WAVE1_CHECK(chosen_divisor == 2);

  // Pre-rotated portrait 1440x3120 @ 1x auto -> 720x1560 (divisor 2).
  PresentExtent rot_1x = ComputeAndroidPresentExtent(
      1440, 3120, 1, 1, 16384, 16384, 1, 1, 0, &chosen_divisor);
  WAVE1_CHECK(rot_1x.width == 720 && rot_1x.height == 1560);
  WAVE1_CHECK(chosen_divisor == 2);

  // Smartphone 1080p 2400x1080 @ 1x auto -> stays native (divisor 1).
  PresentExtent fhd_1x = ComputeAndroidPresentExtent(
      2400, 1080, 1, 1, 16384, 16384, 1, 1, 0, &chosen_divisor);
  WAVE1_CHECK(fhd_1x.width == 2400 && fhd_1x.height == 1080);
  WAVE1_CHECK(chosen_divisor == 1);

  // Tablet 2960x1848 @ 2x auto -> stays native (divisor 1).
  PresentExtent tab_2x = ComputeAndroidPresentExtent(
      2960, 1848, 1, 1, 16384, 16384, 2, 2, 0, &chosen_divisor);
  WAVE1_CHECK(tab_2x.width == 2960 && tab_2x.height == 1848);
  WAVE1_CHECK(chosen_divisor == 1);

  // Explicit native mode (divisor_setting = 1) -> stays 3120x1440.
  PresentExtent native_qhd = ComputeAndroidPresentExtent(
      3120, 1440, 1, 1, 16384, 16384, 1, 1, 1, &chosen_divisor);
  WAVE1_CHECK(native_qhd.width == 3120 && native_qhd.height == 1440);
  WAVE1_CHECK(chosen_divisor == 1);

  // Surface minImageExtent == panel forces native size and reports divisor 1.
  PresentExtent clamped = ComputeAndroidPresentExtent(
      2960, 1848, 2960, 1848, 2960, 1848, 1, 1, 0, &chosen_divisor);
  WAVE1_CHECK(clamped.width == 2960 && clamped.height == 1848);
  WAVE1_CHECK(chosen_divisor == 1);

  const std::string presenter_src = ReadSourceText(
      cpp_dir / "xenia/src/xenia/ui/vulkan/vulkan_presenter.cc");
  WAVE1_CHECK(presenter_src.find("vulkan_android_present_divisor") !=
              std::string::npos);
  WAVE1_CHECK(presenter_src.find("ReadIntCvarOrDefault(\"draw_resolution_scale_x\"") !=
              std::string::npos);
  WAVE1_CHECK(presenter_src.find("DECLARE_int32(draw_resolution_scale_x)") ==
              std::string::npos);
  WAVE1_CHECK(presenter_src.find("!IsZeroFGRequested() && !xe::IsZeroFGMainSurfaceActive()") !=
              std::string::npos);
  WAVE1_CHECK(presenter_src.find("ComputeAndroidPresentExtent(") !=
              std::string::npos);
}

void TestHugePagesWiring(const std::filesystem::path& cpp_dir) {
  const std::string mem_h =
      ReadSourceText(cpp_dir / "xenia/src/xenia/base/memory.h");
  const std::string mem_posix =
      ReadSourceText(cpp_dir / "xenia/src/xenia/base/memory_posix.cc");
  const std::string code_cache =
      ReadSourceText(cpp_dir / "xenia/src/xenia/cpu/backend/code_cache_base.h");
  const std::string memory_cc =
      ReadSourceText(cpp_dir / "xenia/src/xenia/memory.cc");

  WAVE1_CHECK(mem_h.find("void AdviseHugePages(void* base_address, size_t length);") !=
              std::string::npos);
  WAVE1_CHECK(mem_posix.find("DEFINE_bool(\n    memory_huge_pages, true") !=
                  std::string::npos ||
              mem_posix.find("memory_huge_pages, true") != std::string::npos);
  WAVE1_CHECK(mem_posix.find("MADV_HUGEPAGE") != std::string::npos);

  const auto first_advise =
      code_cache.find("xe::memory::AdviseHugePages(generated_code_execute_base_");
  WAVE1_CHECK(first_advise != std::string::npos);
  const auto second_advise = code_cache.find(
      "xe::memory::AdviseHugePages(generated_code_execute_base_", first_advise + 1);
  WAVE1_CHECK(second_advise != std::string::npos);
  WAVE1_CHECK(memory_cc.find("xe::memory::AdviseHugePages(") != std::string::npos);
}

void TestTextureLodBiasAndPipelineCheckpoints(
    const std::filesystem::path& cpp_dir) {
  const std::string flags_h =
      ReadSourceText(cpp_dir / "xenia/src/xenia/gpu/gpu_flags.h");
  const std::string flags_cc =
      ReadSourceText(cpp_dir / "xenia/src/xenia/gpu/gpu_flags.cc");
  const std::string fetch_cc = ReadSourceText(
      cpp_dir / "xenia/src/xenia/gpu/spirv_shader_translator_fetch.cc");
  const std::string pcache_h = ReadSourceText(
      cpp_dir / "xenia/src/xenia/gpu/vulkan/vulkan_pipeline_cache.h");
  const std::string pcache_cc = ReadSourceText(
      cpp_dir / "xenia/src/xenia/gpu/vulkan/vulkan_pipeline_cache.cc");

  WAVE1_CHECK(flags_h.find("DECLARE_int32(gpu_texture_lod_bias);") !=
              std::string::npos);
  WAVE1_CHECK(flags_cc.find("gpu_texture_lod_bias, 0") != std::string::npos);

  const auto lod_pos = fetch_cc.find("cvars::gpu_texture_lod_bias");
  WAVE1_CHECK(lod_pos != std::string::npos);
  const auto computed_guard = fetch_cc.rfind("if (use_computed_lod)", lod_pos);
  WAVE1_CHECK(computed_guard != std::string::npos &&
              lod_pos - computed_guard < 300);

  WAVE1_CHECK(pcache_cc.find(".lodbias{}") != std::string::npos);
  WAVE1_CHECK(pcache_h.find("vk_pipeline_cache_save_mutex_") !=
              std::string::npos);
  WAVE1_CHECK(pcache_h.find("kVkPipelineCachePrecreateSaveIntervalMs = 5000") !=
              std::string::npos);
  WAVE1_CHECK(pcache_h.find("kVkPipelineCachePrecreateSaveBatch = 256") !=
              std::string::npos);
  WAVE1_CHECK(pcache_cc.find("SaveVkPipelineCache();") != std::string::npos);
  WAVE1_CHECK(pcache_cc.find("WaitResult::kTimeout") != std::string::npos);
}

bool ReferenceSradxCA(int64_t rt, uint8_t rb) {
  const uint8_t sh = rb & 0x7F;
  const uint8_t clamp_sh = sh > 63 ? 63 : sh;
  const int64_t v = rt >> clamp_sh;
  const uint64_t uv = static_cast<uint64_t>(v);
  const int64_t shifted_back = static_cast<int64_t>(uv << clamp_sh);
  const bool shifted_out = ((sh & 0x40) != 0) || (shifted_back != rt);
  return (rt < 0) && shifted_out;
}

bool ExactIsaSradxCA(int64_t rt, uint8_t rb) {
  const uint8_t sh = rb & 0x7F;
  if (rt >= 0 || sh == 0) {
    return false;
  }
  if (sh >= 64) {
    return true;
  }
  const uint64_t lost_mask = (uint64_t(1) << sh) - 1u;
  return (static_cast<uint64_t>(rt) & lost_mask) != 0u;
}

bool ReferenceSrawxCA(int32_t rt, uint8_t rb) {
  const uint8_t sh = rb & 0x3F;
  const uint8_t clamp_sh = sh > 31 ? 31 : sh;
  const int32_t v = rt >> clamp_sh;
  const uint32_t uv = static_cast<uint32_t>(v);
  const int32_t shifted_back = static_cast<int32_t>(uv << clamp_sh);
  const bool shifted_out = ((sh & 0x20) != 0) || (shifted_back != rt);
  return (rt < 0) && shifted_out;
}

bool ExactIsaSrawxCA(int32_t rt, uint8_t rb) {
  const uint8_t sh = rb & 0x3F;
  if (rt >= 0 || sh == 0) {
    return false;
  }
  if (sh >= 32) {
    return true;
  }
  const uint32_t lost_mask = (uint32_t(1) << sh) - 1u;
  return (static_cast<uint32_t>(rt) & lost_mask) != 0u;
}

void TestPpcShiftCarryFixes(const std::filesystem::path& cpp_dir) {
  const int64_t test_i64[] = {
      0,
      1,
      -1,
      std::numeric_limits<int64_t>::min(),
      std::numeric_limits<int64_t>::min() + 1,
      std::numeric_limits<int64_t>::max(),
      static_cast<int64_t>(0x8000000000000000ULL | 0x4ULL),
  };
  const uint8_t shifts_64[] = {0, 1, 2, 3, 62, 63, 64, 65, 127};
  for (int64_t rt : test_i64) {
    for (uint8_t rb : shifts_64) {
      WAVE1_CHECK(ReferenceSradxCA(rt, rb) == ExactIsaSradxCA(rt, rb));
    }
  }
  WAVE1_CHECK(ReferenceSradxCA(std::numeric_limits<int64_t>::min(), 64));
  WAVE1_CHECK(ReferenceSradxCA(std::numeric_limits<int64_t>::min(), 127));
  WAVE1_CHECK(!ReferenceSradxCA(std::numeric_limits<int64_t>::min(), 63));
  WAVE1_CHECK(!ReferenceSradxCA(std::numeric_limits<int64_t>::min(), 0));

  const int32_t test_i32[] = {
      0,
      1,
      -1,
      std::numeric_limits<int32_t>::min(),
      std::numeric_limits<int32_t>::min() + 1,
      std::numeric_limits<int32_t>::max(),
      static_cast<int32_t>(0x80000004U),
  };
  const uint8_t shifts_32[] = {0, 1, 2, 3, 30, 31, 32, 33, 63};
  for (int32_t rt : test_i32) {
    for (uint8_t rb : shifts_32) {
      WAVE1_CHECK(ReferenceSrawxCA(rt, rb) == ExactIsaSrawxCA(rt, rb));
    }
  }
  WAVE1_CHECK(ReferenceSrawxCA(std::numeric_limits<int32_t>::min(), 32));
  WAVE1_CHECK(ReferenceSrawxCA(std::numeric_limits<int32_t>::min(), 63));
  WAVE1_CHECK(!ReferenceSrawxCA(std::numeric_limits<int32_t>::min(), 31));
  WAVE1_CHECK(!ReferenceSrawxCA(std::numeric_limits<int32_t>::min(), 0));

  const std::string alu_src =
      ReadSourceText(cpp_dir / "xenia/src/xenia/cpu/ppc/ppc_emit_alu.cc");
  WAVE1_CHECK(alu_src.find("LoadConstantInt8(0x40)") != std::string::npos);
  WAVE1_CHECK(alu_src.find("LoadConstantInt8(0x20)") != std::string::npos);

  const std::string altivec_src =
      ReadSourceText(cpp_dir / "xenia/src/xenia/cpu/ppc/ppc_emit_altivec.cc");
  WAVE1_CHECK(altivec_src.find("A NaN in either operand is out of bounds both ways") !=
              std::string::npos);
  WAVE1_CHECK(altivec_src.find("PACK_TYPE_16_IN_32 | PACK_TYPE_IN_UNSIGNED") !=
              std::string::npos);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "Usage: %s <emulator-core/src/main/cpp>\n", argv[0]);
    return 2;
  }
  const std::filesystem::path cpp_dir(argv[1]);
  TestCriticalSectionSpinCap(cpp_dir);
  TestAndroidPresentDivisor(cpp_dir);
  TestHugePagesWiring(cpp_dir);
  TestTextureLodBiasAndPipelineCheckpoints(cpp_dir);
  TestPpcShiftCarryFixes(cpp_dir);
  std::puts("wave1_native_port_opts_test: PASS");
  return 0;
}
