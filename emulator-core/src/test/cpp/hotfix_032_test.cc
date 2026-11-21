#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "xenia/gpu/vulkan/vulkan_driver_quirks.h"

namespace {

#define HOTFIX_CHECK(cond)                                                 \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "HOTFIX_032 check failed (%s:%d): %s\n",       \
                   __FILE__, __LINE__, #cond);                             \
      std::abort();                                                        \
    }                                                                      \
  } while (false)

using xe::gpu::vulkan::IsNonTurnipMobileTiler;
using xe::gpu::vulkan::IsSamsungXclipseProprietary;
using xe::gpu::vulkan::kVendorIdArm;
using xe::gpu::vulkan::kVendorIdSamsung;
using xe::gpu::vulkan::kVkDriverIdArmProprietary;
using xe::gpu::vulkan::kVkDriverIdMesaPanvk;
using xe::gpu::vulkan::kVkDriverIdMesaRadv;
using xe::gpu::vulkan::kVkDriverIdMesaTurnip;
using xe::gpu::vulkan::kVkDriverIdSamsungProprietary;

void TestSamsungXclipseDetection() {
  // Samsung proprietary driver on Xclipse 940 (#81).
  HOTFIX_CHECK(IsSamsungXclipseProprietary(kVkDriverIdSamsungProprietary,
                                           kVendorIdSamsung,
                                           "Samsung Xclipse 940"));

  // Open-source Mesa RADV on Xclipse 940 keeps Samsung's vendorID (0x144D)
  // and must never be forced through Samsung's proprietary transcode (#81).
  HOTFIX_CHECK(!IsSamsungXclipseProprietary(
      kVkDriverIdMesaRadv, kVendorIdSamsung,
      "AMD Radeon Graphics (RADV MAGELLAN CARL)"));
  HOTFIX_CHECK(!IsSamsungXclipseProprietary(kVkDriverIdMesaRadv,
                                            kVendorIdSamsung, "Xclipse RADV"));

  // Other non-Samsung known drivers with vendorID 0 or other vendorIDs.
  HOTFIX_CHECK(!IsSamsungXclipseProprietary(kVkDriverIdMesaTurnip, 0x5143,
                                            "Turnip Adreno 740"));
  HOTFIX_CHECK(!IsSamsungXclipseProprietary(kVkDriverIdArmProprietary,
                                            kVendorIdArm, "Mali-G77 MC9"));
  HOTFIX_CHECK(!IsSamsungXclipseProprietary(kVkDriverIdMesaPanvk, kVendorIdArm,
                                            "Mali-G77 MC9"));

  // Unknown driverID (0) falls back to vendorID / deviceName heuristics.
  HOTFIX_CHECK(IsSamsungXclipseProprietary(0, kVendorIdSamsung, "Samsung GPU"));
  HOTFIX_CHECK(IsSamsungXclipseProprietary(0, 0x1002, "Samsung Xclipse 920"));
  HOTFIX_CHECK(!IsSamsungXclipseProprietary(0, 0x5143, "Adreno (TM) 740"));
  HOTFIX_CHECK(!IsSamsungXclipseProprietary(0, 0x5143, nullptr));
}

void TestNonTurnipMobileTiler() {
  // Proprietary Samsung, Arm Mali, and PanVK enable mobile tiler AFBC opts.
  HOTFIX_CHECK(IsNonTurnipMobileTiler(kVkDriverIdSamsungProprietary,
                                      kVendorIdSamsung));
  HOTFIX_CHECK(IsNonTurnipMobileTiler(kVkDriverIdArmProprietary, kVendorIdArm));
  HOTFIX_CHECK(IsNonTurnipMobileTiler(kVkDriverIdMesaPanvk, kVendorIdArm));
  HOTFIX_CHECK(IsNonTurnipMobileTiler(0, kVendorIdArm));
  HOTFIX_CHECK(IsNonTurnipMobileTiler(0, kVendorIdSamsung));

  // Turnip and Mesa RADV are excluded even when vendorID matches Samsung.
  HOTFIX_CHECK(!IsNonTurnipMobileTiler(kVkDriverIdMesaTurnip, 0x5143));
  HOTFIX_CHECK(!IsNonTurnipMobileTiler(kVkDriverIdMesaRadv, kVendorIdSamsung));
  HOTFIX_CHECK(!IsNonTurnipMobileTiler(kVkDriverIdMesaRadv, 0x1002));
}

void TestArmv8BaselineInCmake(const std::filesystem::path& cpp_dir) {
  const auto cmake_path = cpp_dir / "CMakeLists.txt";
  std::ifstream in(cmake_path);
  HOTFIX_CHECK(in.good());
  std::ostringstream ss;
  ss << in.rdbuf();
  const std::string content = ss.str();

  // #122: baseline must target ARMv8.0-A with runtime LSE outline atomics, not
  // ARMv8.2-A instructions that SIGILL on Snapdragon 680 / Cortex-A73/A53.
  HOTFIX_CHECK(content.find("add_compile_options(-march=armv8-a -moutline-atomics)") !=
               std::string::npos);
  HOTFIX_CHECK(content.find("-march=armv8.2") == std::string::npos);
}

}  // namespace

int main(int argc, char** argv) {
  TestSamsungXclipseDetection();
  TestNonTurnipMobileTiler();
  if (argc >= 2) {
    TestArmv8BaselineInCmake(argv[1]);
  }
  std::puts("hotfix_032_test: PASS");
  return 0;
}
