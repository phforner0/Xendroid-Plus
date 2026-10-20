// The XNet startup parameters (xenia/kernel/xam/xnet_startup_params.h): what
// XNet runs with for those a title passes to XNetStartup. Zero fields take
// xam's defaults, the others are clamped to its limits, and a struct of
// another size is ignored.

#include <array>
#include <cstdint>
#include <cstdio>

#include "xenia/kernel/xam/xnet_startup_params.h"

namespace {

int failures = 0;

void expect(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAILED: %s\n", what);
    ++failures;
  }
}

}  // namespace

int main() {
  using xe::kernel::xam::NormalizeStartupParams;
  using xe::kernel::xam::StartupParams;

  const StartupParams defaults{13, 0, 8, 32, 16, 16, 8, 32, 64, 2, 3, 8, 2};
  // Lost Planet 2 sets the size and leaves the rest to XNet.
  const uint8_t unset[13] = {13};
  expect(NormalizeStartupParams(unset) == defaults, "zero fields take the defaults");
  expect(NormalizeStartupParams({}) == defaults, "no parameters are the defaults");
  const uint8_t unknown_size[13] = {12, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
  expect(NormalizeStartupParams(unknown_size) == defaults,
         "a struct of another size is ignored");
  const uint8_t given[13] = {13, 0x0F, 255, 1, 100, 63, 255, 1, 16, 9, 9, 9, 9};
  expect(NormalizeStartupParams(given) ==
             StartupParams{13, 0x06, 232, 1, 63, 63, 255, 1, 16, 9, 9, 9, 9},
         "fields are clamped and the security flags cleared");
  std::array<uint8_t, 0x4C> extended{0x4C};
  extended[8] = 16;
  auto from_extended = defaults;
  from_extended[8] = 16;
  expect(NormalizeStartupParams(extended) == from_extended,
         "the extended struct starts with the same fields");

  if (failures) {
    std::fprintf(stderr, "xnet_startup_test: %d failed\n", failures);
    return 1;
  }
  std::printf("xnet_startup_test: ok\n");
  return 0;
}
