#ifndef XENIA_KERNEL_HLE_CENSUS_H_
#define XENIA_KERNEL_HLE_CENSUS_H_

#include "xenia/base/cvar.h"

DECLARE_bool(hle_census);
DECLARE_uint32(hle_census_interval_ms);

namespace xe {
namespace cpu {
class ExportResolver;
}  // namespace cpu
namespace kernel {

// Wraps kernel export trampolines with relaxed atomic counters and starts a
// low-overhead census thread when `cvars::hle_census` is true. No-op when off.
void InstallHleCensus(xe::cpu::ExportResolver* resolver);
void ShutdownHleCensus();

}  // namespace kernel
}  // namespace xe

#endif  // XENIA_KERNEL_HLE_CENSUS_H_
