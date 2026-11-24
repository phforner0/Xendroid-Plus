#ifndef XENIA_GPU_TRACE_BENCH_RUNNER_H_
#define XENIA_GPU_TRACE_BENCH_RUNNER_H_

#include <filesystem>

#include "xenia/base/cvar.h"

DECLARE_string(gpu_trace_bench);
DECLARE_uint32(gpu_trace_bench_loops);
DECLARE_string(gpu_trace_bench_arms);
DECLARE_uint32(gpu_trace_bench_block_loops);
DECLARE_string(gpu_trace_bench_run_id);
DECLARE_bool(gpu_trace_bench_guard);
DECLARE_double(gpu_trace_bench_expect_ms);
DECLARE_double(gpu_trace_bench_slow_x);
DECLARE_uint32(gpu_trace_bench_hang_ms);
DECLARE_bool(gpu_trace_bench_uncapped);

namespace xe {
namespace gpu {

class GraphicsSystem;

// Replays `cvars::gpu_trace_bench` (.xtr) deterministically over TracePlayer,
// switching live CVars in ABBA order with a lock-free hang/slow watchdog.
// Returns 0 on completion, non-zero on error.
int RunTraceBench(GraphicsSystem* graphics_system,
                  const std::filesystem::path& storage_root);

}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_TRACE_BENCH_RUNNER_H_
