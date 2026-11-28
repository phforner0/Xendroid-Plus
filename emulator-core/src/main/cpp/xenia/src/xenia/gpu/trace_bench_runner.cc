#include "xenia/gpu/trace_bench_runner.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "../../../../xe_gpu_bench_logic.h"
#include "xenia/base/cvar.h"
#include "xenia/base/filesystem.h"
#include "xenia/base/frame_stats.h"
#include "xenia/base/logging.h"
#include "xenia/gpu/command_processor.h"
#include "xenia/gpu/gpu_flags.h"
#include "xenia/gpu/graphics_system.h"
#include "xenia/gpu/trace_player.h"
#include "xenia/ui/presenter.h"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wabsolute-value"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#endif
#include "third_party/tomlplusplus/toml.hpp"
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#pragma clang diagnostic ignored "-Wsign-compare"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wsign-compare"
#endif
#include "third_party/stb/stb_image_write.h"
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

DEFINE_string(gpu_trace_bench, "",
              "Path to a .xtr trace file to replay deterministically in a loop "
              "instead of launching a title (empty = off).",
              "GPU");
DEFINE_uint32(gpu_trace_bench_loops, 120,
              "Total replay iterations when no ABBA arms are configured.", "GPU");
DEFINE_string(gpu_trace_bench_arms, "",
              "ABBA live-CVar arms, e.g. \"base:cvar=false;opt:cvar=true\".",
              "GPU");
DEFINE_uint32(gpu_trace_bench_block_loops, 10,
              "Loops per ABBA block (loop 0 of each block is dropped as a "
              "transient).",
              "GPU");
DEFINE_string(gpu_trace_bench_run_id, "",
              "Run identifier echoed on XDROID-BENCH lines.", "GPU");
DEFINE_bool(gpu_trace_bench_guard, true,
            "Enable lock-free hang/slow self-abort watchdog (_exit(3)).",
            "GPU");
DEFINE_double(gpu_trace_bench_expect_ms, 0.0,
              "Expected wall ms/frame (0 disables slow-loop self-abort).",
              "GPU");
DEFINE_double(gpu_trace_bench_slow_x, 1.6,
              "Slow-abort threshold multiplier against gpu_trace_bench_expect_ms.",
              "GPU");
DEFINE_uint32(gpu_trace_bench_hang_ms, 4000,
              "Hang-abort timeout in ms per timed frame (60s during warmup).",
              "GPU");
DEFINE_bool(gpu_trace_bench_uncapped, true,
            "Disable framerate_limit, host present pacing, and guest refresh "
            "cap during trace benchmarking (timedemo mode).",
            "GPU");

namespace xe {
namespace gpu {
namespace {

int64_t NowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

bool ApplyBenchLiveCvar(const std::string& name, const std::string& raw_value) {
  if (!cvar::ConfigVars) return false;
  auto it = cvar::ConfigVars->find(name);
  if (it == cvar::ConfigVars->end() || !it->second) {
    XELOGW("XDROID-BENCH: unknown cvar '{}'", name);
    return false;
  }
  try {
    std::string literal = raw_value;
    if (raw_value != "true" && raw_value != "false") {
      char* end = nullptr;
      std::strtod(raw_value.c_str(), &end);
      const bool is_num = !raw_value.empty() && end && *end == '\0';
      if (!is_num && raw_value.front() != '"') {
        literal = "\"" + raw_value + "\"";
      }
    }
    toml::table tbl = toml::parse("v = " + literal);
    it->second->LoadGameConfigValue(tbl.get("v"));
    return true;
  } catch (const std::exception& e) {
    XELOGW("XDROID-BENCH: failed to parse {}={}: {}", name, raw_value, e.what());
    return false;
  }
}

void SaveLastGuestFramePng(GraphicsSystem* gs,
                           const std::filesystem::path& out) {
  ui::Presenter* presenter = gs ? gs->presenter() : nullptr;
  if (!presenter) return;
  ui::RawImage image;
  if (!presenter->CaptureGuestOutput(image)) {
    XELOGW("XDROID-BENCH: CaptureGuestOutput returned false");
    return;
  }
  FILE* handle = filesystem::OpenFile(out, "wb");
  if (!handle) return;
  auto cb = [](void* ctx, void* data, int size) {
    std::fwrite(data, 1, size_t(size), static_cast<FILE*>(ctx));
  };
  stbi_write_png_to_func(cb, handle, int(image.width), int(image.height), 4,
                         image.data.data(), int(image.stride));
  std::fclose(handle);
  XELOGI("XDROID-BENCH: captured {}x{} frame to {}", image.width, image.height,
         xe::path_to_utf8(out));
}

void WriteAbortJson(const std::filesystem::path& traces_dir, const char* reason,
                    const std::string& run_id, uint32_t frame_idx) {
  std::error_code ec;
  std::filesystem::create_directories(traces_dir, ec);
  FILE* f = filesystem::OpenFile(traces_dir / "abort.json", "w");
  if (!f) return;
  std::fprintf(
      f, "{\"reason\":\"%s\",\"run_id\":\"%s\",\"frame\":%u}\n",
      reason, run_id.c_str(), frame_idx);
  std::fclose(f);
}

}  // namespace

int RunTraceBench(GraphicsSystem* gs,
                  const std::filesystem::path& storage_root) {
  if (!gs || cvars::gpu_trace_bench.empty()) {
    return 1;
  }

  const std::filesystem::path trace_path(cvars::gpu_trace_bench);
  auto player = std::make_unique<TracePlayer>(gs);
  if (!player->Open(xe::path_to_utf8(trace_path))) {
    XELOGE("XDROID-BENCH: failed to open trace {}", xe::path_to_utf8(trace_path));
    return 2;
  }
  if (player->frame_count() <= 0) {
    XELOGE("XDROID-BENCH: trace has 0 frames");
    return 2;
  }

  if (cvars::gpu_trace_bench_uncapped) {
    SetFramerateLimit(0);
    SetGuestDisplayRefreshCap(false);
    ApplyBenchLiveCvar("vulkan_allow_present_mode_immediate", "true");
  }
  ApplyBenchLiveCvar("log_gpu_frame_time_breakdown", "true");

  const std::string run_id = cvars::gpu_trace_bench_run_id;
  const std::string run_suffix = run_id.empty() ? "" : (" run=" + run_id);

  // Lock-free watchdog shared state (atomics only).
  std::atomic<int64_t> frame_start_ns{0};
  std::atomic<bool> in_warmup{true};
  std::atomic<bool> finished{false};
  std::atomic<uint32_t> current_frame{0};

  xendroid::BenchWatchdogConfig wd_cfg;
  wd_cfg.expect_ms = cvars::gpu_trace_bench_expect_ms;
  wd_cfg.slow_multiplier = cvars::gpu_trace_bench_slow_x;
  wd_cfg.hang_ns = uint64_t(cvars::gpu_trace_bench_hang_ms) * 1'000'000ULL;

  const std::filesystem::path traces_dir = storage_root / "traces";
  std::thread watchdog;
  if (cvars::gpu_trace_bench_guard) {
    watchdog = std::thread([&]() {
      while (!finished.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (finished.load(std::memory_order_acquire)) break;
        const auto decision = xendroid::EvaluateWatchdogHang(
            NowNs(), frame_start_ns.load(std::memory_order_acquire),
            in_warmup.load(std::memory_order_relaxed), wd_cfg);
        if (decision == xendroid::BenchWatchdogVerdict::kHangAbort) {
          const uint32_t f = current_frame.load(std::memory_order_relaxed);
          XELOGE("XDROID-BENCH: abort reason=HANG frame={}{}", f, run_suffix);
          WriteAbortJson(traces_dir, "HANG", run_id, f);
          xe::FlushLog();
          std::_Exit(3);
        }
      }
    });
  }

  auto replay_once = [&](double& out_wall_ms, double& out_gpu_ms) {
    // Drain any prior GPU work so timestamps never bleed across loops or ABBA
    // arms.
    if (auto* cp = gs->command_processor()) {
      threading::Fence idle_fence;
      cp->CallInThread([cp, &idle_fence]() {
        cp->WaitUntilGpuIdleForBench();
        idle_fence.Signal();
      });
      idle_fence.Wait();
    }
    const uint64_t gpu_before =
        xe::internal::last_bench_gpu_exec_ns().load(std::memory_order_relaxed);
    frame_start_ns.store(NowNs(), std::memory_order_release);
    for (int fi = 0; fi < player->frame_count(); ++fi) {
      current_frame.store(uint32_t(fi), std::memory_order_relaxed);
      player->ReplayFrame(fi, false);
      player->WaitOnPlayback();
    }
    if (auto* cp = gs->command_processor()) {
      threading::Fence idle_fence;
      cp->CallInThread([cp, &idle_fence]() {
        cp->WaitUntilGpuIdleForBench();
        idle_fence.Signal();
      });
      idle_fence.Wait();
    }
    const int64_t end_ns = NowNs();
    const int64_t start_ns = frame_start_ns.exchange(0, std::memory_order_release);
    out_wall_ms = double(end_ns - start_ns) / 1e6;
    const uint64_t gpu_after =
        xe::internal::last_bench_gpu_exec_ns().load(std::memory_order_relaxed);
    out_gpu_ms = gpu_after >= gpu_before ? double(gpu_after - gpu_before) / 1e6
                                         : 0.0;
  };

  // Warm-up pass (compiles pipelines; 60s hang limit).
  double warm_wall = 0.0, warm_gpu = 0.0;
  replay_once(warm_wall, warm_gpu);
  in_warmup.store(false, std::memory_order_release);
  XELOGI("XDROID-BENCH: warmup wall_ms={:.2f} gpu_ms={:.2f}{}", warm_wall,
         warm_gpu, run_suffix);

  auto arms = xendroid::ParseBenchArmsSpec(cvars::gpu_trace_bench_arms);
  if (arms.empty()) {
    arms.push_back({"default", {}});
  }

  const size_t block_loops =
      std::max<uint32_t>(2, cvars::gpu_trace_bench_block_loops);
  std::vector<size_t> schedule;
  if (arms.size() >= 2) {
    const size_t rounds =
        std::max<size_t>(2, cvars::gpu_trace_bench_loops / (block_loops * 4));
    schedule = xendroid::BuildAbbaSchedule(arms.size(), rounds);
  } else {
    schedule.push_back(0);
  }

  std::vector<double> arm0_medians;
  std::vector<double> arm1_medians;
  uint32_t slow_streak = 0;
  uint32_t total_loops = 0;

  for (size_t blk = 0; blk < schedule.size(); ++blk) {
    const auto& arm = arms[schedule[blk]];
    for (const auto& ov : arm.overrides) {
      ApplyBenchLiveCvar(ov.key, ov.value);
    }
    const size_t loops_here =
        arms.size() == 1 ? cvars::gpu_trace_bench_loops : block_loops;
    std::vector<double> blk_walls;
    for (size_t l = 0; l < loops_here; ++l) {
      double wall_ms = 0.0, gpu_ms = 0.0;
      replay_once(wall_ms, gpu_ms);
      ++total_loops;
      XELOGI(
          "XDROID-BENCH: arm={} block={} loop={} wall_ms={:.3f} "
          "gpu_ms={:.3f}{}",
          arm.name, blk, l, wall_ms, gpu_ms, run_suffix);
      if (l > 0 || loops_here == 1) {
        blk_walls.push_back(wall_ms);
      }
      if (cvars::gpu_trace_bench_guard &&
          xendroid::EvaluateWatchdogLoopWall(wall_ms, slow_streak, wd_cfg) ==
              xendroid::BenchWatchdogVerdict::kSlowAbort) {
        XELOGE("XDROID-BENCH: abort reason=SLOW frame={}{}", total_loops,
               run_suffix);
        WriteAbortJson(traces_dir, "SLOW", run_id, total_loops);
        xe::FlushLog();
        finished.store(true, std::memory_order_release);
        if (watchdog.joinable()) watchdog.join();
        std::_Exit(3);
      }
    }
    const double blk_med = xendroid::ComputeMedian(blk_walls);
    if (schedule[blk] == 0) {
      arm0_medians.push_back(blk_med);
    } else if (schedule[blk] == 1) {
      arm1_medians.push_back(blk_med);
    }

    if (arms.size() >= 2 && (blk % 4 == 3)) {
      const size_t pairs = std::min(arm0_medians.size(), arm1_medians.size());
      std::vector<double> deltas;
      deltas.reserve(pairs);
      for (size_t i = 0; i < pairs; ++i) {
        deltas.push_back(arm1_medians[i] - arm0_medians[i]);
      }
      const auto ci = xendroid::ComputeBootstrapCI(deltas);
      if (xendroid::ShouldStopBenchEarly(ci)) {
        XELOGI(
            "XDROID-BENCH: early-stop pairs={} delta_ms={:.3f} "
            "ci95=[{:.3f},{:.3f}]{}",
            ci.n, ci.median, ci.ci_low, ci.ci_high, run_suffix);
        break;
      }
    }
  }

  finished.store(true, std::memory_order_release);
  if (watchdog.joinable()) {
    watchdog.join();
  }

  std::filesystem::path png_out = trace_path;
  png_out.replace_extension(".last.png");
  SaveLastGuestFramePng(gs, png_out);

  XELOGI("XDROID-BENCH: done loops={}{}", total_loops, run_suffix);
  xe::FlushLog();
  return 0;
}

}  // namespace gpu
}  // namespace xe
