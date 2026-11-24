// Host unit tests for xendroid-bench pure C++20 core logic:
// 1. HLE Census aggregation (including >1024 slot capacity) & CLOCK-OFF / STALLED
// 2. AudioSanityTracker RMS, silence duration, clipping, NaN safety
// 3. PadScript parser (S, SPEC:Nms, hold/release, wait, nested loop, path jail)
// 4. GPU TraceBench ABBA scheduling, bootstrap 95% CI, early stop, lock-free watchdog

#include <cassert>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <vector>

#include "xe_audio_sanity.h"
#include "xe_gpu_bench_logic.h"
#include "xe_hle_census_logic.h"
#include "xe_pad_script_parser.h"

namespace {

void TestHleCensusLogic() {
  uint64_t cur[4] = {10, 500, 120, 0};
  uint64_t last[4] = {10, 100, 20, 0};
  xendroid::HleClockDelta healthy{10000, 10050, 9980, 10000};
  auto sum = xendroid::SummarizeHleCensusWindow(cur, last, 4, healthy, 2);
  assert(sum.total_calls == 500);
  assert(sum.active_exports == 2);
  assert(!sum.clock_off);
  assert(!sum.stalled);
  assert(sum.top_entries.size() == 2);
  assert(sum.top_entries[0].slot == 1 && sum.top_entries[0].delta_calls == 400);
  assert(sum.top_entries[1].slot == 2 && sum.top_entries[1].delta_calls == 100);

  // Stalled + CLOCK-OFF case: no new calls for 10s, guest tick frozen at 0 ms.
  xendroid::HleClockDelta frozen{10000, 10000, 0, 10000};
  auto bad = xendroid::SummarizeHleCensusWindow(cur, last, 4, frozen, 2);
  assert(bad.total_calls == 0);
  assert(bad.stalled);
  assert(bad.clock_off);

  // Large slot table (> 2876 exports across xboxkrnl + xam + xbdm).
  std::vector<uint64_t> large_cur(3200, 0);
  std::vector<uint64_t> large_last(3200, 0);
  large_cur[2950] = 77;
  large_cur[3100] = 99;
  auto big = xendroid::SummarizeHleCensusWindow(large_cur.data(),
                                                large_last.data(), 3200,
                                                healthy, 4);
  assert(big.total_calls == 176);
  assert(big.top_entries[0].slot == 3100);
}

void TestAudioSanityTracker() {
  xendroid::AudioSanityTracker tracker;
  std::vector<float> tone(480 * 2, 0.5f);  // 10 ms at 48 kHz
  tracker.Observe(tone.data(), 480, 48000);
  auto snap1 = tracker.SnapshotAndReset();
  assert(std::fabs(snap1.rms - 0.5f) < 1e-4f);
  assert(std::fabs(snap1.peak - 0.5f) < 1e-4f);
  assert(snap1.silence_ms == 0);
  assert(snap1.clipped_samples == 0);

  // Sustained silence + 1 NaN + 2 clipped samples test
  std::vector<float> silent(4800 * 2, 0.0f);  // 100 ms
  tracker.Observe(silent.data(), 4800, 48000);
  tracker.Observe(silent.data(), 4800, 48000);
  auto snap2 = tracker.SnapshotAndReset();
  assert(snap2.rms < 1e-5f);
  assert(snap2.silence_ms == 200);

  float edge[4] = {1.0f, -1.05f, NAN, 0.25f};
  tracker.Observe(edge, 2, 48000);
  auto snap3 = tracker.SnapshotAndReset();
  assert(snap3.clipped_samples == 2);
  assert(snap3.nan_samples == 1);
}

void TestPadScriptParser() {
  const char* script = R"(# Forza Horizon autonomous test
wait title 4D5309C9 180
START:4
idle 30
wait draws>=1500 90
loop 2
  S 5 1000 0.50 0.00 0.00 0.00 0.00 0.90
  A+DPAD_DOWN:150ms # inline comment
endloop
hold RT=1.0
release RT=1.0
release all
mark race ready checkpoint
end
)";
  std::vector<xendroid::PadScriptCommand> cmds;
  xendroid::PadScriptCommand err;
  assert(xendroid::ParsePadScriptText(script, cmds, err));
  assert(cmds.size() == 13);
  assert(cmds[0].op == xendroid::PadScriptOp::kWaitTitle);
  assert(cmds[0].n == 0x4D5309C9);
  assert(cmds[3].op == xendroid::PadScriptOp::kWaitDrawsAtLeast);
  assert(cmds[3].n == 1500);
  assert(cmds[5].op == xendroid::PadScriptOp::kSet);
  assert(cmds[5].input.buttons == 0x1000);
  assert(cmds[6].op == xendroid::PadScriptOp::kPress);
  assert(cmds[6].ms);
  assert(cmds[6].n == 150);
  assert(cmds[7].op == xendroid::PadScriptOp::kEndLoop);
  assert(cmds[7].n == 4);
  assert(cmds[11].op == xendroid::PadScriptOp::kMark);
  assert(cmds[11].text == "race ready checkpoint");

  // Invalid unclosed loop, zero hold, zero timeout, and NaN axes must fail.
  assert(!xendroid::ParsePadScriptText("loop 2\nA\n", cmds, err));
  assert(err.line == 1);
  assert(!xendroid::ParsePadScriptText("A:0ms\n", cmds, err));
  assert(!xendroid::ParsePadScriptText("wait title 4D5309C9 0\n", cmds, err));
  assert(!xendroid::ParsePadScriptText("hold LX=nan\n", cmds, err));

  // Path jail checks under storage_root.
  const std::filesystem::path root = "/sdcard/Android/data/xendroid/files";
  std::filesystem::path resolved;
  assert(xendroid::pad_detail::ResolvePadScriptPathUnderRoot(
      root, "scripts/menu.pad", resolved));
  assert(!xendroid::pad_detail::ResolvePadScriptPathUnderRoot(
      root, "../escape.pad", resolved));
  assert(!xendroid::pad_detail::ResolvePadScriptPathUnderRoot(
      root, "/etc/passwd", resolved));
  assert(!xendroid::pad_detail::ResolvePadScriptPathUnderRoot(
      root, "", resolved));

  // Live pulse nonce stripping & overlapping button refcounting.
  assert(xendroid::StripPadLiveNonce("A:150#3") == "A:150");
  assert(xendroid::StripPadLiveNonce("START:120") == "START:120");
  assert(xendroid::StripPadLiveNonce("B#0") == "B");

  xendroid::PadLiveButtonRefAccumulator refs;
  assert(refs.Add(0x0010) == 0x0010);
  assert(refs.Add(0x1010) == 0x1010);
  assert(refs.Remove(0x1010) == 0x0010);  // START stays held across overlap
  assert(refs.Remove(0x0010) == 0x0000);
  assert(refs.Remove(0x0010) == 0x0000);  // underflow guard
}

void TestGpuBenchLogic() {
  auto arms = xendroid::ParseBenchArmsSpec(
      "base:vulkan_direct_host_resolve=false,foo=0;opt:vulkan_direct_host_resolve=true,foo=1");
  assert(arms.size() == 2);
  assert(arms[0].name == "base");
  assert(arms[0].overrides.size() == 2);
  assert(arms[1].name == "opt");
  assert(arms[1].overrides[0].value == "true");

  auto sched = xendroid::BuildAbbaSchedule(2, 2);
  assert((sched == std::vector<size_t>{0, 1, 1, 0, 0, 1, 1, 0}));

  std::vector<double> deltas = {-2.5, -2.6, -2.4, -2.55, -2.45, -2.5};
  auto ci = xendroid::ComputeBootstrapCI(deltas);
  assert(ci.excludes_zero);
  assert(ci.median < -2.3);
  assert(xendroid::ShouldStopBenchEarly(ci, 4, 0.15));

  xendroid::BenchWatchdogConfig cfg;
  cfg.expect_ms = 20.0;
  cfg.slow_multiplier = 1.6;
  cfg.slow_streak_limit = 3;
  cfg.hang_ns = 4'000'000'000ULL;
  cfg.warmup_hang_ns = 60'000'000'000ULL;

  // Hang detection: warmup allows 10s, timed run aborts at 5s.
  assert(xendroid::EvaluateWatchdogHang(10'000'000'000LL, 1, true, cfg) ==
         xendroid::BenchWatchdogVerdict::kOk);
  assert(xendroid::EvaluateWatchdogHang(5'000'000'000LL, 1, false, cfg) ==
         xendroid::BenchWatchdogVerdict::kHangAbort);

  // Slow streak detection: 3 consecutive > 32 ms frames trigger kSlowAbort.
  uint32_t streak = 0;
  assert(xendroid::EvaluateWatchdogLoopWall(35.0, streak, cfg) ==
         xendroid::BenchWatchdogVerdict::kOk);
  assert(xendroid::EvaluateWatchdogLoopWall(36.0, streak, cfg) ==
         xendroid::BenchWatchdogVerdict::kOk);
  assert(xendroid::EvaluateWatchdogLoopWall(38.0, streak, cfg) ==
         xendroid::BenchWatchdogVerdict::kSlowAbort);
}

}  // namespace

int main() {
  TestHleCensusLogic();
  TestAudioSanityTracker();
  TestPadScriptParser();
  TestGpuBenchLogic();
  std::printf("bench_tools_logic_test: ok\n");
  return 0;
}
