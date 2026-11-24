#pragma once
// Pure C++20 logic for GPU TraceBench ABBA scheduling, paired bootstrap 95% CI,
// sequential early stopping, and lock-free watchdog abort evaluation.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace xendroid {

struct BenchArmCvar {
  std::string key;
  std::string value;
};

struct BenchArmSpec {
  std::string name;
  std::vector<BenchArmCvar> overrides;
};

// Parses `"base:cvar1=0,cvar2=false;opt:cvar1=1,cvar2=true"`.
inline std::vector<BenchArmSpec> ParseBenchArmsSpec(std::string_view spec) {
  std::vector<BenchArmSpec> arms;
  size_t pos = 0;
  while (pos < spec.size()) {
    size_t end = spec.find(';', pos);
    if (end == std::string_view::npos) end = spec.size();
    std::string_view item = spec.substr(pos, end - pos);
    pos = end + 1;
    const size_t colon = item.find(':');
    if (colon == std::string_view::npos || colon == 0) continue;
    BenchArmSpec arm;
    arm.name = std::string(item.substr(0, colon));
    std::string_view kvs = item.substr(colon + 1);
    size_t kpos = 0;
    while (kpos < kvs.size()) {
      size_t comma = kvs.find(',', kpos);
      if (comma == std::string_view::npos) comma = kvs.size();
      std::string_view kv = kvs.substr(kpos, comma - kpos);
      kpos = comma + 1;
      const size_t eq = kv.find('=');
      if (eq != std::string_view::npos && eq > 0) {
        arm.overrides.push_back(
            {std::string(kv.substr(0, eq)), std::string(kv.substr(eq + 1))});
      }
    }
    arms.push_back(std::move(arm));
  }
  return arms;
}

// Emits ABBA order for `rounds` 4-block cycles: [0, 1, 1, 0, ...].
inline std::vector<size_t> BuildAbbaSchedule(size_t arm_count, size_t rounds) {
  std::vector<size_t> seq;
  if (arm_count == 0 || rounds == 0) return seq;
  if (arm_count == 1) {
    seq.assign(rounds, 0);
    return seq;
  }
  seq.reserve(rounds * 4);
  for (size_t r = 0; r < rounds; ++r) {
    seq.push_back(0);
    seq.push_back(1);
    seq.push_back(1);
    seq.push_back(0);
  }
  return seq;
}

inline double ComputeMedian(std::vector<double> vals) {
  if (vals.empty()) return 0.0;
  std::sort(vals.begin(), vals.end());
  const size_t mid = vals.size() / 2;
  if (vals.size() % 2 == 1) return vals[mid];
  return 0.5 * (vals[mid - 1] + vals[mid]);
}

struct BootstrapResult {
  size_t n = 0;
  double median = 0.0;
  double ci_low = 0.0;
  double ci_high = 0.0;
  double half_width = 0.0;
  bool excludes_zero = false;
};

inline BootstrapResult ComputeBootstrapCI(const std::vector<double>& deltas,
                                          size_t resamples = 1000,
                                          uint64_t seed = 20261123ULL) {
  BootstrapResult out;
  out.n = deltas.size();
  if (deltas.empty()) return out;
  out.median = ComputeMedian(deltas);
  if (deltas.size() == 1) {
    out.ci_low = out.ci_high = out.median;
    return out;
  }
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<size_t> dist(0, deltas.size() - 1);
  std::vector<double> boot;
  boot.reserve(resamples);
  std::vector<double> sample(deltas.size());
  for (size_t r = 0; r < resamples; ++r) {
    for (size_t i = 0; i < deltas.size(); ++i) {
      sample[i] = deltas[dist(rng)];
    }
    boot.push_back(ComputeMedian(sample));
  }
  std::sort(boot.begin(), boot.end());
  const size_t lo = size_t(std::floor(0.025 * double(resamples - 1)));
  const size_t hi = std::min(resamples - 1,
                             size_t(std::ceil(0.975 * double(resamples - 1))));
  out.ci_low = boot[lo];
  out.ci_high = boot[hi];
  out.half_width = 0.5 * (out.ci_high - out.ci_low);
  out.excludes_zero = (out.ci_low > 0.0) || (out.ci_high < 0.0);
  return out;
}

inline bool ShouldStopBenchEarly(const BootstrapResult& ci,
                                 size_t min_pairs = 4,
                                 double target_half_width = 0.15) {
  if (ci.n < min_pairs) return false;
  return ci.excludes_zero ||
         (ci.half_width <= target_half_width &&
          ci.ci_low <= 0.0 && ci.ci_high >= 0.0);
}

enum class BenchWatchdogVerdict : uint8_t {
  kOk = 0,
  kSlowAbort = 1,
  kHangAbort = 2,
};

struct BenchWatchdogConfig {
  double expect_ms = 0.0;
  double slow_multiplier = 1.6;
  uint32_t slow_streak_limit = 3;
  uint64_t hang_ns = 4'000'000'000ULL;
  uint64_t warmup_hang_ns = 60'000'000'000ULL;
};

inline BenchWatchdogVerdict EvaluateWatchdogHang(
    int64_t now_ns,
    int64_t frame_start_ns,
    bool in_warmup,
    const BenchWatchdogConfig& cfg) {
  if (frame_start_ns <= 0 || now_ns <= frame_start_ns) {
    return BenchWatchdogVerdict::kOk;
  }
  const uint64_t elapsed = uint64_t(now_ns - frame_start_ns);
  const uint64_t limit = in_warmup ? cfg.warmup_hang_ns : cfg.hang_ns;
  return elapsed > limit ? BenchWatchdogVerdict::kHangAbort
                         : BenchWatchdogVerdict::kOk;
}

inline BenchWatchdogVerdict EvaluateWatchdogLoopWall(
    double loop_wall_ms,
    uint32_t& consecutive_slow,
    const BenchWatchdogConfig& cfg) {
  if (cfg.expect_ms <= 0.0) {
    consecutive_slow = 0;
    return BenchWatchdogVerdict::kOk;
  }
  if (loop_wall_ms > cfg.expect_ms * cfg.slow_multiplier) {
    if (++consecutive_slow >= cfg.slow_streak_limit) {
      return BenchWatchdogVerdict::kSlowAbort;
    }
  } else {
    consecutive_slow = 0;
  }
  return BenchWatchdogVerdict::kOk;
}

}  // namespace xendroid
