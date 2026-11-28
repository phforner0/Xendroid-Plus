#pragma once
// Pure C++20 logic for HLE export census + guest-vs-host clock drift detection.
// Tested natively on host via emulator-core/src/test/cpp/bench_tools_logic_test.cc.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace xendroid {

struct HleClockDelta {
  uint64_t host_ms = 0;
  uint64_t guest_uptime_ms = 0;
  uint64_t guest_tick_ms = 0;
  uint64_t guest_sys_ms = 0;
};

struct HleCensusEntry {
  size_t slot = 0;
  uint64_t delta_calls = 0;
};

struct HleCensusSummary {
  uint64_t total_calls = 0;
  size_t active_exports = 0;
  bool clock_off = false;
  bool stalled = false;
  std::vector<HleCensusEntry> top_entries;
};

inline bool IsGuestClockOff(uint64_t host_ms, uint64_t guest_ms) {
  if (host_ms < 500) {
    return false;
  }
  // > 20% divergence from host wall time (mirrors XDroid360 HLE census check).
  return guest_ms * 5 < host_ms * 4 || guest_ms * 5 > host_ms * 6;
}

inline HleCensusSummary SummarizeHleCensusWindow(
    const uint64_t* current_counts,
    uint64_t* last_counts,
    size_t count,
    const HleClockDelta& clocks,
    size_t top_n = 16) {
  HleCensusSummary out;
  std::vector<std::pair<uint64_t, size_t>> active;
  for (size_t i = 0; i < count; ++i) {
    const uint64_t cur = current_counts[i];
    const uint64_t delta = cur >= last_counts[i] ? (cur - last_counts[i]) : 0;
    last_counts[i] = cur;
    if (delta > 0) {
      out.total_calls += delta;
      active.emplace_back(delta, i);
    }
  }
  out.active_exports = active.size();
  std::sort(active.begin(), active.end(),
            [](const auto& a, const auto& b) {
              return a.first != b.first ? a.first > b.first : a.second < b.second;
            });
  const size_t limit = std::min(top_n, active.size());
  out.top_entries.reserve(limit);
  for (size_t k = 0; k < limit; ++k) {
    out.top_entries.push_back({active[k].second, active[k].first});
  }
  out.clock_off = IsGuestClockOff(clocks.host_ms, clocks.guest_uptime_ms) ||
                  IsGuestClockOff(clocks.host_ms, clocks.guest_tick_ms) ||
                  IsGuestClockOff(clocks.host_ms, clocks.guest_sys_ms);
  out.stalled = (clocks.host_ms >= 5000 && out.total_calls == 0);
  return out;
}

}  // namespace xendroid
