#include "xenia/kernel/hle_census.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "../../../../xe_hle_census_logic.h"
#include "third_party/fmt/include/fmt/format.h"
#include "xenia/base/clock.h"
#include "xenia/base/logging.h"
#include "xenia/cpu/export_resolver.h"

DEFINE_bool(
    hle_census, false,
    "Wrap kernel export trampolines with relaxed atomic counters and log top "
    "HLE callers plus guest-vs-host clock deltas periodically (for smoke triage).",
    "Kernel");
DEFINE_uint32(hle_census_interval_ms, 10000,
              "Reporting period in milliseconds for hle_census.", "Kernel");

namespace xe {
namespace kernel {
namespace {

constexpr size_t kGroupCount = 64;
constexpr size_t kGroupSize = 64;
constexpr size_t kSlots = kGroupCount * kGroupSize;  // 4096 > xboxkrnl+xam+xbdm
std::array<std::atomic<uint64_t>, kSlots> g_count{};
std::array<xe::cpu::ExportTrampoline, kSlots> g_orig{};
std::array<const char*, kSlots> g_name{};
size_t g_used = 0;
xe::cpu::ExportResolver* g_census_resolver = nullptr;

std::mutex g_census_mutex;
std::condition_variable g_census_cv;
std::thread g_census_thread;
bool g_census_stop = false;

template <size_t I>
void CensusThunk(xe::cpu::ppc::PPCContext* ctx) {
  g_count[I].fetch_add(1, std::memory_order_relaxed);
  g_orig[I](ctx);
}

template <size_t Base, size_t... J>
constexpr std::array<xe::cpu::ExportTrampoline, kGroupSize> MakeGroup(
    std::index_sequence<J...>) {
  return {&CensusThunk<Base + J>...};
}

template <size_t... G>
constexpr std::array<std::array<xe::cpu::ExportTrampoline, kGroupSize>,
                     kGroupCount>
MakeThunkGroups(std::index_sequence<G...>) {
  return {MakeGroup<G * kGroupSize>(std::make_index_sequence<kGroupSize>{})...};
}

void CensusThreadMain(size_t used_count) {
  std::vector<uint64_t> last(used_count, 0);
  std::vector<uint64_t> snapshot(used_count, 0);
  uint64_t gu = xe::Clock::QueryGuestUptimeMillis();
  uint64_t gt = xe::Clock::QueryGuestTickCount();
  uint64_t gs = xe::Clock::QueryGuestSystemTime();
  uint64_t hu = xe::Clock::QueryHostUptimeMillis();
  const uint64_t freq =
      std::max<uint64_t>(1, xe::Clock::guest_tick_frequency());
  const uint32_t interval_ms =
      std::max<uint32_t>(500, cvars::hle_census_interval_ms);

  for (;;) {
    {
      std::unique_lock<std::mutex> lock(g_census_mutex);
      if (g_census_cv.wait_for(lock, std::chrono::milliseconds(interval_ms),
                               [] { return g_census_stop; })) {
        return;
      }
    }

    for (size_t i = 0; i < used_count; ++i) {
      snapshot[i] = g_count[i].load(std::memory_order_relaxed);
    }

    const uint64_t gu2 = xe::Clock::QueryGuestUptimeMillis();
    const uint64_t gt2 = xe::Clock::QueryGuestTickCount();
    const uint64_t gs2 = xe::Clock::QueryGuestSystemTime();
    const uint64_t hu2 = xe::Clock::QueryHostUptimeMillis();

    xendroid::HleClockDelta clocks;
    clocks.host_ms = hu2 >= hu ? (hu2 - hu) : 0;
    clocks.guest_uptime_ms = gu2 >= gu ? (gu2 - gu) : 0;
    clocks.guest_tick_ms = ((gt2 >= gt ? (gt2 - gt) : 0) * 1000ULL) / freq;
    clocks.guest_sys_ms = (gs2 >= gs ? (gs2 - gs) : 0) / 10000ULL;

    const auto summary = xendroid::SummarizeHleCensusWindow(
        snapshot.data(), last.data(), used_count, clocks, 16);

    std::string top;
    for (const auto& entry : summary.top_entries) {
      top += fmt::format(" {}={}", g_name[entry.slot], entry.delta_calls);
    }
    XELOGI(
        "XDROID-HLE: {} s calls {} exports {} top:{} | clock ms host {} up {} "
        "tick {} sys {}{}{}",
        clocks.host_ms / 1000, summary.total_calls, summary.active_exports, top,
        clocks.host_ms, clocks.guest_uptime_ms, clocks.guest_tick_ms,
        clocks.guest_sys_ms, summary.clock_off ? " CLOCK-OFF" : "",
        summary.stalled ? " STALLED" : "");

    gu = gu2;
    gt = gt2;
    gs = gs2;
    hu = hu2;
  }
}

}  // namespace

void InstallHleCensus(xe::cpu::ExportResolver* resolver) {
  if (!cvars::hle_census || !resolver) {
    return;
  }
  ShutdownHleCensus();

  static constexpr auto kThunkGroups =
      MakeThunkGroups(std::make_index_sequence<kGroupCount>{});
  g_used = 0;
  size_t skipped = 0;
  for (xe::cpu::Export* e : resolver->all_exports_by_name()) {
    if (!e || e->get_type() != xe::cpu::Export::Type::kFunction ||
        !e->function_data.trampoline) {
      continue;
    }
    if (g_used == kSlots) {
      ++skipped;
      continue;
    }
    g_count[g_used].store(0, std::memory_order_relaxed);
    g_orig[g_used] = e->function_data.trampoline;
    g_name[g_used] = e->name;
    e->function_data.trampoline =
        kThunkGroups[g_used / kGroupSize][g_used % kGroupSize];
    ++g_used;
  }

  XELOGI("XDROID-HLE: census enabled, {} exports wrapped ({} skipped)", g_used,
         skipped);
  g_census_resolver = resolver;
  resolver->set_hle_census_active(true);
  std::lock_guard<std::mutex> lock(g_census_mutex);
  g_census_stop = false;
  g_census_thread = std::thread(CensusThreadMain, g_used);
}

void ShutdownHleCensus() {
  if (g_census_resolver) {
    g_census_resolver->set_hle_census_active(false);
    g_census_resolver = nullptr;
  }
  {
    std::lock_guard<std::mutex> lock(g_census_mutex);
    g_census_stop = true;
  }
  g_census_cv.notify_all();
  if (g_census_thread.joinable()) {
    g_census_thread.join();
  }
  for (size_t i = 0; i < g_used; ++i) {
    g_orig[i] = nullptr;
    g_name[i] = nullptr;
  }
  g_used = 0;
}

}  // namespace kernel
}  // namespace xe
