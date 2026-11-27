// Wave 7 (Onda 1B) CPU / JIT / W^X / POSIX / Kernel P0 Hardening tests.
// Exercises real host & Linux/WSL2 POSIX syscalls + concurrency invariants +
// source-level safety contracts for:
//   1. W^X boot probe policy (ResolveWritableExecutableMemoryPreferred):
//      no transient RWX mmap probe when a64_wx_dual_mapping=true or
//      writable_executable_memory=false; clean RWX fallback only when W^X
//      dual-mapping probe fails AND writable_executable_memory=true.
//   2. Live Linux/WSL2 POSIX test (when __linux__): real memfd_create +
//      r-xs / rw-s dual-mapping verification via /proc/self/maps + live
//      MAP_FIXED_NOREPLACE non-clobbering (EEXIST) and exact-address mapping,
//      plus live CLOCK_MONOTONIC_RAW vs fixed 1e9 Hz scale verification.
//   3. Upstream xenia-canary@d92b4a2ffe manual-reset Event release_count_
//      multithreaded wakeup across Set()+Reset() race, slice preservation via
//      since, and WaitMultiple wait_any vs wait_all semantics.
//   4. VSCR[SAT] census late false->true transition (late XEX DLL / non-XEX
//      AddModule / EnableDynamicCode), compiled function invalidation,
//      in-flight compile retry, and vscr_sat_census=false kill-switch.
//   5. A64 PIC site eviction (pic_sites_by_gate_ / patched_pic_sites_ /
//      pic_sites_by_caller_) and host-page-aware (4 KB & 16 KB) + parallel
//      compilation loop-header alignment padding.
//   6. Source-level contracts across all 12 owned Wave 7 Onda 1B files.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(__linux__)
#include <cerrno>
#include <ctime>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#if defined(__has_include)
#if __has_include(<linux/memfd.h>)
#include <linux/memfd.h>
#endif
#endif
#ifndef MFD_CLOEXEC
#define MFD_CLOEXEC 0x0001u
#endif
#endif

#include "xenia/base/memory.h"
#include "xenia/cpu/backend/a64/a64_direct_call.h"

namespace {

#define WAVE7_1B_CHECK(cond)                                                 \
  do {                                                                       \
    if (!(cond)) {                                                           \
      std::fprintf(stderr, "WAVE7_1B check failed (%s:%d): %s\n", __FILE__,  \
                   __LINE__, #cond);                                         \
      std::abort();                                                          \
    }                                                                        \
  } while (false)

namespace fs = std::filesystem;
namespace a64 = xe::cpu::backend::a64;

std::string NormalizeNewlines(std::string s) {
  s.erase(std::remove(s.begin(), s.end(), '\r'), s.end());
  return s;
}

std::string ReadSourceText(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  WAVE7_1B_CHECK(in.good());
  std::ostringstream ss;
  ss << in.rdbuf();
  return NormalizeNewlines(ss.str());
}

std::string StripLineComments(const std::string& src) {
  std::string out;
  out.reserve(src.size());
  std::istringstream iss(src);
  std::string line;
  while (std::getline(iss, line)) {
    const size_t cpos = line.find("//");
    if (cpos != std::string::npos) {
      line.resize(cpos);
    }
    out += line;
    out.push_back('\n');
  }
  return out;
}

std::string ExtractFunctionBody(const std::string& src,
                                const std::string& anchor) {
  size_t search_pos = 0;
  while (true) {
    const size_t start = src.find(anchor, search_pos);
    WAVE7_1B_CHECK(start != std::string::npos);
    const size_t open = src.find('{', start);
    const size_t semi = src.find(';', start);
    WAVE7_1B_CHECK(open != std::string::npos);
    // Skip forward declarations like `uint64_t Foo(...);` so we extract the
    // actual definition body rather than the next function's braces.
    if (semi != std::string::npos && semi < open) {
      search_pos = semi + 1;
      continue;
    }
    int depth = 0;
    for (size_t i = open; i < src.size(); ++i) {
      if (src[i] == '{') {
        ++depth;
      } else if (src[i] == '}') {
        --depth;
        if (depth == 0) {
          return src.substr(open, i - open + 1);
        }
      }
    }
    WAVE7_1B_CHECK(false && "Unmatched brace while extracting function body");
    return {};
  }
}

// ---------------------------------------------------------------------------
// 1. W^X boot probe policy (real xe::memory::ResolveWritableExecutableMemoryPreferred)
// ---------------------------------------------------------------------------
void TestWxBootProbePolicy() {
  using xe::memory::DualMappingWxBackend;
  using xe::memory::ResolveWritableExecutableMemoryPreferred;

  // Case 1 (Android Wave 6/7 default): a64_wx_dual_mapping=true,
  // writable_executable_memory=true, dual-mapping succeeds -> prefer W^X
  // (return false) WITHOUT ever invoking the RWX mmap boot probe!
  for (DualMappingWxBackend ok_backend :
       {DualMappingWxBackend::kMemfd, DualMappingWxBackend::kAshmem,
        DualMappingWxBackend::kPosixShm}) {
    int wx_probes = 0;
    int rwx_probes = 0;
    const bool rwx_preferred = ResolveWritableExecutableMemoryPreferred(
        /*wx_dual_mapping_requested=*/true, /*allow_rwx_fallback=*/true,
        [&]() {
          ++wx_probes;
          return ok_backend;
        },
        [&]() {
          ++rwx_probes;
          return true;
        });
    WAVE7_1B_CHECK(!rwx_preferred);
    WAVE7_1B_CHECK(wx_probes == 1);
    WAVE7_1B_CHECK(rwx_probes == 0);
  }

  // Case 2: a64_wx_dual_mapping=true, writable_executable_memory=true,
  // dual-mapping fails (SELinux denied or Windows) -> probes RWX and falls
  // back cleanly to RWX iff host supports RWX.
  {
    int wx_probes = 0;
    int rwx_probes = 0;
    const bool rwx_preferred = ResolveWritableExecutableMemoryPreferred(
        true, true,
        [&]() {
          ++wx_probes;
          return DualMappingWxBackend::kUnsupported;
        },
        [&]() {
          ++rwx_probes;
          return true;
        });
    WAVE7_1B_CHECK(rwx_preferred);
    WAVE7_1B_CHECK(wx_probes == 1);
    WAVE7_1B_CHECK(rwx_probes == 1);
  }
  {
    int wx_probes = 0;
    int rwx_probes = 0;
    const bool rwx_preferred = ResolveWritableExecutableMemoryPreferred(
        true, true,
        [&]() {
          ++wx_probes;
          return DualMappingWxBackend::kUnsupported;
        },
        [&]() {
          ++rwx_probes;
          return false;
        });
    WAVE7_1B_CHECK(!rwx_preferred);
    WAVE7_1B_CHECK(wx_probes == 1);
    WAVE7_1B_CHECK(rwx_probes == 1);
  }

  // Case 3: writable_executable_memory=false -> strictly forbid RWX (return
  // false) with ZERO RWX boot probe regardless of a64_wx_dual_mapping or W^X
  // probe result.
  for (bool wx_req : {false, true}) {
    for (DualMappingWxBackend backend :
         {DualMappingWxBackend::kMemfd, DualMappingWxBackend::kUnsupported}) {
      int wx_probes = 0;
      int rwx_probes = 0;
      const bool rwx_preferred = ResolveWritableExecutableMemoryPreferred(
          wx_req, /*allow_rwx_fallback=*/false,
          [&]() {
            ++wx_probes;
            return backend;
          },
          [&]() {
            ++rwx_probes;
            return true;
          });
      WAVE7_1B_CHECK(!rwx_preferred);
      WAVE7_1B_CHECK(wx_probes == 1);
      WAVE7_1B_CHECK(rwx_probes == 0);
    }
  }

  // Case 4: a64_wx_dual_mapping=false, writable_executable_memory=true ->
  // probe RWX first; if RWX works, stay on RWX with ZERO W^X probe; if RWX is
  // denied by the OS, probe W^X and return false.
  {
    int wx_probes = 0;
    int rwx_probes = 0;
    const bool rwx_preferred = ResolveWritableExecutableMemoryPreferred(
        /*wx_dual_mapping_requested=*/false, /*allow_rwx_fallback=*/true,
        [&]() {
          ++wx_probes;
          return DualMappingWxBackend::kMemfd;
        },
        [&]() {
          ++rwx_probes;
          return true;
        });
    WAVE7_1B_CHECK(rwx_preferred);
    WAVE7_1B_CHECK(rwx_probes == 1);
    WAVE7_1B_CHECK(wx_probes == 0);
  }
  {
    int wx_probes = 0;
    int rwx_probes = 0;
    const bool rwx_preferred = ResolveWritableExecutableMemoryPreferred(
        false, true,
        [&]() {
          ++wx_probes;
          return DualMappingWxBackend::kMemfd;
        },
        [&]() {
          ++rwx_probes;
          return false;
        });
    WAVE7_1B_CHECK(!rwx_preferred);
    WAVE7_1B_CHECK(rwx_probes == 1);
    WAVE7_1B_CHECK(wx_probes == 1);
  }
}

// ---------------------------------------------------------------------------
// 2. Live Linux/WSL2 POSIX memfd dual-mapping + MAP_FIXED_NOREPLACE + clock
// ---------------------------------------------------------------------------
#if defined(__linux__)
std::string LookupProcMapsPerms(uintptr_t addr) {
  std::ifstream maps("/proc/self/maps");
  WAVE7_1B_CHECK(maps.good());
  std::string line;
  while (std::getline(maps, line)) {
    std::istringstream iss(line);
    uintptr_t begin = 0;
    uintptr_t end = 0;
    char dash = 0;
    std::string perms;
    if ((iss >> std::hex >> begin >> dash >> end >> perms) && dash == '-' &&
        addr >= begin && addr < end) {
      return perms;
    }
  }
  return {};
}

void TestPosixRealMemfdDualMappingAndNoReplace() {
  const size_t ps = static_cast<size_t>(getpagesize());
  WAVE7_1B_CHECK(ps >= 4096u && (ps & (ps - 1u)) == 0);

  // 2a. Real memfd_create + r-xs / rw-s dual-mapping without RWX.
#if defined(__NR_memfd_create)
  const int fd =
      static_cast<int>(syscall(__NR_memfd_create, "wave7_1b_wx", MFD_CLOEXEC));
  WAVE7_1B_CHECK(fd >= 0);
  WAVE7_1B_CHECK(ftruncate(fd, static_cast<off_t>(ps * 2u)) == 0);

  void* rx_view = mmap(nullptr, ps, PROT_READ | PROT_EXEC, MAP_SHARED, fd, 0);
  WAVE7_1B_CHECK(rx_view != MAP_FAILED);
  void* rw_view = mmap(nullptr, ps, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  WAVE7_1B_CHECK(rw_view != MAP_FAILED);
  WAVE7_1B_CHECK(rx_view != rw_view);

  const std::string rx_perms =
      LookupProcMapsPerms(reinterpret_cast<uintptr_t>(rx_view));
  const std::string rw_perms =
      LookupProcMapsPerms(reinterpret_cast<uintptr_t>(rw_view));
  WAVE7_1B_CHECK(rx_perms == "r-xs");
  WAVE7_1B_CHECK(rw_perms == "rw-s");

  // Write ARM64 NOP sentinel + branch word through RW view, verify RX sees it.
  auto* rw_words = reinterpret_cast<volatile uint32_t*>(rw_view);
  auto* rx_words = reinterpret_cast<const volatile uint32_t*>(rx_view);
  rw_words[0] = xe::memory::kDualMappingWxProbeSentinel;
  rw_words[1] = 0x14000010u;
  std::atomic_thread_fence(std::memory_order_seq_cst);
  WAVE7_1B_CHECK(rx_words[0] == xe::memory::kDualMappingWxProbeSentinel);
  WAVE7_1B_CHECK(rx_words[1] == 0x14000010u);

  // 2b. Live MAP_FIXED_NOREPLACE: must fail with EEXIST on occupied VA and
  // leave existing mapping untouched (whereas MAP_FIXED would clobber it!).
#ifdef MAP_FIXED_NOREPLACE
  void* occupied = mmap(nullptr, ps, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  WAVE7_1B_CHECK(occupied != MAP_FAILED);
  constexpr uint32_t kOccupiedSentinel = 0xCAFEBABEu;
  *reinterpret_cast<volatile uint32_t*>(occupied) = kOccupiedSentinel;

  errno = 0;
  void* clobber_attempt =
      mmap(occupied, ps, PROT_READ | PROT_EXEC,
           MAP_SHARED | MAP_FIXED_NOREPLACE, fd, 0);
  WAVE7_1B_CHECK(clobber_attempt == MAP_FAILED);
  WAVE7_1B_CHECK(errno == EEXIST);
  WAVE7_1B_CHECK(*reinterpret_cast<const volatile uint32_t*>(occupied) ==
                 kOccupiedSentinel);
  WAVE7_1B_CHECK(munmap(occupied, ps) == 0);

  // Obtain a guaranteed-free page slot by reserving 2 pages and unmapping page 1.
  void* two_pages =
      mmap(nullptr, ps * 2u, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  WAVE7_1B_CHECK(two_pages != MAP_FAILED);
  void* free_slot = static_cast<uint8_t*>(two_pages) + ps;
  WAVE7_1B_CHECK(munmap(free_slot, ps) == 0);

  void* placed_rx = mmap(free_slot, ps, PROT_READ | PROT_EXEC,
                         MAP_SHARED | MAP_FIXED_NOREPLACE, fd, 0);
  WAVE7_1B_CHECK(placed_rx == free_slot);
  WAVE7_1B_CHECK(
      *reinterpret_cast<const volatile uint32_t*>(placed_rx) ==
      xe::memory::kDualMappingWxProbeSentinel);
  WAVE7_1B_CHECK(munmap(placed_rx, ps) == 0);
  WAVE7_1B_CHECK(munmap(two_pages, ps) == 0);
#endif

  WAVE7_1B_CHECK(munmap(rw_view, ps) == 0);
  WAVE7_1B_CHECK(munmap(rx_view, ps) == 0);
  WAVE7_1B_CHECK(close(fd) == 0);
#endif

  // 2c. Live CLOCK_MONOTONIC_RAW vs fixed 1e9 Hz frequency scale.
  timespec t0{}, t1{};
  WAVE7_1B_CHECK(clock_gettime(CLOCK_MONOTONIC_RAW, &t0) == 0);
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  WAVE7_1B_CHECK(clock_gettime(CLOCK_MONOTONIC_RAW, &t1) == 0);
  const uint64_t ticks0 = static_cast<uint64_t>(t0.tv_sec) * 1000000000ull +
                          static_cast<uint64_t>(t0.tv_nsec);
  const uint64_t ticks1 = static_cast<uint64_t>(t1.tv_sec) * 1000000000ull +
                          static_cast<uint64_t>(t1.tv_nsec);
  WAVE7_1B_CHECK(ticks1 > ticks0);
  const uint64_t dt_ticks = ticks1 - ticks0;
  constexpr uint64_t kFixedNsFreq = 1000000000ull;
  const uint64_t elapsed_ms = static_cast<uint64_t>(
      (static_cast<unsigned __int128>(dt_ticks) * 1000ull) / kFixedNsFreq);
  WAVE7_1B_CHECK(elapsed_ms >= 15ull && elapsed_ms <= 500ull);
}
#endif  // defined(__linux__)

// ---------------------------------------------------------------------------
// 3. Clock scale unit parity verification (cross-platform math proof)
// ---------------------------------------------------------------------------
void TestClockPosixScaleParity() {
  auto ticks_to = [](uint64_t ticks, uint64_t units, uint64_t freq) -> uint64_t {
#if defined(__SIZEOF_INT128__)
    return static_cast<uint64_t>(
        (static_cast<unsigned __int128>(ticks) * units) / freq);
#else
    return (ticks / freq) * units + ((ticks % freq) * units) / freq;
#endif
  };
  // Simulate 1.5 seconds of CLOCK_MONOTONIC_RAW nanosecond ticks even when
  // clock_getres() reports a coarse ARM64 timer step (52 ns for 19.2 MHz).
  const uint64_t raw_ns_ticks = 1'500'000'000ull;
  constexpr uint64_t kCorrectFreq = 1'000'000'000ull;
  WAVE7_1B_CHECK(ticks_to(raw_ns_ticks, 1000ull, kCorrectFreq) == 1500ull);
  WAVE7_1B_CHECK(ticks_to(raw_ns_ticks, 10'000'000ull, kCorrectFreq) ==
                 15'000'000ull);

  // Confirm why the old 1e9 / res.tv_nsec formula was broken: for res=52ns it
  // produced ~78,000 ms (52x drift!) instead of 1,500 ms.
  const uint64_t buggy_freq_52ns = 1'000'000'000ull / 52ull;
  WAVE7_1B_CHECK(ticks_to(raw_ns_ticks, 1000ull, buggy_freq_52ns) > 70000ull);
}

// ---------------------------------------------------------------------------
// 4. Manual-reset Event release_count_ semantics (xenia-canary@d92b4a2ffe)
// ---------------------------------------------------------------------------
class ManualResetEventModel {
 public:
  explicit ManualResetEventModel(bool initial_state) : signal_(initial_state) {}

  uint64_t release_count() const {
    return release_count_.load(std::memory_order_acquire);
  }

  void Signal() {
    std::unique_lock<std::mutex> lock(mutex_);
    signal_ = true;
    release_count_.fetch_add(1, std::memory_order_release);
    cv_.notify_all();
  }

  void Reset() {
    std::unique_lock<std::mutex> lock(mutex_);
    signal_ = false;
  }

  bool Wait(std::chrono::milliseconds timeout,
            std::optional<uint64_t> since = std::nullopt,
            std::atomic<int>* parked_counter = nullptr) {
    std::unique_lock<std::mutex> lock(mutex_);
    const uint64_t start = since ? *since : release_count();
    auto predicate = [this, start] {
      return signal_ || release_count() != start;
    };
    if (predicate()) {
      return true;
    }
    if (parked_counter) {
      parked_counter->fetch_add(1, std::memory_order_acq_rel);
    }
    return cv_.wait_for(lock, timeout, predicate);
  }

  static std::pair<bool, size_t> WaitMultiplePoll(
      const std::vector<ManualResetEventModel*>& events, bool wait_all,
      const std::vector<uint64_t>& since) {
    if (wait_all) {
      for (size_t i = 0; i < events.size(); ++i) {
        std::lock_guard<std::mutex> lk(events[i]->mutex_);
        if (!events[i]->signal_) {
          return {false, 0};
        }
      }
      return {true, 0};
    }
    for (size_t i = 0; i < events.size(); ++i) {
      std::lock_guard<std::mutex> lk(events[i]->mutex_);
      if (events[i]->signal_ || events[i]->release_count() != since[i]) {
        return {true, i};
      }
    }
    return {false, 0};
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  bool signal_ = false;
  std::atomic<uint64_t> release_count_{0};
};

void TestManualResetReleaseCountSemantics() {
  // 4a. Multithreaded Set() immediately followed by Reset() must release ALL
  // already-parked waiters even though signal_ is false when they wake.
  ManualResetEventModel ev(false);
  constexpr int kWaiters = 8;
  std::atomic<int> parked{0};
  std::atomic<int> succeeded{0};
  std::vector<std::thread> threads;
  threads.reserve(kWaiters);
  for (int i = 0; i < kWaiters; ++i) {
    threads.emplace_back([&]() {
      if (ev.Wait(std::chrono::milliseconds(1000), std::nullopt, &parked)) {
        succeeded.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }
  while (parked.load(std::memory_order_acquire) < kWaiters) {
    std::this_thread::yield();
  }
  // Give waiters a brief moment to enter cv_.wait_for after incrementing parked.
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  ev.Signal();
  ev.Reset();
  for (auto& t : threads) {
    t.join();
  }
  WAVE7_1B_CHECK(succeeded.load() == kWaiters);
  WAVE7_1B_CHECK(ev.release_count() == 1ull);

  // 4b. A fresh waiter arriving AFTER Reset() sees release_count_==1 at entry
  // and must NOT wake spuriously -> times out cleanly.
  WAVE7_1B_CHECK(!ev.Wait(std::chrono::milliseconds(15)));

  // 4c. Alertable wait slice preservation: capture since before Signal+Reset,
  // then a later slice passing `since` is still satisfied.
  const uint64_t slice_since = ev.release_count();
  ev.Signal();
  ev.Reset();
  WAVE7_1B_CHECK(ev.Wait(std::chrono::milliseconds(0), slice_since));
  WAVE7_1B_CHECK(!ev.Wait(std::chrono::milliseconds(0)));

  // 4d. WaitMultiple wait_any vs wait_all semantics:
  // If ev0 was Set()+Reset() while ev1 was unsignaled, wait_any MUST wake on
  // ev0 via release_count_, whereas wait_all MUST NOT succeed even after ev1
  // later becomes signaled because ev0 is no longer simultaneously signaled.
  ManualResetEventModel ev0(false);
  ManualResetEventModel ev1(false);
  const std::vector<ManualResetEventModel*> handles = {&ev0, &ev1};
  const std::vector<uint64_t> start_counts = {ev0.release_count(),
                                              ev1.release_count()};
  ev0.Signal();
  ev0.Reset();
  const auto any_res =
      ManualResetEventModel::WaitMultiplePoll(handles, false, start_counts);
  WAVE7_1B_CHECK(any_res.first && any_res.second == 0u);

  ev1.Signal();
  const auto all_res_unsync =
      ManualResetEventModel::WaitMultiplePoll(handles, true, start_counts);
  WAVE7_1B_CHECK(!all_res_unsync.first);

  ev0.Signal();
  const auto all_res_sync =
      ManualResetEventModel::WaitMultiplePoll(handles, true, start_counts);
  WAVE7_1B_CHECK(all_res_sync.first);
}

// ---------------------------------------------------------------------------
// 5. VSCR[SAT] census late transition, invalidation, in-flight retry & switch
// ---------------------------------------------------------------------------
struct VscrCensusModel {
  bool cvar_vscr_sat_census = true;
  bool dynamic_code_enabled = false;
  bool vscr_sat_observable_flag = true;
  uint64_t code_sweep_count = 0;

  struct CompiledFn {
    bool is_guest = true;
    bool is_default_behavior = true;
    bool compiled_with_sat = false;
  };
  std::unordered_map<uint32_t, CompiledFn> entry_table;

  bool vscr_sat_observable() const {
    return !cvar_vscr_sat_census || vscr_sat_observable_flag;
  }

  void FinishCensus(bool saw_xex, bool any_xex_reads_vscr,
                    bool any_non_trampoline_non_xex) {
    bool obs = !cvar_vscr_sat_census || dynamic_code_enabled;
    if (!obs) {
      if (!saw_xex || any_xex_reads_vscr || any_non_trampoline_non_xex) {
        obs = true;
      }
    }
    vscr_sat_observable_flag = obs;
  }

  void InvalidateCompiledFunctionsForVscr() {
    ++code_sweep_count;
    for (auto it = entry_table.begin(); it != entry_table.end();) {
      if (it->second.is_guest && it->second.is_default_behavior) {
        it = entry_table.erase(it);
      } else {
        ++it;
      }
    }
  }

  void OnModuleReadsVscr() {
    const bool was = vscr_sat_observable_flag;
    vscr_sat_observable_flag = true;
    if (!was) {
      InvalidateCompiledFunctionsForVscr();
    }
  }

  void AddLateModule(bool is_xex, bool xex_reads_vscr, bool is_trampoline) {
    if (is_xex) {
      if (xex_reads_vscr) {
        OnModuleReadsVscr();
      }
    } else if (!is_trampoline) {
      OnModuleReadsVscr();
    }
  }

  void EnableDynamicCode() {
    dynamic_code_enabled = true;
    const bool was = vscr_sat_observable_flag;
    vscr_sat_observable_flag = true;
    if (!was) {
      InvalidateCompiledFunctionsForVscr();
    }
  }

  void CompileFunction(uint32_t addr, bool simulate_late_vscr_race) {
    for (int attempt = 0; attempt < 4; ++attempt) {
      const bool vscr_at_start = vscr_sat_observable();
      if (simulate_late_vscr_race && attempt == 0) {
        OnModuleReadsVscr();
      }
      const bool vscr_stale = !vscr_at_start && vscr_sat_observable();
      if (vscr_stale) {
        continue;
      }
      entry_table[addr] = CompiledFn{true, true, vscr_at_start};
      return;
    }
    WAVE7_1B_CHECK(false);
  }
};

void TestVscrSatCensusLateTransition() {
  // 5a. Clean XEX with no mfvscr -> census sets observable=false.
  {
    VscrCensusModel m;
    m.FinishCensus(/*saw_xex=*/true, /*any_xex_reads_vscr=*/false,
                   /*any_non_trampoline_non_xex=*/false);
    WAVE7_1B_CHECK(!m.vscr_sat_observable());

    // Compile guest fn (without SAT) and preserve an import thunk.
    m.CompileFunction(0x82010000u, false);
    m.entry_table[0x82090000u] = {true, /*is_default_behavior=*/false, false};
    WAVE7_1B_CHECK(!m.entry_table[0x82010000u].compiled_with_sat);

    // Late non-executable kernel *_trampoline does NOT trigger invalidation.
    m.AddLateModule(/*is_xex=*/false, false, /*is_trampoline=*/true);
    WAVE7_1B_CHECK(!m.vscr_sat_observable());
    WAVE7_1B_CHECK(m.entry_table.count(0x82010000u) == 1u);

    // Late XEX DLL without mfvscr does NOT trigger invalidation.
    m.AddLateModule(/*is_xex=*/true, /*xex_reads_vscr=*/false, false);
    WAVE7_1B_CHECK(!m.vscr_sat_observable());
    WAVE7_1B_CHECK(m.entry_table.count(0x82010000u) == 1u);

    // Late XEX DLL WITH mfvscr flips false -> true and invalidates guest code
    // while keeping import thunks intact.
    m.AddLateModule(/*is_xex=*/true, /*xex_reads_vscr=*/true, false);
    WAVE7_1B_CHECK(m.vscr_sat_observable());
    WAVE7_1B_CHECK(m.code_sweep_count == 1u);
    WAVE7_1B_CHECK(m.entry_table.count(0x82010000u) == 0u);
    WAVE7_1B_CHECK(m.entry_table.count(0x82090000u) == 1u);

    // Recompile now emits VSCR[SAT] updates.
    m.CompileFunction(0x82010000u, false);
    WAVE7_1B_CHECK(m.entry_table[0x82010000u].compiled_with_sat);
  }

  // 5b. Late non-XEX module (e.g. ElfModule/RawModule) and EnableDynamicCode().
  {
    VscrCensusModel m;
    m.FinishCensus(true, false, false);
    m.CompileFunction(0x82020000u, false);
    m.AddLateModule(/*is_xex=*/false, false, /*is_trampoline=*/false);
    WAVE7_1B_CHECK(m.vscr_sat_observable());
    WAVE7_1B_CHECK(m.entry_table.count(0x82020000u) == 0u);
  }
  {
    VscrCensusModel m;
    m.FinishCensus(true, false, false);
    m.CompileFunction(0x82030000u, false);
    m.EnableDynamicCode();
    WAVE7_1B_CHECK(m.vscr_sat_observable());
    WAVE7_1B_CHECK(m.entry_table.count(0x82030000u) == 0u);
  }

  // 5c. Concurrent in-flight compile race: vscr flips false->true mid-compile
  // -> must detect vscr_stale and retry so published function has SAT=true.
  {
    VscrCensusModel m;
    m.FinishCensus(true, false, false);
    m.CompileFunction(0x82040000u, /*simulate_late_vscr_race=*/true);
    WAVE7_1B_CHECK(m.vscr_sat_observable());
    WAVE7_1B_CHECK(m.entry_table[0x82040000u].compiled_with_sat);
  }

  // 5d. Kill-switch vscr_sat_census=false forces observable=true always.
  {
    VscrCensusModel m;
    m.cvar_vscr_sat_census = false;
    m.FinishCensus(true, false, false);
    WAVE7_1B_CHECK(m.vscr_sat_observable());
  }
}

// ---------------------------------------------------------------------------
// 6. PIC site eviction (pic_sites_by_gate_) & 4 KB / 16 KB / parallel loop align
// ---------------------------------------------------------------------------
constexpr size_t ComputeLoopHeaderAlignmentPaddingForPageRef(
    size_t current_offset, size_t entry_mod_page, size_t host_page_size,
    bool allow_page_nudge) {
  if ((current_offset & 3u) != 0) {
    return 0;
  }
  if (!allow_page_nudge || host_page_size < 4096u ||
      (host_page_size & (host_page_size - 1u)) != 0) {
    return (32u - (current_offset & 31u)) & 31u;
  }
  const size_t page_mask = host_page_size - 1u;
  const size_t page_off =
      ((entry_mod_page & page_mask) + current_offset) & page_mask;
  if (page_off >= host_page_size - 64u) {
    return host_page_size - page_off;
  }
  return (32u - (page_off & 31u)) & 31u;
}

class PicRegistryModel {
 public:
  void Register(uintptr_t gate_rx, uintptr_t cmp_lo_rx, uintptr_t cmp_hi_rx,
                uintptr_t call_op_rx, uintptr_t fallback_rx,
                uint32_t caller_guest_addr) {
    if (by_gate_.count(gate_rx)) {
      return;
    }
    a64::IndirectCallPicSite site{};
    site.gate_rx = gate_rx;
    site.cmp_lo_rx = cmp_lo_rx;
    site.cmp_hi_rx = cmp_hi_rx;
    site.call_op_rx = call_op_rx;
    site.fallback_rx = fallback_rx;
    site.is_tail = false;
    site.state = a64::IndirectCallPicState::kUninitialized;
    by_gate_.emplace(gate_rx, site);
    if (caller_guest_addr != 0) {
      caller_by_gate_.emplace(gate_rx, caller_guest_addr);
      by_caller_[caller_guest_addr].push_back(gate_rx);
    }
  }

  bool TryPatch(uintptr_t gate_rx, uint32_t target_guest, uintptr_t target_rx) {
    auto it = by_gate_.find(gate_rx);
    if (it == by_gate_.end() ||
        it->second.state != a64::IndirectCallPicState::kUninitialized) {
      return false;
    }
    a64::IndirectCallPicPatchWords words{};
    if (!a64::ComputeIndirectCallPicPatchWords(it->second, target_guest,
                                               target_rx, &words)) {
      RetireLocked(it);
      return false;
    }
    it->second.cached_guest_target = target_guest;
    it->second.state = a64::IndirectCallPicState::kPatched;
    by_target_[target_guest].push_back(gate_rx);
    return true;
  }

  bool Fallback(uintptr_t gate_rx) {
    auto it = by_gate_.find(gate_rx);
    if (it == by_gate_.end()) {
      return false;
    }
    RetireLocked(it);
    return true;
  }

  void RevertRange(uint32_t guest_start, uint32_t guest_end) {
    std::vector<uintptr_t> gates_to_retire;
    for (auto it = by_target_.begin(); it != by_target_.end();) {
      if (it->first < guest_start || it->first > guest_end) {
        ++it;
        continue;
      }
      gates_to_retire.insert(gates_to_retire.end(), it->second.begin(),
                             it->second.end());
      it = by_target_.erase(it);
    }
    for (auto it = by_caller_.begin(); it != by_caller_.end();) {
      if (it->first < guest_start || it->first > guest_end) {
        ++it;
        continue;
      }
      gates_to_retire.insert(gates_to_retire.end(), it->second.begin(),
                             it->second.end());
      it = by_caller_.erase(it);
    }
    for (uintptr_t gate : gates_to_retire) {
      auto it = by_gate_.find(gate);
      if (it != by_gate_.end()) {
        RetireLocked(it);
      }
    }
  }

  size_t gate_count() const { return by_gate_.size(); }
  size_t target_map_count() const { return by_target_.size(); }
  size_t caller_map_count() const { return by_caller_.size(); }
  size_t caller_by_gate_count() const { return caller_by_gate_.size(); }

 private:
  void RetireLocked(
      std::unordered_map<uintptr_t, a64::IndirectCallPicSite>::iterator it) {
    const uintptr_t gate_rx = it->first;
    if (it->second.state == a64::IndirectCallPicState::kPatched) {
      auto tit = by_target_.find(it->second.cached_guest_target);
      if (tit != by_target_.end()) {
        auto& vec = tit->second;
        vec.erase(std::remove(vec.begin(), vec.end(), gate_rx), vec.end());
        if (vec.empty()) {
          by_target_.erase(tit);
        }
      }
    }
    auto cit = caller_by_gate_.find(gate_rx);
    if (cit != caller_by_gate_.end()) {
      const uint32_t caller = cit->second;
      caller_by_gate_.erase(cit);
      auto bc = by_caller_.find(caller);
      if (bc != by_caller_.end()) {
        auto& vec = bc->second;
        vec.erase(std::remove(vec.begin(), vec.end(), gate_rx), vec.end());
        if (vec.empty()) {
          by_caller_.erase(bc);
        }
      }
    }
    by_gate_.erase(it);
  }

  std::unordered_map<uintptr_t, a64::IndirectCallPicSite> by_gate_;
  std::unordered_map<uint32_t, std::vector<uintptr_t>> by_target_;
  std::unordered_map<uintptr_t, uint32_t> caller_by_gate_;
  std::unordered_map<uint32_t, std::vector<uintptr_t>> by_caller_;
};

void TestPicEvictionAndPageAwareAlignment() {
  // 6a. PIC site eviction across all retirement paths leaves zero leaks.
  PicRegistryModel reg;
  const uintptr_t base_rx = 0x2A0010000ull;
  for (uint32_t i = 0; i < 400; ++i) {
    const uintptr_t g = base_rx + static_cast<uintptr_t>(i) * 128u;
    const uint32_t caller = 0x82100000u + (i / 4u) * 0x100u;
    reg.Register(g, g + 4, g + 8, g + 20, g + 64, caller);
  }
  WAVE7_1B_CHECK(reg.gate_count() == 400u);

  // Patch 0..199 in range.
  for (uint32_t i = 0; i < 200; ++i) {
    const uintptr_t g = base_rx + static_cast<uintptr_t>(i) * 128u;
    WAVE7_1B_CHECK(reg.TryPatch(g, 0x82500000u + (i % 10u) * 0x100u,
                                base_rx + 0x200000ull));
  }
  // Out-of-range (+128 MiB) on 200..249 -> immediately evicted.
  for (uint32_t i = 200; i < 250; ++i) {
    const uintptr_t g = base_rx + static_cast<uintptr_t>(i) * 128u;
    WAVE7_1B_CHECK(!reg.TryPatch(g, 0x82600000u, g + 20 + 0x08000000ull));
  }
  WAVE7_1B_CHECK(reg.gate_count() == 350u);

  // Direct fallback on 250..299 (uninitialized) and 0..49 (already patched).
  for (uint32_t i = 250; i < 300; ++i) {
    WAVE7_1B_CHECK(reg.Fallback(base_rx + static_cast<uintptr_t>(i) * 128u));
    // Second fallback on already-evicted gate is a safe no-op.
    WAVE7_1B_CHECK(!reg.Fallback(base_rx + static_cast<uintptr_t>(i) * 128u));
  }
  for (uint32_t i = 0; i < 50; ++i) {
    WAVE7_1B_CHECK(reg.Fallback(base_rx + static_cast<uintptr_t>(i) * 128u));
  }
  WAVE7_1B_CHECK(reg.gate_count() == 250u);

  // Sweep target range covering 0x82500000..0x82500FFF -> evicts all 150
  // remaining patched sites!
  reg.RevertRange(0x82500000u, 0x82500FFFu);
  WAVE7_1B_CHECK(reg.gate_count() == 100u);
  WAVE7_1B_CHECK(reg.target_map_count() == 0u);

  // Sweep caller range covering remaining 100 uninitialized sites (300..399).
  reg.RevertRange(0x82100000u, 0x82110000u);
  WAVE7_1B_CHECK(reg.gate_count() == 0u);
  WAVE7_1B_CHECK(reg.target_map_count() == 0u);
  WAVE7_1B_CHECK(reg.caller_map_count() == 0u);
  WAVE7_1B_CHECK(reg.caller_by_gate_count() == 0u);

  // 6b. 4 KB vs 16 KB host page loop alignment + parallel compilation mode.
  for (size_t off = 0; off < 4096; off += 4) {
    // 4 KB host must match ComputeLoopHeaderAlignmentPaddingBytes identically.
    const size_t pad4k =
        ComputeLoopHeaderAlignmentPaddingForPageRef(off, 64u, 4096u, true);
    WAVE7_1B_CHECK(pad4k ==
                   a64::ComputeLoopHeaderAlignmentPaddingBytes(off, 64u));
  }

  // 16 KB Android host: interior 4 KB boundaries (0x0FC0, 0x1FC0, 0x2FC0) must
  // NOT incur a 64 B anti-page-split nudge; only 0x3FC0..0x3FFC nudges to 16 KB.
  for (size_t interior : {size_t(0x0FC0u), size_t(0x1FC0u), size_t(0x2FC0u)}) {
    const size_t pad = ComputeLoopHeaderAlignmentPaddingForPageRef(
        interior, 0u, 16384u, true);
    WAVE7_1B_CHECK(pad == 0u);  // already 32 B aligned, not near 16 KB edge!
  }
  WAVE7_1B_CHECK(ComputeLoopHeaderAlignmentPaddingForPageRef(
                     0x3FC0u, 0u, 16384u, true) == 64u);
  WAVE7_1B_CHECK(ComputeLoopHeaderAlignmentPaddingForPageRef(
                     0x3FE4u, 0u, 16384u, true) == 28u);
  WAVE7_1B_CHECK(ComputeLoopHeaderAlignmentPaddingForPageRef(
                     0x3FFCu, 0u, 16384u, true) == 4u);

  // Parallel compilation (allow_page_nudge=false): strictly <= 28 B NOPs and
  // 32 B aligned relative to any 64 B-aligned entry address.
  for (size_t off = 0; off < 16384; off += 4) {
    const size_t pad =
        ComputeLoopHeaderAlignmentPaddingForPageRef(off, 0x3F80u, 16384u, false);
    WAVE7_1B_CHECK(pad <= 28u);
    WAVE7_1B_CHECK(((off + pad) & 31u) == 0u);
    WAVE7_1B_CHECK(((64u + off + pad) & 31u) == 0u);
  }
}

// ---------------------------------------------------------------------------
// 7. Source-level safety contracts across all 12 owned Wave 7 Onda 1B files
// ---------------------------------------------------------------------------
void TestSourceContracts(const fs::path& cpp_root) {
  const fs::path xenia = cpp_root / "xenia/src/xenia";

  // 7a. W^X boot probe policy + MapFileView MAP_FIXED_NOREPLACE.
  const std::string mem_h = ReadSourceText(xenia / "base/memory.h");
  const std::string mem_cc = ReadSourceText(xenia / "base/memory.cc");
  const std::string mem_posix = ReadSourceText(xenia / "base/memory_posix.cc");
  WAVE7_1B_CHECK(mem_h.find("ResolveWritableExecutableMemoryPreferred") !=
                 std::string::npos);
  WAVE7_1B_CHECK(mem_cc.find("writable_executable_memory, true,") !=
                 std::string::npos);
  WAVE7_1B_CHECK(mem_cc.find("a64_wx_dual_mapping, true,") !=
                 std::string::npos);
  const std::string wx_pref_body = ExtractFunctionBody(
      mem_cc, "bool IsWritableExecutableMemoryPreferred()");
  WAVE7_1B_CHECK(wx_pref_body.find("ResolveWritableExecutableMemoryPreferred(") !=
                 std::string::npos);
  WAVE7_1B_CHECK(wx_pref_body.find("if (!IsWritableExecutableMemorySupported())") ==
                 std::string::npos);

  const std::string map_fv_body =
      ExtractFunctionBody(mem_posix, "void* MapFileView(");
  WAVE7_1B_CHECK(map_fv_body.find("MAP_FIXED_NOREPLACE") != std::string::npos);
  WAVE7_1B_CHECK(map_fv_body.find("if (over_reservation || base_address != nullptr)") ==
                 std::string::npos);
  WAVE7_1B_CHECK(map_fv_body.find("EOPNOTSUPP") != std::string::npos);

  // 7b. VSCR[SAT] census: AddModule, late invalidation, kill-switch, in-flight retry.
  const std::string proc_h = ReadSourceText(xenia / "cpu/processor.h");
  const std::string proc_cc = ReadSourceText(xenia / "cpu/processor.cc");
  const std::string xex_cc = ReadSourceText(xenia / "cpu/xex_module.cc");
  WAVE7_1B_CHECK(proc_h.find("DECLARE_bool(vscr_sat_census);") !=
                 std::string::npos);
  WAVE7_1B_CHECK(proc_cc.find("vscr_sat_census, true,") != std::string::npos);
  const std::string add_mod_body =
      ExtractFunctionBody(proc_cc, "bool Processor::AddModule(");
  WAVE7_1B_CHECK(add_mod_body.find("OnModuleReadsVscr()") != std::string::npos);
  const std::string on_vscr_body =
      ExtractFunctionBody(proc_cc, "void Processor::OnModuleReadsVscr()");
  WAVE7_1B_CHECK(on_vscr_body.find("InvalidateCompiledFunctionsForVscr()") !=
                 std::string::npos);
  const std::string enable_dyn_body =
      ExtractFunctionBody(proc_cc, "void Processor::EnableDynamicCode()");
  WAVE7_1B_CHECK(enable_dyn_body.find("InvalidateCompiledFunctionsForVscr()") !=
                 std::string::npos);
  const std::string resolve_fn_body =
      ExtractFunctionBody(proc_cc, "Function* Processor::ResolveFunction(");
  WAVE7_1B_CHECK(resolve_fn_body.find("vscr_stale") != std::string::npos);
  const std::string reads_vscr_body =
      ExtractFunctionBody(xex_cc, "bool XexModule::ReadsVscr() const");
  WAVE7_1B_CHECK(reads_vscr_body.find("if (!ptr)") != std::string::npos);

  // 7c. Unified FlushCodeRange, ResetIndirection, PIC eviction, tail-call PIC,
  //     and thread-safe 16 KB page-aware loop alignment.
  const std::string cc_base =
      ReadSourceText(xenia / "cpu/backend/code_cache_base.h");
  const std::string place_guest =
      ExtractFunctionBody(cc_base, "void PlaceGuestCode(");
  // Count `FlushCodeRange(` call sites inside PlaceGuestCode: must be unified
  // into 1 requires-guarded dispatch (3 mentions: requires + 3-arg + 2-arg).
  size_t flush_mentions = 0;
  for (size_t p = 0;
       (p = place_guest.find("self().FlushCodeRange(", p)) != std::string::npos;
       p += 22) {
    ++flush_mentions;
  }
  WAVE7_1B_CHECK(flush_mentions == 3u);
  WAVE7_1B_CHECK(cc_base.find("void ResetIndirection(") != std::string::npos);
  WAVE7_1B_CHECK(cc_base.find("EnterCodeEmitter()") != std::string::npos);
  WAVE7_1B_CHECK(cc_base.find("std::memory_order_acquire") != std::string::npos);

  const std::string a64_cc_h =
      ReadSourceText(xenia / "cpu/backend/a64/a64_code_cache.h");
  const std::string a64_cc_cc =
      ReadSourceText(xenia / "cpu/backend/a64/a64_code_cache.cc");
  const std::string a64_be_cc =
      ReadSourceText(xenia / "cpu/backend/a64/a64_backend.cc");
  const std::string a64_em_cc =
      ReadSourceText(xenia / "cpu/backend/a64/a64_emitter.cc");
  WAVE7_1B_CHECK(a64_cc_h.find("ComputeLoopHeaderAlignmentPaddingForPage") !=
                 std::string::npos);
  WAVE7_1B_CHECK(a64_cc_h.find("TrackedPicSiteCount()") != std::string::npos);
  WAVE7_1B_CHECK(a64_cc_cc.find("RetirePicSiteLocked(") != std::string::npos);
  const std::string fb_pic = ExtractFunctionBody(
      a64_cc_cc, "bool A64CodeCache::FallbackIndirectCallPicSite(");
  WAVE7_1B_CHECK(fb_pic.find("RetirePicSiteLocked(") != std::string::npos);
  const std::string rev_pic = ExtractFunctionBody(
      a64_cc_cc, "void A64CodeCache::RevertDirectCallSitesInRange(");
  WAVE7_1B_CHECK(rev_pic.find("RetirePicSiteLocked(") != std::string::npos);
  WAVE7_1B_CHECK(rev_pic.find("pic_sites_by_caller_") != std::string::npos);

  const std::string call_ind = ExtractFunctionBody(
      a64_em_cc, "void A64Emitter::CallIndirect(");
  WAVE7_1B_CHECK(call_ind.find("cvars::a64_indirect_call_pic && !is_tail_call") !=
                 std::string::npos);
  WAVE7_1B_CHECK(a64_em_cc.find("EnterCodeEmitter()") != std::string::npos);
  WAVE7_1B_CHECK(a64_em_cc.find("ComputeLoopHeaderAlignmentPaddingForPage(") !=
                 std::string::npos);

  const std::string patch_pic_be = ExtractFunctionBody(
      a64_be_cc, "uint64_t PatchAndResolveIndirectCallPic(");
  WAVE7_1B_CHECK(patch_pic_be.find("dynamic_target_in_body = 0;") !=
                 std::string::npos);
  const std::string inv_dyn = ExtractFunctionBody(
      a64_be_cc, "void A64Backend::InvalidateDynamicCalls(");
  WAVE7_1B_CHECK(inv_dyn.find("ResetIndirection(") != std::string::npos);

  // 7d. POSIX threading (d92b4a2ffe release_count_, ESRCH, pre-cleanup wait,
  //     GetCurrentThread unique_ptr), XEvent::SyncFromGuest, xam_task null-check,
  //     clock_posix 1e9 Hz parity.
  const std::string thr_posix =
      ReadSourceText(xenia / "base/threading_posix.cc");
  WAVE7_1B_CHECK(thr_posix.find("release_count_") != std::string::npos);
  WAVE7_1B_CHECK(thr_posix.find("ReleaseCounts(") != std::string::npos);
  const std::string term_body =
      ExtractFunctionBody(thr_posix, "void Terminate(int exit_code)");
  WAVE7_1B_CHECK(term_body.find("ESRCH") != std::string::npos);
  WAVE7_1B_CHECK(term_body.find("State::kUninitialized") != std::string::npos);
  WAVE7_1B_CHECK(term_body.find("assert_always()") == std::string::npos);
  const std::string get_curr =
      ExtractFunctionBody(thr_posix, "Thread* Thread::GetCurrentThread()");
  WAVE7_1B_CHECK(get_curr.find("owned_current_thread_") != std::string::npos);
  WAVE7_1B_CHECK(get_curr.find("new PosixThread") == std::string::npos);

  const std::string xevent_cc = ReadSourceText(xenia / "kernel/xevent.cc");
  const std::string sync_guest =
      ExtractFunctionBody(xevent_cc, "void XEvent::SyncFromGuest()");
  const size_t sg_lock = sync_guest.find("global_critical_region_.Acquire()");
  const size_t sg_epoch = sync_guest.find("pulse_epoch_.fetch_add(1)");
  const size_t sg_wake = sync_guest.find("WakeCooperativeWaiters()");
  WAVE7_1B_CHECK(sg_lock != std::string::npos);
  WAVE7_1B_CHECK(sg_epoch != std::string::npos);
  WAVE7_1B_CHECK(sg_wake != std::string::npos);
  WAVE7_1B_CHECK(sg_lock < sg_epoch && sg_epoch < sg_wake);

  const std::string xam_task = ReadSourceText(xenia / "kernel/xam/xam_task.cc");
  const std::string xam_sched =
      ExtractFunctionBody(xam_task, "dword_result_t XamTaskSchedule_entry(");
  WAVE7_1B_CHECK(xam_sched.find("if (title_process && kthread)") !=
                 std::string::npos);
  WAVE7_1B_CHECK(xam_sched.find("if (handle_ptr)") != std::string::npos);

  const std::string clock_posix =
      ReadSourceText(xenia / "base/clock_posix.cc");
  const std::string freq_plat = ExtractFunctionBody(
      clock_posix, "uint64_t Clock::host_tick_frequency_platform()");
  const std::string freq_plat_code = StripLineComments(freq_plat);
  WAVE7_1B_CHECK(freq_plat_code.find("clock_getres") == std::string::npos);
  WAVE7_1B_CHECK(freq_plat_code.find("1000000000ull") != std::string::npos);

  // 7e. CodeCacheBase #161/#162 P0 fix: round_up(..., false) at offset 0 and
  //     unwind_reservation.data_size == 0.
  const std::string pred_off =
      ExtractFunctionBody(cc_base, "size_t predicted_next_code_offset(");
  WAVE7_1B_CHECK(pred_off.find("xe::round_up(hint, alignment, false)") !=
                 std::string::npos);
  WAVE7_1B_CHECK(
      place_guest.find("xe::round_up(pad_start_offset, entry_alignment, false)") !=
      std::string::npos);
  WAVE7_1B_CHECK(
      place_guest.find("xe::round_up(unwind_reservation.data_size, 16, false)") !=
      std::string::npos);
  WAVE7_1B_CHECK(place_guest.find("if (end_write_address > tail_write_address)") !=
                 std::string::npos);

  // 7f. A64Emitter #168 P0 deadlock fix: AddToTail returns heap-stable
  //     NewCachedLabel() (never dangling tail_code_.back().label), tail loop is
  //     index-based, ~A64Emitter cleans up, and Emit rejects undefined labels.
  const std::string add_to_tail = StripLineComments(
      ExtractFunctionBody(a64_em_cc, "Label& A64Emitter::AddToTail("));
  WAVE7_1B_CHECK(add_to_tail.find("NewCachedLabel()") != std::string::npos);
  WAVE7_1B_CHECK(add_to_tail.find("e.L(stable_label)") != std::string::npos);
  WAVE7_1B_CHECK(add_to_tail.find("return stable_label;") != std::string::npos);
  WAVE7_1B_CHECK(add_to_tail.find("tail_code_.back().label") ==
                 std::string::npos);
  WAVE7_1B_CHECK(a64_em_cc.find("for (size_t tail_idx = 0; tail_idx < tail_code_.size();") !=
                 std::string::npos);
  WAVE7_1B_CHECK(a64_em_cc.find("if (hasUndefinedLabel())") !=
                 std::string::npos);
  WAVE7_1B_CHECK(a64_em_cc.find("A64Emitter::~A64Emitter() { ResetPerFunctionState(); }") !=
                 std::string::npos);
}

// ---------------------------------------------------------------------------
// 8. CodeCacheBase #161/#162 P0: round_up(0, align, false) placement &
//    zero-unwind zero-waste + single unified FlushCodeRange invariant
// ---------------------------------------------------------------------------
void TestCodeCacheBaseZeroOffsetAndZeroUnwind() {
  // Confirm the underlying math.h default footgun that caused #161:
  WAVE7_1B_CHECK(xe::round_up<size_t>(0u, 64u) == 64u);
  WAVE7_1B_CHECK(xe::round_up<size_t>(0u, 64u, false) == 0u);
  WAVE7_1B_CHECK(xe::round_up<size_t>(0u, 16u, false) == 0u);

  // Exact placement simulator mirroring CodeCacheBase::PlaceGuestCode &
  // predicted_next_code_offset.
  struct SimCodeCache {
    size_t offset = 0;
    size_t total_fill_calls = 0;
    size_t total_fill_bytes = 0;
    size_t total_flush_calls = 0;
    size_t last_flush_start = 0;
    size_t last_flush_size = 0;

    size_t PredictedNextOffset(size_t alignment) const {
      return alignment ? xe::round_up(offset, alignment, false) : offset;
    }

    size_t Place(size_t code_size, size_t entry_alignment,
                 size_t unwind_data_size) {
      const size_t pad_start_offset = offset;
      const size_t aligned_entry_offset =
          entry_alignment > 16
              ? xe::round_up(pad_start_offset, entry_alignment, false)
              : pad_start_offset;
      offset = aligned_entry_offset;
      offset += xe::round_up(code_size, size_t(16));
      const size_t tail_offset = offset;
      offset += xe::round_up(unwind_data_size, size_t(16), false);
      const size_t end_offset = offset;

      if (aligned_entry_offset > pad_start_offset) {
        ++total_fill_calls;
        total_fill_bytes += aligned_entry_offset - pad_start_offset;
      }
      if (end_offset > tail_offset) {
        ++total_fill_calls;
        total_fill_bytes += end_offset - tail_offset;
      }
      const size_t flush_size = offset - pad_start_offset;
      if (flush_size > 0) {
        ++total_flush_calls;
        last_flush_start = pad_start_offset;
        last_flush_size = flush_size;
      }
      return aligned_entry_offset;
    }
  };

  SimCodeCache sim;
  // (a) First placement (resolve_function_thunk_, >= 64 B -> 64 B alignment,
  //     unwind_data_size == 0 on Android): MUST start at offset 0 with zero
  //     leading BRK #0 padding, zero FillCode calls, zero wasted tail bytes,
  //     and exactly 1 unified FlushCodeRange.
  WAVE7_1B_CHECK(sim.PredictedNextOffset(64u) == 0u);
  const size_t thunk_off = sim.Place(/*code_size=*/80u, /*entry_alignment=*/64u,
                                     /*unwind_data_size=*/0u);
  WAVE7_1B_CHECK(thunk_off == 0u);
  WAVE7_1B_CHECK(sim.offset == 80u);
  WAVE7_1B_CHECK(sim.total_fill_calls == 0u);
  WAVE7_1B_CHECK(sim.total_fill_bytes == 0u);
  WAVE7_1B_CHECK(sim.total_flush_calls == 1u);
  WAVE7_1B_CHECK(sim.last_flush_start == 0u);
  WAVE7_1B_CHECK(sim.last_flush_size == 80u);

  // (b) Subsequent 64 B-aligned placement with unwind_data_size == 0 (Android):
  //     predicted offset matches actual aligned_entry_offset (128), fills only
  //     the 48 B pre-entry alignment gap (zero tail fill), and issues 1 flush
  //     covering [80, 192).
  WAVE7_1B_CHECK(sim.PredictedNextOffset(64u) == 128u);
  const size_t fn2_off = sim.Place(/*code_size=*/64u, /*entry_alignment=*/64u,
                                   /*unwind_data_size=*/0u);
  WAVE7_1B_CHECK(fn2_off == 128u);
  WAVE7_1B_CHECK(sim.offset == 192u);
  WAVE7_1B_CHECK(sim.total_fill_calls == 1u);
  WAVE7_1B_CHECK(sim.total_fill_bytes == 48u);
  WAVE7_1B_CHECK(sim.total_flush_calls == 2u);
  WAVE7_1B_CHECK(sim.last_flush_start == 80u);
  WAVE7_1B_CHECK(sim.last_flush_size == 112u);

  // (c) Placement with non-zero unwind_data_size (e.g. Windows/Mac 12 B -> 16 B)
  //     at already 64 B-aligned offset 192: zero pre-entry fill, 16 B tail fill,
  //     and still 1 single unified FlushCodeRange covering [192, 240).
  WAVE7_1B_CHECK(sim.PredictedNextOffset(64u) == 192u);
  const size_t fn3_off = sim.Place(/*code_size=*/32u, /*entry_alignment=*/64u,
                                   /*unwind_data_size=*/12u);
  WAVE7_1B_CHECK(fn3_off == 192u);
  WAVE7_1B_CHECK(sim.offset == 240u);
  WAVE7_1B_CHECK(sim.total_fill_calls == 2u);
  WAVE7_1B_CHECK(sim.total_fill_bytes == 64u);
  WAVE7_1B_CHECK(sim.total_flush_calls == 3u);
  WAVE7_1B_CHECK(sim.last_flush_start == 192u);
  WAVE7_1B_CHECK(sim.last_flush_size == 48u);
}

// ---------------------------------------------------------------------------
// 9. A64Emitter #168 P0 deadlock: AddToTail label stability across vector
//    reallocations and nested tail appends (eliminating unbound `b.ne .+0`)
// ---------------------------------------------------------------------------
void TestAddToTailStableLabelResolution() {
  // Mirrors xbyak_aarch64::Label + LabelManager forward-branch backpatching
  // and A64Emitter::AddToTail + index-based tail emission.
  struct SimLabel {
    int id = 0;
  };
  struct SimEmitter {
    int next_label_id = 1;
    std::vector<uint32_t> code;
    std::unordered_map<int, size_t> defined_offsets;
    std::vector<std::pair<int, size_t>> undef_bne_sites;
    std::vector<std::unique_ptr<SimLabel>> label_cache;
    std::vector<std::function<void(SimEmitter&)>> tail_code;

    SimLabel& NewCachedLabel() {
      label_cache.push_back(std::make_unique<SimLabel>());
      return *label_cache.back();
    }

    int GetId(SimLabel& lbl) {
      if (lbl.id == 0) {
        lbl.id = next_label_id++;
      }
      return lbl.id;
    }

    void EmitNop() { code.push_back(0xD503201Fu); }

    // Encodes ARM64 `b.ne <rel19>` (placeholder 0x54000001 = `b.ne .+0` when
    // forward/unresolved, patched to exact byte delta when L() defines label).
    size_t EmitBne(SimLabel& lbl) {
      const size_t idx = code.size();
      const int id = GetId(lbl);
      auto it = defined_offsets.find(id);
      if (it == defined_offsets.end()) {
        code.push_back(0x54000001u);  // b.ne .+0 placeholder
        undef_bne_sites.emplace_back(id, idx);
      } else {
        const int64_t delta_words =
            static_cast<int64_t>(it->second) - static_cast<int64_t>(idx);
        const uint32_t imm19 = static_cast<uint32_t>(delta_words) & 0x7FFFFu;
        code.push_back(0x54000001u | (imm19 << 5));
      }
      return idx;
    }

    void L(SimLabel& lbl) {
      const int id = GetId(lbl);
      const size_t target_idx = code.size();
      defined_offsets[id] = target_idx;
      std::vector<std::pair<int, size_t>> remaining;
      for (const auto& site : undef_bne_sites) {
        if (site.first == id) {
          const int64_t delta_words =
              static_cast<int64_t>(target_idx) - static_cast<int64_t>(site.second);
          const uint32_t imm19 = static_cast<uint32_t>(delta_words) & 0x7FFFFu;
          code[site.second] = 0x54000001u | (imm19 << 5);
        } else {
          remaining.push_back(site);
        }
      }
      undef_bne_sites = std::move(remaining);
    }

    SimLabel& AddToTail(std::function<void(SimEmitter&, SimLabel&)> cb) {
      SimLabel& stable_label = NewCachedLabel();
      tail_code.push_back(
          [cb = std::move(cb), &stable_label](SimEmitter& e) {
            e.L(stable_label);
            cb(e, stable_label);
          });
      return stable_label;
    }

    void EmitTails() {
      for (size_t i = 0; i < tail_code.size(); ++i) {
        auto fn = tail_code[i];
        fn(*this);
      }
    }

    bool HasUndefinedLabel() const { return !undef_bne_sites.empty(); }
  };

  SimEmitter em;
  // Simulate PushStackpoint() adding 1 tail (tail_code size=1, cap=1), then
  // multiple CallIndirect PIC sites each adding TWO tails (pic_fallback_lookup
  // then pic_miss_stub) forcing repeated std::vector reallocations (1->2->4->8)
  // BEFORE emitting b.ne pic_fallback_lookup!
  SimLabel& overflow = em.AddToTail([](SimEmitter& e, SimLabel&) {
    e.EmitNop();
  });
  const size_t overflow_br = em.EmitBne(overflow);

  std::vector<size_t> pic_bne_sites;
  for (int site = 0; site < 10; ++site) {
    SimLabel& pic_fallback = em.AddToTail([site](SimEmitter& e, SimLabel&) {
      e.EmitNop();
      // Also exercise nested AddToTail inside a tail callback on site 0.
      if (site == 0) {
        SimLabel& nested = e.AddToTail([](SimEmitter& ne, SimLabel&) {
          ne.EmitNop();
        });
        e.EmitBne(nested);
      }
    });
    SimLabel& pic_miss = em.AddToTail([](SimEmitter& e, SimLabel&) {
      e.EmitNop();
    });
    em.EmitBne(pic_miss);
    pic_bne_sites.push_back(em.EmitBne(pic_fallback));
  }
  em.EmitTails();

  WAVE7_1B_CHECK(!em.HasUndefinedLabel());
  WAVE7_1B_CHECK(em.code[overflow_br] != 0x54000001u);
  for (size_t bne_idx : pic_bne_sites) {
    // None of the `b.ne pic_fallback` instructions may remain `b.ne .+0`!
    WAVE7_1B_CHECK(em.code[bne_idx] != 0x54000001u);
    const uint32_t imm19 = (em.code[bne_idx] >> 5) & 0x7FFFFu;
    WAVE7_1B_CHECK(imm19 > 0u && bne_idx + imm19 < em.code.size());
  }
}

}  // namespace

int main(int argc, char** argv) {
  TestWxBootProbePolicy();
#if defined(__linux__)
  TestPosixRealMemfdDualMappingAndNoReplace();
#endif
  TestClockPosixScaleParity();
  TestManualResetReleaseCountSemantics();
  TestVscrSatCensusLateTransition();
  TestPicEvictionAndPageAwareAlignment();
  TestCodeCacheBaseZeroOffsetAndZeroUnwind();
  TestAddToTailStableLabelResolution();
  if (argc >= 2 && argv[1] && argv[1][0] != '\0') {
    TestSourceContracts(fs::path(argv[1]));
  }
  std::printf("wave7_onda1b_cpu_wx_test: PASS\n");
  return 0;
}
