/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_KERNEL_ADPF_TID_REGISTRY_H_
#define XENIA_KERNEL_ADPF_TID_REGISTRY_H_

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <mutex>
#include <utility>
#include <vector>

#if defined(__ANDROID__) && !defined(XE_DISABLE_ANDROID_DLOPEN)
#include <dlfcn.h>
#include <unistd.h>
#elif defined(__ANDROID__) || defined(__linux__)
#include <unistd.h>
#endif

namespace xe {
namespace kernel {

template <typename Pred>
std::vector<int32_t> FilterAliveTids(const std::vector<int32_t>& candidate_tids,
                                     Pred&& is_alive) {
  std::vector<int32_t> result;
  result.reserve(candidate_tids.size());
  for (int32_t tid : candidate_tids) {
    if (tid <= 0) {
      continue;
    }
    if (std::find(result.begin(), result.end(), tid) != result.end()) {
      continue;
    }
    if (is_alive(tid)) {
      result.push_back(tid);
    }
  }
  return result;
}

inline bool IsProcSelfTaskAlive(int32_t tid) {
#if defined(__ANDROID__) || defined(__linux__)
  if (tid <= 0) {
    return false;
  }
  char path[48];
  std::snprintf(path, sizeof(path), "/proc/self/task/%d", tid);
  return access(path, F_OK) == 0;
#else
  (void)tid;
  return false;
#endif
}

class GuestCpuTidRegistry {
 public:
  void RegisterTid(int32_t tid) {
    if (tid <= 0) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (std::find(tids_.begin(), tids_.end(), tid) == tids_.end()) {
      tids_.push_back(tid);
      generation_.fetch_add(1, std::memory_order_acq_rel);
    }
  }

  void UnregisterTid(int32_t tid) {
    if (tid <= 0) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = std::find(tids_.begin(), tids_.end(), tid);
    if (it != tids_.end()) {
      tids_.erase(it);
      generation_.fetch_add(1, std::memory_order_acq_rel);
    }
  }

  void Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!tids_.empty()) {
      tids_.clear();
      generation_.fetch_add(1, std::memory_order_acq_rel);
    }
  }

  template <typename Pred>
  std::vector<int32_t> SnapshotLiveTids(Pred&& is_alive) const {
    std::vector<int32_t> copy;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      copy = tids_;
    }
    return FilterAliveTids(copy, std::forward<Pred>(is_alive));
  }

  std::vector<int32_t> SnapshotLiveTids() const {
    return SnapshotLiveTids(IsProcSelfTaskAlive);
  }

  uint64_t generation() const {
    return generation_.load(std::memory_order_acquire);
  }

 private:
  mutable std::mutex mutex_;
  std::vector<int32_t> tids_;
  std::atomic<uint64_t> generation_{0};
};

class CpuPerformanceHintSession {
 public:
  CpuPerformanceHintSession() = default;
  ~CpuPerformanceHintSession() { Shutdown(); }

  CpuPerformanceHintSession(const CpuPerformanceHintSession&) = delete;
  CpuPerformanceHintSession& operator=(const CpuPerformanceHintSession&) =
      delete;

  void OnFrameTick(bool enabled, uint64_t frame_duration_ns,
                   const GuestCpuTidRegistry& registry) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!enabled) {
      ShutdownLocked();
      failed_ = false;
      return;
    }
#if defined(__ANDROID__) && !defined(XE_DISABLE_ANDROID_DLOPEN)
    if (failed_) {
      return;
    }
    if (!EnsureApiLoadedLocked()) {
      failed_ = true;
      return;
    }
    if (frame_duration_ns > 0) {
      interval_sum_ns_ += frame_duration_ns;
      ++interval_count_;
    }
    if (!session_) {
      if (interval_count_ < kWarmupFrames) {
        return;
      }
      const uint64_t gen = registry.generation();
      std::vector<int32_t> live_tids = registry.SnapshotLiveTids();
      if (live_tids.empty()) {
        return;
      }
      const int64_t target_ns = ClampTargetNs(interval_sum_ns_ / interval_count_);
      session_ = create_session_(manager_, live_tids.data(), live_tids.size(),
                                 target_ns);
      if (!session_) {
        std::vector<int32_t> retried =
            FilterAliveTids(live_tids, IsProcSelfTaskAlive);
        if (!retried.empty() && retried.size() != live_tids.size()) {
          live_tids.swap(retried);
          session_ = create_session_(manager_, live_tids.data(),
                                     live_tids.size(), target_ns);
        }
      }
      if (!session_) {
        failed_ = true;
        return;
      }
      target_ns_ = target_ns;
      active_tids_ = std::move(live_tids);
      last_generation_ = gen;
      interval_sum_ns_ = 0;
      interval_count_ = 0;
      return;
    }

    const uint64_t gen = registry.generation();
    if (gen != last_generation_) {
      std::vector<int32_t> live_tids = registry.SnapshotLiveTids();
      if (live_tids.empty()) {
        CloseSessionOnlyLocked();
        last_generation_ = gen;
        return;
      }
      if (live_tids != active_tids_) {
        if (!UpdateSessionThreadsLocked(live_tids)) {
          CloseSessionOnlyLocked();
          live_tids = registry.SnapshotLiveTids();
          if (live_tids.empty()) {
            last_generation_ = gen;
            return;
          }
          session_ = create_session_(manager_, live_tids.data(),
                                     live_tids.size(), target_ns_);
          if (!session_) {
            failed_ = true;
            return;
          }
        }
        active_tids_ = std::move(live_tids);
      }
      last_generation_ = gen;
    }

    if (session_ && report_actual_work_ && frame_duration_ns > 0) {
      const uint64_t clamped = std::min<uint64_t>(
          frame_duration_ns, static_cast<uint64_t>(std::numeric_limits<int64_t>::max()));
      report_actual_work_(session_, static_cast<int64_t>(clamped));
    }
    if (session_ && update_target_work_ && interval_count_ >= kTargetUpdateFrames) {
      const int64_t new_target_ns =
          ClampTargetNs(interval_sum_ns_ / interval_count_);
      update_target_work_(session_, new_target_ns);
      target_ns_ = new_target_ns;
      interval_sum_ns_ = 0;
      interval_count_ = 0;
    }
#else
    (void)frame_duration_ns;
    (void)registry;
#endif
  }

  void Shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    ShutdownLocked();
    failed_ = false;
  }

  bool active() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return session_ != nullptr;
  }

 private:
  static constexpr uint32_t kWarmupFrames = 60;
  static constexpr uint32_t kTargetUpdateFrames = 120;

  static int64_t ClampTargetNs(uint64_t avg_ns) {
    const uint64_t min_ns = 1'000'000ull;
    const uint64_t max_ns = 100'000'000ull;
    return static_cast<int64_t>(std::clamp(avg_ns, min_ns, max_ns));
  }

  void CloseSessionOnlyLocked() {
#if defined(__ANDROID__) && !defined(XE_DISABLE_ANDROID_DLOPEN)
    if (session_ && close_session_) {
      close_session_(session_);
    }
#endif
    session_ = nullptr;
    active_tids_.clear();
  }

  void ShutdownLocked() {
    CloseSessionOnlyLocked();
    interval_sum_ns_ = 0;
    interval_count_ = 0;
    last_generation_ = 0;
    target_ns_ = 16'666'667;
  }

#if defined(__ANDROID__) && !defined(XE_DISABLE_ANDROID_DLOPEN)
  bool EnsureApiLoadedLocked() {
    if (api_checked_) {
      return manager_ && create_session_ && report_actual_work_ &&
             close_session_;
    }
    api_checked_ = true;
    void* lib = dlopen("libandroid.so", RTLD_NOW | RTLD_NOLOAD);
    if (!lib) {
      lib = dlopen("libandroid.so", RTLD_NOW | RTLD_LOCAL);
    }
    if (!lib) {
      return false;
    }
    auto get_manager = reinterpret_cast<void* (*)()>(
        dlsym(lib, "APerformanceHint_getManager"));
    create_session_ = reinterpret_cast<void* (*)(
        void*, const int32_t*, size_t, int64_t)>(
        dlsym(lib, "APerformanceHint_createSession"));
    set_threads_ = reinterpret_cast<int (*)(void*, const int32_t*, size_t)>(
        dlsym(lib, "APerformanceHint_setThreads"));
    report_actual_work_ = reinterpret_cast<int (*)(void*, int64_t)>(
        dlsym(lib, "APerformanceHint_reportActualWorkDuration"));
    update_target_work_ = reinterpret_cast<int (*)(void*, int64_t)>(
        dlsym(lib, "APerformanceHint_updateTargetWorkDuration"));
    close_session_ = reinterpret_cast<void (*)(void*)>(
        dlsym(lib, "APerformanceHint_closeSession"));
    manager_ = get_manager ? get_manager() : nullptr;
    return manager_ && create_session_ && report_actual_work_ &&
           close_session_;
  }

  bool UpdateSessionThreadsLocked(std::vector<int32_t>& live_tids) {
    if (!session_ || !set_threads_ || live_tids.empty()) {
      return false;
    }
    if (set_threads_(session_, live_tids.data(), live_tids.size()) == 0) {
      return true;
    }
    std::vector<int32_t> retried =
        FilterAliveTids(live_tids, IsProcSelfTaskAlive);
    if (!retried.empty() && retried.size() != live_tids.size() &&
        set_threads_(session_, retried.data(), retried.size()) == 0) {
      live_tids.swap(retried);
      return true;
    }
    return false;
  }

  bool api_checked_ = false;
  void* manager_ = nullptr;
  void* (*create_session_)(void*, const int32_t*, size_t, int64_t) = nullptr;
  int (*set_threads_)(void*, const int32_t*, size_t) = nullptr;
  int (*report_actual_work_)(void*, int64_t) = nullptr;
  int (*update_target_work_)(void*, int64_t) = nullptr;
  void (*close_session_)(void*) = nullptr;
#endif

  mutable std::mutex mutex_;
  void* session_ = nullptr;
  bool failed_ = false;
  uint64_t interval_sum_ns_ = 0;
  uint32_t interval_count_ = 0;
  int64_t target_ns_ = 16'666'667;
  uint64_t last_generation_ = 0;
  std::vector<int32_t> active_tids_;
};

}  // namespace kernel
}  // namespace xe

#endif  // XENIA_KERNEL_ADPF_TID_REGISTRY_H_
