#ifndef XENDROID_XE_AUDIO_ADPF_LOGIC_H_
#define XENDROID_XE_AUDIO_ADPF_LOGIC_H_

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace xendroid {

inline std::string_view TrimThreadComm(std::string_view comm) {
  while (!comm.empty()) {
    const char ch = comm.back();
    if (ch == '\n' || ch == '\r' || ch == '\0' || ch == ' ') {
      comm.remove_suffix(1);
    } else {
      break;
    }
  }
  return comm;
}

// Audio ADPF session tracks ONLY latency-critical audio threads (Audio Worker
// and XMA Decoder), never Guest CPU 0..5 or Guest I/O*.
inline bool IsAudioAdpfThreadName(std::string_view comm) {
  const std::string_view trimmed = TrimThreadComm(comm);
  return trimmed == "Audio Worker" || trimmed == "XMA Decoder";
}

struct ThreadCommEntry {
  int32_t tid = 0;
  std::string comm;
};

// Keeps the lowest live positive TID per audio thread name, returning sorted
// unique TIDs (at most 2: Audio Worker + XMA Decoder).
inline std::vector<int32_t> SelectAudioAdpfThreadTids(
    const std::vector<ThreadCommEntry>& entries) {
  std::map<std::string, int32_t> min_tid_by_name;
  for (const auto& entry : entries) {
    if (entry.tid <= 0) {
      continue;
    }
    const std::string_view name = TrimThreadComm(entry.comm);
    if (!IsAudioAdpfThreadName(name)) {
      continue;
    }
    auto [it, inserted] =
        min_tid_by_name.emplace(std::string(name), entry.tid);
    if (!inserted && entry.tid < it->second) {
      it->second = entry.tid;
    }
  }
  std::vector<int32_t> tids;
  tids.reserve(min_tid_by_name.size());
  for (const auto& [name, tid] : min_tid_by_name) {
    (void)name;
    tids.push_back(tid);
  }
  std::sort(tids.begin(), tids.end());
  tids.erase(std::unique(tids.begin(), tids.end()), tids.end());
  return tids;
}

enum class AudioAdpfThreadRole {
  kAudioWorker,
  kXmaDecoder,
};

// Lock-free registry for Audio Worker and XMA Decoder host TIDs so the ADPF
// session can start immediately and dynamically update via setThreads when
// either thread comes up or restarts.
class AudioAdpfThreadRegistry {
 public:
  void SetAudioWorkerTid(int32_t tid) { SetSlot(audio_worker_tid_, tid); }
  void ClearAudioWorkerTid(int32_t tid) { ClearSlot(audio_worker_tid_, tid); }
  void SetXmaDecoderTid(int32_t tid) { SetSlot(xma_decoder_tid_, tid); }
  void ClearXmaDecoderTid(int32_t tid) { ClearSlot(xma_decoder_tid_, tid); }

  uint64_t generation() const {
    return generation_.load(std::memory_order_acquire);
  }

  template <typename Pred>
  std::vector<int32_t> SnapshotLiveTids(Pred&& is_alive) const {
    const int32_t worker = audio_worker_tid_.load(std::memory_order_acquire);
    const int32_t xma = xma_decoder_tid_.load(std::memory_order_acquire);
    std::vector<int32_t> tids;
    tids.reserve(2);
    if (worker > 0 && is_alive(worker)) {
      tids.push_back(worker);
    }
    if (xma > 0 && xma != worker && is_alive(xma)) {
      tids.push_back(xma);
    }
    std::sort(tids.begin(), tids.end());
    return tids;
  }

 private:
  void SetSlot(std::atomic<int32_t>& slot, int32_t tid) {
    if (tid <= 0) {
      return;
    }
    const int32_t prev = slot.exchange(tid, std::memory_order_acq_rel);
    if (prev != tid) {
      generation_.fetch_add(1, std::memory_order_acq_rel);
    }
  }

  void ClearSlot(std::atomic<int32_t>& slot, int32_t tid) {
    if (tid <= 0) {
      return;
    }
    int32_t expected = tid;
    if (slot.compare_exchange_strong(expected, 0, std::memory_order_acq_rel)) {
      generation_.fetch_add(1, std::memory_order_acq_rel);
    }
  }

  std::atomic<int32_t> audio_worker_tid_{0};
  std::atomic<int32_t> xma_decoder_tid_{0};
  std::atomic<uint64_t> generation_{0};
};

inline AudioAdpfThreadRegistry& GlobalAudioAdpfThreadRegistry() {
  static AudioAdpfThreadRegistry registry;
  return registry;
}

class ScopedAudioAdpfThread {
 public:
  ScopedAudioAdpfThread(AudioAdpfThreadRegistry& registry,
                        AudioAdpfThreadRole role, int32_t tid)
      : registry_(registry), role_(role), tid_(tid) {
    if (role_ == AudioAdpfThreadRole::kAudioWorker) {
      registry_.SetAudioWorkerTid(tid_);
    } else {
      registry_.SetXmaDecoderTid(tid_);
    }
  }

  ~ScopedAudioAdpfThread() {
    if (role_ == AudioAdpfThreadRole::kAudioWorker) {
      registry_.ClearAudioWorkerTid(tid_);
    } else {
      registry_.ClearXmaDecoderTid(tid_);
    }
  }

  ScopedAudioAdpfThread(const ScopedAudioAdpfThread&) = delete;
  ScopedAudioAdpfThread& operator=(const ScopedAudioAdpfThread&) = delete;

 private:
  AudioAdpfThreadRegistry& registry_;
  AudioAdpfThreadRole role_;
  int32_t tid_;
};

struct AudioAdpfOps {
  void* (*create_session)(void* user, const int32_t* tids, size_t count,
                          int64_t target_ns) = nullptr;
  int (*set_threads)(void* user, void* session, const int32_t* tids,
                     size_t count) = nullptr;
  int (*report_actual_work)(void* user, void* session,
                            int64_t actual_ns) = nullptr;
  void (*close_session)(void* user, void* session) = nullptr;
  bool (*is_tid_alive)(void* user, int32_t tid) = nullptr;
  void* user = nullptr;
};

struct CallbackTimeBreakdown {
  uint64_t wall_us = 0;
  uint64_t cpu_us = 0;
  uint64_t run_queue_wait_us = 0;
  uint64_t blocked_sleep_us = 0;
};

// Separates guest callback wall time into on-CPU execution, OS run-queue wait,
// and blocked/sleeping time (waiting on guest events, locks, or XMA kicks).
inline CallbackTimeBreakdown ComputeCallbackTimeBreakdown(
    uint64_t wall_us, uint64_t cpu_us, uint64_t run_queue_wait_us) {
  CallbackTimeBreakdown out;
  out.wall_us = wall_us;
  out.cpu_us = cpu_us;
  out.run_queue_wait_us = run_queue_wait_us;
  const uint64_t accounted = cpu_us + run_queue_wait_us;
  const uint64_t capped =
      (accounted < cpu_us) ? wall_us : std::min(wall_us, accounted);
  out.blocked_sleep_us = wall_us - capped;
  return out;
}

// Active work duration to report to ADPF: cpu_us + run_queue_wait_us, strictly
// excluding blocked_sleep_us so waiting on guest locks/events/XMA never
// inflates ADPF reports and causes WALT scheduler stalls.
inline uint64_t ComputeAdpfActiveWorkUs(
    const CallbackTimeBreakdown& breakdown) {
  const uint64_t active = breakdown.cpu_us + breakdown.run_queue_wait_us;
  if (active < breakdown.cpu_us) {
    return breakdown.wall_us;
  }
  return std::min(breakdown.wall_us, active);
}

inline uint64_t ComputeAdpfActiveWorkUs(uint64_t wall_us, uint64_t cpu_us,
                                        uint64_t run_queue_wait_us) {
  return ComputeAdpfActiveWorkUs(
      ComputeCallbackTimeBreakdown(wall_us, cpu_us, run_queue_wait_us));
}

// RAII ADPF session manager for the audio deadline (5333 us).
// Handles dynamic setThreads updates when XMA Decoder starts/restarts after
// Audio Worker, dead-TID filtering, recreate fallback, and RAII closeSession.
class AudioAdpfSession {
 public:
  explicit AudioAdpfSession(int64_t target_ns, AudioAdpfOps ops = {})
      : target_ns_(target_ns), ops_(ops) {}

  ~AudioAdpfSession() { Close(); }

  AudioAdpfSession(const AudioAdpfSession&) = delete;
  AudioAdpfSession& operator=(const AudioAdpfSession&) = delete;

  void SetOps(const AudioAdpfOps& ops) {
    Close();
    ops_ = ops;
    permanently_failed_ = false;
  }

  void Close() {
    if (session_ && ops_.close_session) {
      ops_.close_session(ops_.user, session_);
    }
    session_ = nullptr;
    active_tids_.clear();
  }

  void SyncAndReport(bool enabled, uint64_t actual_us,
                     const AudioAdpfThreadRegistry& registry) {
    if (!enabled) {
      Close();
      permanently_failed_ = false;
      return;
    }
    if (permanently_failed_) {
      return;
    }
    if (!ops_.create_session || !ops_.report_actual_work ||
        !ops_.close_session) {
      permanently_failed_ = true;
      return;
    }

    auto is_alive = [this](int32_t tid) -> bool {
      return ops_.is_tid_alive ? ops_.is_tid_alive(ops_.user, tid) : true;
    };

    const uint64_t gen = registry.generation();
    if (!session_) {
      std::vector<int32_t> tids = registry.SnapshotLiveTids(is_alive);
      if (tids.empty()) {
        return;
      }
      session_ = ops_.create_session(ops_.user, tids.data(), tids.size(),
                                     target_ns_);
      if (!session_) {
        std::vector<int32_t> retried = registry.SnapshotLiveTids(is_alive);
        if (!retried.empty() && retried != tids) {
          tids.swap(retried);
          session_ = ops_.create_session(ops_.user, tids.data(), tids.size(),
                                         target_ns_);
        }
      }
      if (!session_) {
        permanently_failed_ = true;
        return;
      }
      active_tids_ = std::move(tids);
      last_generation_ = gen;
    } else if (gen != last_generation_) {
      std::vector<int32_t> live_tids = registry.SnapshotLiveTids(is_alive);
      if (live_tids.empty()) {
        Close();
        last_generation_ = gen;
        return;
      }
      if (live_tids != active_tids_) {
        bool updated = false;
        if (ops_.set_threads) {
          if (ops_.set_threads(ops_.user, session_, live_tids.data(),
                               live_tids.size()) == 0) {
            updated = true;
          } else {
            std::vector<int32_t> retried = registry.SnapshotLiveTids(is_alive);
            if (!retried.empty() && retried != live_tids &&
                ops_.set_threads(ops_.user, session_, retried.data(),
                                 retried.size()) == 0) {
              live_tids.swap(retried);
              updated = true;
            }
          }
        }
        if (!updated) {
          Close();
          live_tids = registry.SnapshotLiveTids(is_alive);
          if (live_tids.empty()) {
            last_generation_ = gen;
            return;
          }
          session_ = ops_.create_session(ops_.user, live_tids.data(),
                                         live_tids.size(), target_ns_);
          if (!session_) {
            permanently_failed_ = true;
            return;
          }
        }
        active_tids_ = std::move(live_tids);
      }
      last_generation_ = gen;
    }

    if (session_) {
      const uint64_t clamped_us = std::max<uint64_t>(actual_us, 1ULL);
      const uint64_t max_us =
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / 1000ULL;
      const int64_t actual_ns =
          static_cast<int64_t>(std::min(clamped_us, max_us) * 1000ULL);
      ops_.report_actual_work(ops_.user, session_, actual_ns);
    }
  }

  void SyncAndReportBreakdown(bool enabled,
                              const CallbackTimeBreakdown& breakdown,
                              const AudioAdpfThreadRegistry& registry) {
    SyncAndReport(enabled, ComputeAdpfActiveWorkUs(breakdown), registry);
  }

  bool active() const { return session_ != nullptr; }
  bool permanently_failed() const { return permanently_failed_; }
  int64_t target_ns() const { return target_ns_; }
  const std::vector<int32_t>& active_tids() const { return active_tids_; }

 private:
  int64_t target_ns_ = 5333 * int64_t(1000);
  AudioAdpfOps ops_{};
  void* session_ = nullptr;
  std::vector<int32_t> active_tids_;
  uint64_t last_generation_ = 0;
  bool permanently_failed_ = false;
};

inline bool IsAudioFrameSamplesSilent(const float* samples,
                                      size_t sample_count) {
  if (!samples || sample_count == 0) {
    return true;
  }
  uint32_t any = 0;
  for (size_t i = 0; i < sample_count; ++i) {
    uint32_t word = 0;
    std::memcpy(&word, samples + i, sizeof(word));
    any |= word;
  }
  return any == 0;
}

inline bool HasAnyEnabledXmaContext(const std::atomic<uint64_t>* words,
                                    size_t word_count) {
  if (!words) {
    return false;
  }
  for (size_t i = 0; i < word_count; ++i) {
    if (words[i].load(std::memory_order_seq_cst) != 0) {
      return true;
    }
  }
  return false;
}

}  // namespace xendroid

#endif  // XENDROID_XE_AUDIO_ADPF_LOGIC_H_
