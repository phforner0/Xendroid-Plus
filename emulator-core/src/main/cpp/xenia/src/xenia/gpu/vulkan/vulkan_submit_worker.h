/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_VULKAN_VULKAN_SUBMIT_WORKER_H_
#define XENIA_GPU_VULKAN_VULKAN_SUBMIT_WORKER_H_

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>

namespace xe {
namespace gpu {
namespace vulkan {
namespace submit_worker {

// Bounded SPSC chunk ring capacity (Onda 3A): 1 active chunk recording on the
// GPU Commands thread + 1 chunk replaying on the Submit & Replay Worker + 2
// chunks absorbing bursts of drained mid-frame submissions without stalling.
constexpr size_t kSubmitChunkRingCapacity = 4;

// Immutable per-submission timestamp snapshot captured by the Recorder before
// pushing a SubmitChunk into the SPSC ring so the Submit Worker never touches
// Recorder-owned pass/resolve timestamp arrays or counters.
struct SubmitTimestampSnapshot {
  uint32_t fs_timestamp_slot = UINT32_MAX;
  uint32_t setup_ts_pair = UINT32_MAX;
  uint32_t resolve_ts_base = 0;
  uint32_t resolve_ts_count = 0;
  uint32_t pass_ts_base = 0;
  uint32_t pass_ts_count = 0;
  uint32_t resolve_ring_slot = 0;
  uint32_t pass_ring_slot = 0;
  bool reset_resolve_timestamps = false;
  bool reset_pass_timestamps = false;
};

// Bounded single-producer / single-consumer slot coordinator with cacheline-
// isolated head_ and tail_ cursors, zero locks on fast paths, and separate
// early-slot-release (after DeferredCommandBuffer::Execute/Reset) vs. full
// worker-completion tracking (after vkQueueSubmit and pool bookkeeping).
template <size_t Capacity = kSubmitChunkRingCapacity>
class SpscSlotRing {
  static_assert(Capacity >= 2 && (Capacity & (Capacity - 1)) == 0,
                "SpscSlotRing Capacity must be a power of two >= 2");

 public:
  static constexpr size_t kMask = Capacity - 1;

  SpscSlotRing() = default;
  SpscSlotRing(const SpscSlotRing&) = delete;
  SpscSlotRing& operator=(const SpscSlotRing&) = delete;

  constexpr size_t capacity() const { return Capacity; }

  size_t HeadTailDistanceBytes() const {
    const uintptr_t tail_addr = reinterpret_cast<uintptr_t>(&tail_);
    const uintptr_t head_addr = reinterpret_cast<uintptr_t>(&head_);
    return tail_addr > head_addr ? static_cast<size_t>(tail_addr - head_addr)
                                 : static_cast<size_t>(head_addr - tail_addr);
  }

  bool IsEmpty() const {
    return head_.load(std::memory_order_acquire) ==
           tail_.load(std::memory_order_acquire);
  }

  uint64_t PendingCount() const {
    const uint64_t t = tail_.load(std::memory_order_acquire);
    const uint64_t h = head_.load(std::memory_order_acquire);
    return t >= h ? (t - h) : 0;
  }

  // Producer-only: acquires the next writable slot index, blocking on
  // free_cv_ only when all Capacity slots are currently owned by the consumer.
  // Returns false if Stop was requested while waiting.
  bool AcquireWritableSlot(size_t& slot_index_out) {
    const uint64_t t = tail_.load(std::memory_order_relaxed);
    if ((t - head_.load(std::memory_order_acquire)) < Capacity) {
      slot_index_out = static_cast<size_t>(t & kMask);
      return true;
    }
    std::unique_lock<std::mutex> lock(free_mutex_);
    free_cv_.wait(lock, [this, t]() {
      return (t - head_.load(std::memory_order_acquire)) < Capacity ||
             stop_requested_.load(std::memory_order_acquire);
    });
    if (stop_requested_.load(std::memory_order_acquire)) {
      return false;
    }
    slot_index_out = static_cast<size_t>(t & kMask);
    return true;
  }

  // Producer-only: publishes the slot previously returned by
  // AcquireWritableSlot and wakes the consumer without lost wakeups.
  void CommitWritableSlot() {
    const uint64_t t = tail_.load(std::memory_order_relaxed);
    tail_.store(t + 1, std::memory_order_release);
    {
      std::lock_guard<std::mutex> lock(ready_mutex_);
    }
    ready_cv_.notify_one();
  }

  // Consumer-only: waits until a committed slot is available or Stop is
  // requested (draining all committed slots before returning false).
  bool WaitAndPeekReadableSlot(size_t& slot_index_out) {
    const uint64_t h = head_.load(std::memory_order_relaxed);
    if (h < tail_.load(std::memory_order_acquire)) {
      slot_index_out = static_cast<size_t>(h & kMask);
      return true;
    }
    std::unique_lock<std::mutex> lock(ready_mutex_);
    ready_cv_.wait(lock, [this, h]() {
      return h < tail_.load(std::memory_order_acquire) ||
             stop_requested_.load(std::memory_order_acquire);
    });
    if (h < tail_.load(std::memory_order_acquire)) {
      slot_index_out = static_cast<size_t>(h & kMask);
      return true;
    }
    return false;
  }

  // Consumer-only: releases the chunk slot back to the producer after replay
  // and chunk Reset() have completed (may be called before vkQueueSubmit).
  void ReleaseReadableSlot() {
    const uint64_t h = head_.load(std::memory_order_relaxed);
    head_.store(h + 1, std::memory_order_release);
    {
      std::lock_guard<std::mutex> lock(free_mutex_);
    }
    free_cv_.notify_one();
  }

  // Consumer-only: marks the entire chunk (including vkQueueSubmit and
  // command-pool/semaphore bookkeeping) complete so WaitUntilIdle() can return.
  void MarkWorkerChunkComplete() {
    const uint64_t done =
        worker_completed_.load(std::memory_order_relaxed) + 1;
    worker_completed_.store(done, std::memory_order_release);
    {
      std::lock_guard<std::mutex> lock(idle_mutex_);
    }
    idle_cv_.notify_all();
  }

  // Producer-only: blocks until every chunk committed so far has finished both
  // slot release and post-submit worker bookkeeping.
  void WaitUntilIdle() {
    const uint64_t target = tail_.load(std::memory_order_relaxed);
    if (worker_completed_.load(std::memory_order_acquire) >= target) {
      return;
    }
    std::unique_lock<std::mutex> lock(idle_mutex_);
    idle_cv_.wait(lock, [this, target]() {
      return worker_completed_.load(std::memory_order_acquire) >= target ||
             stop_requested_.load(std::memory_order_acquire);
    });
  }

  void RequestStop() {
    stop_requested_.store(true, std::memory_order_release);
    {
      std::lock_guard<std::mutex> lock(ready_mutex_);
    }
    ready_cv_.notify_all();
    {
      std::lock_guard<std::mutex> lock(free_mutex_);
    }
    free_cv_.notify_all();
    {
      std::lock_guard<std::mutex> lock(idle_mutex_);
    }
    idle_cv_.notify_all();
  }

  // Must only be called while the consumer thread is stopped/joined.
  void Reset() {
    tail_.store(0, std::memory_order_relaxed);
    head_.store(0, std::memory_order_relaxed);
    worker_completed_.store(0, std::memory_order_relaxed);
    stop_requested_.store(false, std::memory_order_relaxed);
  }

 private:
  alignas(64) std::atomic<uint64_t> tail_{0};
  alignas(64) std::atomic<uint64_t> head_{0};
  alignas(64) std::atomic<uint64_t> worker_completed_{0};
  alignas(64) std::atomic<bool> stop_requested_{false};

  std::mutex ready_mutex_;
  std::condition_variable ready_cv_;
  std::mutex free_mutex_;
  std::condition_variable free_cv_;
  std::mutex idle_mutex_;
  std::condition_variable idle_cv_;
};

// SPSC ring owning an inline array of default-constructible ChunkT slots.
template <typename ChunkT, size_t Capacity = kSubmitChunkRingCapacity>
class SpscChunkRing {
 public:
  constexpr size_t capacity() const { return ring_.capacity(); }
  bool IsEmpty() const { return ring_.IsEmpty(); }
  uint64_t PendingCount() const { return ring_.PendingCount(); }
  size_t HeadTailDistanceBytes() const {
    return ring_.HeadTailDistanceBytes();
  }

  bool AcquireWritableSlot(size_t& slot_index_out) {
    return ring_.AcquireWritableSlot(slot_index_out);
  }
  void CommitWritableSlot() { ring_.CommitWritableSlot(); }
  bool WaitAndPeekReadableSlot(size_t& slot_index_out) {
    return ring_.WaitAndPeekReadableSlot(slot_index_out);
  }
  void ReleaseReadableSlot() { ring_.ReleaseReadableSlot(); }
  void MarkWorkerChunkComplete() { ring_.MarkWorkerChunkComplete(); }
  void WaitUntilIdle() { ring_.WaitUntilIdle(); }
  void RequestStop() { ring_.RequestStop(); }
  void Reset() { ring_.Reset(); }

  ChunkT& slot(size_t index) { return slots_[index]; }
  const ChunkT& slot(size_t index) const { return slots_[index]; }

 private:
  SpscSlotRing<Capacity> ring_;
  std::array<ChunkT, Capacity> slots_{};
};

// Pure-logic 3-pointer monotonic timeline coordinator enforcing:
//   completed_submission_ <= submitted_submission_ < recorded_submission_
// with immediate error / VK_ERROR_DEVICE_LOST wakeup on WaitUntilSubmitted.
class SubmissionTimelineState {
 public:
  uint64_t recorded_submission() const {
    return recorded_submission_.load(std::memory_order_acquire);
  }
  uint64_t submitted_submission() const {
    return submitted_submission_.load(std::memory_order_acquire);
  }
  uint64_t completed_submission() const {
    return completed_submission_.load(std::memory_order_acquire);
  }

  bool HasError() const { return failed_.load(std::memory_order_acquire); }
  int32_t error_code() const {
    return error_code_.load(std::memory_order_acquire);
  }

  void ResetError() {
    std::lock_guard<std::mutex> lock(submit_mutex_);
    failed_.store(false, std::memory_order_release);
    error_code_.store(0, std::memory_order_release);
  }

  // Producer-only: reserves the current logical submission S and advances
  // recorded_submission_ to S + 1.
  uint64_t ReserveNextSubmission() {
    const uint64_t s = recorded_submission_.load(std::memory_order_relaxed);
    recorded_submission_.store(s + 1, std::memory_order_release);
    return s;
  }

  // Consumer-only: publishes that submission S succeeded on the GPU queue.
  void MarkSubmitted(uint64_t submission_index) {
    {
      std::lock_guard<std::mutex> lock(submit_mutex_);
      const uint64_t prev =
          submitted_submission_.load(std::memory_order_relaxed);
      if (submission_index > prev) {
        submitted_submission_.store(submission_index,
                                    std::memory_order_release);
      }
    }
    submit_cv_.notify_all();
  }

  // Consumer-only: propagates a queue/device error (e.g. VK_ERROR_DEVICE_LOST)
  // and advances submitted_submission_ so WaitUntilSubmitted never deadlocks.
  void MarkSubmissionFailed(uint64_t submission_index, int32_t err_code) {
    {
      std::lock_guard<std::mutex> lock(submit_mutex_);
      if (!failed_.load(std::memory_order_relaxed)) {
        error_code_.store(err_code, std::memory_order_release);
        failed_.store(true, std::memory_order_release);
      }
      const uint64_t prev =
          submitted_submission_.load(std::memory_order_relaxed);
      if (submission_index > prev) {
        submitted_submission_.store(submission_index,
                                    std::memory_order_release);
      }
    }
    submit_cv_.notify_all();
  }

  // Producer-only: waits until submitted_submission_ >= target_submission or
  // an error occurs. Returns true only when submitted without error.
  bool WaitUntilSubmitted(uint64_t target_submission) {
    if (target_submission == 0) {
      return !failed_.load(std::memory_order_acquire);
    }
    if (submitted_submission_.load(std::memory_order_acquire) >=
        target_submission) {
      return !failed_.load(std::memory_order_acquire);
    }
    std::unique_lock<std::mutex> lock(submit_mutex_);
    submit_cv_.wait(lock, [this, target_submission]() {
      return submitted_submission_.load(std::memory_order_acquire) >=
                 target_submission ||
             failed_.load(std::memory_order_acquire);
    });
    return !failed_.load(std::memory_order_acquire) &&
           submitted_submission_.load(std::memory_order_acquire) >=
               target_submission;
  }

  // Producer-only: advances completed_submission_ monotonically (never beyond
  // submitted_submission_).
  void AdvanceCompleted(uint64_t new_completed) {
    const uint64_t sub = submitted_submission_.load(std::memory_order_acquire);
    const uint64_t clamped = std::min(new_completed, sub);
    const uint64_t prev = completed_submission_.load(std::memory_order_relaxed);
    if (clamped > prev) {
      completed_submission_.store(clamped, std::memory_order_release);
    }
  }

 private:
  alignas(64) std::atomic<uint64_t> recorded_submission_{1};
  alignas(64) std::atomic<uint64_t> submitted_submission_{0};
  alignas(64) std::atomic<uint64_t> completed_submission_{0};
  std::atomic<bool> failed_{false};
  std::atomic<int32_t> error_code_{0};
  mutable std::mutex submit_mutex_;
  std::condition_variable submit_cv_;
};

// Threaded single-consumer Submit & Replay Worker engine built directly on
// SpscChunkRing for pure-logic unit/stress tests and reusable worker lifecycle.
template <typename ChunkT, size_t Capacity = kSubmitChunkRingCapacity>
class AsyncSubmitWorkerEngine {
 public:
  AsyncSubmitWorkerEngine() = default;
  AsyncSubmitWorkerEngine(const AsyncSubmitWorkerEngine&) = delete;
  AsyncSubmitWorkerEngine& operator=(const AsyncSubmitWorkerEngine&) = delete;
  ~AsyncSubmitWorkerEngine() { Stop(); }

  bool is_running() const { return running_; }

  template <typename Fn>
  void Start(Fn&& execute_chunk) {
    if (running_) {
      return;
    }
    ring_.Reset();
    running_ = true;
    worker_thread_ = std::thread([this, fn = std::forward<Fn>(execute_chunk)]() mutable {
      size_t slot_idx = 0;
      while (ring_.WaitAndPeekReadableSlot(slot_idx)) {
        bool released = false;
        auto release_slot = [&]() {
          if (!released) {
            released = true;
            ring_.ReleaseReadableSlot();
          }
        };
        fn(ring_.slot(slot_idx), release_slot);
        release_slot();
        ring_.MarkWorkerChunkComplete();
      }
    });
  }

  bool AcquireWritableSlot(size_t& slot_index_out) {
    return ring_.AcquireWritableSlot(slot_index_out);
  }
  ChunkT& slot(size_t index) { return ring_.slot(index); }
  void CommitWritableSlot() { ring_.CommitWritableSlot(); }

  void Drain() {
    if (running_) {
      ring_.WaitUntilIdle();
    }
  }

  void Stop() {
    if (!running_) {
      return;
    }
    ring_.WaitUntilIdle();
    ring_.RequestStop();
    if (worker_thread_.joinable()) {
      worker_thread_.join();
    }
    running_ = false;
    ring_.Reset();
  }

 private:
  SpscChunkRing<ChunkT, Capacity> ring_;
  std::thread worker_thread_;
  bool running_ = false;
};

}  // namespace submit_worker
}  // namespace vulkan
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_VULKAN_VULKAN_SUBMIT_WORKER_H_
