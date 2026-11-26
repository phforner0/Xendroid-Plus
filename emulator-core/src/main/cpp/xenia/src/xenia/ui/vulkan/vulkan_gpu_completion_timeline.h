/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_UI_VULKAN_VULKAN_GPU_COMPLETION_TIMELINE_H_
#define XENIA_UI_VULKAN_VULKAN_GPU_COMPLETION_TIMELINE_H_

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include "xenia/base/assert.h"
#include "xenia/ui/gpu_completion_timeline.h"
#include "xenia/ui/vulkan/vulkan_device.h"

namespace xe {
namespace ui {
namespace vulkan {

class VulkanGPUCompletionTimeline : public GPUCompletionTimeline {
 public:
  explicit VulkanGPUCompletionTimeline(VulkanDevice* const vulkan_device,
                                       const char* const name = "")
      : vulkan_device_(vulkan_device), name_(name) {}

  const char* name() const { return name_; }
  size_t pending_submission_count() const {
    std::lock_guard<std::mutex> lock(timeline_mutex_);
    return pending_submission_fences_.size();
  }
  uint64_t front_pending_submission() const {
    std::lock_guard<std::mutex> lock(timeline_mutex_);
    return pending_submission_fences_.empty()
               ? 0
               : pending_submission_fences_.front().first;
  }

  // Three-pointer monotonic submission timeline (Onda 3A):
  //   completed_submission_ <= submitted_submission_ < recorded_submission_
  // Safe to read from any thread (Recorder, Submit Worker, creation threads).
  uint64_t GetRecordedSubmission() const {
    return recorded_submission_.load(std::memory_order_acquire);
  }
  uint64_t GetSubmittedSubmission() const {
    return submitted_submission_.load(std::memory_order_acquire);
  }
  uint64_t GetCompletedSubmissionCached() const {
    return completed_submission_cached_.load(std::memory_order_acquire);
  }
  bool HasWorkerSubmitError() const {
    return worker_failed_.load(std::memory_order_acquire);
  }
  VkResult GetWorkerSubmitError() const {
    return worker_submit_error_.load(std::memory_order_acquire);
  }
  void ClearWorkerSubmitError() {
    std::lock_guard<std::mutex> lock(timeline_mutex_);
    worker_failed_.store(false, std::memory_order_release);
    worker_submit_error_.store(VK_SUCCESS, std::memory_order_release);
  }

  // Recorder-only: reserves the current logical submission index S and
  // advances upcoming/recorded submission to S + 1 before pushing the chunk
  // to the asynchronous Submit & Replay Worker.
  uint64_t ReserveNextSubmission() {
    const uint64_t reserved = GetUpcomingSubmission();
    IncrementUpcomingSubmission();
    recorded_submission_.store(GetUpcomingSubmission(),
                               std::memory_order_release);
    return reserved;
  }

  // Worker-only: acquires a free VkFence (short lock), calls vkQueueSubmit
  // OUTSIDE timeline_mutex_, then registers {submission_index, fence} and
  // publishes submitted_submission_ (or propagates error / DEVICE_LOST).
  VkResult SubmitReservedSubmission(
      uint64_t submission_index, uint32_t queue_family_index,
      uint32_t queue_index, uint32_t submit_count, const VkSubmitInfo* submits,
      uint64_t* queue_lock_wait_ns_out = nullptr,
      uint64_t* submit_host_ns_out = nullptr);

  // Worker-only: propagates a pre-submit failure (e.g., vkQueueBindSparse or
  // vkEndCommandBuffer failure) and wakes WaitUntilSubmitted without deadlock.
  void ReportReservedSubmissionFailed(uint64_t submission_index,
                                      VkResult error_result);

  // Recorder-only: waits until submitted_submission_ >= target_submission or a
  // worker submission error / VK_ERROR_DEVICE_LOST occurs. Never deadlocks.
  bool WaitUntilSubmitted(uint64_t target_submission);

  VulkanGPUCompletionTimeline(const VulkanGPUCompletionTimeline&) = delete;
  VulkanGPUCompletionTimeline& operator=(const VulkanGPUCompletionTimeline&) =
      delete;
  VulkanGPUCompletionTimeline(VulkanGPUCompletionTimeline&&) = delete;
  VulkanGPUCompletionTimeline& operator=(VulkanGPUCompletionTimeline&&) =
      delete;

  ~VulkanGPUCompletionTimeline();

  class FenceAcquisition {
   public:
    explicit FenceAcquisition(
        VulkanGPUCompletionTimeline* const completion_timeline,
        const VkFence fence)
        : completion_timeline_(completion_timeline), fence_(fence) {
      assert_not_null(completion_timeline);
      assert_true(fence != VK_NULL_HANDLE);
    }

    FenceAcquisition(const FenceAcquisition&) = delete;
    FenceAcquisition& operator=(const FenceAcquisition&) = delete;

    FenceAcquisition(FenceAcquisition&& other)
        : completion_timeline_(other.completion_timeline_),
          fence_(other.fence_),
          submission_successful_(other.submission_successful_) {
      other.completion_timeline_ = nullptr;
      other.fence_ = VK_NULL_HANDLE;
      other.submission_successful_.reset();
    }

    ~FenceAcquisition() {
      if (completion_timeline_ && fence_) {
        bool notify_submitted = false;
        {
          std::lock_guard<std::mutex> lock(
              completion_timeline_->timeline_mutex_);
#ifndef NDEBUG
          assert_not_zero(completion_timeline_->fences_acquired_);
          --completion_timeline_->fences_acquired_;
#endif
          if (submission_successful_.value_or(false)) {
            const uint64_t submission =
                completion_timeline_->GetUpcomingSubmission();
            assert_true(
                completion_timeline_->pending_submission_fences_.empty() ||
                completion_timeline_->pending_submission_fences_.front().first <
                    submission);
            completion_timeline_->pending_submission_fences_.emplace_back(
                submission, fence_);
            completion_timeline_->IncrementUpcomingSubmission();
            completion_timeline_->recorded_submission_.store(
                completion_timeline_->GetUpcomingSubmission(),
                std::memory_order_release);
            if (submission >
                completion_timeline_->submitted_submission_.load(
                    std::memory_order_relaxed)) {
              completion_timeline_->submitted_submission_.store(
                  submission, std::memory_order_release);
            }
            notify_submitted = true;
          } else {
            completion_timeline_->free_fences_.push_back(fence_);
          }
        }
        if (notify_submitted) {
          completion_timeline_->submit_cv_.notify_all();
        }
      }
    }

    VkFence GetFenceForSubmitting() {
      // Don't mark the fence as used in a submission if already tried to
      // submit, but failed.
      assert_true(!submission_successful_.has_value() ||
                  *submission_successful_);
      submission_successful_ = true;
      return fence_;
    }

    void SetSubmissionFailedOrAborted() { submission_successful_ = false; }

   private:
    VulkanGPUCompletionTimeline* completion_timeline_ = nullptr;

    VkFence fence_ = VK_NULL_HANDLE;

    std::optional<bool> submission_successful_;
  };

  // If the submission has succeeded (`GetFenceForSubmitting` was called, but
  // `SetSubmissionFailedOrAborted` was not), will advance to the next
  // submission once the acquisition is released.
  //
  // It's possible to acquire a fence not right before submitting, but also well
  // in advance, such as before recording the command buffer, for instance, to
  // skip recording it if fence acquisition has failed.
  //
  // Acquiring a fence also updates the completed submission in order to reuse
  // fences if this completion timeline is used without regular checks or waits
  // (for instance, if it's supplementary to another completion timeline, and
  // awaited only before destroying something).
  [[nodiscard]] std::optional<FenceAcquisition> AcquireFenceForSubmission(
      VkResult* result_out_opt = nullptr);

  VkResult AcquireFenceAndSubmit(uint32_t queue_family_index,
                                 uint32_t queue_index, uint32_t submit_count,
                                 const VkSubmitInfo* submits,
                                 uint64_t* queue_lock_wait_ns_out = nullptr,
                                 uint64_t* submit_host_ns_out = nullptr);

  void UpdateCompletedSubmission() override;

  // debug.xendroid.fence_collect experiment. A status poll of a still-pending
  // fence blocks until it retires on Turnip/kgsl, so an unbounded walk over
  // the pending fences waits for the GPU to drain completely. With bounded
  // collection, awaits poll only up to the awaited submission, and reclaiming
  // fences for new submissions only polls those older than the newest few.
  static void SetBoundedCollection(bool bounded) {
    bounded_collection_.store(bounded, std::memory_order_relaxed);
  }
  static bool bounded_collection() {
    return bounded_collection_.load(std::memory_order_relaxed);
  }

  // Driver fence status polls and waits done by this timeline, and the time
  // spent in them, since the previous call.
  struct DriverWaitStats {
    uint64_t polls = 0;
    // Longer than 50 us - the fence was still pending, and the poll waited.
    uint64_t slow_polls = 0;
    uint64_t poll_ns = 0;
    uint64_t waits = 0;
    uint64_t wait_ns = 0;
  };
  DriverWaitStats TakeDriverWaitStats() {
    DriverWaitStats stats = driver_wait_stats_;
    driver_wait_stats_ = DriverWaitStats();
    return stats;
  }

 protected:
  void AwaitSubmissionImpl(uint64_t awaited_submission) override;
  void UpdateCompletedSubmissionUpTo(uint64_t awaited_submission) override;

 private:
  // Polls the pending fences in submission order, stopping at the first one
  // that is not signaled or whose submission is past `last_submission`.
  void PollPendingFences(uint64_t last_submission);

  inline static std::atomic<bool> bounded_collection_{false};
  DriverWaitStats driver_wait_stats_;

  VulkanDevice* const vulkan_device_;
  const char* const name_;

  // Protects ONLY free_fences_ and pending_submission_fences_ (short swaps).
  // NEVER held across vkQueueSubmit, vkWaitForFences, vkGetFenceStatus,
  // vkResetFences, or vkCreateFence.
  mutable std::mutex timeline_mutex_;
  std::condition_variable submit_cv_;

  std::atomic<uint64_t> recorded_submission_{1};
  std::atomic<uint64_t> submitted_submission_{0};
  std::atomic<uint64_t> completed_submission_cached_{0};
  std::atomic<bool> worker_failed_{false};
  std::atomic<VkResult> worker_submit_error_{VK_SUCCESS};

  std::vector<VkFence> free_fences_;

  // <Submission index, fence>, in submission index order.
  std::deque<std::pair<uint64_t, VkFence>> pending_submission_fences_;

#ifndef NDEBUG
  size_t fences_acquired_ = 0;
#endif
};

}  // namespace vulkan
}  // namespace ui
}  // namespace xe

#endif  // XENIA_UI_VULKAN_VULKAN_GPU_COMPLETION_TIMELINE_H_
