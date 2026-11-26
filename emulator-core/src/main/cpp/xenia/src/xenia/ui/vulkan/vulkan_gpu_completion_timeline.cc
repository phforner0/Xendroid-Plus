/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/ui/vulkan/vulkan_gpu_completion_timeline.h"

#include <algorithm>
#include <chrono>
#include <iterator>

#include "xenia/base/assert.h"
#include "xenia/base/cvar.h"
#include "xenia/base/logging.h"

DEFINE_bool(vulkan_completion_wait_telemetry, false,
            "Log existing host completion/submit telemetry without extra GPU polls.",
            "Vulkan");

namespace xe {
namespace ui {
namespace vulkan {

namespace {
uint64_t TimelineNowNs() {
  return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::steady_clock::now().time_since_epoch())
                      .count());
}
constexpr uint64_t kSlowFencePollNs = 50000;
// Bounded collection: submissions this new are left alone when reclaiming.
constexpr size_t kReclaimKeepNewest = 4;
}  // namespace

VulkanGPUCompletionTimeline::~VulkanGPUCompletionTimeline() {
#ifndef NDEBUG
  assert_zero(fences_acquired_);
#endif

  if (!pending_submission_fences_.empty()) {
    if (vulkan_device_->functions().vkWaitForFences(
            vulkan_device_->device(), 1,
            &pending_submission_fences_.back().second, VK_TRUE,
            UINT64_MAX) == VK_ERROR_DEVICE_LOST) {
      XELOGE(
          "VulkanGPUCompletionTimeline[{}]: DEVICE_LOST waiting on final "
          "submission {} during destruction ({} fences pending)",
          name_, pending_submission_fences_.back().first,
          pending_submission_fences_.size());
      if (vulkan_device_->SetLost()) {
        vulkan_device_->LogFaultInfo();
      }
    }

    while (!pending_submission_fences_.empty()) {
      vulkan_device_->functions().vkDestroyFence(
          vulkan_device_->device(), pending_submission_fences_.back().second,
          nullptr);
      pending_submission_fences_.pop_back();
    }
  }

  while (!free_fences_.empty()) {
    vulkan_device_->functions().vkDestroyFence(vulkan_device_->device(),
                                               free_fences_.back(), nullptr);
    free_fences_.pop_back();
  }
}

std::optional<VulkanGPUCompletionTimeline::FenceAcquisition>
VulkanGPUCompletionTimeline::AcquireFenceForSubmission(
    VkResult* const result_out_opt) {
  // Prefer reusing fences already reclaimed by awaits without polling the
  // driver: the status query of a still-pending fence blocks until it retires
  // on Turnip/kgsl (vkGetFenceStatus is not non-blocking there), which would
  // serialize the CPU with the GPU on every submit. Only poll when fences
  // would otherwise accumulate (users like the presenter's guest output
  // refresher never await in steady state) - by the time this many
  // submissions are pending, the oldest one has long retired, so the poll
  // reclaims it without a meaningful wait.
  constexpr size_t kMaxPendingBeforeReclaimPoll = 8;
  static_assert(kMaxPendingBeforeReclaimPoll > kReclaimKeepNewest);
  bool need_reclaim_poll = false;
  uint64_t reclaim_poll_limit = UINT64_MAX;
  {
    std::lock_guard<std::mutex> lock(timeline_mutex_);
    if (free_fences_.empty() &&
        pending_submission_fences_.size() >= kMaxPendingBeforeReclaimPoll) {
      need_reclaim_poll = true;
      if (bounded_collection()) {
        // Only the older submissions: polling the newest ones too would wait
        // for the whole queue to drain (see SetBoundedCollection).
        reclaim_poll_limit =
            pending_submission_fences_[pending_submission_fences_.size() -
                                       kReclaimKeepNewest - 1]
                .first;
      }
    }
  }
  if (need_reclaim_poll) {
    if (reclaim_poll_limit != UINT64_MAX) {
      PollPendingFences(reclaim_poll_limit);
    } else {
      UpdateCompletedSubmission();
    }
  }

  VkFence fence = VK_NULL_HANDLE;
  {
    std::lock_guard<std::mutex> lock(timeline_mutex_);
    if (!free_fences_.empty()) {
      fence = free_fences_.back();
      free_fences_.pop_back();
    }
  }

  if (fence != VK_NULL_HANDLE) {
    const VkResult fence_reset_result =
        vulkan_device_->functions().vkResetFences(vulkan_device_->device(), 1,
                                                  &fence);
    if (fence_reset_result != VK_SUCCESS) {
      XELOGE("Failed to reset a Vulkan fence: {}",
             vk::to_string(vk::Result(fence_reset_result)));
      vulkan_device_->functions().vkDestroyFence(vulkan_device_->device(),
                                                 fence, nullptr);
      fence = VK_NULL_HANDLE;
    }
  }

  if (fence == VK_NULL_HANDLE) {
    const VkFenceCreateInfo fence_create_info = {
        VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    const VkResult fence_create_result =
        vulkan_device_->functions().vkCreateFence(
            vulkan_device_->device(), &fence_create_info, nullptr, &fence);
    if (fence_create_result != VK_SUCCESS) {
      XELOGE("Failed to create a Vulkan fence: {}",
             vk::to_string(vk::Result(fence_create_result)));
      if (result_out_opt != nullptr) {
        *result_out_opt = fence_create_result;
      }
      return std::nullopt;
    }
  }

#ifndef NDEBUG
  {
    std::lock_guard<std::mutex> lock(timeline_mutex_);
    ++fences_acquired_;
  }
#endif

  if (result_out_opt != nullptr) {
    *result_out_opt = VK_SUCCESS;
  }
  return FenceAcquisition(this, fence);
}

VkResult VulkanGPUCompletionTimeline::SubmitReservedSubmission(
    const uint64_t submission_index, const uint32_t queue_family_index,
    const uint32_t queue_index, const uint32_t submit_count,
    const VkSubmitInfo* const submits, uint64_t* queue_lock_wait_ns_out,
    uint64_t* submit_host_ns_out) {
  if (queue_lock_wait_ns_out) *queue_lock_wait_ns_out = 0;
  if (submit_host_ns_out) *submit_host_ns_out = 0;

  // Step 1: Pop a reusable fence under a brief lock (NEVER call Vulkan under
  // timeline_mutex_).
  VkFence fence = VK_NULL_HANDLE;
  {
    std::lock_guard<std::mutex> lock(timeline_mutex_);
    if (!free_fences_.empty()) {
      fence = free_fences_.back();
      free_fences_.pop_back();
    }
  }

  // Step 2: Reset or create the fence OUTSIDE timeline_mutex_.
  if (fence != VK_NULL_HANDLE) {
    const VkResult fence_reset_result =
        vulkan_device_->functions().vkResetFences(vulkan_device_->device(), 1,
                                                  &fence);
    if (fence_reset_result != VK_SUCCESS) {
      XELOGE("VulkanGPUCompletionTimeline[{}]: vkResetFences failed: {}",
             name_, vk::to_string(vk::Result(fence_reset_result)));
      vulkan_device_->functions().vkDestroyFence(vulkan_device_->device(),
                                                 fence, nullptr);
      fence = VK_NULL_HANDLE;
    }
  }
  if (fence == VK_NULL_HANDLE) {
    const VkFenceCreateInfo fence_create_info = {
        VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    const VkResult fence_create_result =
        vulkan_device_->functions().vkCreateFence(
            vulkan_device_->device(), &fence_create_info, nullptr, &fence);
    if (fence_create_result != VK_SUCCESS) {
      XELOGE("VulkanGPUCompletionTimeline[{}]: vkCreateFence failed: {}",
             name_, vk::to_string(vk::Result(fence_create_result)));
      ReportReservedSubmissionFailed(submission_index, fence_create_result);
      return fence_create_result;
    }
  }

  // Step 3: Submit to the Vulkan queue OUTSIDE timeline_mutex_.
  VkResult submit_result;
  const bool measure_host = queue_lock_wait_ns_out || submit_host_ns_out;
  const uint64_t queue_begin_ns = measure_host ? TimelineNowNs() : 0;
  {
    const VulkanDevice::Queue::Acquisition queue_acquisition =
        vulkan_device_->AcquireQueue(queue_family_index, queue_index);
    const uint64_t submit_begin_ns = measure_host ? TimelineNowNs() : 0;
    if (queue_lock_wait_ns_out) {
      *queue_lock_wait_ns_out = submit_begin_ns - queue_begin_ns;
    }
    submit_result = vulkan_device_->SubmitAndUpdateLost(
        queue_acquisition.queue(), submit_count, submits, fence);
    if (submit_host_ns_out) {
      *submit_host_ns_out = TimelineNowNs() - submit_begin_ns;
    }
  }

  // Step 4: Publish {submission_index, fence} and advance submitted_submission_
  // under a brief lock, then wake WaitUntilSubmitted waiters.
  {
    std::lock_guard<std::mutex> lock(timeline_mutex_);
    if (submit_result == VK_SUCCESS) {
      assert_true(pending_submission_fences_.empty() ||
                  pending_submission_fences_.back().first < submission_index);
      pending_submission_fences_.emplace_back(submission_index, fence);
      const uint64_t prev =
          submitted_submission_.load(std::memory_order_relaxed);
      if (submission_index > prev) {
        submitted_submission_.store(submission_index,
                                    std::memory_order_release);
      }
    } else {
      free_fences_.push_back(fence);
      if (!worker_failed_.load(std::memory_order_relaxed)) {
        worker_submit_error_.store(submit_result, std::memory_order_release);
        worker_failed_.store(true, std::memory_order_release);
      }
      const uint64_t prev =
          submitted_submission_.load(std::memory_order_relaxed);
      if (submission_index > prev) {
        submitted_submission_.store(submission_index,
                                    std::memory_order_release);
      }
    }
  }
  submit_cv_.notify_all();
  return submit_result;
}

void VulkanGPUCompletionTimeline::ReportReservedSubmissionFailed(
    const uint64_t submission_index, const VkResult error_result) {
  if (error_result == VK_ERROR_DEVICE_LOST) {
    if (vulkan_device_->SetLost()) {
      vulkan_device_->LogFaultInfo();
    }
  }
  {
    std::lock_guard<std::mutex> lock(timeline_mutex_);
    if (!worker_failed_.load(std::memory_order_relaxed)) {
      worker_submit_error_.store(
          error_result != VK_SUCCESS ? error_result : VK_ERROR_UNKNOWN,
          std::memory_order_release);
      worker_failed_.store(true, std::memory_order_release);
    }
    const uint64_t prev =
        submitted_submission_.load(std::memory_order_relaxed);
    if (submission_index > prev) {
      submitted_submission_.store(submission_index, std::memory_order_release);
    }
  }
  submit_cv_.notify_all();
}

bool VulkanGPUCompletionTimeline::WaitUntilSubmitted(
    const uint64_t target_submission) {
  if (target_submission == 0) {
    return !worker_failed_.load(std::memory_order_acquire);
  }
  if (submitted_submission_.load(std::memory_order_acquire) >=
      target_submission) {
    return !worker_failed_.load(std::memory_order_acquire);
  }
  std::unique_lock<std::mutex> lock(timeline_mutex_);
  submit_cv_.wait(lock, [this, target_submission]() {
    return submitted_submission_.load(std::memory_order_acquire) >=
               target_submission ||
           worker_failed_.load(std::memory_order_acquire);
  });
  return !worker_failed_.load(std::memory_order_acquire) &&
         submitted_submission_.load(std::memory_order_acquire) >=
             target_submission;
}

VkResult VulkanGPUCompletionTimeline::AcquireFenceAndSubmit(
    const uint32_t queue_family_index, const uint32_t queue_index,
    const uint32_t submit_count, const VkSubmitInfo* const submits,
    uint64_t* queue_lock_wait_ns_out, uint64_t* submit_host_ns_out) {
  if (queue_lock_wait_ns_out) *queue_lock_wait_ns_out = 0;
  if (submit_host_ns_out) *submit_host_ns_out = 0;
  VkResult fence_acquire_result;
  std::optional<FenceAcquisition> fence_acquisition =
      AcquireFenceForSubmission(&fence_acquire_result);
  if (!fence_acquisition.has_value()) {
    return fence_acquire_result;
  }

  VkResult submit_result;
  const bool measure_host = queue_lock_wait_ns_out || submit_host_ns_out;
  const uint64_t queue_begin_ns = measure_host ? TimelineNowNs() : 0;
  {
    const VulkanDevice::Queue::Acquisition queue_acquisition =
        vulkan_device_->AcquireQueue(queue_family_index, queue_index);
    const uint64_t submit_begin_ns = measure_host ? TimelineNowNs() : 0;
    if (queue_lock_wait_ns_out)
      *queue_lock_wait_ns_out = submit_begin_ns - queue_begin_ns;
    submit_result = vulkan_device_->SubmitAndUpdateLost(
        queue_acquisition.queue(), submit_count, submits,
        fence_acquisition->GetFenceForSubmitting());
    if (submit_host_ns_out) *submit_host_ns_out = TimelineNowNs() - submit_begin_ns;
  }
  if (submit_result != VK_SUCCESS) {
    fence_acquisition->SetSubmissionFailedOrAborted();
  }
  return submit_result;
}

void VulkanGPUCompletionTimeline::UpdateCompletedSubmission() {
  PollPendingFences(UINT64_MAX);
}

void VulkanGPUCompletionTimeline::UpdateCompletedSubmissionUpTo(
    const uint64_t awaited_submission) {
  PollPendingFences(bounded_collection() ? awaited_submission : UINT64_MAX);
}

void VulkanGPUCompletionTimeline::PollPendingFences(
    const uint64_t last_submission) {
  // Snapshot up to 16 pending {submission, fence} pairs under a brief lock,
  // then call vkGetFenceStatus OUTSIDE timeline_mutex_ so the Submit Worker
  // can concurrently enqueue new submissions without contention.
  constexpr size_t kPollBatchSize = 16;
  std::pair<uint64_t, VkFence> batch[kPollBatchSize];

  for (;;) {
    size_t batch_count = 0;
    size_t total_pending_snapshot = 0;
    {
      std::lock_guard<std::mutex> lock(timeline_mutex_);
      total_pending_snapshot = pending_submission_fences_.size();
      for (const auto& entry : pending_submission_fences_) {
        if (entry.first > last_submission || batch_count >= kPollBatchSize) {
          break;
        }
        batch[batch_count++] = entry;
      }
    }
    if (batch_count == 0) {
      return;
    }

    size_t signaled_count = 0;
    uint64_t highest_completed = 0;
    bool stopped = false;
    for (size_t i = 0; i < batch_count; ++i) {
      const uint64_t poll_begin_ns = TimelineNowNs();
      const VkResult fence_status = vulkan_device_->functions().vkGetFenceStatus(
          vulkan_device_->device(), batch[i].second);
      const uint64_t poll_ns = TimelineNowNs() - poll_begin_ns;
      ++driver_wait_stats_.polls;
      driver_wait_stats_.poll_ns += poll_ns;
      if (poll_ns > kSlowFencePollNs) {
        ++driver_wait_stats_.slow_polls;
      }
      if (fence_status != VK_SUCCESS) {
        // Not ready, or an error.
        if (fence_status == VK_ERROR_DEVICE_LOST) {
          XELOGE(
              "VulkanGPUCompletionTimeline[{}]: DEVICE_LOST polling fence for "
              "submission {} ({} fences pending)",
              name_, batch[i].first, total_pending_snapshot);
          if (vulkan_device_->SetLost()) {
            vulkan_device_->LogFaultInfo();
          }
        }
        stopped = true;
        break;
      }
      highest_completed = batch[i].first;
      ++signaled_count;
    }

    if (signaled_count > 0) {
      {
        std::lock_guard<std::mutex> lock(timeline_mutex_);
        for (size_t i = 0; i < signaled_count; ++i) {
          assert_false(pending_submission_fences_.empty());
          assert_true(pending_submission_fences_.front().first ==
                      batch[i].first);
          free_fences_.push_back(pending_submission_fences_.front().second);
          pending_submission_fences_.pop_front();
        }
      }
      SetCompletedSubmission(highest_completed);
      completed_submission_cached_.store(highest_completed,
                                         std::memory_order_release);
    }

    if (stopped || batch_count < kPollBatchSize) {
      return;
    }
  }
}

void VulkanGPUCompletionTimeline::AwaitSubmissionImpl(
    const uint64_t awaited_submission) {
  // According to the Vulkan 1.4.335 specification:
  // "The first synchronization scope includes every batch submitted in the same
  // queue submission command. Fence signal operations that are defined by
  // vkQueueSubmit or vkQueueSubmit2 additionally include in the first
  // synchronization scope all commands that occur earlier in submission order.
  // Fence signal operations that are defined by vkQueueSubmit or vkQueueSubmit2
  // or vkQueueBindSparse additionally include in the first synchronization
  // scope any semaphore and fence signal operations that occur earlier in
  // signal operation order."

  // If the awaited submission is still queued in the async Submit Worker, wait
  // for vkQueueSubmit (or immediate wakeup on worker failure / DEVICE_LOST).
  if (awaited_submission >
      submitted_submission_.load(std::memory_order_acquire)) {
    if (!WaitUntilSubmitted(awaited_submission)) {
      return;
    }
  }

  VkFence target_fence = VK_NULL_HANDLE;
  uint64_t target_submission = 0;
  size_t pending_count_snapshot = 0;
  {
    std::lock_guard<std::mutex> lock(timeline_mutex_);
    pending_count_snapshot = pending_submission_fences_.size();
    for (const auto& entry : pending_submission_fences_) {
      if (entry.first > awaited_submission) {
        break;
      }
      target_submission = entry.first;
      target_fence = entry.second;
    }
  }

  if (target_fence != VK_NULL_HANDLE) {
    const uint64_t wait_begin_ns = TimelineNowNs();
    const VkResult fence_wait_result =
        vulkan_device_->functions().vkWaitForFences(
            vulkan_device_->device(), 1, &target_fence, VK_TRUE, UINT64_MAX);
    ++driver_wait_stats_.waits;
    driver_wait_stats_.wait_ns += TimelineNowNs() - wait_begin_ns;
    if (fence_wait_result != VK_SUCCESS) {
      XELOGE(
          "VulkanGPUCompletionTimeline[{}]: vkWaitForFences -> {} (awaited "
          "submission {}, last-pending {}, {} fences pending)",
          name_, vk::to_string(vk::Result(fence_wait_result)),
          awaited_submission, target_submission, pending_count_snapshot);
      if (fence_wait_result == VK_ERROR_DEVICE_LOST) {
        if (vulkan_device_->SetLost()) {
          vulkan_device_->LogFaultInfo();
        }
      }
      return;
    }
    {
      std::lock_guard<std::mutex> lock(timeline_mutex_);
      while (!pending_submission_fences_.empty() &&
             pending_submission_fences_.front().first <= target_submission) {
        free_fences_.push_back(pending_submission_fences_.front().second);
        pending_submission_fences_.pop_front();
      }
    }
  }

  if (!worker_failed_.load(std::memory_order_acquire) &&
      GetCompletedSubmissionFromLastUpdate() < awaited_submission) {
    SetCompletedSubmission(awaited_submission);
    completed_submission_cached_.store(awaited_submission,
                                       std::memory_order_release);
  }
}

}  // namespace vulkan
}  // namespace ui
}  // namespace xe
