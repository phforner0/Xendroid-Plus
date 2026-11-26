#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#define XENIA_GPU_SUBMIT_WORKER_PURE_LOGIC_ONLY
#include "xenia/gpu/vulkan/vulkan_submit_worker.h"

#define XENIA_GPU_VULKAN_SHARED_MEMORY_PURE_LOGIC_ONLY
#include "xenia/gpu/vulkan/vulkan_shared_memory.h"

namespace {

#define SUBMIT_WORKER_CHECK(cond)                                             \
  do {                                                                        \
    if (!(cond)) {                                                            \
      std::fprintf(stderr,                                                    \
                   "GPU_SUBMIT_WORKER check failed (%s:%d): %s\n", __FILE__,  \
                   __LINE__, #cond);                                          \
      std::abort();                                                           \
    }                                                                         \
  } while (false)

namespace fs = std::filesystem;
namespace sw = xe::gpu::vulkan::submit_worker;

std::string NormalizeNewlines(std::string s) {
  s.erase(std::remove(s.begin(), s.end(), '\r'), s.end());
  return s;
}

std::string ReadSourceText(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  SUBMIT_WORKER_CHECK(in.good());
  std::ostringstream ss;
  ss << in.rdbuf();
  return NormalizeNewlines(ss.str());
}

struct StressChunk {
  uint64_t submission_index = 0;
  uint64_t payload_hash = 0;
  uint32_t word_count = 0;
  bool is_swap = false;
  uint64_t required_pipeline_ticket = 0;
  sw::SubmitTimestampSnapshot timestamps{};
};

void TestSpscRingLayoutAndCapacity() {
  static_assert(sw::kSubmitChunkRingCapacity == 4,
                "Onda 3A requires bounded SPSC ring K = 4");
  sw::SpscSlotRing<sw::kSubmitChunkRingCapacity> ring;
  SUBMIT_WORKER_CHECK(ring.capacity() == 4);
  SUBMIT_WORKER_CHECK(ring.IsEmpty());
  SUBMIT_WORKER_CHECK(ring.HeadTailDistanceBytes() >= 64);
}

void TestSpscStress100kPlus() {
  constexpr uint64_t kTotalChunks = 120000;
  sw::SpscChunkRing<StressChunk, sw::kSubmitChunkRingCapacity> ring;

  std::atomic<uint64_t> consumed_count{0};
  std::atomic<bool> consumer_ok{true};

  std::thread consumer([&]() {
    uint64_t expected_seq = 1;
    size_t slot_idx = 0;
    while (ring.WaitAndPeekReadableSlot(slot_idx)) {
      StressChunk& c = ring.slot(slot_idx);
      const uint64_t expected_hash =
          (expected_seq * UINT64_C(0x9E3779B97F4A7C15)) ^
          (uint64_t(c.word_count) << 32);
      if (c.submission_index != expected_seq ||
          c.payload_hash != expected_hash ||
          c.is_swap != ((expected_seq % 5) == 0) ||
          c.required_pipeline_ticket != (expected_seq / 3)) {
        consumer_ok.store(false, std::memory_order_release);
      }
      // Exercise variable consumer latency bursts (full/empty backpressure).
      if ((expected_seq & 0x3FFu) == 0) {
        std::this_thread::yield();
      }
      c.submission_index = 0;
      c.payload_hash = 0;
      ring.ReleaseReadableSlot();
      ring.MarkWorkerChunkComplete();
      consumed_count.store(expected_seq, std::memory_order_release);
      ++expected_seq;
    }
  });

  for (uint64_t seq = 1; seq <= kTotalChunks; ++seq) {
    size_t slot_idx = 0;
    SUBMIT_WORKER_CHECK(ring.AcquireWritableSlot(slot_idx));
    StressChunk& c = ring.slot(slot_idx);
    c.submission_index = seq;
    c.word_count = static_cast<uint32_t>((seq * 17u) & 0xFFFu);
    c.payload_hash = (seq * UINT64_C(0x9E3779B97F4A7C15)) ^
                     (uint64_t(c.word_count) << 32);
    c.is_swap = (seq % 5) == 0;
    c.required_pipeline_ticket = seq / 3;
    c.timestamps.fs_timestamp_slot = static_cast<uint32_t>(seq % 32);
    ring.CommitWritableSlot();
  }

  ring.WaitUntilIdle();
  ring.RequestStop();
  consumer.join();

  SUBMIT_WORKER_CHECK(consumer_ok.load(std::memory_order_acquire));
  SUBMIT_WORKER_CHECK(consumed_count.load(std::memory_order_acquire) ==
                      kTotalChunks);
  SUBMIT_WORKER_CHECK(ring.IsEmpty());
}

void TestThreePointerMonotonicity() {
  sw::SubmissionTimelineState timeline;
  SUBMIT_WORKER_CHECK(timeline.completed_submission() == 0);
  SUBMIT_WORKER_CHECK(timeline.submitted_submission() == 0);
  SUBMIT_WORKER_CHECK(timeline.recorded_submission() == 1);

  sw::SpscChunkRing<StressChunk, sw::kSubmitChunkRingCapacity> ring;
  constexpr uint64_t kSubmissions = 2500;
  std::atomic<bool> invariant_ok{true};

  std::thread worker([&]() {
    size_t slot_idx = 0;
    while (ring.WaitAndPeekReadableSlot(slot_idx)) {
      const uint64_t s = ring.slot(slot_idx).submission_index;
      ring.ReleaseReadableSlot();
      if ((s & 0x3Fu) == 0) {
        std::this_thread::yield();
      }
      timeline.MarkSubmitted(s);
      ring.MarkWorkerChunkComplete();
    }
  });

  for (uint64_t i = 1; i <= kSubmissions; ++i) {
    size_t slot_idx = 0;
    SUBMIT_WORKER_CHECK(ring.AcquireWritableSlot(slot_idx));
    const uint64_t s = timeline.ReserveNextSubmission();
    SUBMIT_WORKER_CHECK(s == i);
    ring.slot(slot_idx).submission_index = s;
    ring.CommitWritableSlot();

    // Check 3-pointer monotonic invariant: completed <= submitted < recorded.
    const uint64_t completed = timeline.completed_submission();
    const uint64_t submitted = timeline.submitted_submission();
    const uint64_t recorded = timeline.recorded_submission();
    if (!(completed <= submitted && submitted < recorded)) {
      invariant_ok.store(false, std::memory_order_release);
    }

    // Periodically await submission and advance completed.
    if ((i % 13) == 0) {
      SUBMIT_WORKER_CHECK(timeline.WaitUntilSubmitted(i));
      const uint64_t sub_now = timeline.submitted_submission();
      SUBMIT_WORKER_CHECK(sub_now >= i);
      if (sub_now > 1) {
        timeline.AdvanceCompleted(sub_now - 1);
      }
      const uint64_t c2 = timeline.completed_submission();
      const uint64_t s2 = timeline.submitted_submission();
      const uint64_t r2 = timeline.recorded_submission();
      if (!(c2 <= s2 && s2 < r2)) {
        invariant_ok.store(false, std::memory_order_release);
      }
    }
  }

  SUBMIT_WORKER_CHECK(timeline.WaitUntilSubmitted(kSubmissions));
  ring.WaitUntilIdle();
  timeline.AdvanceCompleted(kSubmissions);
  SUBMIT_WORKER_CHECK(timeline.completed_submission() == kSubmissions);
  SUBMIT_WORKER_CHECK(timeline.submitted_submission() == kSubmissions);
  SUBMIT_WORKER_CHECK(timeline.recorded_submission() == kSubmissions + 1);
  SUBMIT_WORKER_CHECK(invariant_ok.load(std::memory_order_acquire));

  ring.RequestStop();
  worker.join();
}

void TestZeroCopyTrackerLag1To3Submissions() {
  // Onda 2A ZeroCopyPageSubmissionTracker interaction under 1..3 submission
  // lag between Recorder (recorded_submission) and Submit Worker (submitted).
  xe::gpu::vulkan::ZeroCopyPageSubmissionTracker tracker;
  tracker.Initialize(20);  // 1 MiB (256 pages of 4 KiB)

  sw::SubmissionTimelineState timeline;
  constexpr uint32_t kPage = 4096u;

  // Chunk S=1 uses page 4, sealed into SPSC (NOT yet submitted: lag = 1).
  const uint64_t s1 = timeline.ReserveNextSubmission();
  SUBMIT_WORKER_CHECK(s1 == 1);
  tracker.MarkByteRangeUsed(4u * kPage, kPage, s1);

  // Chunk S=2 (lag 1): page 4 MUST NOT be zero-copied; untouched page 10 CAN.
  SUBMIT_WORKER_CHECK(timeline.submitted_submission() == 0);
  SUBMIT_WORKER_CHECK(timeline.completed_submission() == 0);
  SUBMIT_WORKER_CHECK(!tracker.IsByteRangeSafeForDirectWrite(
      4u * kPage, kPage, timeline.completed_submission(), true, 64));
  SUBMIT_WORKER_CHECK(tracker.IsByteRangeSafeForDirectWrite(
      10u * kPage, kPage, timeline.completed_submission(), true, 64));

  // Chunk S=2 uses page 5, Chunk S=3 uses page 6 (lag 2 and lag 3 in SPSC!).
  const uint64_t s2 = timeline.ReserveNextSubmission();
  tracker.MarkByteRangeUsed(5u * kPage, kPage, s2);
  const uint64_t s3 = timeline.ReserveNextSubmission();
  tracker.MarkByteRangeUsed(6u * kPage, kPage, s3);

  // Worker submits S=1 and S=2, GPU has NOT retired S=1 yet (completed == 0).
  timeline.MarkSubmitted(s1);
  timeline.MarkSubmitted(s2);
  SUBMIT_WORKER_CHECK(timeline.submitted_submission() == 2);
  SUBMIT_WORKER_CHECK(timeline.completed_submission() == 0);

  // All 3 pages (4, 5, 6) MUST still be rejected for zero-copy direct write!
  for (uint32_t p : {4u, 5u, 6u}) {
    SUBMIT_WORKER_CHECK(!tracker.IsByteRangeSafeForDirectWrite(
        p * kPage, kPage, timeline.completed_submission(), true, 64));
  }

  // GPU completes S=1: page 4 becomes safe, while page 5 (submitted, in GPU)
  // and page 6 (still queued in SPSC!) remain unsafe (routed to staging).
  timeline.AdvanceCompleted(s1);
  SUBMIT_WORKER_CHECK(tracker.IsByteRangeSafeForDirectWrite(
      4u * kPage, kPage, timeline.completed_submission(), true, 64));
  SUBMIT_WORKER_CHECK(!tracker.IsByteRangeSafeForDirectWrite(
      5u * kPage, kPage, timeline.completed_submission(), true, 64));
  SUBMIT_WORKER_CHECK(!tracker.IsByteRangeSafeForDirectWrite(
      6u * kPage, kPage, timeline.completed_submission(), true, 64));

  // Worker submits S=3, GPU completes S=3 -> all pages become safe.
  timeline.MarkSubmitted(s3);
  timeline.AdvanceCompleted(s3);
  for (uint32_t p : {4u, 5u, 6u}) {
    SUBMIT_WORKER_CHECK(tracker.IsByteRangeSafeForDirectWrite(
        p * kPage, kPage, timeline.completed_submission(), true, 64));
  }
}

void TestIssueSwapOrderingQueueA0() {
  // Proves that EndSubmission(true) -> WaitUntilSubmitted(S_swap) strictly
  // orders S_swap on Queue A:0 BEFORE VulkanPresenter / ZeroFG handoff_submit.
  struct QueueEvent {
    uint64_t submission;
    bool is_handoff;
  };
  std::mutex queue_a0_mutex;
  std::vector<QueueEvent> queue_a0_log;

  sw::SubmissionTimelineState timeline;
  sw::AsyncSubmitWorkerEngine<StressChunk, sw::kSubmitChunkRingCapacity> worker;

  worker.Start([&](StressChunk& chunk, auto&& release_slot) {
    const uint64_t s = chunk.submission_index;
    const bool is_swap = chunk.is_swap;
    // Release chunk slot before vkQueueSubmit (matching §2.2.1).
    release_slot();
    if (is_swap) {
      // Simulate replay/submit latency before hitting Queue A:0.
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    {
      std::lock_guard<std::mutex> lock(queue_a0_mutex);
      queue_a0_log.push_back({s, false});
    }
    timeline.MarkSubmitted(s);
  });

  constexpr uint64_t kFrames = 200;
  for (uint64_t frame = 1; frame <= kFrames; ++frame) {
    // 2 async mid-frame submissions (non-blocking).
    for (int mid = 0; mid < 2; ++mid) {
      size_t idx = 0;
      SUBMIT_WORKER_CHECK(worker.AcquireWritableSlot(idx));
      const uint64_t s = timeline.ReserveNextSubmission();
      worker.slot(idx).submission_index = s;
      worker.slot(idx).is_swap = false;
      worker.CommitWritableSlot();
    }
    // Final swap submission: EndSubmission(true) MUST WaitUntilSubmitted.
    size_t swap_idx = 0;
    SUBMIT_WORKER_CHECK(worker.AcquireWritableSlot(swap_idx));
    const uint64_t s_swap = timeline.ReserveNextSubmission();
    worker.slot(swap_idx).submission_index = s_swap;
    worker.slot(swap_idx).is_swap = true;
    worker.CommitWritableSlot();

    SUBMIT_WORKER_CHECK(timeline.WaitUntilSubmitted(s_swap));
    // Now presenter / ZeroFG submits handoff_submit on Queue A:0.
    {
      std::lock_guard<std::mutex> lock(queue_a0_mutex);
      queue_a0_log.push_back({s_swap, true});
    }
  }

  worker.Drain();
  worker.Stop();

  // Verify every handoff_submit(S_swap) occurs IMMEDIATELY AFTER submit(S_swap)
  // in Queue A:0 order, never before!
  uint64_t last_submitted = 0;
  uint64_t handoff_count = 0;
  for (const QueueEvent& ev : queue_a0_log) {
    if (!ev.is_handoff) {
      SUBMIT_WORKER_CHECK(ev.submission == last_submitted + 1);
      last_submitted = ev.submission;
    } else {
      SUBMIT_WORKER_CHECK(last_submitted >= ev.submission);
      ++handoff_count;
    }
  }
  SUBMIT_WORKER_CHECK(handoff_count == kFrames);
  SUBMIT_WORKER_CHECK(last_submitted == kFrames * 3);
}

void TestDeviceLostPropagationNoDeadlock() {
  constexpr int32_t kVkErrorDeviceLost = -4;
  sw::SubmissionTimelineState timeline;
  sw::AsyncSubmitWorkerEngine<StressChunk, sw::kSubmitChunkRingCapacity> worker;

  worker.Start([&](StressChunk& chunk, auto&& release_slot) {
    const uint64_t s = chunk.submission_index;
    release_slot();
    if (s == 5) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      timeline.MarkSubmissionFailed(s, kVkErrorDeviceLost);
      return;
    }
    if (timeline.HasError()) {
      timeline.MarkSubmissionFailed(s, kVkErrorDeviceLost);
      return;
    }
    timeline.MarkSubmitted(s);
  });

  // Push chunks 1..6; chunk 5 triggers VK_ERROR_DEVICE_LOST.
  for (uint64_t i = 1; i <= 6; ++i) {
    size_t idx = 0;
    SUBMIT_WORKER_CHECK(worker.AcquireWritableSlot(idx));
    const uint64_t s = timeline.ReserveNextSubmission();
    worker.slot(idx).submission_index = s;
    worker.CommitWritableSlot();
  }

  // Producer waiting on S=5 MUST wake up with failure (false) without deadlock.
  const auto t0 = std::chrono::steady_clock::now();
  const bool ok5 = timeline.WaitUntilSubmitted(5);
  const bool ok6 = timeline.WaitUntilSubmitted(6);
  const auto elapsed_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - t0)
          .count();

  SUBMIT_WORKER_CHECK(!ok5);
  SUBMIT_WORKER_CHECK(!ok6);
  SUBMIT_WORKER_CHECK(timeline.HasError());
  SUBMIT_WORKER_CHECK(timeline.error_code() == kVkErrorDeviceLost);
  SUBMIT_WORKER_CHECK(elapsed_ms < 2000);

  worker.Drain();
  worker.Stop();
}

void TestShutdownClearCacheAndLiveToggle() {
  sw::SubmissionTimelineState timeline;
  sw::AsyncSubmitWorkerEngine<StressChunk, sw::kSubmitChunkRingCapacity> worker;
  std::atomic<uint64_t> total_executed{0};

  auto worker_fn = [&](StressChunk& chunk, auto&& release_slot) {
    const uint64_t s = chunk.submission_index;
    release_slot();
    timeline.MarkSubmitted(s);
    total_executed.fetch_add(1, std::memory_order_relaxed);
  };

  uint64_t expected_total = 0;
  for (int cycle = 0; cycle < 10; ++cycle) {
    // Live toggle 0 -> 1: start worker.
    worker.Start(worker_fn);
    SUBMIT_WORKER_CHECK(worker.is_running());

    // Push burst of 20 chunks.
    for (int i = 0; i < 20; ++i) {
      size_t idx = 0;
      SUBMIT_WORKER_CHECK(worker.AcquireWritableSlot(idx));
      const uint64_t s = timeline.ReserveNextSubmission();
      worker.slot(idx).submission_index = s;
      worker.CommitWritableSlot();
      ++expected_total;
    }

    // ClearCache / Pause barrier: Drain must complete all queued work.
    worker.Drain();
    SUBMIT_WORKER_CHECK(
        timeline.WaitUntilSubmitted(timeline.recorded_submission() - 1));
    timeline.AdvanceCompleted(timeline.recorded_submission() - 1);

    // Push another burst (Resume).
    for (int i = 0; i < 15; ++i) {
      size_t idx = 0;
      SUBMIT_WORKER_CHECK(worker.AcquireWritableSlot(idx));
      const uint64_t s = timeline.ReserveNextSubmission();
      worker.slot(idx).submission_index = s;
      worker.CommitWritableSlot();
      ++expected_total;
    }

    // Live toggle 1 -> 0: Drain + Stop cleanly.
    worker.Drain();
    SUBMIT_WORKER_CHECK(
        timeline.WaitUntilSubmitted(timeline.recorded_submission() - 1));
    worker.Stop();
    SUBMIT_WORKER_CHECK(!worker.is_running());

    // Synchronous submissions while disabled (0 mode).
    for (int i = 0; i < 5; ++i) {
      const uint64_t s = timeline.ReserveNextSubmission();
      timeline.MarkSubmitted(s);
      timeline.AdvanceCompleted(s);
    }
    SUBMIT_WORKER_CHECK(timeline.completed_submission() ==
                        timeline.submitted_submission());
    SUBMIT_WORKER_CHECK(timeline.submitted_submission() + 1 ==
                        timeline.recorded_submission());
  }

  SUBMIT_WORKER_CHECK(total_executed.load(std::memory_order_acquire) ==
                      expected_total);
}

void TestSourceInvariants(const fs::path& cpp_root) {
  const fs::path xenia_src = cpp_root / "xenia" / "src" / "xenia";
  const fs::path vk_gpu = xenia_src / "gpu" / "vulkan";
  const fs::path vk_ui = xenia_src / "ui" / "vulkan";

  const std::string sw_h = ReadSourceText(vk_gpu / "vulkan_submit_worker.h");
  SUBMIT_WORKER_CHECK(sw_h.find("alignas(64)") != std::string::npos);
  SUBMIT_WORKER_CHECK(sw_h.find("kSubmitChunkRingCapacity = 4") !=
                      std::string::npos);

  const std::string tl_h =
      ReadSourceText(vk_ui / "vulkan_gpu_completion_timeline.h");
  const std::string tl_cc =
      ReadSourceText(vk_ui / "vulkan_gpu_completion_timeline.cc");
  SUBMIT_WORKER_CHECK(tl_h.find("ReserveNextSubmission") != std::string::npos);
  SUBMIT_WORKER_CHECK(tl_h.find("SubmitReservedSubmission") !=
                      std::string::npos);
  SUBMIT_WORKER_CHECK(tl_h.find("ReportReservedSubmissionFailed") !=
                      std::string::npos);
  SUBMIT_WORKER_CHECK(tl_h.find("WaitUntilSubmitted") != std::string::npos);
  SUBMIT_WORKER_CHECK(tl_h.find("submitted_submission_") != std::string::npos);
  SUBMIT_WORKER_CHECK(tl_cc.find("submit_cv_.notify_all()") !=
                      std::string::npos);
  SUBMIT_WORKER_CHECK(tl_cc.find("worker_failed_") != std::string::npos);

  const std::string pc_h = ReadSourceText(vk_gpu / "vulkan_pipeline_cache.h");
  const std::string pc_cc = ReadSourceText(vk_gpu / "vulkan_pipeline_cache.cc");
  SUBMIT_WORKER_CHECK(pc_h.find("PrepareAsyncSubmissionOnRecorder") !=
                      std::string::npos);
  SUBMIT_WORKER_CHECK(pc_h.find("AwaitPipelineTicket") != std::string::npos);
  SUBMIT_WORKER_CHECK(pc_h.find("ScopedWorkerReplay") != std::string::npos);
  SUBMIT_WORKER_CHECK(pc_cc.find("WaitForPipelineCompletionFromWorker") !=
                      std::string::npos);
  SUBMIT_WORKER_CHECK(pc_cc.find("IsInAsyncReplayThread()") !=
                      std::string::npos);
  SUBMIT_WORKER_CHECK(pc_cc.find("lto_args.creation_ticket = 0;") !=
                      std::string::npos);

  const std::string zpd_h =
      ReadSourceText(vk_gpu / "vulkan_zpd_query_pool.h");
  const std::string zpd_cc =
      ReadSourceText(vk_gpu / "vulkan_zpd_query_pool.cc");
  SUBMIT_WORKER_CHECK(zpd_h.find("RecordResolveBatch(DeferredCommandBuffer&") !=
                      std::string::npos);
  SUBMIT_WORKER_CHECK(zpd_cc.find("CmdVkCopyQueryPoolResults") !=
                      std::string::npos);
  SUBMIT_WORKER_CHECK(zpd_cc.find("CmdVkPipelineBarrier") !=
                      std::string::npos);

  const std::string cp_h =
      ReadSourceText(vk_gpu / "vulkan_command_processor.h");
  const std::string cp_cc =
      ReadSourceText(vk_gpu / "vulkan_command_processor.cc");
  SUBMIT_WORKER_CHECK(cp_cc.find("vulkan_async_submit_worker, false") !=
                      std::string::npos);
  SUBMIT_WORKER_CHECK(cp_cc.find("vulkan_async_submit_worker_nice, -10") !=
                      std::string::npos);
  SUBMIT_WORKER_CHECK(
      cp_cc.find("UPDATE_from_bool(vulkan_async_submit_worker") ==
      std::string::npos);
  SUBMIT_WORKER_CHECK(
      cp_cc.find("UPDATE_from_int32(vulkan_async_submit_worker_nice") ==
      std::string::npos);
  SUBMIT_WORKER_CHECK(cp_cc.find("debug.xendroid.async_submit_worker") !=
                      std::string::npos);
  SUBMIT_WORKER_CHECK(cp_cc.find("debug.xendroid.submit_worker_nice") !=
                      std::string::npos);
  SUBMIT_WORKER_CHECK(cp_h.find("DrainSubmitWorker") != std::string::npos);
  SUBMIT_WORKER_CHECK(cp_cc.find("WaitUntilSubmitted(") != std::string::npos);
  SUBMIT_WORKER_CHECK(
      cp_cc.find("zpd_host_query_pool_->RecordResolveBatch(deferred_command_buffer())") !=
      std::string::npos);
}

}  // namespace

int main(int argc, char** argv) {
  TestSpscRingLayoutAndCapacity();
  TestSpscStress100kPlus();
  TestThreePointerMonotonicity();
  TestZeroCopyTrackerLag1To3Submissions();
  TestIssueSwapOrderingQueueA0();
  TestDeviceLostPropagationNoDeadlock();
  TestShutdownClearCacheAndLiveToggle();
  if (argc > 1) {
    TestSourceInvariants(fs::path(argv[1]));
  }
  std::printf("gpu_submit_worker_test: PASS\n");
  return 0;
}
