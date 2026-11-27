#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#define XENIA_GPU_SUBMIT_WORKER_PURE_LOGIC_ONLY
#include "xenia/gpu/vulkan/vulkan_submit_worker.h"

#define XENIA_GPU_DRAW_CUTS_PURE_LOGIC_ONLY
#include "xenia/gpu/vulkan/vulkan_command_processor.h"

#define XENIA_GPU_VULKAN_RESOLVE_OPTS_PURE_LOGIC_ONLY
#include "xenia/gpu/vulkan/vulkan_render_target_cache.h"

#define XENIA_GPU_VULKAN_SHARED_MEMORY_PURE_LOGIC_ONLY
#include "xenia/gpu/vulkan/vulkan_shared_memory.h"

namespace {

#define W7_CHECK(cond)                                                     \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "WAVE7_ONDA1A_GPU_P0 check failed (%s:%d): %s\n", \
                   __FILE__, __LINE__, #cond);                             \
      std::abort();                                                        \
    }                                                                      \
  } while (false)

namespace fs = std::filesystem;
namespace sw = xe::gpu::vulkan::submit_worker;
namespace dc = xe::gpu::draw_cuts;
namespace ro = xe::gpu::vulkan::resolve_opts;

std::string NormalizeNewlines(std::string s) {
  s.erase(std::remove(s.begin(), s.end(), '\r'), s.end());
  return s;
}

std::string ReadSourceText(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  W7_CHECK(in.good());
  std::ostringstream ss;
  ss << in.rdbuf();
  return NormalizeNewlines(ss.str());
}

std::string StripLineComments(const std::string& src) {
  std::string out;
  out.reserve(src.size());
  size_t i = 0;
  while (i < src.size()) {
    if (i + 1 < src.size() && src[i] == '/' && src[i + 1] == '/') {
      while (i < src.size() && src[i] != '\n') {
        ++i;
      }
      continue;
    }
    out.push_back(src[i++]);
  }
  return out;
}

std::string ExtractFunctionBody(const std::string& src,
                                const std::string& signature_needle) {
  const size_t start = src.find(signature_needle);
  W7_CHECK(start != std::string::npos);
  const size_t brace_open = src.find('{', start);
  W7_CHECK(brace_open != std::string::npos);
  int depth = 0;
  for (size_t i = brace_open; i < src.size(); ++i) {
    if (src[i] == '{') {
      ++depth;
    } else if (src[i] == '}') {
      --depth;
      if (depth == 0) {
        return src.substr(start, i - start + 1);
      }
    }
  }
  W7_CHECK(false);
  return {};
}

// -----------------------------------------------------------------------------
// 1. P0 Pause() draining Submit Worker (concurrency + deadlock-freedom)
// -----------------------------------------------------------------------------
void TestPauseDrainConcurrencyAndDeadlockFreedom(const fs::path& xenia_gpu) {
  // Real multithreaded SPSC ring drain test mirroring CommandProcessor::Pause()
  // -> VulkanCommandProcessor::OnPause() -> DrainSubmitWorker().
  sw::SpscSlotRing<4> ring;
  std::array<uint64_t, 4> slot_submission{};
  std::atomic<uint64_t> submitted{0};
  std::atomic<bool> pause_fence_signalled{false};
  std::atomic<bool> submit_after_pause_detected{false};

  std::thread worker([&]() {
    size_t slot = 0;
    while (ring.WaitAndPeekReadableSlot(slot)) {
      const uint64_t sub = slot_submission[slot];
      // Simulate replay + vkQueueSubmit latency.
      std::this_thread::sleep_for(std::chrono::microseconds(150));
      ring.ReleaseReadableSlot();
      if (pause_fence_signalled.load(std::memory_order_acquire)) {
        submit_after_pause_detected.store(true, std::memory_order_release);
      }
      submitted.store(sub, std::memory_order_release);
      ring.MarkWorkerChunkComplete();
    }
  });

  constexpr uint64_t kBurstSubmissions = 12;
  for (uint64_t s = 1; s <= kBurstSubmissions; ++s) {
    size_t slot = 0;
    W7_CHECK(ring.AcquireWritableSlot(slot));
    slot_submission[slot] = s;
    ring.CommitWritableSlot();
  }

  // OnPause(): drain ring BEFORE signalling Pause() fence.
  ring.WaitUntilIdle();
  W7_CHECK(ring.IsEmpty());
  W7_CHECK(submitted.load(std::memory_order_acquire) == kBurstSubmissions);
  pause_fence_signalled.store(true, std::memory_order_release);

  // Calling WaitUntilIdle again when already drained must never deadlock.
  ring.WaitUntilIdle();
  W7_CHECK(!submit_after_pause_detected.load(std::memory_order_acquire));

  // Shutdown worker cleanly; WaitUntilIdle after stop must also return cleanly.
  ring.RequestStop();
  worker.join();
  ring.WaitUntilIdle();

  // Source contracts for CommandProcessor::Pause() & VulkanCommandProcessor.
  const std::string cp_h = ReadSourceText(xenia_gpu / "command_processor.h");
  const std::string cp_cc = ReadSourceText(xenia_gpu / "command_processor.cc");
  const std::string vcp_h =
      ReadSourceText(xenia_gpu / "vulkan" / "vulkan_command_processor.h");
  const std::string vcp_cc =
      ReadSourceText(xenia_gpu / "vulkan" / "vulkan_command_processor.cc");

  W7_CHECK(cp_h.find("virtual void OnPause()") != std::string::npos);
  const std::string pause_body =
      ExtractFunctionBody(cp_cc, "void CommandProcessor::Pause()");
  const size_t pos_on_pause = pause_body.find("OnPause();");
  const size_t pos_fence_sig = pause_body.find("fence.Signal();");
  const size_t pos_wait_evt =
      pause_body.find("threading::Wait(pause_resume_event_.get()");
  W7_CHECK(pos_on_pause != std::string::npos);
  W7_CHECK(pos_fence_sig != std::string::npos);
  W7_CHECK(pos_wait_evt != std::string::npos);
  W7_CHECK(pos_on_pause < pos_fence_sig);
  W7_CHECK(pos_fence_sig < pos_wait_evt);

  W7_CHECK(vcp_h.find("void OnPause() override;") != std::string::npos);
  const std::string vcp_on_pause =
      ExtractFunctionBody(vcp_cc, "void VulkanCommandProcessor::OnPause()");
  W7_CHECK(vcp_on_pause.find("DrainSubmitWorker();") != std::string::npos);
  W7_CHECK(vcp_on_pause.find("HasWorkerSubmitError()") != std::string::npos);
  W7_CHECK(vcp_on_pause.find("OnHostGpuLossFromAnyThread(true)") !=
           std::string::npos);

  const std::string drain_body = ExtractFunctionBody(
      vcp_cc, "void VulkanCommandProcessor::DrainSubmitWorker()");
  W7_CHECK(drain_body.find("!async_submit_worker_active_ || "
                           "!submit_worker_thread_.joinable()") !=
           std::string::npos);
  W7_CHECK(drain_body.find("!device_lost_") != std::string::npos);
  W7_CHECK(drain_body.find("!completion_timeline_.HasWorkerSubmitError()") !=
           std::string::npos);
}

// -----------------------------------------------------------------------------
// 2a-2d. Submit Worker coherence, nice reapply, PrepareForWait, Timestamp ring
// -----------------------------------------------------------------------------
void TestTimestampSlotRingGuardWraparound(const fs::path& xenia_gpu) {
  dc::TimestampSlotRingGuard<32> guard;
  W7_CHECK(guard.dropped_count() == 0);

  // Submission 0 is invalid.
  W7_CHECK(guard.TryAcquireSlot(0, 0, 0) ==
           dc::TimestampSlotRingGuard<32>::kInvalidSlot);
  W7_CHECK(guard.dropped_count() == 1);
  guard.Reset();

  // Fill slots for submissions 1..31 with completed = 0 (delta 1..31 < 32).
  for (uint64_t sub = 1; sub <= 31; ++sub) {
    const uint32_t slot = guard.TryAcquireSlot(sub, /*completed=*/0,
                                               /*oldest_unread=*/1);
    W7_CHECK(slot == uint32_t(sub % 32));
    guard.CommitSlot(slot, sub);
  }
  W7_CHECK(guard.dropped_count() == 0);

  // Submission 32 with completed=0 -> delta=32 >= 32 -> MUST reject!
  W7_CHECK(guard.TryAcquireSlot(32, /*completed=*/0, /*oldest_unread=*/1) ==
           dc::TimestampSlotRingGuard<32>::kInvalidSlot);
  W7_CHECK(guard.dropped_count() == 1);

  // Submission 33 with completed=1 -> delta=32 >= 32 -> MUST reject!
  W7_CHECK(guard.TryAcquireSlot(33, /*completed=*/1, /*oldest_unread=*/1) ==
           dc::TimestampSlotRingGuard<32>::kInvalidSlot);
  W7_CHECK(guard.dropped_count() == 2);

  // Submission 33 with completed=2 -> delta=31 < 32, slot 1's prev_sub=1 <= 2,
  // BUT oldest_unread=1 (CPU hasn't popped vk_submit_times_ record 1 yet) ->
  // MUST STILL REJECT to avoid clobbering unread mapped timestamp readback!
  W7_CHECK(guard.TryAcquireSlot(33, /*completed=*/2, /*oldest_unread=*/1) ==
           dc::TimestampSlotRingGuard<32>::kInvalidSlot);
  W7_CHECK(guard.dropped_count() == 3);

  // Once CPU consumes record 1 (oldest_unread=2), slot 1 is safe to reuse!
  const uint32_t slot_33 =
      guard.TryAcquireSlot(33, /*completed=*/2, /*oldest_unread=*/2);
  W7_CHECK(slot_33 == 1u);
  guard.CommitSlot(slot_33, 33);
  W7_CHECK(guard.slot_submission(1u) == 33u);

  // If slot 2 still holds in-flight prev_sub=2 with completed=1, even if
  // caller asks for an earlier-looking check, prev_sub > completed rejects.
  W7_CHECK(guard.TryAcquireSlot(34, /*completed=*/2, /*oldest_unread=*/0) ==
           dc::TimestampSlotRingGuard<32>::kInvalidSlot);
  W7_CHECK(guard.TryAcquireSlot(34, /*completed=*/3, /*oldest_unread=*/0) ==
           2u);

  guard.Reset();
  W7_CHECK(guard.slot_submission(1u) == 0u);
  W7_CHECK(guard.dropped_count() == 0u);

  // Source contracts in vulkan_command_processor.cc.
  const std::string vcp_cc =
      ReadSourceText(xenia_gpu / "vulkan" / "vulkan_command_processor.cc");
  W7_CHECK(vcp_cc.find("timestamp_slot_guard_.TryAcquireSlot(") !=
           std::string::npos);
  W7_CHECK(vcp_cc.find("timestamp_slot_guard_.CommitSlot(") !=
           std::string::npos);
  const std::string open_pass_ts =
      ExtractFunctionBody(vcp_cc, "void VulkanCommandProcessor::OpenPassTimestamp(");
  W7_CHECK(open_pass_ts.find("current_submission_timestamp_slot_ == UINT32_MAX") !=
           std::string::npos);
  const std::string capture_ts = ExtractFunctionBody(
      vcp_cc, "void VulkanCommandProcessor::CaptureTimestampSnapshotOnRecorder(");
  W7_CHECK(capture_ts.find("slot == UINT32_MAX") != std::string::npos);
  W7_CHECK(capture_ts.find("current_submission_timestamp_slot_ = UINT32_MAX;") !=
           std::string::npos);
}

void TestWorkerNiceReapplyAndAwaitQueryResolve(const fs::path& xenia_gpu) {
  dc::WorkerNiceReapplyTracker tracker;
  constexpr uint64_t kSec = dc::WorkerNiceReapplyTracker::kReapplyIntervalNs;

  // Initial check applies -10 and logs once.
  auto c0 = tracker.Check(-10, 100);
  W7_CHECK(c0.valid_range && c0.should_apply && !c0.should_log_invalid);
  W7_CHECK(!tracker.already_logged(-10));
  W7_CHECK(tracker.ShouldLogOutcome(-10));
  W7_CHECK(tracker.already_logged(-10));

  // Sub-1s checks do not trigger reapply.
  auto c1 = tracker.Check(-10, 100 + kSec - 1);
  W7_CHECK(c1.valid_range && !c1.should_apply);

  // At >= 1s, reapply triggers to survive Android cgroup resets, with zero log
  // spam because ShouldLogOutcome(-10) returns false.
  auto c2 = tracker.Check(-10, 100 + kSec);
  W7_CHECK(c2.valid_range && c2.should_apply);
  W7_CHECK(tracker.already_logged(-10));
  W7_CHECK(!tracker.ShouldLogOutcome(-10));

  // Next check 500ms later does not reapply.
  auto c3 = tracker.Check(-10, 100 + kSec + kSec / 2);
  W7_CHECK(c3.valid_range && !c3.should_apply);

  // Runtime change to -6 applies immediately and logs once.
  auto c4 = tracker.Check(-6, 100 + kSec + kSec / 2 + 10);
  W7_CHECK(c4.valid_range && c4.should_apply);
  W7_CHECK(!tracker.already_logged(-6));
  W7_CHECK(tracker.ShouldLogOutcome(-6));
  W7_CHECK(!tracker.ShouldLogOutcome(-6));

  // Out-of-range value logs once and does not spam.
  auto c5 = tracker.Check(-25, 100 + 3 * kSec);
  W7_CHECK(!c5.valid_range && !c5.should_apply && c5.should_log_invalid);
  auto c6 = tracker.Check(-25, 100 + 5 * kSec);
  W7_CHECK(!c6.valid_range && !c6.should_apply && !c6.should_log_invalid);

  // Returning to desired == 0 applies once, then skips periodic 1s reapply.
  auto c7 = tracker.Check(0, 100 + 6 * kSec);
  W7_CHECK(c7.valid_range && c7.should_apply);
  auto c8 = tracker.Check(0, 100 + 10 * kSec);
  W7_CHECK(c8.valid_range && !c8.should_apply);

  // Source contracts for AwaitQueryResolve & SubmitWorkerThreadMain.
  const std::string vcp_cc =
      ReadSourceText(xenia_gpu / "vulkan" / "vulkan_command_processor.cc");
  const std::string await_qr = ExtractFunctionBody(
      vcp_cc, "bool VulkanCommandProcessor::AwaitQueryResolve(");
  W7_CHECK(await_qr.find("HasWorkerSubmitError()") != std::string::npos);
  W7_CHECK(await_qr.find("WaitUntilSubmitted(wait_for_submission)") !=
           std::string::npos);
  W7_CHECK(await_qr.find(
               "CheckSubmissionCompletionAndDeviceLoss(wait_for_submission)") !=
           std::string::npos);
  W7_CHECK(await_qr.find("AwaitSubmissionAndUpdateCompleted(") ==
           std::string::npos);

  const std::string submit_main = ExtractFunctionBody(
      vcp_cc, "void VulkanCommandProcessor::SubmitWorkerThreadMain()");
  W7_CHECK(submit_main.find("WorkerNiceReapplyTracker") != std::string::npos);
  W7_CHECK(submit_main.find("getpriority(PRIO_PROCESS, 0)") !=
           std::string::npos);
  W7_CHECK(submit_main.find("setpriority(PRIO_PROCESS, 0, desired)") !=
           std::string::npos);
}

void TestPrepareForWaitFlushAndSmallReadbackNewest(const fs::path& xenia_gpu) {
  // Verify SmallReadbackRingState::NewestSubmission() including wraparound.
  dc::SmallReadbackRingState ring;
  W7_CHECK(ring.NewestSubmission() == 0u);
  uint32_t slot = 0;
  for (uint64_t i = 1; i <= 64; ++i) {
    W7_CHECK(ring.TryEnqueue(i, 0x1000u, 256u, slot));
    W7_CHECK(ring.NewestSubmission() == i);
  }
  W7_CHECK(ring.full());
  W7_CHECK(ring.DrainCompleted(10, [](const dc::SmallReadbackRingSlot&) {}) ==
           10u);
  // Push across wrap boundary (tail wraps from 0 to 1..3).
  W7_CHECK(ring.TryEnqueue(65, 0x2000u, 256u, slot));
  W7_CHECK(slot == 0u);
  W7_CHECK(ring.NewestSubmission() == 65u);
  W7_CHECK(ring.TryEnqueue(66, 0x3000u, 256u, slot));
  W7_CHECK(slot == 1u);
  W7_CHECK(ring.NewestSubmission() == 66u);

  // Case 1: Nothing pending -> no-op.
  auto a0 = dc::EvaluatePrepareForWaitFlush(
      /*device_lost=*/false, /*submission_open=*/true, /*async=*/true,
      /*cur=*/5, /*submitted=*/4, /*completed=*/3, false, 0, false, false, 0,
      false, false);
  W7_CHECK(!a0.close_open_submission && !a0.drain_submit_worker &&
           !a0.poll_completion && a0.await_submission == 0);

  // Case 2: Device lost -> never stall/close/drain.
  auto a_lost = dc::EvaluatePrepareForWaitFlush(
      /*device_lost=*/true, /*submission_open=*/true, /*async=*/true,
      /*cur=*/5, /*submitted=*/4, /*completed=*/3, true, 5, true, true, 5, true,
      true);
  W7_CHECK(!a_lost.close_open_submission && !a_lost.drain_submit_worker &&
           !a_lost.poll_completion && a_lost.await_submission == 0);

  // Case 3: Small readback recorded in open submission (sub 5) -> MUST close
  // submission, drain async worker, and await sub 5!
  auto a_sr_open = dc::EvaluatePrepareForWaitFlush(
      false, /*submission_open=*/true, /*async=*/true, /*cur=*/5,
      /*submitted=*/4, /*completed=*/3, /*has_sr=*/true, /*sr_newest=*/5, false,
      false, 0, false, false);
  W7_CHECK(a_sr_open.close_open_submission);
  W7_CHECK(a_sr_open.drain_submit_worker);
  W7_CHECK(a_sr_open.await_submission == 5);
  W7_CHECK(!a_sr_open.poll_completion);

  // Case 4: Small readback in closed submission 4 still queued in async worker
  // (submitted=3) -> don't close open sub 5, DO drain worker, DO await sub 4.
  auto a_sr_queued = dc::EvaluatePrepareForWaitFlush(
      false, /*submission_open=*/true, /*async=*/true, /*cur=*/5,
      /*submitted=*/3, /*completed=*/2, /*has_sr=*/true, /*sr_newest=*/4, false,
      false, 0, false, false);
  W7_CHECK(!a_sr_queued.close_open_submission);
  W7_CHECK(a_sr_queued.drain_submit_worker);
  W7_CHECK(a_sr_queued.await_submission == 4);

  // Case 5: Small readback already completed (sub 3 <= completed 3) -> poll
  // only (drains to guest RAM without closing open sub 5 or draining worker).
  auto a_sr_done = dc::EvaluatePrepareForWaitFlush(
      false, /*submission_open=*/true, /*async=*/true, /*cur=*/5,
      /*submitted=*/4, /*completed=*/3, /*has_sr=*/true, /*sr_newest=*/3, false,
      false, 0, false, false);
  W7_CHECK(!a_sr_done.close_open_submission);
  W7_CHECK(!a_sr_done.drain_submit_worker);
  W7_CHECK(a_sr_done.await_submission == 0);
  W7_CHECK(a_sr_done.poll_completion);

  // Case 6: Pending ZPD resolve batch in open submission -> close + drain + poll.
  auto a_zpd_batch = dc::EvaluatePrepareForWaitFlush(
      false, /*submission_open=*/true, /*async=*/true, /*cur=*/7,
      /*submitted=*/6, /*completed=*/5, false, 0, /*zpd_batch=*/true,
      /*zpd_in_flight=*/true, /*zpd_newest=*/7, false, false);
  W7_CHECK(a_zpd_batch.close_open_submission);
  W7_CHECK(a_zpd_batch.drain_submit_worker);
  W7_CHECK(a_zpd_batch.await_submission == 0);
  W7_CHECK(a_zpd_batch.poll_completion);

  // Case 7: Pending strict ZPD retire with unsubmitted prior submission -> drain
  // worker + poll without closing unrelated open submission.
  auto a_zpd_retire = dc::EvaluatePrepareForWaitFlush(
      false, /*submission_open=*/false, /*async=*/true, /*cur=*/8,
      /*submitted=*/6, /*completed=*/5, false, 0, false, false, 0,
      /*zpd_retire=*/true, false);
  W7_CHECK(!a_zpd_retire.close_open_submission);
  W7_CHECK(a_zpd_retire.drain_submit_worker);
  W7_CHECK(a_zpd_retire.poll_completion);

  // Source contract on PrepareForWait().
  const std::string vcp_cc =
      ReadSourceText(xenia_gpu / "vulkan" / "vulkan_command_processor.cc");
  const std::string pfw_body =
      ExtractFunctionBody(vcp_cc, "void VulkanCommandProcessor::PrepareForWait()");
  W7_CHECK(pfw_body.find("AwaitMemexportReadbacks(") != std::string::npos);
  W7_CHECK(pfw_body.find("EvaluatePrepareForWaitFlush(") != std::string::npos);
  W7_CHECK(pfw_body.find("EndSubmission(false)") != std::string::npos);
  W7_CHECK(pfw_body.find("DrainSubmitWorker()") != std::string::npos);
  W7_CHECK(pfw_body.find("CheckSubmissionCompletionAndDeviceLoss(") !=
           std::string::npos);
}

// -----------------------------------------------------------------------------
// 3. vulkan_rt_rows_on_demand=true (transfers, band replicate, OOM, grow barriers)
// -----------------------------------------------------------------------------
void TestRtRowsOnDemandAndGrowBarriers(const fs::path& xenia_gpu) {
  // 3a. ComputeTransferNeededGuestHeight tests.
  // 1x MSAA, 32bpp, pitch=16 tiles (1280px), base=0, tiles [0, 720) -> 45 rows * 16 = 720px.
  W7_CHECK(ro::ComputeTransferNeededGuestHeight(0, 720, 0, 16, ro::kMsaa1X) ==
           720u);
  // 2x MSAA (tile_h = 8), pitch=16, base=0, tiles [0, 720) -> 45 rows * 8 = 360px.
  W7_CHECK(ro::ComputeTransferNeededGuestHeight(0, 720, 0, 16, ro::kMsaa2X) ==
           360u);
  // 4x MSAA (tile_h = 8), pitch=32, base=0, tiles [0, 1440) -> 45 rows * 8 = 360px.
  W7_CHECK(ro::ComputeTransferNeededGuestHeight(0, 1440, 0, 32, ro::kMsaa4X) ==
           360u);
  // Partial row rounds up: 17 tiles with pitch 16 -> 2 tile rows * 16 = 32px.
  W7_CHECK(ro::ComputeTransferNeededGuestHeight(0, 17, 0, 16, ro::kMsaa1X) ==
           32u);
  // 64bpp (GetPitchTiles() == 32 for 16-tile-at-32bpp): 32 tiles -> 1 row * 16 = 16px.
  W7_CHECK(ro::ComputeTransferNeededGuestHeight(0, 32, 0, 32, ro::kMsaa1X) ==
           16u);
  // Non-zero base, non-wrapped: base=500, [500, 532), pitch=16 -> 2 rows * 16 = 32px.
  W7_CHECK(ro::ComputeTransferNeededGuestHeight(500, 532, 500, 16, ro::kMsaa1X) ==
           32u);
  // Wrapped around 2048: base=2000, [0, 16), pitch=16 -> local_end=(2048-2000)+16=64 -> 4 rows * 16 = 64px.
  W7_CHECK(ro::ComputeTransferNeededGuestHeight(0, 16, 2000, 16, ro::kMsaa1X) ==
           64u);
  // Straddling base across wrap: base=100, [50, 150) -> full 2048 tiles -> 128 rows * 16 = 2048px.
  W7_CHECK(ro::ComputeTransferNeededGuestHeight(50, 150, 100, 16, ro::kMsaa1X) ==
           2048u);
  // Empty / degenerate inputs return 0.
  W7_CHECK(ro::ComputeTransferNeededGuestHeight(10, 10, 0, 16, ro::kMsaa1X) ==
           0u);
  W7_CHECK(ro::ComputeTransferNeededGuestHeight(0, 16, 0, 0, ro::kMsaa1X) == 0u);

  // Regression (#173): RenderTargetHeightCoversNeeded clamps needed_guest_height
  // to max_guest_height so narrow/wrapped RTs already at max_guest_height are
  // NEVER dropped silently, while genuine OOM-undersized RTs are still skipped.
  const uint32_t narrow_wrap_needed =
      ro::ComputeTransferNeededGuestHeight(0, 16, 100, 1, ro::kMsaa1X);
  W7_CHECK(narrow_wrap_needed == 31424u);  // (2048 - 100 + 16) * 16 > 8192!
  // 1x scale cap (max_guest_height = 8192): allocated == max -> accepted.
  W7_CHECK(ro::RenderTargetHeightCoversNeeded(8192u, 8192u, narrow_wrap_needed));
  // 2x scale cap (max_guest_height = 4096): allocated == max -> accepted.
  W7_CHECK(ro::RenderTargetHeightCoversNeeded(4096u, 4096u, narrow_wrap_needed));
  // Genuine OOM on narrow/wrapped RT (allocated < min(needed, max)) -> skipped.
  W7_CHECK(!ro::RenderTargetHeightCoversNeeded(512u, 8192u, narrow_wrap_needed));
  W7_CHECK(!ro::RenderTargetHeightCoversNeeded(2048u, 4096u, narrow_wrap_needed));
  // Normal RT (needed < max): exact threshold check.
  W7_CHECK(ro::RenderTargetHeightCoversNeeded(720u, 8192u, 720u));
  W7_CHECK(!ro::RenderTargetHeightCoversNeeded(704u, 8192u, 720u));

  // 3d. ResolveGrowBarrierStates VUID-03900 & stage/access invariants.
  constexpr uint32_t kTop = ro::kVkStageTopOfPipeBit;
  constexpr uint32_t kAll = ro::kVkStageAllCommandsBit;
  constexpr uint32_t kFrag = ro::kVkStageFragmentShaderBit;
  constexpr uint32_t kComp = ro::kVkStageComputeShaderBit;
  constexpr uint32_t kColorOut = 0x00000400u;  // COLOR_ATTACHMENT_OUTPUT
  constexpr uint32_t kColorRW = 0x00000180u;   // COLOR_ATTACHMENT_READ|WRITE
  constexpr uint32_t kShaderR = ro::kVkAccessShaderReadBit;

  // Case A: old_stage == 0, old_access == 0, attachment layout ->
  // pre_src_stage = TOP_OF_PIPE with access 0, post_dst_stage = draw_stage (NO TOP_OF_PIPE!).
  auto p0 = ro::ResolveGrowBarrierStates(0, 0, /*layout=*/2, kColorOut,
                                         kColorRW, /*draw_layout=*/2, false,
                                         false);
  W7_CHECK(p0.pre_src_stage == kTop && p0.pre_src_access == 0u);
  W7_CHECK((p0.post_dst_stage & kTop) == 0u && p0.post_dst_stage == kColorOut);
  W7_CHECK(p0.post_dst_access == kColorRW && p0.post_layout == 2u);

  // Case B: old_stage == 0, old_access != 0 -> MUST NOT pair TOP_OF_PIPE with
  // non-zero access on EITHER pre or post barrier!
  auto p1 = ro::ResolveGrowBarrierStates(0, kColorRW, /*layout=*/2, kColorOut,
                                         kColorRW, /*draw_layout=*/2, false,
                                         false);
  W7_CHECK(p1.pre_src_stage == kAll && p1.pre_src_access == kColorRW);
  W7_CHECK((p1.post_dst_stage & kTop) == 0u && p1.post_dst_stage == kColorOut);
  W7_CHECK(p1.post_dst_access == kColorRW);

  // Case C: old_stage == TOP_OF_PIPE, old_access != 0 -> strips TOP_OF_PIPE.
  auto p2 = ro::ResolveGrowBarrierStates(kTop, kShaderR, /*layout=*/5,
                                         kColorOut, kColorRW, 2, false, true);
  W7_CHECK((p2.pre_src_stage & kTop) == 0u && p2.pre_src_stage == kAll);
  W7_CHECK((p2.post_dst_stage & kTop) == 0u);
  W7_CHECK(p2.post_dst_stage == (kFrag | kComp));
  W7_CHECK(p2.post_dst_access == kShaderR && p2.post_layout == 5u);

  // Case D: old_layout == SHADER_READ_ONLY_OPTIMAL with valid stage -> preserves stage.
  auto p3 = ro::ResolveGrowBarrierStates(kFrag, kShaderR, /*layout=*/5,
                                         kColorOut, kColorRW, 2, false, true);
  W7_CHECK(p3.pre_src_stage == kFrag && p3.pre_src_access == kShaderR);
  W7_CHECK(p3.post_dst_stage == kFrag && p3.post_dst_access == kShaderR);
  W7_CHECK(p3.post_layout == 5u);

  // Case E: transient/transfer layout -> transitions to draw usage & layout.
  auto p4 = ro::ResolveGrowBarrierStates(0x00001000u, 0x00000800u, /*layout=*/6,
                                         kColorOut, kColorRW, /*draw_layout=*/2,
                                         true, false);
  W7_CHECK((p4.post_dst_stage & kTop) == 0u);
  W7_CHECK(p4.post_dst_stage == kColorOut && p4.post_dst_access == kColorRW &&
           p4.post_layout == 2u);

  // Source contracts in vulkan_render_target_cache.cc.
  const std::string rtc_cc = ReadSourceText(
      xenia_gpu / "vulkan" / "vulkan_render_target_cache.cc");

  // ReplicateTilingBandRenderTargetRows checks OOM + rebuilds FB + stages clears.
  const std::string rep_body = ExtractFunctionBody(
      rtc_cc,
      "void VulkanRenderTargetCache::ReplicateTilingBandRenderTargetRows(");
  W7_CHECK(rep_body.find("if (!EnsureRenderTargetHeight(") != std::string::npos);
  W7_CHECK(rep_body.find("!last_update_framebuffer_") != std::string::npos);
  W7_CHECK(rep_body.find("GetHostRenderTargetsFramebuffer(") !=
           std::string::npos);
  W7_CHECK(rep_body.find("StageDeferredResolveClearsForLastUpdate();") !=
           std::string::npos);

  // NoteTilingBandDraw checks EnsureRenderTargetHeight return value (both color & depth).
  const std::string note_body = ExtractFunctionBody(
      rtc_cc, "uint32_t VulkanRenderTargetCache::NoteTilingBandDraw(");
  const size_t first_oom = note_body.find("if (!EnsureRenderTargetHeight(");
  W7_CHECK(first_oom != std::string::npos);
  const size_t second_oom =
      note_body.find("if (!EnsureRenderTargetHeight(", first_oom + 1);
  W7_CHECK(second_oom != std::string::npos);

  // DumpRenderTargets handles EnsureRenderTargetHeight OOM by filtering rects.
  const std::string dump_body = ExtractFunctionBody(
      rtc_cc, "void VulkanRenderTargetCache::DumpRenderTargets(");
  W7_CHECK(dump_body.find("if (!EnsureRenderTargetHeight(rt, needed_guest_height))") !=
           std::string::npos);
  W7_CHECK(dump_body.find("dump_rectangles_.erase(") != std::string::npos);

  // PerformTransfersAndResolveClears & Update ensure transfer RT heights.
  const std::string ptrc_body = ExtractFunctionBody(
      rtc_cc,
      "void VulkanRenderTargetCache::PerformTransfersAndResolveClears(");
  W7_CHECK(ptrc_body.find("EnsureTransferRenderTargetsHeight(") !=
           std::string::npos);
  W7_CHECK(ptrc_body.find("transfer_targets_tall_enough(") !=
           std::string::npos);
  W7_CHECK(ptrc_body.find("RenderTargetHeightCoversNeeded(") !=
           std::string::npos);

  const std::string ensure_body = ExtractFunctionBody(
      rtc_cc,
      "bool VulkanRenderTargetCache::EnsureTransferRenderTargetsHeight(");
  W7_CHECK(ensure_body.find("transfer.source") != std::string::npos);
  W7_CHECK(ensure_body.find("transfer.host_depth_source") != std::string::npos);
  W7_CHECK(ensure_body.find("RenderTargetHeightCoversNeeded(") !=
           std::string::npos);
  W7_CHECK(ensure_body.find("EnsureRenderTargetHeight(") != std::string::npos);

  // GrowRenderTarget uses ResolveGrowBarrierStates and flushes post-copy barrier.
  const std::string grow_body = ExtractFunctionBody(
      rtc_cc, "bool VulkanRenderTargetCache::GrowRenderTarget(");
  W7_CHECK(grow_body.find("resolve_opts::ResolveGrowBarrierStates(") !=
           std::string::npos);
  const size_t copy_pos = grow_body.find("CmdVkCopyImage(");
  W7_CHECK(copy_pos != std::string::npos);
  W7_CHECK(grow_body.find("command_processor_.SubmitBarriers(true);",
                          copy_pos) != std::string::npos);
}

// -----------------------------------------------------------------------------
// 4 & 5. Late-swap cache, sysconst dirty gate + 0x2200/0x2203, SharedMem log
// -----------------------------------------------------------------------------
void TestLateSwapSysconstAndSharedMemory(const fs::path& xenia_gpu) {
  // 4. Late-swap cache in ThrottlePresentation().
  const std::string cp_cc = ReadSourceText(xenia_gpu / "command_processor.cc");
  W7_CHECK(cp_cc.find("DECLARE_bool(vulkan_late_swap_present);") !=
           std::string::npos);
  W7_CHECK(cp_cc.find("debug.xendroid.late_swap") != std::string::npos);
  const std::string throttle_body = StripLineComments(
      ExtractFunctionBody(cp_cc, "void CommandProcessor::ThrottlePresentation()"));
  W7_CHECK(throttle_body.find("effective_value()") == std::string::npos);
  W7_CHECK(throttle_body.find("std::string") == std::string::npos);
  W7_CHECK(throttle_body.find("__system_property_get") == std::string::npos);
  W7_CHECK(throttle_body.find("cvars::vulkan_late_swap_present") !=
           std::string::npos);
  W7_CHECK(throttle_body.find("late_swap_prop_override_") != std::string::npos);
  W7_CHECK(throttle_body.find("late_swap_cvar_resolved_") != std::string::npos);

  const std::string worker_body =
      ExtractFunctionBody(cp_cc, "void CommandProcessor::WorkerThreadMain()");
  W7_CHECK(worker_body.find("PollLateSwapProperty();") != std::string::npos);
  const std::string bintrace_body =
      ExtractFunctionBody(cp_cc, "void CommandProcessor::BinTracePoll()");
  W7_CHECK(bintrace_body.find("PollLateSwapProperty();") != std::string::npos);

  // 5a. Sysconst classification: RB_DEPTHCONTROL (0x2200) & RB_COLOR_MASK (0x2203).
  W7_CHECK((dc::ClassifySysConstRegisterDirtyGroups(0x2200u) &
            dc::kSysConstGroupFsi) != 0u);
  W7_CHECK((dc::ClassifySysConstRegisterDirtyGroups(0x2203u) &
            dc::kSysConstGroupFsi) != 0u);
  // Exhaustive soundness check across [0x0000, 0x4FFF]: range(r, 1) must
  // ALWAYS be a superset of single-register classification (zero missed dirty
  // bits), and exact across [0x21FE, 0x220B] covering 0x2200 & 0x2203.
  for (uint32_t r = 0x0000u; r <= 0x4FFFu; ++r) {
    const uint32_t single = dc::ClassifySysConstRegisterDirtyGroups(r);
    const uint32_t range1 = dc::ClassifySysConstRangeDirtyGroups(r, 1u);
    W7_CHECK((range1 & single) == single);
    if (r >= 0x21FEu && r <= 0x220Bu) {
      W7_CHECK(single == range1);
    }
  }
  // Multi-register spans covering [0x2200..0x2203] match union of singles.
  for (uint32_t start = 0x21FEu; start <= 0x2206u; ++start) {
    for (uint32_t len = 1u; len <= 12u; ++len) {
      uint32_t expected = 0u;
      for (uint32_t i = 0; i < len; ++i) {
        expected |= dc::ClassifySysConstRegisterDirtyGroups(start + i);
      }
      W7_CHECK(dc::ClassifySysConstRangeDirtyGroups(start, len) == expected);
    }
  }

  // Verify hot sysconst dirty tracker writes are gated on
  // cvars::vulkan_memoize_system_constants in vulkan_command_processor.cc.
  const std::string vcp_cc =
      ReadSourceText(xenia_gpu / "vulkan" / "vulkan_command_processor.cc");
  const std::string wr_body = ExtractFunctionBody(
      vcp_cc, "void VulkanCommandProcessor::WriteRegister(");
  W7_CHECK(wr_body.find("cvars::vulkan_memoize_system_constants") !=
           std::string::npos);
  const std::string rr_body = ExtractFunctionBody(
      vcp_cc, "void VulkanCommandProcessor::RestoreRegisters(");
  W7_CHECK(rr_body.find("cvars::vulkan_memoize_system_constants") !=
           std::string::npos);
  const std::string wf_body = ExtractFunctionBody(
      vcp_cc, "void VulkanCommandProcessor::WriteFetchFromMem(");
  W7_CHECK(wf_body.find("cvars::vulkan_memoize_system_constants") !=
           std::string::npos);
  const std::string wps_body = ExtractFunctionBody(
      vcp_cc, "void VulkanCommandProcessor::WritePossiblySpecialRegistersFromMem(");
  W7_CHECK(wps_body.find("track_sysconst_dirty") != std::string::npos);

  // 5b. SharedMemory reason helper + low-RAM sparse fallback + cvar kept true!
  using xe::gpu::vulkan::DescribeDenseSharedMemoryReason;
  const std::string r_zc_only =
      DescribeDenseSharedMemoryReason(false, true, false);
  W7_CHECK(r_zc_only.find("zero-copy") != std::string::npos);
  W7_CHECK(r_zc_only.find("resolve/memexport") == std::string::npos);

  const std::string r_uma_only =
      DescribeDenseSharedMemoryReason(true, false, false);
  W7_CHECK(r_uma_only.find("UMA") != std::string::npos);

  const std::string r_no_host =
      DescribeDenseSharedMemoryReason(false, false, true);
  W7_CHECK(r_no_host.find("resolve/memexport") != std::string::npos);

  const std::string r_all = DescribeDenseSharedMemoryReason(true, true, true);
  W7_CHECK(r_all.find("UMA") != std::string::npos &&
           r_all.find("zero-copy") != std::string::npos &&
           r_all.find("resolve/memexport") != std::string::npos);

  const std::string sm_cc =
      ReadSourceText(xenia_gpu / "vulkan" / "vulkan_shared_memory.cc");
  // ABBA Onda 2 winner MUST remain default true!
  W7_CHECK(sm_cc.find("vulkan_shared_memory_zero_copy_upload, true,") !=
           std::string::npos);
  W7_CHECK(sm_cc.find("DescribeDenseSharedMemoryReason(") != std::string::npos);
  W7_CHECK(sm_cc.find("host-mapped for resolve/memexport readback") ==
           std::string::npos);
  W7_CHECK(sm_cc.find("if (dense_alloc_failed)") != std::string::npos);
  W7_CHECK(sm_cc.find("try_create_sparse_buffer();") != std::string::npos);
}

// -----------------------------------------------------------------------------
// 6. P0 — 3 Upstream Ports (28c265ed71, 8f189544e8, 902c0a447e)
// -----------------------------------------------------------------------------
void TestThreeUpstreamPortsContracts(const fs::path& xenia_gpu) {
  // Port 1: xenia-canary @ 28c265ed71 (PM4_EVENT_WRITE_CFL)
  const std::string pm4_h =
      ReadSourceText(xenia_gpu / "pm4_command_processor_implement.h");
  const std::string disasm_cc =
      ReadSourceText(xenia_gpu / "packet_disassembler.cc");
  W7_CHECK(pm4_h.find("opcode == PM4_EVENT_WRITE_CFL") != std::string::npos);
  W7_CHECK(pm4_h.find("case PM4_EVENT_WRITE_CFL:\n      case PM4_EVENT_WRITE_SHD:") !=
           std::string::npos);
  W7_CHECK(disasm_cc.find("case PM4_EVENT_WRITE_CFL:") != std::string::npos);
  W7_CHECK(disasm_cc.find("\"PM4_EVENT_WRITE_CFL\"") != std::string::npos);

  // Port 2: has207/xenia-edge @ 8f189544e8 (FSI depth slope outside non-uniform CF)
  const std::string spirv_cc =
      ReadSourceText(xenia_gpu / "spirv_shader_translator.cc");
  const std::string spirv_rb_cc =
      ReadSourceText(xenia_gpu / "spirv_shader_translator_rb.cc");
  W7_CHECK(spirv_cc.find(
               "if (DSV_IsApplyingPolygonOffset() ||\n"
               "      (edram_fragment_shader_interlock_ && "
               "!current_shader().writes_depth()))") != std::string::npos);
  const std::string fsi_ds_body = ExtractFunctionBody(
      spirv_rb_cc,
      "void SpirvShaderTranslator::FSI_DepthStencilTest(");
  W7_CHECK(fsi_ds_body.find("main_fbo_depth_unbiased_") != std::string::npos);
  W7_CHECK(fsi_ds_body.find("main_fbo_depth_derivatives_") !=
           std::string::npos);
  W7_CHECK(fsi_ds_body.find("OpDPdxCoarse") == std::string::npos);
  W7_CHECK(fsi_ds_body.find("OpDPdyCoarse") == std::string::npos);

  // Port 3: rexglue-sdk @ 902c0a447e (dynamic rendering pass break after transfer)
  const std::string vcp_h =
      ReadSourceText(xenia_gpu / "vulkan" / "vulkan_command_processor.h");
  const std::string vcp_cc =
      ReadSourceText(xenia_gpu / "vulkan" / "vulkan_command_processor.cc");
  W7_CHECK(vcp_h.find("VkImageView current_transfer_dest_view_ = VK_NULL_HANDLE;") !=
           std::string::npos);
  W7_CHECK(vcp_cc.find("current_transfer_dest_view_ == VK_NULL_HANDLE") !=
           std::string::npos);
  W7_CHECK(vcp_cc.find("current_transfer_dest_view_ == transfer_dest_view") !=
           std::string::npos);
  W7_CHECK(vcp_cc.find("current_transfer_dest_view_ = transfer_dest_view;") !=
           std::string::npos);
  const std::string end_rp =
      ExtractFunctionBody(vcp_cc, "void VulkanCommandProcessor::EndRenderPass()");
  W7_CHECK(end_rp.find("current_transfer_dest_view_ = VK_NULL_HANDLE;") !=
           std::string::npos);
  const std::string begin_sub = ExtractFunctionBody(
      vcp_cc, "bool VulkanCommandProcessor::BeginSubmission(");
  W7_CHECK(begin_sub.find("current_transfer_dest_view_ = VK_NULL_HANDLE;") !=
           std::string::npos);
}

}  // namespace

int main(int argc, char** argv) {
  fs::path cpp_dir;
  if (argc >= 2 && argv[1] && argv[1][0] != '\0') {
    cpp_dir = fs::path(argv[1]);
  } else {
    cpp_dir = fs::path(__FILE__).parent_path().parent_path().parent_path() /
              "main" / "cpp";
  }
  const fs::path xenia_gpu = cpp_dir / "xenia" / "src" / "xenia" / "gpu";

  TestPauseDrainConcurrencyAndDeadlockFreedom(xenia_gpu);
  TestTimestampSlotRingGuardWraparound(xenia_gpu);
  TestWorkerNiceReapplyAndAwaitQueryResolve(xenia_gpu);
  TestPrepareForWaitFlushAndSmallReadbackNewest(xenia_gpu);
  TestRtRowsOnDemandAndGrowBarriers(xenia_gpu);
  TestLateSwapSysconstAndSharedMemory(xenia_gpu);
  TestThreeUpstreamPortsContracts(xenia_gpu);

  std::printf("wave7_onda1a_gpu_p0_test: ALL CHECKS PASSED\n");
  return 0;
}
