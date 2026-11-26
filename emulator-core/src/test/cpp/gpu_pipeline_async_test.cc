#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#define XENIA_GPU_SHADER_STORAGE_PURE_LOGIC_ONLY
#include "xenia/gpu/shader_storage.h"

namespace {

#define GPU_ASYNC_CHECK(cond)                                                 \
  do {                                                                        \
    if (!(cond)) {                                                            \
      std::fprintf(stderr, "GPU_PIPELINE_ASYNC check failed (%s:%d): %s\n",   \
                   __FILE__, __LINE__, #cond);                                \
      std::abort();                                                           \
    }                                                                         \
  } while (false)

namespace fs = std::filesystem;

std::string NormalizeNewlines(std::string s) {
  s.erase(std::remove(s.begin(), s.end(), '\r'), s.end());
  return s;
}

std::string ReadSourceText(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  GPU_ASYNC_CHECK(in.good());
  std::ostringstream ss;
  ss << in.rdbuf();
  return NormalizeNewlines(ss.str());
}

void TestCoalescingSnapshotSlot() {
  xe::gpu::CoalescingSnapshotSlot slot;
  GPU_ASYNC_CHECK(!slot.HasPending());

  int executed_id = 0;
  int destroyed_id = 0;

  // Enqueue and take a single snapshot.
  bool superseded = slot.Enqueue([&](bool execute) {
    if (execute) {
      executed_id = 1;
    } else {
      destroyed_id = 1;
    }
  });
  GPU_ASYNC_CHECK(!superseded);
  GPU_ASYNC_CHECK(slot.HasPending());
  auto t1 = slot.Take();
  GPU_ASYNC_CHECK(static_cast<bool>(t1));
  GPU_ASYNC_CHECK(!slot.HasPending());
  t1(true);
  GPU_ASYNC_CHECK(executed_id == 1);
  GPU_ASYNC_CHECK(destroyed_id == 0);

  // Enqueue #2 then #3 before #2 is taken: #2 must be superseded (execute=false)
  // and #3 must execute (execute=true).
  std::vector<std::pair<int, bool>> events;
  GPU_ASYNC_CHECK(!slot.Enqueue([&](bool execute) {
    events.emplace_back(2, execute);
  }));
  GPU_ASYNC_CHECK(slot.Enqueue([&](bool execute) {
    events.emplace_back(3, execute);
  }));
  GPU_ASYNC_CHECK(events.size() == 1);
  GPU_ASYNC_CHECK(events[0].first == 2 && events[0].second == false);

  auto t3 = slot.Take();
  GPU_ASYNC_CHECK(static_cast<bool>(t3));
  t3(true);
  GPU_ASYNC_CHECK(events.size() == 2);
  GPU_ASYNC_CHECK(events[1].first == 3 && events[1].second == true);

  // Concurrent producer/worker test: worker blocks inside #4 while producer
  // enqueues #5 then #6; #5 is superseded without blocking producer, #6 runs on
  // drain.
  std::atomic<bool> in_task4{false};
  std::atomic<bool> release_task4{false};
  std::atomic<int> task5_discarded{0};
  std::atomic<int> task6_executed{0};

  GPU_ASYNC_CHECK(!slot.Enqueue([&](bool execute) {
    GPU_ASYNC_CHECK(execute);
    in_task4.store(true, std::memory_order_release);
    while (!release_task4.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
  }));

  std::thread worker([&]() {
    auto first = slot.Take();
    GPU_ASYNC_CHECK(static_cast<bool>(first));
    first(true);
    // Drain remaining pending snapshot on shutdown.
    if (auto pending = slot.Take()) {
      pending(true);
    }
  });

  while (!in_task4.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  GPU_ASYNC_CHECK(!slot.Enqueue([&](bool execute) {
    if (!execute) {
      task5_discarded.fetch_add(1, std::memory_order_relaxed);
    }
  }));
  GPU_ASYNC_CHECK(slot.Enqueue([&](bool execute) {
    if (execute) {
      task6_executed.fetch_add(1, std::memory_order_relaxed);
    }
  }));
  GPU_ASYNC_CHECK(task5_discarded.load(std::memory_order_relaxed) == 1);

  release_task4.store(true, std::memory_order_release);
  worker.join();
  GPU_ASYNC_CHECK(task6_executed.load(std::memory_order_relaxed) == 1);
  GPU_ASYNC_CHECK(!slot.HasPending());
}

void TestIsProprietaryMaliDriver() {
  constexpr uint32_t kVendorArm = 0x13B5;
  constexpr uint32_t kVendorQualcomm = 0x5143;
  constexpr uint32_t kDriverArmProprietary = 9;
  constexpr uint32_t kDriverMesaTurnip = 18;
  constexpr uint32_t kDriverMesaPanvk = 20;

  // Arm proprietary (driverID == 9 or legacy driverID == 0 with vendorID == 0x13B5).
  GPU_ASYNC_CHECK(xe::gpu::IsProprietaryMaliDriver(kVendorArm,
                                                   kDriverArmProprietary));
  GPU_ASYNC_CHECK(xe::gpu::IsProprietaryMaliDriver(kVendorArm, 0));

  // PanVK / malisx2 (vendorID == 0x13B5, driverID == 20) MUST NOT match!
  GPU_ASYNC_CHECK(
      !xe::gpu::IsProprietaryMaliDriver(kVendorArm, kDriverMesaPanvk));

  // Turnip / other vendors must not match.
  GPU_ASYNC_CHECK(
      !xe::gpu::IsProprietaryMaliDriver(kVendorQualcomm, kDriverMesaTurnip));
  GPU_ASYNC_CHECK(!xe::gpu::IsProprietaryMaliDriver(kVendorQualcomm, 0));
}

void TestShouldShrinkRenderArea() {
  // Non-Turnip drivers keep vulkan_tight_render_area controllable via cvar.
  GPU_ASYNC_CHECK(xe::gpu::ShouldShrinkRenderArea(true, false, false, false));
  GPU_ASYNC_CHECK(!xe::gpu::ShouldShrinkRenderArea(false, false, false, false));
  GPU_ASYNC_CHECK(xe::gpu::ShouldShrinkRenderArea(false, false, true, false));

  // Turnip disables the refuted shrink by default unless opted back in via
  // vulkan_tight_render_area_turnip (or render_area_dirty_extent).
  GPU_ASYNC_CHECK(!xe::gpu::ShouldShrinkRenderArea(true, false, false, true));
  GPU_ASYNC_CHECK(xe::gpu::ShouldShrinkRenderArea(true, true, false, true));
  GPU_ASYNC_CHECK(!xe::gpu::ShouldShrinkRenderArea(false, true, false, true));
  GPU_ASYNC_CHECK(xe::gpu::ShouldShrinkRenderArea(false, false, true, true));
}

void TestReplayPipelineBindStateMachine() {
  using xe::gpu::ReplayPipelineBindOutcome;

  // Legacy by-value mode: null drops draws, non-null binds.
  bool unready = false;
  GPU_ASYNC_CHECK(xe::gpu::EvaluateReplayPipelineBind(
                      /*has_slot=*/false, /*resolved_non_null=*/false,
                      unready) == ReplayPipelineBindOutcome::kDropDraws);
  GPU_ASYNC_CHECK(unready);
  GPU_ASYNC_CHECK(xe::gpu::EvaluateReplayPipelineBind(
                      /*has_slot=*/false, /*resolved_non_null=*/true,
                      unready) == ReplayPipelineBindOutcome::kBind);
  GPU_ASYNC_CHECK(!unready);

  // Replay-resolution mode: cold pipeline compiled by replay time binds cleanly;
  // unready/failed pipeline after replay wait drops only until next valid bind.
  GPU_ASYNC_CHECK(xe::gpu::EvaluateReplayPipelineBind(
                      /*has_slot=*/true, /*resolved_non_null=*/true, unready) ==
                  ReplayPipelineBindOutcome::kBind);
  GPU_ASYNC_CHECK(!unready);
  GPU_ASYNC_CHECK(xe::gpu::EvaluateReplayPipelineBind(
                      /*has_slot=*/true, /*resolved_non_null=*/false,
                      unready) == ReplayPipelineBindOutcome::kDropDraws);
  GPU_ASYNC_CHECK(unready);
  GPU_ASYNC_CHECK(xe::gpu::EvaluateReplayPipelineBind(
                      /*has_slot=*/true, /*resolved_non_null=*/true, unready) ==
                  ReplayPipelineBindOutcome::kBind);
  GPU_ASYNC_CHECK(!unready);
}

void TestSourceInvariants(const fs::path& cpp_root) {
  const fs::path xenia_gpu = cpp_root / "xenia" / "src" / "xenia" / "gpu";
  const fs::path vk_dir = xenia_gpu / "vulkan";

  const std::string shader_storage =
      ReadSourceText(xenia_gpu / "shader_storage.h");
  GPU_ASYNC_CHECK(shader_storage.find("setpriority(PRIO_PROCESS, 0, 10)") !=
                  std::string::npos);
  GPU_ASYNC_CHECK(shader_storage.find("QueuePipelineCacheSave") !=
                  std::string::npos);
  GPU_ASYNC_CHECK(shader_storage.find("CoalescingSnapshotSlot") !=
                  std::string::npos);

  const std::string pcache_h =
      ReadSourceText(vk_dir / "vulkan_pipeline_cache.h");
  const std::string pcache_cc =
      ReadSourceText(vk_dir / "vulkan_pipeline_cache.cc");
  GPU_ASYNC_CHECK(pcache_h.find("ResolvePipelineForReplay") !=
                  std::string::npos);
  GPU_ASYNC_CHECK(pcache_cc.find("vulkan_async_pipeline_bind_on_replay, false") !=
                  std::string::npos);
  GPU_ASYNC_CHECK(pcache_cc.find("UPDATE_from_bool(vulkan_async_pipeline_bind_on_replay") ==
                  std::string::npos);
  GPU_ASYNC_CHECK(pcache_cc.find("vkMergePipelineCaches") != std::string::npos);
  GPU_ASYNC_CHECK(pcache_cc.find("QueuePipelineCacheSave") != std::string::npos);

  const std::string dcb_h =
      ReadSourceText(vk_dir / "deferred_command_buffer.h");
  const std::string dcb_cc =
      ReadSourceText(vk_dir / "deferred_command_buffer.cc");
  GPU_ASYNC_CHECK(dcb_h.find("void* pipeline_slot") != std::string::npos);
  GPU_ASYNC_CHECK(dcb_cc.find("ResolvePipelineForReplay") != std::string::npos);

  const std::string shmem_h =
      ReadSourceText(vk_dir / "vulkan_shared_memory.h");
  const std::string shmem_cc =
      ReadSourceText(vk_dir / "vulkan_shared_memory.cc");
  GPU_ASYNC_CHECK(shmem_h.find("buffer_device_address()") != std::string::npos);
  GPU_ASYNC_CHECK(shmem_cc.find("vkGetBufferDeviceAddress") !=
                  std::string::npos);

  const std::string cp_cc =
      ReadSourceText(vk_dir / "vulkan_command_processor.cc");
  GPU_ASYNC_CHECK(cp_cc.find("shared_memory_->buffer_device_address()") !=
                  std::string::npos);
  GPU_ASYNC_CHECK(cp_cc.find("vkGetBufferDeviceAddress") == std::string::npos);
  GPU_ASYNC_CHECK(cp_cc.find("vulkan_async_pipeline_bind_on_replay") !=
                  std::string::npos);
  GPU_ASYNC_CHECK(cp_cc.find("current_pipeline != VK_NULL_HANDLE &&") !=
                  std::string::npos);
  GPU_ASYNC_CHECK(cp_cc.find("vulkan_tight_render_area_turnip, false") !=
                  std::string::npos);
  GPU_ASYNC_CHECK(cp_cc.find("ShouldShrinkRenderArea(") != std::string::npos);

  const std::string rtc_cc =
      ReadSourceText(vk_dir / "vulkan_render_target_cache.cc");
  GPU_ASYNC_CHECK(rtc_cc.find("IsProprietaryMaliDriver(") !=
                  std::string::npos);
}

}  // namespace

int main(int argc, char** argv) {
  TestCoalescingSnapshotSlot();
  TestIsProprietaryMaliDriver();
  TestShouldShrinkRenderArea();
  TestReplayPipelineBindStateMachine();
  if (argc >= 2) {
    TestSourceInvariants(fs::path(argv[1]));
  } else {
    std::fprintf(stderr, "Usage: gpu_pipeline_async_test <cpp_root>\n");
    return 1;
  }
  std::printf("gpu_pipeline_async_test: PASS\n");
  return 0;
}
