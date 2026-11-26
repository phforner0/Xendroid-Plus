#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef XE_DISABLE_ANDROID_DLOPEN
#define XE_DISABLE_ANDROID_DLOPEN 1
#endif

#include "xe_audio_adpf_logic.h"
#include "xenia/kernel/adpf_tid_registry.h"

namespace {

#define ADPF_CHECK(cond)                                                 \
  do {                                                                   \
    if (!(cond)) {                                                       \
      std::fprintf(stderr, "AUDIO_ADPF check failed (%s:%d): %s\n",      \
                   __FILE__, __LINE__, #cond);                           \
      std::abort();                                                      \
    }                                                                    \
  } while (false)

namespace fs = std::filesystem;

std::string ReadSourceText(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  ADPF_CHECK(in.good());
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

struct FakeAdpfState {
  int create_calls = 0;
  int set_threads_calls = 0;
  int report_calls = 0;
  int close_calls = 0;
  int power_eff_calls = 0;

  bool fail_create = false;
  bool fail_set_threads = false;
  bool fail_power_eff = false;
  bool allow_set_threads = true;

  int next_handle = 1;
  void* active_session = nullptr;
  int64_t last_target_ns = 0;
  int64_t last_report_ns = 0;
  bool last_power_eff_enabled = false;
  std::vector<int32_t> last_tids;
  std::vector<int32_t> dead_tids;

  static void* Create(void* user, const int32_t* tids, size_t count,
                      int64_t target_ns) {
    auto* s = static_cast<FakeAdpfState*>(user);
    ++s->create_calls;
    if (s->fail_create || count == 0) {
      return nullptr;
    }
    for (size_t i = 0; i < count; ++i) {
      for (int32_t dead : s->dead_tids) {
        if (tids[i] == dead) {
          return nullptr;
        }
      }
    }
    s->last_tids.assign(tids, tids + count);
    s->last_target_ns = target_ns;
    s->active_session = reinterpret_cast<void*>(uintptr_t(s->next_handle++));
    return s->active_session;
  }

  static int SetThreads(void* user, void* session, const int32_t* tids,
                        size_t count) {
    auto* s = static_cast<FakeAdpfState*>(user);
    ++s->set_threads_calls;
    if (s->fail_set_threads || session == nullptr || count == 0) {
      return -1;
    }
    for (size_t i = 0; i < count; ++i) {
      for (int32_t dead : s->dead_tids) {
        if (tids[i] == dead) {
          return -1;
        }
      }
    }
    s->last_tids.assign(tids, tids + count);
    return 0;
  }

  static int Report(void* user, void* session, int64_t actual_ns) {
    auto* s = static_cast<FakeAdpfState*>(user);
    ++s->report_calls;
    if (!session || actual_ns <= 0) {
      return -1;
    }
    s->last_report_ns = actual_ns;
    return 0;
  }

  static void Close(void* user, void* session) {
    auto* s = static_cast<FakeAdpfState*>(user);
    ++s->close_calls;
    if (s->active_session == session) {
      s->active_session = nullptr;
    }
  }

  static bool IsAlive(void* user, int32_t tid) {
    auto* s = static_cast<FakeAdpfState*>(user);
    if (tid <= 0) {
      return false;
    }
    for (int32_t dead : s->dead_tids) {
      if (tid == dead) {
        return false;
      }
    }
    return true;
  }

  static int SetPowerEff(void* user, void* session, bool enabled) {
    auto* s = static_cast<FakeAdpfState*>(user);
    ++s->power_eff_calls;
    if (s->fail_power_eff || !session) {
      return -1;
    }
    s->last_power_eff_enabled = enabled;
    return 0;
  }

  xendroid::AudioAdpfOps AudioOps() {
    xendroid::AudioAdpfOps ops;
    ops.create_session = &Create;
    ops.set_threads = allow_set_threads ? &SetThreads : nullptr;
    ops.report_actual_work = &Report;
    ops.close_session = &Close;
    ops.is_tid_alive = &IsAlive;
    ops.user = this;
    return ops;
  }

  xe::kernel::BackgroundPowerHintSession::Ops BgOps(bool with_power_eff) {
    xe::kernel::BackgroundPowerHintSession::Ops ops;
    ops.create_session = &Create;
    ops.set_threads = allow_set_threads ? &SetThreads : nullptr;
    ops.set_prefer_power_efficiency = with_power_eff ? &SetPowerEff : nullptr;
    ops.close_session = &Close;
    ops.is_tid_alive = &IsAlive;
    ops.user = this;
    return ops;
  }
};

void TestAudioThreadNameFiltering() {
  ADPF_CHECK(xendroid::IsAudioAdpfThreadName("Audio Worker"));
  ADPF_CHECK(xendroid::IsAudioAdpfThreadName("Audio Worker\n"));
  ADPF_CHECK(xendroid::IsAudioAdpfThreadName("XMA Decoder"));
  ADPF_CHECK(xendroid::IsAudioAdpfThreadName("XMA Decoder\r\n"));

  ADPF_CHECK(!xendroid::IsAudioAdpfThreadName("Guest CPU 0"));
  ADPF_CHECK(!xendroid::IsAudioAdpfThreadName("Guest CPU 5\n"));
  ADPF_CHECK(!xendroid::IsAudioAdpfThreadName("Guest I/O"));
  ADPF_CHECK(!xendroid::IsAudioAdpfThreadName("Guest I/O 0"));
  ADPF_CHECK(!xendroid::IsAudioAdpfThreadName(""));
  ADPF_CHECK(!xendroid::IsAudioAdpfThreadName("Audio"));

  const std::vector<xendroid::ThreadCommEntry> entries = {
      {1001, "Guest CPU 0\n"},
      {1002, "Guest CPU 1\n"},
      {1003, "Guest CPU 2\n"},
      {1004, "Guest CPU 3\n"},
      {1005, "Guest CPU 4\n"},
      {1006, "Guest CPU 5\n"},
      {1010, "Guest I/O\n"},
      {1025, "Audio Worker\n"},
      {1020, "Audio Worker\n"},
      {1035, "XMA Decoder\n"},
      {1030, "XMA Decoder\n"},
      {-1, "Audio Worker"},
  };
  const auto selected = xendroid::SelectAudioAdpfThreadTids(entries);
  ADPF_CHECK(selected.size() == 2u);
  ADPF_CHECK(selected[0] == 1020);
  ADPF_CHECK(selected[1] == 1030);
}

void TestAudioAdpfSessionLifecycleAndDynamicThreads() {
  xendroid::AudioAdpfThreadRegistry reg;
  ADPF_CHECK(reg.generation() == 0u);

  // Audio Worker starts first (XMA Decoder not yet registered).
  reg.SetAudioWorkerTid(2001);
  ADPF_CHECK(reg.generation() == 1u);

  FakeAdpfState fake;
  {
    xendroid::AudioAdpfSession session(5333 * 1000LL, fake.AudioOps());
    session.SyncAndReport(true, 1200, reg);
    ADPF_CHECK(session.active());
    ADPF_CHECK(fake.create_calls == 1);
    ADPF_CHECK(fake.last_target_ns == 5333000LL);
    ADPF_CHECK(fake.last_tids == std::vector<int32_t>({2001}));
    ADPF_CHECK(fake.report_calls == 1);
    ADPF_CHECK(fake.last_report_ns == 1200000LL);

    // XMA Decoder starts after first pump -> dynamic setThreads updates to 2.
    reg.SetXmaDecoderTid(2005);
    session.SyncAndReport(true, 0, reg);
    ADPF_CHECK(fake.create_calls == 1);
    ADPF_CHECK(fake.set_threads_calls == 1);
    ADPF_CHECK(session.active_tids() == std::vector<int32_t>({2001, 2005}));
    // 0 us clamped to >= 1000 ns so Android never rejects with EINVAL.
    ADPF_CHECK(fake.last_report_ns == 1000LL);

    // Mismatched clear does not erase active TID.
    reg.ClearXmaDecoderTid(9999);
    ADPF_CHECK(session.active_tids() == std::vector<int32_t>({2001, 2005}));

    // Recreate fallback when setThreads fails.
    reg.ClearXmaDecoderTid(2005);
    reg.SetXmaDecoderTid(2010);
    fake.fail_set_threads = true;
    session.SyncAndReport(true, 800, reg);
    ADPF_CHECK(fake.set_threads_calls == 2);
    ADPF_CHECK(fake.close_calls == 1);
    ADPF_CHECK(fake.create_calls == 2);
    ADPF_CHECK(session.active());
    ADPF_CHECK(session.active_tids() == std::vector<int32_t>({2001, 2010}));
    fake.fail_set_threads = false;

    // Disabling closes immediately; re-enabling opens again.
    session.SyncAndReport(false, 500, reg);
    ADPF_CHECK(!session.active());
    ADPF_CHECK(fake.close_calls == 2);

    session.SyncAndReport(true, 900, reg);
    ADPF_CHECK(session.active());
    ADPF_CHECK(fake.create_calls == 3);
  }
  // Leaving scope triggers RAII closeSession.
  ADPF_CHECK(fake.close_calls == 3);
  ADPF_CHECK(fake.active_session == nullptr);

  // Dead TID filtering avoids passing stale TIDs to createSession.
  FakeAdpfState dead_fake;
  dead_fake.dead_tids = {2010};
  {
    xendroid::AudioAdpfSession session(5333 * 1000LL, dead_fake.AudioOps());
    session.SyncAndReport(true, 600, reg);
    ADPF_CHECK(session.active());
    ADPF_CHECK(dead_fake.last_tids == std::vector<int32_t>({2001}));
  }
  ADPF_CHECK(dead_fake.close_calls == 1);

  // Android API 33 (no setThreads symbol): closes and recreates when XMA joins,
  // and closes cleanly if all audio threads unregister.
  xendroid::AudioAdpfThreadRegistry api33_reg;
  api33_reg.SetAudioWorkerTid(4001);
  FakeAdpfState api33_fake;
  api33_fake.allow_set_threads = false;
  {
    xendroid::AudioAdpfSession session(5333 * 1000LL, api33_fake.AudioOps());
    session.SyncAndReport(true, 700, api33_reg);
    ADPF_CHECK(session.active());
    ADPF_CHECK(api33_fake.create_calls == 1);
    ADPF_CHECK(api33_fake.set_threads_calls == 0);

    api33_reg.SetXmaDecoderTid(4002);
    session.SyncAndReport(true, 800, api33_reg);
    ADPF_CHECK(session.active());
    ADPF_CHECK(api33_fake.close_calls == 1);
    ADPF_CHECK(api33_fake.create_calls == 2);
    ADPF_CHECK(session.active_tids() == std::vector<int32_t>({4001, 4002}));

    api33_reg.ClearAudioWorkerTid(4001);
    api33_reg.ClearXmaDecoderTid(4002);
    session.SyncAndReport(true, 800, api33_reg);
    ADPF_CHECK(!session.active());
    ADPF_CHECK(!session.permanently_failed());
    ADPF_CHECK(api33_fake.close_calls == 2);
  }
}

void TestContentionBreakdownAndHelpers() {
  const auto bd = xendroid::ComputeCallbackTimeBreakdown(28468, 404, 475);
  ADPF_CHECK(bd.wall_us == 28468);
  ADPF_CHECK(bd.cpu_us == 404);
  ADPF_CHECK(bd.run_queue_wait_us == 475);
  ADPF_CHECK(bd.blocked_sleep_us == 28468 - (404 + 475));

  // Clock skew and uint64 addition overflow guards: never underflow.
  const auto skewed = xendroid::ComputeCallbackTimeBreakdown(100, 90, 30);
  ADPF_CHECK(skewed.blocked_sleep_us == 0);
  const auto overflow = xendroid::ComputeCallbackTimeBreakdown(
      500, UINT64_MAX, 100);
  ADPF_CHECK(overflow.blocked_sleep_us == 0);

  // Audio silence scan helper.
  float samples[16] = {};
  ADPF_CHECK(xendroid::IsAudioFrameSamplesSilent(samples, 16));
  samples[7] = 0.001f;
  ADPF_CHECK(!xendroid::IsAudioFrameSamplesSilent(samples, 16));

  // XMA enabled mask helper.
  std::atomic<uint64_t> mask[5] = {};
  ADPF_CHECK(!xendroid::HasAnyEnabledXmaContext(mask, 5));
  mask[4].store(1ULL << 63);
  ADPF_CHECK(xendroid::HasAnyEnabledXmaContext(mask, 5));
}

void TestPowerEfficiencyHelperAndBackgroundSession() {
  using xe::kernel::BackgroundPowerHintSession;
  using xe::kernel::GuestCpuTidRegistry;
  using xe::kernel::SetAdpfPreferPowerEfficiency;

  FakeAdpfState fake;
  int dummy_session = 1;
  ADPF_CHECK(!SetAdpfPreferPowerEfficiency(nullptr, true, &FakeAdpfState::SetPowerEff, &fake));
  ADPF_CHECK(!SetAdpfPreferPowerEfficiency(&dummy_session, true, nullptr, &fake));
  fake.fail_power_eff = true;
  ADPF_CHECK(!SetAdpfPreferPowerEfficiency(&dummy_session, true, &FakeAdpfState::SetPowerEff, &fake));
  fake.fail_power_eff = false;
  ADPF_CHECK(SetAdpfPreferPowerEfficiency(&dummy_session, true, &FakeAdpfState::SetPowerEff, &fake));
  ADPF_CHECK(fake.last_power_eff_enabled);

  // API < 35 (no setPreferPowerEfficiency symbol): never create a boost session for Guest I/O.
  GuestCpuTidRegistry io_reg;
  io_reg.RegisterTid(3001);
  {
    FakeAdpfState api34_fake;
    BackgroundPowerHintSession bg(api34_fake.BgOps(/*with_power_eff=*/false));
    bg.SyncThreads(true, io_reg);
    ADPF_CHECK(!bg.active());
    ADPF_CHECK(api34_fake.create_calls == 0);
  }

  // API 35+: if setPreferPowerEfficiency returns error, close immediately.
  {
    FakeAdpfState fail_fake;
    fail_fake.fail_power_eff = true;
    BackgroundPowerHintSession bg(fail_fake.BgOps(/*with_power_eff=*/true));
    bg.SyncThreads(true, io_reg);
    ADPF_CHECK(!bg.active());
    ADPF_CHECK(fail_fake.create_calls == 1);
    ADPF_CHECK(fail_fake.close_calls == 1);
  }

  // API 35+ happy path: create + preferPowerEfficiency(true) + dynamic setThreads + RAII close.
  FakeAdpfState ok_fake;
  {
    BackgroundPowerHintSession bg(ok_fake.BgOps(/*with_power_eff=*/true));
    bg.SyncThreads(true, io_reg);
    ADPF_CHECK(bg.active());
    ADPF_CHECK(ok_fake.create_calls == 1);
    ADPF_CHECK(ok_fake.power_eff_calls == 1);
    ADPF_CHECK(ok_fake.last_power_eff_enabled);
    ADPF_CHECK(ok_fake.last_tids == std::vector<int32_t>({3001}));

    io_reg.RegisterTid(3002);
    bg.SyncThreads(true, io_reg);
    ADPF_CHECK(ok_fake.set_threads_calls == 1);
    ADPF_CHECK(ok_fake.last_tids == std::vector<int32_t>({3001, 3002}));

    // Unregistering all I/O threads closes session without failing; re-register reopens.
    io_reg.UnregisterTid(3001);
    io_reg.UnregisterTid(3002);
    bg.SyncThreads(true, io_reg);
    ADPF_CHECK(!bg.active());
    ADPF_CHECK(ok_fake.close_calls == 1);

    io_reg.RegisterTid(3003);
    bg.SyncThreads(true, io_reg);
    ADPF_CHECK(bg.active());
    ADPF_CHECK(ok_fake.create_calls == 2);
    ADPF_CHECK(ok_fake.power_eff_calls == 2);
    ADPF_CHECK(ok_fake.last_tids == std::vector<int32_t>({3003}));
  }
  ADPF_CHECK(ok_fake.close_calls == 2);
}

void TestSourceContracts(const fs::path& cpp_dir) {
  const std::string audio_cc =
      ReadSourceText(cpp_dir / "xenia/src/xenia/apu/audio_system.cc");
  ADPF_CHECK(audio_cc.find("\"Guest CPU\"") == std::string::npos);
  ADPF_CHECK(audio_cc.find("AudioAdpfSession") != std::string::npos);
  ADPF_CHECK(audio_cc.find("APerformanceHint_closeSession") != std::string::npos);
  ADPF_CHECK(audio_cc.find("APerformanceHint_setThreads") != std::string::npos);
  ADPF_CHECK(audio_cc.find("AudioSystem::kAudioPumpInterval * int64_t(1000)") !=
             std::string::npos);
  ADPF_CHECK(audio_cc.find("GlobalAudioAdpfThreadRegistry") != std::string::npos);
  ADPF_CHECK(audio_cc.find("ComputeCallbackTimeBreakdown") != std::string::npos);
  ADPF_CHECK(audio_cc.find("blocked_sleep") != std::string::npos);
  ADPF_CHECK(audio_cc.find("IsAudioFrameSamplesSilent") != std::string::npos);

  const std::string xma_cc =
      ReadSourceText(cpp_dir / "xenia/src/xenia/apu/xma_decoder.cc");
  ADPF_CHECK(xma_cc.find("GlobalAudioAdpfThreadRegistry") != std::string::npos);
  ADPF_CHECK(xma_cc.find("HasAnyEnabledXmaContext") != std::string::npos);
  ADPF_CHECK(xma_cc.find("setpriority(PRIO_PROCESS, 0, -12)") != std::string::npos);

  const std::string aaudio_cpp =
      ReadSourceText(cpp_dir / "xe_aaudio_audio_driver.cpp");
  ADPF_CHECK(aaudio_cpp.find("semaphore_->Release(static_cast<int>(releases), nullptr)") !=
             std::string::npos);

  const std::string adpf_h =
      ReadSourceText(cpp_dir / "xenia/src/xenia/kernel/adpf_tid_registry.h");
  ADPF_CHECK(adpf_h.find("APerformanceHint_setPreferPowerEfficiency") !=
             std::string::npos);
  ADPF_CHECK(adpf_h.find("BackgroundPowerHintSession") != std::string::npos);

  const std::string sched_cc =
      ReadSourceText(cpp_dir / "xenia/src/xenia/kernel/guest_scheduler.cc");
  ADPF_CHECK(sched_cc.find("kIoNice = 10") != std::string::npos);
  ADPF_CHECK(sched_cc.find("setpriority(PRIO_PROCESS, 0, kIoNice)") !=
             std::string::npos);
  ADPF_CHECK(sched_cc.find("ApplyBackgroundIoPriority(\"Guest I/O\")") !=
             std::string::npos);
  ADPF_CHECK(sched_cc.find("ApplyBackgroundIoPriority(\"Guest I/O pool\")") !=
             std::string::npos);
  ADPF_CHECK(sched_cc.find("BackgroundPowerHintSession") != std::string::npos);
  ADPF_CHECK(sched_cc.find("GuestIoPowerHintManager::Instance().Shutdown()") !=
             std::string::npos);
}

}  // namespace

int main(int argc, char** argv) {
  TestAudioThreadNameFiltering();
  TestAudioAdpfSessionLifecycleAndDynamicThreads();
  TestContentionBreakdownAndHelpers();
  TestPowerEfficiencyHelperAndBackgroundSession();
  if (argc >= 2 && argv[1] && argv[1][0] != '\0') {
    TestSourceContracts(fs::path(argv[1]));
  }
  std::printf("audio_adpf_logic_test: PASS\n");
  return 0;
}
