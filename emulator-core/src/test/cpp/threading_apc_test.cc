// Native test of Thread::QueueUserCallback (the host side of guest APCs and
// NtAlertThread): callbacks queued at the same time to different threads must
// each run on their own target thread, during an alertable wait, and never on
// another thread or with no target at all. Also multi-object waits given the
// same handle twice, which the cooperative scheduler polls with a 0 ms timeout.
//
// Built for Android (the fork's XE_PLATFORM_xendroid path) as a static x86_64
// executable by tools/test-native-logic.sh, with the logging calls stubbed.

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "xenia/base/logging.h"
#include "xenia/base/threading.h"

namespace xe::logging {
bool ShouldLog(LogLevel, uint32_t) { return false; }
namespace internal {
std::pair<char*, size_t> GetThreadBuffer() {
  static thread_local char buffer[1024];
  return {buffer, sizeof(buffer)};
}
void AppendLogLine(LogLevel, const char, size_t) {}
}  // namespace internal
}  // namespace xe::logging

using namespace std::chrono_literals;
using xe::threading::Thread;

namespace {

int failures = 0;

void check(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAILED: %s\n", what);
    ++failures;
  }
}

// One callback, on its own thread, during an alertable sleep.
void single_callback_runs_on_its_thread() {
  std::atomic<bool> stop{false};
  std::atomic<int> ran{0};
  std::atomic<Thread*> where{nullptr};
  Thread::CreationParameters params = {};
  auto thread = Thread::Create(params, [&] {
    while (!stop.load()) xe::threading::AlertableSleep(1ms);
  });
  Thread* target = thread.get();
  for (int i = 0; i < 200 && ran.load() == 0; ++i) {
    thread->QueueUserCallback([&] {
      where.store(Thread::GetCurrentThread());
      ran.fetch_add(1);
    });
    std::this_thread::sleep_for(2ms);
  }
  stop.store(true);
  check(xe::threading::Wait(thread.get(), false, 2s) ==
            xe::threading::WaitResult::kSuccess,
        "the alerted thread did not finish");
  check(ran.load() > 0, "the callback never ran during an alertable sleep");
  check(where.load() == target, "the callback ran on another thread");
}

// Several queuers alerting several targets at once: each callback must find
// itself on the thread it was queued to.
void concurrent_callbacks_stay_on_their_targets() {
  constexpr int kTargets = 4;
  constexpr int kQueuers = 3;
  constexpr int kRounds = 3000;
  std::atomic<bool> stop{false};
  std::atomic<int> ran[kTargets] = {};
  std::atomic<int> wrong{0};
  std::vector<std::unique_ptr<Thread>> targets;
  Thread::CreationParameters params = {};
  for (int i = 0; i < kTargets; ++i) {
    targets.push_back(Thread::Create(params, [&] {
      while (!stop.load()) xe::threading::AlertableSleep(500us);
    }));
  }
  std::vector<std::thread> queuers;
  for (int q = 0; q < kQueuers; ++q) {
    queuers.emplace_back([&, q] {
      for (int round = 0; round < kRounds; ++round) {
        int j = (round + q) % kTargets;
        Thread* target = targets[j].get();
        target->QueueUserCallback([&, j, target] {
          if (Thread::GetCurrentThread() != target) wrong.fetch_add(1);
          ran[j].fetch_add(1);
        });
      }
    });
  }
  for (auto& queuer : queuers) queuer.join();
  std::this_thread::sleep_for(20ms);
  stop.store(true);
  for (auto& target : targets) {
    check(xe::threading::Wait(target.get(), false, 2s) ==
              xe::threading::WaitResult::kSuccess,
          "an alerted thread did not finish");
  }
  int total = 0;
  for (auto& count : ran) total += count.load();
  std::printf("concurrent alerts: %d callbacks ran, %d on the wrong thread\n",
              total, wrong.load());
  check(wrong.load() == 0, "a callback ran on a thread it was not queued to");
  check(total > 0, "no callback ran");
}

// A handle passed twice: a wait-any still times out, then succeeds on the
// first slot, and a wait-all fails as on Windows. Each used to spin forever.
void repeated_handle_waits_end() {
  using xe::threading::WaitResult;
  auto event = xe::threading::Event::CreateManualResetEvent(false);
  auto other = xe::threading::Event::CreateManualResetEvent(false);
  auto any = xe::threading::WaitAny({event.get(), other.get(), event.get()},
                                    false, 0ms);
  check(any.first == WaitResult::kTimeout,
        "a wait-any on a repeated handle did not time out");
  event->Set();
  any = xe::threading::WaitAny({event.get(), other.get(), event.get()}, false,
                               0ms);
  check(any.first == WaitResult::kSuccess && any.second == 0,
        "a wait-any on a repeated handle did not end on its first slot");
  check(xe::threading::WaitAll({event.get(), other.get(), event.get()}, false,
                               0ms) == WaitResult::kFailed,
        "a wait-all on a repeated handle did not fail");
}

// Event::Pulse wakes only callers already waiting (all of them for manual-reset,
// one for auto-reset), leaves the event unsignaled, and does not sleep.
void pulse_event_releases_only_current_waiters() {
  using xe::threading::Event;
  using xe::threading::WaitResult;
  for (bool manual_reset : {false, true}) {
    auto evt = manual_reset ? Event::CreateManualResetEvent(false)
                            : Event::CreateAutoResetEvent(false);
    check(evt != nullptr, "event creation failed");

    // With nobody waiting, a pulse is lost and leaves the event unsignaled.
    evt->Pulse();
    check(xe::threading::Wait(evt.get(), false, 20ms) == WaitResult::kTimeout,
          "pulse with no waiters left the event signaled");
    check(evt->Query().state == 0,
          "pulse left Query().state nonzero");

    auto other = Event::CreateAutoResetEvent(false);
    std::atomic<unsigned> started{0};
    std::atomic<unsigned> released{0};
    std::thread single_waiter([&] {
      started.fetch_add(1);
      if (xe::threading::Wait(evt.get(), false, 2s) == WaitResult::kSuccess) {
        released.fetch_add(1);
      }
    });
    std::thread multi_waiter([&] {
      started.fetch_add(1);
      auto res = xe::threading::WaitAny({evt.get(), other.get()}, false, 2s);
      if (res.first == WaitResult::kSuccess && res.second == 0) {
        released.fetch_add(1);
      }
    });

    for (int i = 0; i < 200 && started.load() < 2; ++i) {
      std::this_thread::sleep_for(1ms);
    }
    std::this_thread::sleep_for(40ms);

    evt->Pulse();
    const unsigned expected = manual_reset ? 2u : 1u;
    for (int i = 0; i < 500 && released.load() < expected; ++i) {
      std::this_thread::sleep_for(1ms);
    }
    std::this_thread::sleep_for(30ms);
    check(released.load() == expected,
          manual_reset ? "manual-reset pulse did not release all waiters"
                       : "auto-reset pulse did not release exactly one waiter");

    if (!manual_reset) {
      evt->Pulse();
      for (int i = 0; i < 500 && released.load() < 2u; ++i) {
        std::this_thread::sleep_for(1ms);
      }
    }

    single_waiter.join();
    multi_waiter.join();
    check(released.load() == 2u, "second waiter was never released");
    check(xe::threading::Wait(evt.get(), false, 20ms) == WaitResult::kTimeout,
          "event remained signaled after pulse woke waiters");
  }
}

// SignalAndWaitOn (enqueue-before-signal): under auto-reset handoff, the waiter
// must already be queued on `wait_on` before `to_signal` wakes the peer, so the
// peer's immediate Set/Pulse cannot race ahead and get lost or stolen.
void signal_and_wait_handoff_enqueue_before_signal() {
  using xe::threading::Event;
  using xe::threading::WaitResult;

  xe::threading::SetAutoResetEventHandoff(true);
  auto req = Event::CreateAutoResetEvent(false);
  auto ack = Event::CreateAutoResetEvent(false);
  auto aux = Event::CreateAutoResetEvent(false);
  std::atomic<bool> stop{false};
  std::atomic<int> peer_rounds{0};
  std::atomic<int> multi_steals{0};

  // Competing WaitAny observer on `ack` + `aux` exercises multi-wait interaction.
  std::thread multi_observer([&] {
    while (!stop.load(std::memory_order_acquire)) {
      auto res = xe::threading::WaitAny({ack.get(), aux.get()}, false, 1ms);
      if (res.first == WaitResult::kSuccess && res.second == 0) {
        multi_steals.fetch_add(1, std::memory_order_relaxed);
        // Hand it back so the main loop does not stall if a race ever occurred.
        ack->Set();
      }
    }
  });

  // Peer waits on `req` and immediately responds on `ack` (alternating Set and
  // Pulse to verify ScopedParked + FIFO release).
  constexpr int kRounds = 400;
  std::thread peer([&] {
    for (int i = 0; i < kRounds; ++i) {
      if (xe::threading::Wait(req.get(), false, 2s) != WaitResult::kSuccess) {
        return;
      }
      if ((i & 3) == 3) {
        ack->Pulse();
      } else {
        ack->Set();
      }
      peer_rounds.fetch_add(1, std::memory_order_release);
    }
  });

  int main_ok = 0;
  for (int i = 0; i < kRounds; ++i) {
    auto res = xe::threading::SignalAndWait(req.get(), ack.get(), false, 2s);
    if (res == WaitResult::kSuccess) {
      ++main_ok;
    } else {
      break;
    }
  }

  stop.store(true, std::memory_order_release);
  aux->Set();
  peer.join();
  multi_observer.join();
  xe::threading::SetAutoResetEventHandoff(false);

  check(main_ok == kRounds && peer_rounds.load() == kRounds,
        "SignalAndWait handoff lost a wakeup under enqueue-before-signal");
  check(multi_steals.load() == 0,
        "WaitAny stole a token from a SignalAndWait queued waiter");
}

// SignalAndWaitOn failure & same-object safety: if to_signal->Signal() fails
// (e.g. semaphore at max count), SignalAndWait must return kFailed and must not
// drop a pre-existing or concurrently delivered token on the waited event.
void signal_and_wait_rollback_and_self_signal() {
  using xe::threading::Event;
  using xe::threading::Semaphore;
  using xe::threading::WaitResult;

  for (bool handoff : {false, true}) {
    xe::threading::SetAutoResetEventHandoff(handoff);
    auto full_sem = Semaphore::Create(1, 1);
    auto evt = Event::CreateAutoResetEvent(true);
    check(xe::threading::SignalAndWait(full_sem.get(), evt.get(), false,
                                       50ms) == WaitResult::kFailed,
          "SignalAndWait did not fail when to_signal->Signal() failed");
    // The pre-existing token on `evt` must remain intact.
    check(xe::threading::Wait(evt.get(), false, 0ms) == WaitResult::kSuccess,
          "failed SignalAndWait consumed or lost the event token");

    // Self-signal (same object) must not deadlock.
    auto self_evt = Event::CreateAutoResetEvent(false);
    check(xe::threading::SignalAndWait(self_evt.get(), self_evt.get(), false,
                                       500ms) == WaitResult::kSuccess,
          "SignalAndWait on the same event failed or deadlocked");
  }
  xe::threading::SetAutoResetEventHandoff(false);
}

// Thread lifetime safety: destroying the Thread handle from inside the start
// routine just before returning normally (simulating XThread ReleaseHandle ->
// ~XThread -> thread_.reset()) must not UAF in ThreadStartRoutine, and
// Thread::Exit() must signal cleanly without leaking or crashing.
void thread_lifetime_normal_return_and_exit() {
  using xe::threading::WaitResult;
  Thread::CreationParameters params = {};

  // 1. Normal return after the owning unique_ptr<Thread> is reset inside the
  //    worker routine itself (exact UAF trigger of ThreadStartRoutine).
  for (int i = 0; i < 64; ++i) {
    std::atomic<bool> ready{false};
    std::atomic<bool> destroyed{false};
    std::unique_ptr<Thread> holder;
    holder = Thread::Create(params, [&] {
      while (!ready.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      holder.reset();
      destroyed.store(true, std::memory_order_release);
    });
    check(holder != nullptr, "Thread::Create failed in self-destroy test");
    ready.store(true, std::memory_order_release);
    for (int spin = 0; spin < 2000 && !destroyed.load(std::memory_order_acquire);
         ++spin) {
      std::this_thread::sleep_for(1ms);
    }
    check(destroyed.load(std::memory_order_acquire),
          "self-destroying thread did not run to completion");
    // Give ThreadStartRoutine time to execute post-start_routine signaling.
    std::this_thread::sleep_for(2ms);
  }

  // 2. Explicit Thread::Exit() path (pthread_exit) waited on by caller.
  for (int i = 0; i < 64; ++i) {
    auto t = Thread::Create(params, [] { Thread::Exit(0); });
    check(t != nullptr, "Thread::Create failed in Thread::Exit test");
    check(xe::threading::Wait(t.get(), false, 2s) == WaitResult::kSuccess,
          "Thread::Exit thread did not signal completion");
  }

  // 3. Fire-and-forget parent destruction racing with ThreadStartRoutine
  //    startup (must not UAF on PosixThread* before or during Thread::Exit).
  std::atomic<int> detached_done{0};
  constexpr int kDetachedCount = 64;
  for (int i = 0; i < kDetachedCount; ++i) {
    const bool use_exit = (i & 1) != 0;
    auto t = Thread::Create(params, [&, use_exit] {
      detached_done.fetch_add(1, std::memory_order_release);
      if (use_exit) {
        Thread::Exit(0);
      }
    });
    check(t != nullptr, "Thread::Create failed in fire-and-forget test");
    t.reset();  // Destroy PosixThread immediately while child is starting.
  }
  for (int spin = 0;
       spin < 5000 && detached_done.load(std::memory_order_acquire) < kDetachedCount;
       ++spin) {
    std::this_thread::sleep_for(1ms);
  }
  check(detached_done.load(std::memory_order_acquire) == kDetachedCount,
        "fire-and-forget threads did not all complete cleanly");
  std::this_thread::sleep_for(10ms);
}

}  // namespace

int main() {
  single_callback_runs_on_its_thread();
  concurrent_callbacks_stay_on_their_targets();
  repeated_handle_waits_end();
  pulse_event_releases_only_current_waiters();
  signal_and_wait_handoff_enqueue_before_signal();
  signal_and_wait_rollback_and_self_signal();
  thread_lifetime_normal_return_and_exit();
  if (failures) {
    std::fprintf(stderr, "threading_apc_test: %d failure(s)\n", failures);
    return 1;
  }
  std::printf("threading_apc_test: ok\n");
  return 0;
}
