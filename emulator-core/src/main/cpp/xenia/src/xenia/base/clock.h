/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_BASE_CLOCK_H_
#define XENIA_BASE_CLOCK_H_

#include <atomic>
#include <cstdint>

namespace xe {

// Atomic refcounted hold counter for guest time progression.
// Supports nested holds (e.g. Emulator::Pause() overlapping with synchronous
// Vulkan pipeline compilation waits) without ever resuming while an outer hold
// is active, and guards against underflow on redundant releases (e.g.
// Emulator::Shutdown() calling HoldGuestTime(false) when already unheld).
class GuestTimeHoldRefCounter {
 public:
  // Increments the hold count. Returns true only on the 0 -> 1 transition
  // (when guest time actually freezes).
  bool Acquire() {
    return count_.fetch_add(1, std::memory_order_acq_rel) == 0;
  }

  // Decrements the hold count if > 0. Returns true only on the 1 -> 0
  // transition (when guest time actually resumes). Never underflows below 0.
  bool Release() {
    int32_t current = count_.load(std::memory_order_acquire);
    while (current > 0) {
      if (count_.compare_exchange_weak(current, current - 1,
                                       std::memory_order_acq_rel,
                                       std::memory_order_acquire)) {
        return current == 1;
      }
    }
    return false;
  }

  int32_t count() const { return count_.load(std::memory_order_acquire); }
  bool is_held() const { return count_.load(std::memory_order_acquire) > 0; }

 private:
  std::atomic<int32_t> count_{0};
};

}  // namespace xe

#if !defined(XENIA_BASE_CLOCK_PURE_LOGIC_ONLY)

#include <chrono>
#include <utility>

#include "xenia/base/cvar.h"
#include "xenia/base/platform.h"

#if XE_ARCH_AMD64 || XE_ARCH_ARM64
#define XE_CLOCK_RAW_AVAILABLE 1
#endif

DECLARE_bool(clock_no_scaling);
DECLARE_bool(clock_source_raw);

namespace xe {

// chrono APIs in xenia/base/chrono.h are preferred

class Clock {
 public:
  // Host ticks-per-second. Generally QueryHostTickFrequency should be used.
  // Either from platform suplied time source or from hardware directly.
  static uint64_t host_tick_frequency_platform();
#if XE_CLOCK_RAW_AVAILABLE
  XE_NOINLINE
  static uint64_t host_tick_frequency_raw();
#endif
  // Host tick count. Generally QueryHostTickCount() should be used.
  static uint64_t host_tick_count_platform();
#if XE_CLOCK_RAW_AVAILABLE
  // chrispy: the way msvc was ordering the branches was causing rdtsc to be
  // speculatively executed each time the branch history was lost
  XE_NOINLINE
  static uint64_t host_tick_count_raw();
#endif

  // Queries the host tick frequency.
  static uint64_t QueryHostTickFrequency();
  // Queries the current host tick count.
  static uint64_t QueryHostTickCount();
  // Host time, in FILETIME format.
  static uint64_t QueryHostSystemTime();
  // Queries the milliseconds since the host began.
  static uint64_t QueryHostUptimeMillis();

  static uint64_t QueryHostInterruptTime();

  // Guest time scalar.
  static double guest_time_scalar();
  // Sets the guest time scalar, adjusting tick and wall clock speed.
  // Ex: 1x=normal, 2x=double speed, 1/2x=half speed.
  static void set_guest_time_scalar(double scalar);
  // Get the tick ration between host and guest including time scaling if set.
  static std::pair<uint64_t, uint64_t> guest_tick_ratio();
  // Guest ticks-per-second.
  static uint64_t guest_tick_frequency();
  // Sets the guest ticks-per-second.
  static void set_guest_tick_frequency(uint64_t frequency);

  // Queries the current guest tick count, accounting for frequency adjustment
  // and scaling.
  static uint64_t QueryGuestTickCount();

  static uint64_t* GetGuestTickCountPointer();
  // Queries the guest time, in FILETIME format, accounting for scaling.
  static uint64_t QueryGuestSystemTime();
  // Queries the milliseconds since the guest began, accounting for scaling.
  static uint32_t QueryGuestUptimeMillis();

  static uint64_t QueryGuestInterruptTime();

  // Sets the system time of the guest.
  static void SetGuestSystemTime(uint64_t system_time);

  // Freezes or resumes guest tick/system time progression via an atomic
  // reference count so nested holds (pause + pipeline compilation waits) never
  // unpause the emulator prematurely.
  static void HoldGuestTime(bool hold);
  static int32_t GuestTimeHoldCount();

  class ScopedGuestTimeHold {
   public:
    explicit ScopedGuestTimeHold(bool active = true) : active_(active) {
      if (active_) {
        Clock::HoldGuestTime(true);
      }
    }
    ~ScopedGuestTimeHold() {
      if (active_) {
        Clock::HoldGuestTime(false);
      }
    }
    ScopedGuestTimeHold(const ScopedGuestTimeHold&) = delete;
    ScopedGuestTimeHold& operator=(const ScopedGuestTimeHold&) = delete;

   private:
    bool active_;
  };

  // Scales a time duration in milliseconds, from guest time.
  static uint32_t ScaleGuestDurationMillis(uint32_t guest_ms);
  // Scales a time duration in 100ns ticks like FILETIME, from guest time.
  static int64_t ScaleGuestDurationFileTime(int64_t guest_file_time);
  // Scales a time duration represented as a timeval, from guest time.
  static void ScaleGuestDurationTimeval(int32_t* tv_sec, int32_t* tv_usec);
};

}  // namespace xe

#endif  // !defined(XENIA_BASE_CLOCK_PURE_LOGIC_ONLY)

#endif  // XENIA_BASE_CLOCK_H_
