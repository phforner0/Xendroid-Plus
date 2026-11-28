/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_BASE_ANDROID_THERMAL_H_
#define XENIA_BASE_ANDROID_THERMAL_H_

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>

#if defined(__ANDROID__) && !defined(XE_DISABLE_ANDROID_DLOPEN)
#include <dlfcn.h>
#include <sys/system_properties.h>
#endif

namespace xe {

// ADEVICE_THERMAL_STATUS_* from <android/thermal.h> (API 30+).
// Resolved dynamically via dlsym so builds targeting lower API levels and
// non-Android hosts compile cleanly without linking libandroid directly.
inline constexpr int kThermalStatusUnavailable = -1;
inline constexpr int kThermalStatusNone = 0;
inline constexpr int kThermalStatusLight = 1;
inline constexpr int kThermalStatusModerate = 2;
inline constexpr int kThermalStatusSevere = 3;
inline constexpr int kThermalStatusCritical = 4;
inline constexpr int kThermalStatusEmergency = 5;
inline constexpr int kThermalStatusShutdown = 6;

// Android 16 (API 36+) SystemHealthManager compute headroom TTL (ms).
inline constexpr uint64_t kComputeHeadroomTtlMs = 10000;
inline constexpr float kComputeHeadroomMin = 0.0f;
inline constexpr float kComputeHeadroomMax = 100.0f;

struct ThermalSample {
  int status = kThermalStatusUnavailable;
  float headroom = -1.0f;
  bool has_status = false;
  bool has_headroom = false;
  // Android 16 (API 36+) SystemHealthManager CPU/GPU headroom in [0, 100]
  // (inverted scale: 0 = no margin left, 100 = full margin available).
  float cpu_headroom = -1.0f;
  float gpu_headroom = -1.0f;
  bool has_cpu_headroom = false;
  bool has_gpu_headroom = false;
};

namespace internal {

struct AndroidComputeHeadroomState {
  std::atomic<uint64_t> seq{0};
  std::atomic<float> cpu_headroom{-1.0f};
  std::atomic<float> gpu_headroom{-1.0f};
  std::atomic<uint64_t> sample_time_ms{0};
  std::atomic<bool> has_sample{false};
  std::atomic<bool> enabled{false};
};

inline AndroidComputeHeadroomState& ComputeHeadroomState() {
  static AndroidComputeHeadroomState state;
  return state;
}

inline bool IsValidComputeHeadroom(float value) {
  return std::isfinite(value) && value >= kComputeHeadroomMin &&
         value <= kComputeHeadroomMax;
}

}  // namespace internal

inline uint64_t AndroidThermalMonotonicMs() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

inline void SetAndroidComputeHeadroomEnabled(bool enabled) {
  internal::ComputeHeadroomState().enabled.store(enabled,
                                                 std::memory_order_release);
}

inline bool IsAndroidComputeHeadroomEnabled() {
  return internal::ComputeHeadroomState().enabled.load(
      std::memory_order_acquire);
}

inline void SetAndroidComputeHeadroom(float cpu_headroom, float gpu_headroom,
                                      uint64_t now_ms) {
  auto& state = internal::ComputeHeadroomState();
  const bool valid_cpu = internal::IsValidComputeHeadroom(cpu_headroom);
  const bool valid_gpu = internal::IsValidComputeHeadroom(gpu_headroom);
  const uint64_t s = state.seq.load(std::memory_order_relaxed);
  state.seq.store(s + 1u, std::memory_order_release);
  state.cpu_headroom.store(valid_cpu ? cpu_headroom : -1.0f,
                           std::memory_order_relaxed);
  state.gpu_headroom.store(valid_gpu ? gpu_headroom : -1.0f,
                           std::memory_order_relaxed);
  state.sample_time_ms.store(now_ms, std::memory_order_relaxed);
  state.has_sample.store(valid_cpu || valid_gpu, std::memory_order_relaxed);
  state.seq.store(s + 2u, std::memory_order_release);
}

inline void ResetAndroidComputeHeadroom() {
  auto& state = internal::ComputeHeadroomState();
  const uint64_t s = state.seq.load(std::memory_order_relaxed);
  state.seq.store(s + 1u, std::memory_order_release);
  state.cpu_headroom.store(-1.0f, std::memory_order_relaxed);
  state.gpu_headroom.store(-1.0f, std::memory_order_relaxed);
  state.sample_time_ms.store(0, std::memory_order_relaxed);
  state.has_sample.store(false, std::memory_order_relaxed);
  state.seq.store(s + 2u, std::memory_order_release);
  state.enabled.store(false, std::memory_order_release);
}

inline void PopulateAndroidComputeHeadroom(ThermalSample& sample,
                                           uint64_t now_ms) {
  sample.cpu_headroom = -1.0f;
  sample.gpu_headroom = -1.0f;
  sample.has_cpu_headroom = false;
  sample.has_gpu_headroom = false;

  auto& state = internal::ComputeHeadroomState();
  for (int attempt = 0; attempt < 3; ++attempt) {
    const uint64_t s0 = state.seq.load(std::memory_order_acquire);
    if (s0 & 1u) {
      continue;
    }
    const bool has_sample = state.has_sample.load(std::memory_order_relaxed);
    const uint64_t ts_ms = state.sample_time_ms.load(std::memory_order_relaxed);
    const float cpu = state.cpu_headroom.load(std::memory_order_relaxed);
    const float gpu = state.gpu_headroom.load(std::memory_order_relaxed);
    const uint64_t s1 = state.seq.load(std::memory_order_acquire);
    if (s0 != s1) {
      continue;
    }
    if (!has_sample || now_ms < ts_ms ||
        (now_ms - ts_ms) > kComputeHeadroomTtlMs) {
      return;
    }
    if (internal::IsValidComputeHeadroom(cpu)) {
      sample.cpu_headroom = cpu;
      sample.has_cpu_headroom = true;
    }
    if (internal::IsValidComputeHeadroom(gpu)) {
      sample.gpu_headroom = gpu;
      sample.has_gpu_headroom = true;
    }
    return;
  }
}

inline void PollAndroidAutoVrsHeadroomProperty() {
#if defined(__ANDROID__) && !defined(XE_DISABLE_ANDROID_DLOPEN)
  char prop[PROP_VALUE_MAX] = {};
  if (__system_property_get("debug.xendroid.auto_vrs_headroom", prop) > 0 &&
      (prop[0] == '0' || prop[0] == '1') && prop[1] == '\0') {
    SetAndroidComputeHeadroomEnabled(prop[0] == '1');
  }
#endif
}

#if defined(__ANDROID__) && !defined(XE_DISABLE_ANDROID_DLOPEN)

inline void* AndroidThermalLibHandle() {
  static void* const lib = dlopen("libandroid.so", RTLD_NOW | RTLD_LOCAL);
  return lib;
}

inline void* AcquireAndroidThermalManager() {
  using PFN_AThermal_acquireManager = void* (*)();
  static void* const manager = []() -> void* {
    void* const lib = AndroidThermalLibHandle();
    if (!lib) {
      return nullptr;
    }
    auto acquire = reinterpret_cast<PFN_AThermal_acquireManager>(
        dlsym(lib, "AThermal_acquireManager"));
    return acquire ? acquire() : nullptr;
  }();
  return manager;
}

inline int ReadAndroidThermalStatus() {
  void* const manager = AcquireAndroidThermalManager();
  if (!manager) {
    return kThermalStatusUnavailable;
  }
  using PFN_AThermal_getCurrentThermalStatus = int (*)(void*);
  static const PFN_AThermal_getCurrentThermalStatus get_status =
      []() -> PFN_AThermal_getCurrentThermalStatus {
    void* const lib = AndroidThermalLibHandle();
    return lib ? reinterpret_cast<PFN_AThermal_getCurrentThermalStatus>(
                     dlsym(lib, "AThermal_getCurrentThermalStatus"))
               : nullptr;
  }();
  if (!get_status) {
    return kThermalStatusUnavailable;
  }
  const int status = get_status(manager);
  return status >= kThermalStatusNone ? status : kThermalStatusUnavailable;
}

inline float ReadAndroidThermalHeadroom(int32_t forecast_seconds = 10) {
  void* const manager = AcquireAndroidThermalManager();
  if (!manager) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  using PFN_AThermal_getThermalHeadroom = float (*)(void*, int);
  static const PFN_AThermal_getThermalHeadroom get_headroom =
      []() -> PFN_AThermal_getThermalHeadroom {
    void* const lib = AndroidThermalLibHandle();
    return lib ? reinterpret_cast<PFN_AThermal_getThermalHeadroom>(
                     dlsym(lib, "AThermal_getThermalHeadroom"))
               : nullptr;
  }();
  if (!get_headroom) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  return get_headroom(manager, static_cast<int>(forecast_seconds));
}

#else

inline void* AcquireAndroidThermalManager() { return nullptr; }

inline int ReadAndroidThermalStatus() { return kThermalStatusUnavailable; }

inline float ReadAndroidThermalHeadroom(int32_t forecast_seconds = 10) {
  (void)forecast_seconds;
  return std::numeric_limits<float>::quiet_NaN();
}

#endif

inline ThermalSample ReadAndroidThermalSampleAt(int32_t forecast_seconds,
                                                uint64_t now_ms) {
  PollAndroidAutoVrsHeadroomProperty();
  ThermalSample sample;
  const int status = ReadAndroidThermalStatus();
  sample.has_status = status >= kThermalStatusNone;
  sample.status = sample.has_status ? status : kThermalStatusUnavailable;
  const float headroom = ReadAndroidThermalHeadroom(forecast_seconds);
  sample.has_headroom = std::isfinite(headroom) && headroom >= 0.0f;
  sample.headroom = sample.has_headroom ? headroom : -1.0f;
  PopulateAndroidComputeHeadroom(sample, now_ms);
  return sample;
}

inline ThermalSample ReadAndroidThermalSample(int32_t forecast_seconds = 10) {
  return ReadAndroidThermalSampleAt(forecast_seconds,
                                    AndroidThermalMonotonicMs());
}

}  // namespace xe

#endif  // XENIA_BASE_ANDROID_THERMAL_H_
