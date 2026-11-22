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

#include <cmath>
#include <cstdint>
#include <limits>

#if defined(__ANDROID__) && !defined(XE_DISABLE_ANDROID_DLOPEN)
#include <dlfcn.h>
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

struct ThermalSample {
  int status = kThermalStatusUnavailable;
  float headroom = -1.0f;
  bool has_status = false;
  bool has_headroom = false;
};

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

inline ThermalSample ReadAndroidThermalSample(int32_t forecast_seconds = 10) {
  ThermalSample sample;
  const int status = ReadAndroidThermalStatus();
  sample.has_status = status >= kThermalStatusNone;
  sample.status = sample.has_status ? status : kThermalStatusUnavailable;
  const float headroom = ReadAndroidThermalHeadroom(forecast_seconds);
  sample.has_headroom = std::isfinite(headroom) && headroom >= 0.0f;
  sample.headroom = sample.has_headroom ? headroom : -1.0f;
  return sample;
}

}  // namespace xe

#endif  // XENIA_BASE_ANDROID_THERMAL_H_
