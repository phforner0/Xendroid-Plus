#pragma once
// Native PadScript player and live impulse injector for automated QA & benchmarks.
// Uses lock-free atomics for input-driver sampling in AndroidInputDriver::GetState().

#include <cstdint>
#include <filesystem>
#include <string_view>

#include "xenia/base/cvar.h"

DECLARE_string(pad_script);
DECLARE_string(pad_record);

namespace xendroid {

void PadScriptInstallCallbacks();
void PadScriptStart(const std::filesystem::path& storage_root,
                    std::string_view spec);
void PadScriptApplyLive(std::string_view spec);
void PadScriptShutdown();

void PadRecordStart(const std::filesystem::path& storage_root,
                    std::string_view spec);
void PadRecordStop();
void PadRecordObserve(uint16_t buttons, int16_t thumb_lx, int16_t thumb_ly,
                      int16_t thumb_rx, int16_t thumb_ry,
                      uint8_t left_trigger, uint8_t right_trigger);

// Lock-free snapshot read on every XInputGetState(0) poll.
// `axes` order matches xe::hid::android: LX, LY, RX, RY, LT, RT in [-1..1] / [0..1].
void PadScriptSnapshot(bool& active_out, uint16_t& buttons_out,
                       float axes_out[6]);

}  // namespace xendroid
