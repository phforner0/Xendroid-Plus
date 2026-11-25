#include "xe_pad_script.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "xe_pad_script_parser.h"
#include "xenia/base/cvar.h"
#include "xenia/base/frame_stats.h"
#include "xenia/base/logging.h"

DEFINE_string(pad_script, "",
              "Relative path under storage_root to a .pad script replayed over "
              "XInput slot 0 (empty = disabled).",
              "HID");
DEFINE_string(pad_record, "",
              "Relative path under storage_root to record XInput slot 0 "
              "inputs as a swap-clocked .pad script (empty = disabled).",
              "HID");

namespace xendroid {
namespace {

using SteadyClock = std::chrono::steady_clock;

// Layer 1: Script-driven state (written only by the script runner thread).
std::atomic<bool> g_script_active{false};
std::atomic<uint16_t> g_script_buttons{0};
std::atomic<int32_t> g_script_axes_milli[6] = {};

// Layer 2: Live property impulse overlay (never wiped by script or vice versa).
std::atomic<bool> g_live_active{false};
std::atomic<uint16_t> g_live_buttons{0};
std::atomic<int32_t> g_live_axes_milli[6] = {};
std::atomic<uint8_t> g_live_axis_mask{0};
std::mutex g_live_mutex;
PadLiveButtonRefAccumulator g_live_accum_buttons;
float g_live_accum_axes[6] = {};
uint8_t g_live_accum_axis_refs[6] = {};

std::mutex g_control_mutex;
std::thread g_player_thread;
std::atomic<bool> g_player_stop{false};
std::atomic<bool> g_live_pulse_latched{false};
std::atomic<uint64_t> g_live_generation{1};
std::string g_last_spec;
std::string g_last_live_pulse;
std::filesystem::path g_storage_root;

// Layer 3: Asynchronous input recorder (observed on XInputGetState(0)).
struct PadRecordEvent {
  uint64_t tick = 0;
  PadRecordSample sample;
};

std::atomic<bool> g_record_active{false};
std::atomic<bool> g_record_stop{false};
std::mutex g_rec_buf_mutex;
std::vector<PadRecordEvent> g_rec_queue;
bool g_rec_has_last = false;
PadRecordSample g_rec_last_sample;
std::thread g_rec_thread;
std::string g_last_rec_spec;
std::string g_last_rec_prop;

void PublishScriptState(bool active, uint16_t buttons, const float axes[6]) {
  for (int i = 0; i < 6; ++i) {
    g_script_axes_milli[i].store(
        int32_t(std::clamp(axes[i], -1.0f, 1.0f) * 1000.0f),
        std::memory_order_relaxed);
  }
  g_script_buttons.store(buttons, std::memory_order_relaxed);
  g_script_active.store(active, std::memory_order_release);
}

void ClearScriptState() {
  const float zeros[6] = {};
  PublishScriptState(false, 0, zeros);
}

void PublishLiveLocked() {
  uint8_t mask = 0;
  for (int i = 0; i < 6; ++i) {
    const bool has_ax = g_live_accum_axis_refs[i] > 0;
    if (has_ax) mask |= uint8_t(1u << i);
    const float v = has_ax ? g_live_accum_axes[i] : 0.0f;
    g_live_axes_milli[i].store(int32_t(std::clamp(v, -1.0f, 1.0f) * 1000.0f),
                               std::memory_order_relaxed);
  }
  const uint16_t buttons_mask = g_live_accum_buttons.Mask();
  g_live_axis_mask.store(mask, std::memory_order_relaxed);
  g_live_buttons.store(buttons_mask, std::memory_order_relaxed);
  g_live_active.store(buttons_mask != 0 || mask != 0,
                      std::memory_order_release);
}

void ClearLiveState() {
  std::lock_guard<std::mutex> lk(g_live_mutex);
  g_live_accum_buttons.Clear();
  for (int i = 0; i < 6; ++i) {
    g_live_accum_axes[i] = 0.0f;
    g_live_accum_axis_refs[i] = 0;
  }
  PublishLiveLocked();
}

struct ScriptPadAccumulator {
  uint16_t buttons = 0;
  float axes[6] = {};
  uint8_t axis_mask = 0;

  void Press(const PadSpecInput& in) {
    buttons |= in.buttons;
    for (const auto& [ax, val] : in.axes) {
      axes[size_t(ax)] = val;
      axis_mask |= uint8_t(1u << size_t(ax));
    }
    PublishScriptState(true, buttons, axes);
  }

  void Release(const PadSpecInput& in) {
    buttons &= ~in.buttons;
    for (const auto& [ax, val] : in.axes) {
      (void)val;
      axes[size_t(ax)] = 0.0f;
      axis_mask &= uint8_t(~(1u << size_t(ax)));
    }
    PublishScriptState(buttons != 0 || axis_mask != 0, buttons, axes);
  }

  void Set(uint16_t new_buttons, const float new_axes[6]) {
    buttons = new_buttons;
    axis_mask = 0;
    for (int i = 0; i < 6; ++i) {
      axes[i] = new_axes[i];
      if (std::fabs(new_axes[i]) > 0.005f) {
        axis_mask |= uint8_t(1u << i);
      }
    }
    PublishScriptState(true, buttons, axes);
  }

  void ReleaseAll() {
    buttons = 0;
    axis_mask = 0;
    for (int i = 0; i < 6; ++i) axes[i] = 0.0f;
    ClearScriptState();
  }
};

uint64_t CurrentTick() {
  const uint64_t swaps =
      xe::internal::source_issue_total().load(std::memory_order_acquire);
  return swaps ? swaps : uint64_t(xe::logging::GetFrameNumber());
}

void RunScriptThread(std::vector<PadScriptCommand> commands) {
  ScriptPadAccumulator pad;
  constexpr auto kForever = std::chrono::hours(24);
  auto wait_cond = [](auto&& pred, SteadyClock::duration timeout) -> const char* {
    const auto end = SteadyClock::now() + timeout;
    while (!pred()) {
      if (g_player_stop.load(std::memory_order_relaxed)) return "stopped";
      if (SteadyClock::now() >= end) return "timeout";
      std::this_thread::sleep_for(std::chrono::milliseconds(4));
    }
    return nullptr;
  };

  uint64_t base_tick = CurrentTick();
  auto wait_ticks = [&](uint64_t ticks) {
    base_tick += ticks;
    return wait_cond([&] { return CurrentTick() >= base_tick; }, kForever);
  };

  std::vector<uint64_t> loop_counts;
  const char* abort_reason = nullptr;
  size_t steps = 0;
  size_t pc = 0;
  for (; pc < commands.size(); ++pc) {
    if (g_player_stop.load(std::memory_order_relaxed)) {
      abort_reason = "stopped";
      break;
    }
    const PadScriptCommand& c = commands[pc];
    ++steps;
    switch (c.op) {
      case PadScriptOp::kSet:
        if (!(abort_reason = wait_ticks(c.n))) {
          pad.Set(c.input.buttons, c.values);
        }
        break;
      case PadScriptOp::kIdle:
        abort_reason = wait_ticks(c.n);
        break;
      case PadScriptOp::kPress:
        pad.Press(c.input);
        if (c.ms) {
          const auto end =
              SteadyClock::now() + std::chrono::milliseconds(c.n);
          abort_reason = wait_cond(
              [&] { return SteadyClock::now() >= end; }, kForever);
          base_tick = CurrentTick();
        } else {
          base_tick = std::max(base_tick, CurrentTick());
          abort_reason = wait_ticks(c.n);
        }
        pad.Release(c.input);
        break;
      case PadScriptOp::kHold:
        pad.Press(c.input);
        break;
      case PadScriptOp::kRelease:
        pad.Release(c.input);
        break;
      case PadScriptOp::kReleaseAll:
        pad.ReleaseAll();
        break;
      case PadScriptOp::kWaitTitle:
      case PadScriptOp::kWaitFramesAtLeast:
      case PadScriptOp::kWaitDrawsAtLeast:
      case PadScriptOp::kWaitDrawsBelow: {
        abort_reason = wait_cond(
            [&] {
              if (c.op == PadScriptOp::kWaitTitle) {
                return xe::internal::active_title_id_atomic().load(
                           std::memory_order_relaxed) == uint32_t(c.n);
              }
              if (c.op == PadScriptOp::kWaitFramesAtLeast) {
                return CurrentTick() >= c.n;
              }
              const uint32_t draws =
                  xe::internal::last_frame_draw_count().load(
                      std::memory_order_relaxed);
              return c.op == PadScriptOp::kWaitDrawsAtLeast ? (draws >= c.n)
                                                            : (draws < c.n);
            },
            std::chrono::seconds(c.timeout_s));
        if (!abort_reason) {
          base_tick = CurrentTick();
          XELOGI("XDROID-PAD wait-done line={} what={} tick={}", c.line,
                 c.text, base_tick);
        }
        break;
      }
      case PadScriptOp::kLoop:
        loop_counts.push_back(c.n);
        break;
      case PadScriptOp::kEndLoop:
        if (--loop_counts.back() > 0) {
          pc = size_t(c.n);
        } else {
          loop_counts.pop_back();
        }
        break;
      case PadScriptOp::kMark:
        XELOGI("XDROID-PAD mark {} tick={}", c.text, CurrentTick());
        break;
      case PadScriptOp::kEnd:
        pc = commands.size();
        break;
    }
    if (abort_reason) break;
  }

  pad.ReleaseAll();
  if (abort_reason) {
    const int line = pc < commands.size() ? commands[pc].line : 0;
    const std::string what =
        pc < commands.size() ? commands[pc].text : std::string();
    XELOGW("XDROID-PAD abort line={} reason={} what={}", line, abort_reason,
           what);
  } else {
    XELOGI("XDROID-PAD done steps={} ticks={}", steps, CurrentTick());
  }
}

void LivePropCallback(const char* value) {
  if (!value || !value[0]) {
    if (g_live_pulse_latched.load(std::memory_order_acquire)) {
      std::lock_guard<std::mutex> lock(g_control_mutex);
      g_last_live_pulse.clear();
      g_live_pulse_latched.store(false, std::memory_order_release);
    }
    return;
  }
  PadScriptApplyLive(value);
}

void RunRecordThread(std::filesystem::path out_file) {
  std::error_code ec;
  if (out_file.has_parent_path()) {
    std::filesystem::create_directories(out_file.parent_path(), ec);
  }
  std::ofstream out(out_file, std::ios::out | std::ios::trunc);
  if (!out.is_open()) {
    XELOGW("XDROID-PAD record abort reason=open_failed file={}",
           out_file.string());
    g_record_active.store(false, std::memory_order_release);
    return;
  }
  out << "# Recorded by XenDroid pad_record\n";

  uint32_t title_id =
      xe::internal::active_title_id_atomic().load(std::memory_order_relaxed);
  bool wrote_wait_title = false;
  if (title_id != 0) {
    char tbuf[32] = {};
    std::snprintf(tbuf, sizeof(tbuf), "wait title %08X 60\n", title_id);
    out << tbuf;
    wrote_wait_title = true;
  }

  uint64_t last_tick = CurrentTick();
  uint64_t event_count = 0;
  auto last_flush = SteadyClock::now();
  std::vector<PadRecordEvent> batch;
  batch.reserve(64);

  auto flush_batch = [&]() {
    batch.clear();
    {
      std::lock_guard<std::mutex> lk(g_rec_buf_mutex);
      batch.swap(g_rec_queue);
    }
    if (!wrote_wait_title) {
      const uint32_t tid = xe::internal::active_title_id_atomic().load(
          std::memory_order_relaxed);
      if (tid != 0) {
        char tbuf[32] = {};
        std::snprintf(tbuf, sizeof(tbuf), "wait title %08X 60\n", tid);
        out << tbuf;
        wrote_wait_title = true;
      }
    }
    for (const auto& ev : batch) {
      const uint64_t delta = ev.tick >= last_tick ? ev.tick - last_tick : 0;
      if (ev.tick > last_tick) last_tick = ev.tick;
      out << FormatPadSetLine(delta, ev.sample) << '\n';
      ++event_count;
    }
  };

  while (!g_record_stop.load(std::memory_order_acquire)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(16));
    flush_batch();
    const auto now = SteadyClock::now();
    if (now - last_flush >= std::chrono::milliseconds(200)) {
      out.flush();
      last_flush = now;
    }
  }
  flush_batch();
  out << "release all\nend\n";
  out.flush();
  XELOGI("XDROID-PAD record stop file={} events={} ticks={}",
         out_file.string(), event_count, last_tick);
}

void StopRecordLocked() {
  g_record_active.store(false, std::memory_order_release);
  if (g_rec_thread.joinable()) {
    g_record_stop.store(true, std::memory_order_release);
    g_rec_thread.join();
  }
  {
    std::lock_guard<std::mutex> lk(g_rec_buf_mutex);
    g_rec_queue.clear();
    g_rec_has_last = false;
  }
  g_last_rec_spec.clear();
}

void StartRecordLocked(const std::filesystem::path& storage_root,
                       std::string_view spec) {
  g_storage_root = storage_root;
  if (spec.empty() || spec == g_last_rec_spec) {
    return;
  }
  std::filesystem::path full;
  if (!pad_detail::ResolvePadScriptPathUnderRoot(storage_root, spec, full)) {
    XELOGW("XDROID-PAD record abort reason=unsafe_path spec={}", spec);
    return;
  }
  StopRecordLocked();
  g_last_rec_spec = std::string(spec);
  g_record_stop.store(false, std::memory_order_release);
  g_record_active.store(true, std::memory_order_release);
  XELOGI("XDROID-PAD record start file={}", spec);
  g_rec_thread = std::thread(RunRecordThread, std::move(full));
}

void ScriptPropCallback(const char* value) {
  if (!value || !value[0]) return;
  std::filesystem::path root;
  {
    std::lock_guard<std::mutex> lock(g_control_mutex);
    root = g_storage_root;
  }
  if (!root.empty()) {
    PadScriptStart(root, value);
  }
}

void RecordPropCallback(const char* raw) {
  const std::string_view spec(raw ? raw : "");
  std::lock_guard<std::mutex> lock(g_control_mutex);
  if (spec == "__stop__") {
    g_last_rec_prop = "__stop__";
    StopRecordLocked();
    return;
  }
  if (spec.empty()) {
    if (!g_last_rec_prop.empty() && g_last_rec_prop != "__stop__") {
      StopRecordLocked();
    }
    g_last_rec_prop.clear();
    return;
  }
  if (spec == g_last_rec_prop || g_storage_root.empty()) {
    return;
  }
  g_last_rec_prop = std::string(spec);
  StartRecordLocked(g_storage_root, spec);
}

}  // namespace

void PadScriptInstallCallbacks() {
  xe::internal::pad_live_callback().store(&LivePropCallback,
                                          std::memory_order_release);
  xe::internal::pad_script_live_callback().store(&ScriptPropCallback,
                                                 std::memory_order_release);
  xe::internal::pad_record_live_callback().store(&RecordPropCallback,
                                                 std::memory_order_release);
}

void PadScriptStart(const std::filesystem::path& storage_root,
                    std::string_view spec) {
  std::lock_guard<std::mutex> lock(g_control_mutex);
  g_storage_root = storage_root;
  if (spec.empty() || spec == g_last_spec) {
    return;
  }
  g_last_spec = std::string(spec);
  if (g_player_thread.joinable()) {
    g_player_stop.store(true, std::memory_order_relaxed);
    g_player_thread.join();
  }
  ClearScriptState();
  if (spec == "stop") {
    return;
  }

  std::filesystem::path full;
  if (!pad_detail::ResolvePadScriptPathUnderRoot(storage_root, spec, full)) {
    XELOGW("XDROID-PAD abort line=0 reason=path what={}", spec);
    return;
  }
  std::error_code ec;
  if (!std::filesystem::is_regular_file(full, ec)) {
    XELOGW("XDROID-PAD abort line=0 reason=open what={} file={}", spec,
           full.string());
    return;
  }
  std::ifstream file(full);
  if (!file) {
    XELOGW("XDROID-PAD abort line=0 reason=open what={} file={}", spec,
           full.string());
    return;
  }
  std::ostringstream ss;
  ss << file.rdbuf();
  std::vector<PadScriptCommand> cmds;
  PadScriptCommand err;
  if (!ParsePadScriptText(ss.str(), cmds, err)) {
    XELOGW("XDROID-PAD abort line={} reason=parse what={} file={}", err.line,
           err.text, spec);
    return;
  }
  XELOGI("XDROID-PAD start file={} lines={}", spec, cmds.size());
  g_player_stop.store(false, std::memory_order_relaxed);
  g_player_thread = std::thread(RunScriptThread, std::move(cmds));
}

void PadScriptApplyLive(std::string_view spec) {
  if (spec.empty()) {
    std::lock_guard<std::mutex> lock(g_control_mutex);
    g_last_live_pulse.clear();
    g_live_pulse_latched.store(false, std::memory_order_release);
    return;
  }
  {
    std::lock_guard<std::mutex> lock(g_control_mutex);
    if (spec == g_last_live_pulse) {
      return;
    }
    g_last_live_pulse = std::string(spec);
    g_live_pulse_latched.store(true, std::memory_order_release);
  }
  std::string_view raw = StripPadLiveNonce(spec);
  uint32_t hold_ms = 120;
  if (size_t colon = raw.find(':'); colon != std::string_view::npos) {
    hold_ms = std::clamp<uint32_t>(
        uint32_t(std::strtoul(std::string(raw.substr(colon + 1)).c_str(),
                              nullptr, 10)),
        1, 10000);
    raw = raw.substr(0, colon);
  }
  PadSpecInput in;
  if (!ParsePadSpecTokens(raw, in).empty() ||
      (!in.buttons && in.axes.empty())) {
    XELOGW("XDROID-PAD live unknown input '{}'", spec);
    return;
  }
  const uint64_t gen = g_live_generation.load(std::memory_order_acquire);
  XELOGI("XDROID-PAD live {} ({} ms)", spec, hold_ms);
  std::thread([in = std::move(in), hold_ms, gen]() {
    {
      std::lock_guard<std::mutex> lk(g_live_mutex);
      if (g_live_generation.load(std::memory_order_acquire) != gen) return;
      g_live_accum_buttons.Add(in.buttons);
      for (const auto& [ax, val] : in.axes) {
        g_live_accum_axes[size_t(ax)] = val;
        ++g_live_accum_axis_refs[size_t(ax)];
      }
      PublishLiveLocked();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(hold_ms));
    {
      std::lock_guard<std::mutex> lk(g_live_mutex);
      if (g_live_generation.load(std::memory_order_acquire) != gen) return;
      g_live_accum_buttons.Remove(in.buttons);
      for (const auto& [ax, val] : in.axes) {
        (void)val;
        if (g_live_accum_axis_refs[size_t(ax)] > 0) {
          --g_live_accum_axis_refs[size_t(ax)];
        }
      }
      PublishLiveLocked();
    }
  }).detach();
}

void PadScriptShutdown() {
  xe::internal::pad_live_callback().store(nullptr, std::memory_order_release);
  xe::internal::pad_script_live_callback().store(nullptr,
                                                 std::memory_order_release);
  xe::internal::pad_record_live_callback().store(nullptr,
                                                 std::memory_order_release);
  g_live_generation.fetch_add(1, std::memory_order_acq_rel);
  std::lock_guard<std::mutex> lock(g_control_mutex);
  StopRecordLocked();
  g_last_rec_prop.clear();
  if (g_player_thread.joinable()) {
    g_player_stop.store(true, std::memory_order_relaxed);
    g_player_thread.join();
  }
  g_last_spec.clear();
  g_last_live_pulse.clear();
  g_live_pulse_latched.store(false, std::memory_order_release);
  ClearScriptState();
  ClearLiveState();
}

void PadRecordStart(const std::filesystem::path& storage_root,
                    std::string_view spec) {
  std::lock_guard<std::mutex> lock(g_control_mutex);
  StartRecordLocked(storage_root, spec);
}

void PadRecordStop() {
  std::lock_guard<std::mutex> lock(g_control_mutex);
  StopRecordLocked();
}

void PadRecordObserve(uint16_t buttons, int16_t thumb_lx, int16_t thumb_ly,
                      int16_t thumb_rx, int16_t thumb_ry,
                      uint8_t left_trigger, uint8_t right_trigger) {
  if (!g_record_active.load(std::memory_order_acquire)) {
    return;
  }
  const PadRecordSample sample = QuantizePadSample(
      buttons, thumb_lx, thumb_ly, thumb_rx, thumb_ry, left_trigger,
      right_trigger);
  const uint64_t tick = CurrentTick();
  std::lock_guard<std::mutex> lk(g_rec_buf_mutex);
  if (!g_record_active.load(std::memory_order_relaxed)) {
    return;
  }
  if (g_rec_has_last && sample == g_rec_last_sample) {
    return;
  }
  g_rec_has_last = true;
  g_rec_last_sample = sample;
  g_rec_queue.push_back({tick, sample});
}

void PadScriptSnapshot(bool& active_out, uint16_t& buttons_out,
                       float axes_out[6]) {
  const bool script_on = g_script_active.load(std::memory_order_acquire);
  const bool live_on = g_live_active.load(std::memory_order_acquire);
  active_out = script_on || live_on;
  if (!active_out) {
    buttons_out = 0;
    for (int i = 0; i < 6; ++i) axes_out[i] = 0.0f;
    return;
  }
  buttons_out = 0;
  for (int i = 0; i < 6; ++i) axes_out[i] = 0.0f;
  if (script_on) {
    buttons_out |= g_script_buttons.load(std::memory_order_relaxed);
    for (int i = 0; i < 6; ++i) {
      axes_out[i] =
          float(g_script_axes_milli[i].load(std::memory_order_relaxed)) /
          1000.0f;
    }
  }
  if (live_on) {
    buttons_out |= g_live_buttons.load(std::memory_order_relaxed);
    const uint8_t mask = g_live_axis_mask.load(std::memory_order_relaxed);
    for (int i = 0; i < 6; ++i) {
      if (mask & (1u << i)) {
        axes_out[i] =
            float(g_live_axes_milli[i].load(std::memory_order_relaxed)) /
            1000.0f;
      }
    }
  }
}

}  // namespace xendroid
