#pragma once
// Pure C++20 PadScript parser & state representation (no Android/Xenia deps).
// Supports swap-clocked steps (`S`), button/axis specs (`A+DPAD_DOWN:150ms`),
// `hold`, `release`, `idle`, `wait title/frames>=/draws>=/draws<`, `loop`/`endloop`,
// `mark`, and `end`.

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace xendroid {

enum class PadAxis : uint8_t {
  kLeftX = 0,
  kLeftY = 1,
  kRightX = 2,
  kRightY = 3,
  kLeftTrigger = 4,
  kRightTrigger = 5,
};

struct PadNameMask {
  std::string_view name;
  uint16_t mask;
};

inline constexpr PadNameMask kPadButtonTable[] = {
    {"DPAD_UP", 0x0001},    {"DPAD_DOWN", 0x0002}, {"DPAD_LEFT", 0x0004},
    {"DPAD_RIGHT", 0x0008}, {"START", 0x0010},     {"BACK", 0x0020},
    {"LTHUMB", 0x0040},     {"RTHUMB", 0x0080},    {"LB", 0x0100},
    {"RB", 0x0200},         {"GUIDE", 0x0400},     {"A", 0x1000},
    {"B", 0x2000},          {"X", 0x4000},         {"Y", 0x8000},
};

struct PadAxisDef {
  std::string_view name;
  PadAxis axis;
};

inline constexpr PadAxisDef kPadAxisTable[] = {
    {"LX", PadAxis::kLeftX},       {"LY", PadAxis::kLeftY},
    {"RX", PadAxis::kRightX},      {"RY", PadAxis::kRightY},
    {"LT", PadAxis::kLeftTrigger}, {"RT", PadAxis::kRightTrigger},
};

struct PadSpecInput {
  uint16_t buttons = 0;
  std::vector<std::pair<PadAxis, float>> axes;
};

inline std::string_view ParsePadSpecTokens(std::string_view spec,
                                           PadSpecInput& out) {
  std::string_view unknown;
  while (!spec.empty()) {
    const size_t plus = spec.find('+');
    const std::string_view tok =
        plus == std::string_view::npos ? spec : spec.substr(0, plus);
    spec = plus == std::string_view::npos ? std::string_view()
                                          : spec.substr(plus + 1);
    const size_t eq = tok.find('=');
    bool known = false;
    if (eq != std::string_view::npos) {
      const std::string_view key = tok.substr(0, eq);
      const std::string val_str(tok.substr(eq + 1));
      char* end = nullptr;
      const float val = std::strtof(val_str.c_str(), &end);
      if (!val_str.empty() && end && *end == '\0' && std::isfinite(val)) {
        for (const auto& ad : kPadAxisTable) {
          if (ad.name == key) {
            out.axes.emplace_back(ad.axis, val);
            known = true;
            break;
          }
        }
      }
    } else {
      for (const auto& bd : kPadButtonTable) {
        if (bd.name == tok) {
          out.buttons |= bd.mask;
          known = true;
          break;
        }
      }
    }
    if (!known && unknown.empty()) {
      unknown = tok;
    }
  }
  return unknown;
}

inline std::string_view StripPadLiveNonce(std::string_view spec) {
  const size_t hash = spec.find('#');
  return hash == std::string_view::npos ? spec : spec.substr(0, hash);
}

struct PadLiveButtonRefAccumulator {
  uint8_t refs[16] = {};

  uint16_t Add(uint16_t mask) {
    for (uint32_t bit = 0; bit < 16; ++bit) {
      if ((mask & uint16_t(1u << bit)) && refs[bit] < 255) {
        ++refs[bit];
      }
    }
    return Mask();
  }

  uint16_t Remove(uint16_t mask) {
    for (uint32_t bit = 0; bit < 16; ++bit) {
      if ((mask & uint16_t(1u << bit)) && refs[bit] > 0) {
        --refs[bit];
      }
    }
    return Mask();
  }

  uint16_t Mask() const {
    uint16_t out = 0;
    for (uint32_t bit = 0; bit < 16; ++bit) {
      if (refs[bit] > 0) {
        out |= uint16_t(1u << bit);
      }
    }
    return out;
  }

  void Clear() {
    for (uint32_t bit = 0; bit < 16; ++bit) {
      refs[bit] = 0;
    }
  }
};

enum class PadScriptOp : uint8_t {
  kSet,
  kPress,
  kHold,
  kRelease,
  kReleaseAll,
  kIdle,
  kWaitTitle,
  kWaitFramesAtLeast,
  kWaitDrawsAtLeast,
  kWaitDrawsBelow,
  kLoop,
  kEndLoop,
  kMark,
  kEnd,
};

struct PadScriptCommand {
  PadScriptOp op = PadScriptOp::kEnd;
  int line = 0;
  std::string text;
  PadSpecInput input;
  float values[6] = {};
  uint64_t n = 0;
  bool ms = false;
  uint32_t timeout_s = 120;
};

namespace pad_detail {
inline bool ParseUint(std::string_view text, uint64_t& out, int base = 10) {
  if (text.empty() || text.front() == '-') {
    return false;
  }
  const std::string s(text);
  char* end = nullptr;
  out = std::strtoull(s.c_str(), &end, base);
  return end && *end == '\0';
}

inline bool ParseFloat(std::string_view text, float& out) {
  if (text.empty()) {
    return false;
  }
  const std::string s(text);
  char* end = nullptr;
  out = std::strtof(s.c_str(), &end);
  return end && *end == '\0' && std::isfinite(out);
}

inline bool ResolvePadScriptPathUnderRoot(
    const std::filesystem::path& storage_root,
    std::string_view rel_spec,
    std::filesystem::path& out_path) {
  out_path.clear();
  if (storage_root.empty() || rel_spec.empty() ||
      rel_spec.find('\0') != std::string_view::npos) {
    return false;
  }
  const std::filesystem::path rel(rel_spec);
  if (rel.is_absolute() || rel.has_root_name() || rel.has_root_directory()) {
    return false;
  }
  for (const auto& part : rel) {
    if (part == "..") {
      return false;
    }
  }
  const std::filesystem::path norm_root = storage_root.lexically_normal();
  const std::filesystem::path candidate = (norm_root / rel).lexically_normal();
  auto it_r = norm_root.begin();
  auto it_c = candidate.begin();
  for (; it_r != norm_root.end() && it_c != candidate.end(); ++it_r, ++it_c) {
    if (*it_r != *it_c) {
      return false;
    }
  }
  if (it_r != norm_root.end()) {
    return false;
  }
  out_path = candidate;
  return true;
}
}  // namespace pad_detail

inline bool ParsePadScriptText(std::string_view script_text,
                               std::vector<PadScriptCommand>& commands,
                               PadScriptCommand& error_cmd) {
  commands.clear();
  std::vector<size_t> loop_stack;
  size_t pos = 0;
  int line_no = 0;
  while (pos <= script_text.size()) {
    ++line_no;
    size_t eol = script_text.find('\n', pos);
    if (eol == std::string_view::npos) {
      eol = script_text.size();
    }
    std::string_view line = script_text.substr(pos, eol - pos);
    pos = eol + 1;
    if (line_no == 1 && line.size() >= 3 && line[0] == '\xEF' &&
        line[1] == '\xBB' && line[2] == '\xBF') {
      line.remove_prefix(3);
    }
    if (size_t hash = line.find('#'); hash != std::string_view::npos) {
      line = line.substr(0, hash);
    }
    std::vector<std::string_view> t;
    while (!line.empty()) {
      const size_t first = line.find_first_not_of(" \t\r");
      if (first == std::string_view::npos) break;
      line.remove_prefix(first);
      const size_t sep = std::min(line.find_first_of(" \t\r"), line.size());
      t.push_back(line.substr(0, sep));
      line.remove_prefix(sep);
    }
    if (t.empty()) {
      if (pos > script_text.size()) break;
      continue;
    }

    PadScriptCommand c;
    c.line = line_no;
    c.text = std::string(t[0]);
    const std::string_view op = t[0];
    const size_t n = t.size();
    bool ok = false;

    if (op == "S") {
      uint64_t buttons = 0;
      if (n == 9 && pad_detail::ParseUint(t[1], c.n) &&
          pad_detail::ParseUint(t[2], buttons, 16)) {
        c.op = PadScriptOp::kSet;
        c.input.buttons = uint16_t(buttons);
        ok = true;
        for (int i = 0; i < 6; ++i) {
          ok &= pad_detail::ParseFloat(t[3 + i], c.values[i]);
        }
      }
    } else if (op == "release" && n == 2 && t[1] == "all") {
      c.op = PadScriptOp::kReleaseAll;
      ok = true;
    } else if ((op == "hold" || op == "release") && n == 2) {
      c.op = op == "hold" ? PadScriptOp::kHold : PadScriptOp::kRelease;
      ok = ParsePadSpecTokens(t[1], c.input).empty() &&
           (c.input.buttons || !c.input.axes.empty());
    } else if ((op == "idle" || op == "loop") && n == 2) {
      c.op = op == "idle" ? PadScriptOp::kIdle : PadScriptOp::kLoop;
      ok = pad_detail::ParseUint(t[1], c.n) && (c.op == PadScriptOp::kIdle || c.n > 0);
    } else if ((op == "endloop" || op == "end") && n == 1) {
      c.op = op == "end" ? PadScriptOp::kEnd : PadScriptOp::kEndLoop;
      ok = true;
    } else if (op == "mark" && n >= 2) {
      c.op = PadScriptOp::kMark;
      c.text.clear();
      for (size_t idx = 1; idx < n; ++idx) {
        if (idx > 1) c.text.push_back(' ');
        c.text.append(t[idx]);
      }
      ok = true;
    } else if (op == "wait" && n >= 2) {
      size_t timeout_idx = 2;
      if (t[1] == "title" && n >= 3) {
        c.op = PadScriptOp::kWaitTitle;
        timeout_idx = 3;
        ok = pad_detail::ParseUint(t[2], c.n, 16);
      } else if (t[1].substr(0, 8) == "frames>=") {
        c.op = PadScriptOp::kWaitFramesAtLeast;
        ok = pad_detail::ParseUint(t[1].substr(8), c.n);
      } else if (t[1].substr(0, 7) == "draws>=") {
        c.op = PadScriptOp::kWaitDrawsAtLeast;
        ok = pad_detail::ParseUint(t[1].substr(7), c.n);
      } else if (t[1].substr(0, 6) == "draws<") {
        c.op = PadScriptOp::kWaitDrawsBelow;
        ok = pad_detail::ParseUint(t[1].substr(6), c.n);
      }
      if (ok && n > timeout_idx) {
        uint64_t to = 120;
        ok = (n == timeout_idx + 1) &&
             pad_detail::ParseUint(t[timeout_idx], to) && to >= 1;
        c.timeout_s = uint32_t(std::min<uint64_t>(to, 86400));
      }
      c.text = std::string(t[1]);
    } else if (n == 1) {
      c.op = PadScriptOp::kPress;
      c.n = 4;
      std::string_view spec = op;
      if (size_t colon = spec.find(':'); colon != std::string_view::npos) {
        std::string_view hold = spec.substr(colon + 1);
        spec = spec.substr(0, colon);
        if (hold.size() > 2 && hold.substr(hold.size() - 2) == "ms") {
          c.ms = true;
          hold.remove_suffix(2);
        }
        ok = pad_detail::ParseUint(hold, c.n) && c.n >= 1;
      } else {
        ok = true;
      }
      if (ok) {
        ok = ParsePadSpecTokens(spec, c.input).empty() &&
             (c.input.buttons || !c.input.axes.empty());
      }
    }

    if (ok && c.op == PadScriptOp::kLoop) {
      ok = loop_stack.size() < 4;
      loop_stack.push_back(commands.size());
    } else if (ok && c.op == PadScriptOp::kEndLoop) {
      ok = !loop_stack.empty();
      if (ok) {
        c.n = loop_stack.back();
        loop_stack.pop_back();
      }
    }
    if (!ok) {
      error_cmd = std::move(c);
      return false;
    }
    commands.push_back(std::move(c));
    if (pos > script_text.size()) break;
  }
  if (!loop_stack.empty()) {
    error_cmd = commands[loop_stack.back()];
    return false;
  }
  return true;
}

struct PadRecordSample {
  uint16_t buttons = 0;
  int16_t axes_milli[6] = {};

  bool operator==(const PadRecordSample& o) const {
    if (buttons != o.buttons) return false;
    for (int i = 0; i < 6; ++i) {
      if (axes_milli[i] != o.axes_milli[i]) return false;
    }
    return true;
  }
  bool operator!=(const PadRecordSample& o) const { return !(*this == o); }
};

inline int16_t QuantizePadStickMilli(int16_t raw) {
  const int32_t scaled =
      raw >= 0 ? int32_t((int64_t(raw) * 1000 + 16383) / 32767)
               : int32_t((int64_t(raw) * 1000 - 16384) / 32768);
  const int32_t clamped = std::clamp<int32_t>(scaled, -1000, 1000);
  if (clamped >= -40 && clamped <= 40) return 0;
  if (clamped >= 975) return 1000;
  if (clamped <= -975) return -1000;
  const int32_t mag = (std::abs(clamped) + 12) / 25 * 25;
  return int16_t(clamped < 0 ? -std::min<int32_t>(mag, 1000)
                             : std::min<int32_t>(mag, 1000));
}

inline int16_t QuantizePadTriggerMilli(uint8_t raw) {
  const int32_t scaled = int32_t((uint32_t(raw) * 1000u + 127u) / 255u);
  const int32_t clamped = std::clamp<int32_t>(scaled, 0, 1000);
  if (clamped <= 25) return 0;
  if (clamped >= 975) return 1000;
  return int16_t(std::min<int32_t>(((clamped + 12) / 25) * 25, 1000));
}

inline PadRecordSample QuantizePadSample(uint16_t buttons, int16_t lx,
                                         int16_t ly, int16_t rx, int16_t ry,
                                         uint8_t lt, uint8_t rt) {
  PadRecordSample s;
  s.buttons = buttons;
  s.axes_milli[0] = QuantizePadStickMilli(lx);
  s.axes_milli[1] = QuantizePadStickMilli(ly);
  s.axes_milli[2] = QuantizePadStickMilli(rx);
  s.axes_milli[3] = QuantizePadStickMilli(ry);
  s.axes_milli[4] = QuantizePadTriggerMilli(lt);
  s.axes_milli[5] = QuantizePadTriggerMilli(rt);
  return s;
}

inline std::string FormatPadSetLine(uint64_t delta_ticks,
                                    const PadRecordSample& s) {
  char buf[160] = {};
  const int n = std::snprintf(
      buf, sizeof(buf),
      "S %" PRIu64 " %04X %.3f %.3f %.3f %.3f %.3f %.3f", delta_ticks,
      unsigned(s.buttons), double(s.axes_milli[0]) / 1000.0,
      double(s.axes_milli[1]) / 1000.0, double(s.axes_milli[2]) / 1000.0,
      double(s.axes_milli[3]) / 1000.0, double(s.axes_milli[4]) / 1000.0,
      double(s.axes_milli[5]) / 1000.0);
  return n > 0 ? std::string(buf, size_t(n)) : std::string();
}

}  // namespace xendroid
