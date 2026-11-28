/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2013 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/cpu/cpu_flags.h"

#include <cstdio>

DEFINE_string(cpu, "any", "Does nothing. CPU backend [any, x64].", "CPU");

DEFINE_string(
    load_module_map, "",
    "Loads a .map for symbol names and to diff with the generated symbol "
    "database.",
    "CPU");

DEFINE_string(dump_functions_at, "",
              "Comma-separated guest addresses (hex) whose functions' PPC "
              "source, optimized HIR and host machine code are written to "
              "<log dir>/fndump_<function address>.txt when first "
              "translated. Any address in a function selects it.",
              "CPU");
DEFINE_bool(disassemble_functions, false,
            "Disassemble functions during generation.", "CPU");

DEFINE_string(log_guest_calls_at, "",
              "Comma-separated guest addresses (hex) whose entry and return "
              "are logged with the guest argument registers. arm64 only.",
              "CPU");
DEFINE_string(log_guest_call_fields, "",
              "Comma-separated guest words to dump on every log_guest_calls_at "
              "entry. Each term is <gpr>+<hex offset>, optionally followed by "
              "><hex offset> to dereference it and re-offset, and by :<count> "
              "for several words, e.g. \"r4+30,r4+4>F8,r4+4>B0:4\".",
              "CPU");
DEFINE_uint32(log_guest_calls_limit, 1000,
              "Entry and return lines logged per address by "
              "log_guest_calls_at; 0 is unlimited. A return whose value "
              "differs from the last one is always logged.",
              "CPU");
DEFINE_uint32(log_guest_calls_ring, 0,
              "With log_guest_calls_at, keep the last this many entry and "
              "return lines in memory instead of logging each, and log them "
              "when the guest crashes: the calls leading up to a crash without "
              "the per-call logging changing the timing. 0 logs each line.",
              "CPU");

DEFINE_bool(trace_functions, false, "Generate tracing for function statistics.",
            "CPU");
DEFINE_bool(trace_function_coverage, false,
            "Generate tracing for function instruction coverage statistics.",
            "CPU");
DEFINE_bool(trace_function_references, false,
            "Generate tracing for function address references.", "CPU");
DEFINE_bool(trace_function_data, false,
            "Generate tracing for function result data.", "CPU");

DEFINE_uint32(
    cpu_trace_mask, 0,
    "JIT execution trace modes to log (bitmask): 1=instructions, 2=data, "
    "4=function calls (7=all). Each mode must be compiled in to be usable.",
    "CPU");

DEFINE_bool(validate_hir, false,
            "Perform validation checks on the HIR during compilation.", "CPU");
DEFINE_bool(
    accurate_fpscr_status, false,
    "Maintain FPSCR exception summary bits (FX/FEX/VX/OX) after scalar FP "
    "instructions. Disabled by default: titles rarely inspect them and "
    "skipping the FPSCR read-modify-write avoids overhead while preserving "
    "CR1 copies when Rc=1.",
    "CPU");

// https://github.com/bitsh1ft3r/Xenon/blob/091e8cd4dc4a7c697b4979eb200be7c9dee3590b/Xenon/Core/XCPU/PPU/PowerPC.h#L370
DEFINE_uint64(
    pvr, 0x710700,
    "Known PVR's.\n"
    " 0x710200 = Used by Zephyr \n"
    " 0x710300 = Used by Zephyr\n"
    " 0x710500 = Used by Jasper\n"
    " 0x710700 = Default\n"
    " 0x710800 = Used by Corona V1 & V2\n"
    "Processor version and revision number.\nBits 0 to 15 are the version "
    "number.\nBits 16 to 31 are the revision number.\nNote: Some XEXs (such as "
    "mfgbootlauncher.xex) may check for a value that's less than 0x710700.",
    "CPU");

// Breakpoints:
DEFINE_uint64(break_on_instruction, 0,
              "int3 before the given guest address is executed.", "CPU");
DEFINE_int32(break_condition_gpr, -1, "GPR compared to", "CPU");
DEFINE_uint64(break_condition_value, 0, "value compared against", "CPU");
DEFINE_string(break_condition_op, "eq", "comparison operator", "CPU");
DEFINE_bool(break_condition_truncate, true, "truncate value to 32-bits", "CPU");

DEFINE_bool(break_on_debugbreak, true, "int3 on JITed __debugbreak requests.",
            "CPU");
DEFINE_bool(
    adpf_cpu_performance_hint, false,
    "Android: register live guest CPU worker threads with an ADPF performance "
    "hint session to stabilize CPU clocks under heavy load.",
    "CPU");
DEFINE_bool(
    jit_warmup_jprof, true,
    "Persist discovered guest function addresses in a per-title .jprof cache "
    "and pre-translate them in the background on next launch.",
    "CPU");
DEFINE_uint32(
    jit_warmup_jprof_threads, 2,
    "Background worker threads for .jprof JIT pre-translation (clamped 1..4).",
    "CPU");
DEFINE_uint32(
    jit_warmup_jprof_save_interval_ms, 10000,
    "Minimum interval in ms between periodic .jprof LMK-safe checkpoints.",
    "CPU");
DEFINE_uint32(
    jit_warmup_jprof_save_batch, 256,
    "New resolved functions required before triggering a periodic .jprof "
    "checkpoint.",
    "CPU");

namespace xe {
namespace cpu {

bool GuestAddressInList(const std::string& list, uint32_t address) {
  return GuestAddressInList(list, address, address);
}

bool GuestAddressInList(const std::string& list, uint32_t first,
                        uint32_t last) {
  for (size_t pos = 0; pos < list.size();) {
    size_t end = list.find(',', pos);
    if (end == std::string::npos) {
      end = list.size();
    }
    std::string entry = list.substr(pos, end - pos);
    pos = end + 1;
    size_t text = entry.find_first_not_of(" \t");
    if (text == std::string::npos) {
      continue;
    }
    entry = entry.substr(text, entry.find_last_not_of(" \t") - text + 1);
    if (entry.size() > 2 && entry[0] == '0' &&
        (entry[1] == 'x' || entry[1] == 'X')) {
      entry = entry.substr(2);
    }
    if (entry.size() > 4 && (entry.compare(0, 4, "sub_") == 0 ||
                             entry.compare(0, 4, "SUB_") == 0)) {
      entry = entry.substr(4);
    }
    uint32_t parsed = 0;
    if (std::sscanf(entry.c_str(), "%X", &parsed) == 1 && parsed >= first &&
        parsed <= last) {
      return true;
    }
  }
  return false;
}

}  // namespace cpu
}  // namespace xe
