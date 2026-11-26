// Wave 6 (Onda 1) CPU / JIT / Kernel / POSIX / Clock logic & contract tests.
// Validates pure-logic invariants and source-level safety contracts for:
//   1. accurate_fpscr_status=false (30f5d45c adapted)
//   2. VSCR[SAT] observability census & mfvscr detection (6716efc1)
//   3. Leaf functions without stackpoint push/pop (2c64043e)
//   4. STVL/STVR 16-byte aligned fast path with cached labels (7441d651+ef49ec63)
//   5. XEvent pulse_epoch_ ordering inside global_lock (bbe33a08)
//   6. XamTaskSchedule CPU 5 & foreground priority policy (0bb3875c)
//   7. PosixCondition<Event>::SignalAndWaitOn enqueue-before-signal + rollback
//   8. ThreadStartRoutine lifetime safety without Bionic pthread_exit leak
//   9. clock_posix / clock_arm64 caching, cntfrq_el0==0 fallback, __int128 math

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "xenia/cpu/hir/opcodes.h"

namespace {

#define WAVE6_CHECK(cond)                                                \
  do {                                                                   \
    if (!(cond)) {                                                       \
      std::fprintf(stderr, "WAVE6_CPU check failed (%s:%d): %s\n",       \
                   __FILE__, __LINE__, #cond);                           \
      std::abort();                                                      \
    }                                                                    \
  } while (false)

namespace fs = std::filesystem;

std::string NormalizeNewlines(std::string s) {
  s.erase(std::remove(s.begin(), s.end(), '\r'), s.end());
  return s;
}

std::string ReadSourceText(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  WAVE6_CHECK(in.good());
  std::ostringstream ss;
  ss << in.rdbuf();
  return NormalizeNewlines(ss.str());
}

std::string ExtractFunctionBody(const std::string& src,
                                const std::string& anchor) {
  const size_t start = src.find(anchor);
  WAVE6_CHECK(start != std::string::npos);
  const size_t open = src.find('{', start);
  WAVE6_CHECK(open != std::string::npos);
  int depth = 0;
  for (size_t i = open; i < src.size(); ++i) {
    if (src[i] == '{') {
      ++depth;
    } else if (src[i] == '}') {
      --depth;
      if (depth == 0) {
        return src.substr(open, i - open + 1);
      }
    }
  }
  WAVE6_CHECK(false && "Unmatched brace while extracting function body");
  return {};
}

// Reference PPC mfvscr instruction detector:
// VX-form: opcode=4 (bits 0..5), vD (bits 6..10), RA=0, RB=0 (bits 11..20),
// XO=1540 (0x604, bits 21..31).
constexpr bool IsPpcMfvscrInstructionRef(uint32_t code) {
  return (code & 0xFC1FFFFFu) == 0x10000604u;
}

// Reference HIR leaf/call classifier (2c64043e).
constexpr bool HirOpcodeMakesCallRef(xe::cpu::hir::Opcode op) {
  using namespace xe::cpu::hir;
  switch (op) {
    case OPCODE_CALL:
    case OPCODE_CALL_TRUE:
    case OPCODE_CALL_INDIRECT:
    case OPCODE_CALL_INDIRECT_TRUE:
    case OPCODE_CALL_EXTERN:
    case OPCODE_TRAP:
    case OPCODE_TRAP_TRUE:
    case OPCODE_DEBUG_BREAK:
    case OPCODE_DEBUG_BREAK_TRUE:
      return true;
    default:
      return false;
  }
}

// Reference XamTaskSchedule policy decoder (0bb3875c).
struct XamTaskPolicy {
  uint32_t creation_flags;
  bool use_title_priority;
};

constexpr XamTaskPolicy ComputeXamTaskPolicyRef(uint32_t task_flags,
                                                uint32_t xex_system_flags) {
  constexpr uint32_t kAffinityCpu5 = 0x20000000u;
  constexpr uint32_t kXamTaskForegroundFlag = 0x00000002u;
  constexpr uint32_t kXexSystemForegroundTasks = 0x00000002u;
  const bool foreground =
      ((task_flags & kXamTaskForegroundFlag) != 0) ||
      ((xex_system_flags & kXexSystemForegroundTasks) != 0);
  return XamTaskPolicy{kAffinityCpu5, foreground};
}

// Reference 128-bit overflow-safe TicksTo conversion.
uint64_t TicksTo128Ref(uint64_t ticks, uint64_t units, uint64_t freq) {
  if (!freq) {
    return 0;
  }
#if defined(__SIZEOF_INT128__)
  return static_cast<uint64_t>(
      (static_cast<unsigned __int128>(ticks) * units) / freq);
#else
  return (ticks / freq) * units + ((ticks % freq) * units) / freq;
#endif
}

void TestMfvscrOpcodeMatrix() {
  // All 32 destination vector registers v0..v31 must match mfvscr.
  for (uint32_t vd = 0; vd < 32; ++vd) {
    const uint32_t insn = 0x10000604u | (vd << 21);
    WAVE6_CHECK(IsPpcMfvscrInstructionRef(insn));
  }
  // Non-mfvscr instructions and malformed reserved fields must be rejected.
  WAVE6_CHECK(!IsPpcMfvscrInstructionRef(0x10000644u));  // mtvscr v0
  WAVE6_CHECK(!IsPpcMfvscrInstructionRef(0x10000300u));  // vaddsbs
  WAVE6_CHECK(!IsPpcMfvscrInstructionRef(0x4E800020u));  // blr
  WAVE6_CHECK(!IsPpcMfvscrInstructionRef(0x60000000u));  // nop
  WAVE6_CHECK(!IsPpcMfvscrInstructionRef(0x00000000u));
  WAVE6_CHECK(!IsPpcMfvscrInstructionRef(0xFFFFFFFFu));
  WAVE6_CHECK(!IsPpcMfvscrInstructionRef(0x10010604u));  // RA != 0
  WAVE6_CHECK(!IsPpcMfvscrInstructionRef(0x10000E04u));  // RB != 0
  WAVE6_CHECK(!IsPpcMfvscrInstructionRef(0x14000604u));  // primary opcode 5
}

void TestMakesCallOpcodeMatrix() {
  using namespace xe::cpu::hir;
  const std::array<Opcode, 9> call_ops = {
      OPCODE_CALL,        OPCODE_CALL_TRUE,         OPCODE_CALL_INDIRECT,
      OPCODE_CALL_INDIRECT_TRUE, OPCODE_CALL_EXTERN, OPCODE_TRAP,
      OPCODE_TRAP_TRUE,   OPCODE_DEBUG_BREAK,       OPCODE_DEBUG_BREAK_TRUE,
  };
  for (Opcode op : call_ops) {
    WAVE6_CHECK(HirOpcodeMakesCallRef(op));
  }
  const std::array<Opcode, 14> leaf_ops = {
      OPCODE_NOP,          OPCODE_COMMENT,       OPCODE_SOURCE_OFFSET,
      OPCODE_RETURN,       OPCODE_RETURN_TRUE,   OPCODE_BRANCH,
      OPCODE_BRANCH_TRUE,  OPCODE_ADD,           OPCODE_LOAD,
      OPCODE_STORE,        OPCODE_VECTOR_ADD,    OPCODE_STVL,
      OPCODE_STVR,         OPCODE_CHECK_PREEMPT,
  };
  for (Opcode op : leaf_ops) {
    WAVE6_CHECK(!HirOpcodeMakesCallRef(op));
  }
}

void TestXamTaskPolicyMatrix() {
  const auto bg = ComputeXamTaskPolicyRef(0, 0);
  WAVE6_CHECK(bg.creation_flags == 0x20000000u);
  WAVE6_CHECK(!bg.use_title_priority);

  const auto flag1 = ComputeXamTaskPolicyRef(1, 0);
  WAVE6_CHECK(flag1.creation_flags == 0x20000000u);
  WAVE6_CHECK(!flag1.use_title_priority);

  const auto type2 = ComputeXamTaskPolicyRef(2, 0);
  WAVE6_CHECK(type2.creation_flags == 0x20000000u);
  WAVE6_CHECK(type2.use_title_priority);

  const auto xex_fg = ComputeXamTaskPolicyRef(0, 0x2);
  WAVE6_CHECK(xex_fg.creation_flags == 0x20000000u);
  WAVE6_CHECK(xex_fg.use_title_priority);

  const auto both = ComputeXamTaskPolicyRef(3, 0x2);
  WAVE6_CHECK(both.creation_flags == 0x20000000u);
  WAVE6_CHECK(both.use_title_priority);
}

void TestClock128Math() {
  WAVE6_CHECK(TicksTo128Ref(0, 1000, 0) == 0);
  // 1 GHz clock, 1 hour (3.6e12 ticks): naive ticks*10^7 overflows uint64 (> 1.84e19).
  const uint64_t one_hour_ticks_1ghz = 3600ull * 1000000000ull;
  WAVE6_CHECK(TicksTo128Ref(one_hour_ticks_1ghz, 1000ull, 1000000000ull) ==
              3600000ull);
  WAVE6_CHECK(TicksTo128Ref(one_hour_ticks_1ghz, 10000000ull, 1000000000ull) ==
              36000000000ull);
  // 100 days at 1 GHz (~8.64e15 ticks) * 10^7 = 8.64e22 (far above 2^64).
  const uint64_t hundred_days_1ghz = 100ull * 86400ull * 1000000000ull;
  WAVE6_CHECK(TicksTo128Ref(hundred_days_1ghz, 10000000ull, 1000000000ull) ==
              100ull * 86400ull * 10000000ull);
  // 19.2 MHz ARM generic timer (Snapdragon typical cntfrq_el0):
  const uint64_t freq_19_2mhz = 19200000ull;
  const uint64_t ten_days_19_2 = 10ull * 86400ull * freq_19_2mhz;
  WAVE6_CHECK(TicksTo128Ref(ten_days_19_2, 10000000ull, freq_19_2mhz) ==
              10ull * 86400ull * 10000000ull);
  // Sub-second rounding check:
  WAVE6_CHECK(TicksTo128Ref(freq_19_2mhz / 2, 1000ull, freq_19_2mhz) == 500ull);
}

void TestSourceContracts(const fs::path& cpp_root) {
  const fs::path xenia = cpp_root / "xenia/src/xenia";

  // 1. accurate_fpscr_status=false (30f5d45c adapted)
  const std::string cpu_flags_h = ReadSourceText(xenia / "cpu/cpu_flags.h");
  const std::string cpu_flags_cc = ReadSourceText(xenia / "cpu/cpu_flags.cc");
  WAVE6_CHECK(cpu_flags_h.find("DECLARE_bool(accurate_fpscr_status);") !=
              std::string::npos);
  WAVE6_CHECK(cpu_flags_cc.find("DEFINE_bool(\n    accurate_fpscr_status, false,") !=
                  std::string::npos ||
              cpu_flags_cc.find("DEFINE_bool(accurate_fpscr_status, false,") !=
                  std::string::npos);
  WAVE6_CHECK(cpu_flags_cc.find("UPDATE_from_") == std::string::npos);

  const std::string hir_h =
      ReadSourceText(xenia / "cpu/ppc/ppc_hir_builder.h");
  const std::string hir_cc =
      ReadSourceText(xenia / "cpu/ppc/ppc_hir_builder.cc");
  WAVE6_CHECK(hir_h.find("track_fpscr_status_") != std::string::npos);
  WAVE6_CHECK(hir_cc.find("track_fpscr_status_ = cvars::accurate_fpscr_status;") !=
              std::string::npos);
  const std::string update_fpscr_body =
      ExtractFunctionBody(hir_cc, "void PPCHIRBuilder::UpdateFPSCR(");
  const size_t guard_pos = update_fpscr_body.find("if (!track_fpscr_status_)");
  const size_t copy_pos = update_fpscr_body.find("CopyFPSCRToCR1();");
  const size_t cr1_if_pos = update_fpscr_body.find("if (update_cr1)");
  const size_t load_fpscr_pos = update_fpscr_body.find("LoadFPSCR()");
  WAVE6_CHECK(guard_pos != std::string::npos);
  WAVE6_CHECK(cr1_if_pos != std::string::npos && cr1_if_pos > guard_pos);
  WAVE6_CHECK(copy_pos != std::string::npos && copy_pos > cr1_if_pos);
  WAVE6_CHECK(load_fpscr_pos != std::string::npos && copy_pos < load_fpscr_pos);

  // 2. VSCR[SAT] census (6716efc1)
  const std::string xex_h = ReadSourceText(xenia / "cpu/xex_module.h");
  const std::string xex_cc = ReadSourceText(xenia / "cpu/xex_module.cc");
  WAVE6_CHECK(xex_h.find("bool ReadsVscr() const;") != std::string::npos);
  const std::string reads_vscr_body =
      ExtractFunctionBody(xex_cc, "bool XexModule::ReadsVscr() const");
  WAVE6_CHECK(reads_vscr_body.find("0x10000604") != std::string::npos);
  WAVE6_CHECK(reads_vscr_body.find("finished_load_") != std::string::npos);
  WAVE6_CHECK(reads_vscr_body.find("is_patch()") != std::string::npos);
  WAVE6_CHECK(reads_vscr_body.find("page_descriptors") != std::string::npos);

  const std::string proc_h = ReadSourceText(xenia / "cpu/processor.h");
  const std::string proc_cc = ReadSourceText(xenia / "cpu/processor.cc");
  WAVE6_CHECK(proc_h.find("BeginVscrSatCensus()") != std::string::npos);
  WAVE6_CHECK(proc_h.find("FinishVscrSatCensus()") != std::string::npos);
  WAVE6_CHECK(proc_h.find("vscr_sat_observable()") != std::string::npos);
  const std::string enable_dyn_body =
      ExtractFunctionBody(proc_cc, "void Processor::EnableDynamicCode()");
  WAVE6_CHECK(enable_dyn_body.find("vscr_sat_observable_.store(true") !=
              std::string::npos);
  const std::string finish_census_body =
      ExtractFunctionBody(proc_cc, "void Processor::FinishVscrSatCensus()");
  WAVE6_CHECK(finish_census_body.find("_trampoline") != std::string::npos);
  WAVE6_CHECK(finish_census_body.find("is_patch()") != std::string::npos);
  WAVE6_CHECK(finish_census_body.find("saw_xex") != std::string::npos);

  const std::string emu_cc = ReadSourceText(xenia / "emulator.cc");
  const std::string complete_launch =
      ExtractFunctionBody(emu_cc, "Emulator::CompleteLaunch(");
  const size_t begin_census = complete_launch.find("BeginVscrSatCensus()");
  const size_t load_plugins = complete_launch.find("LoadTitlePlugins(");
  const size_t finish_census = complete_launch.find("FinishVscrSatCensus()");
  const size_t start_jprof = complete_launch.find("StartJitProfileSession(");
  WAVE6_CHECK(begin_census != std::string::npos);
  WAVE6_CHECK(load_plugins != std::string::npos);
  WAVE6_CHECK(finish_census != std::string::npos);
  WAVE6_CHECK(start_jprof != std::string::npos);
  WAVE6_CHECK(begin_census < load_plugins);
  WAVE6_CHECK(load_plugins < finish_census);
  WAVE6_CHECK(finish_census < start_jprof);

  const std::string altivec_cc =
      ReadSourceText(xenia / "cpu/ppc/ppc_emit_altivec.cc");
  WAVE6_CHECK(altivec_cc.find("PACK_TYPE_SET_SAT") == std::string::npos);
  // Every StoreSAT(f.DidSaturate(...)) must be guarded by vscr_sat_observable().
  size_t search_pos = 0;
  size_t sat_sites = 0;
  while ((search_pos = altivec_cc.find("f.StoreSAT(", search_pos)) !=
         std::string::npos) {
    ++sat_sites;
    const size_t ctx_start = search_pos > 80 ? search_pos - 80 : 0;
    const std::string ctx = altivec_cc.substr(ctx_start, search_pos - ctx_start);
    WAVE6_CHECK(ctx.find("vscr_sat_observable()") != std::string::npos);
    search_pos += 11;
  }
  WAVE6_CHECK(sat_sites >= 20);

  // 3. Leaf functions without stackpoints (2c64043e)
  const std::string a64_h =
      ReadSourceText(xenia / "cpu/backend/a64/a64_emitter.h");
  const std::string a64_cc =
      ReadSourceText(xenia / "cpu/backend/a64/a64_emitter.cc");
  WAVE6_CHECK(a64_h.find("uses_stackpoints_") != std::string::npos);
  const std::string makes_call_body =
      ExtractFunctionBody(a64_cc, "MakesCall(const hir::HIRBuilder*");
  for (const char* tok : {"OPCODE_CALL", "OPCODE_CALL_TRUE",
                          "OPCODE_CALL_INDIRECT", "OPCODE_CALL_INDIRECT_TRUE",
                          "OPCODE_CALL_EXTERN", "OPCODE_TRAP",
                          "OPCODE_TRAP_TRUE", "OPCODE_DEBUG_BREAK",
                          "OPCODE_DEBUG_BREAK_TRUE"}) {
    WAVE6_CHECK(makes_call_body.find(tok) != std::string::npos);
  }
  const std::string push_sp =
      ExtractFunctionBody(a64_cc, "void A64Emitter::PushStackpoint()");
  const std::string pop_sp =
      ExtractFunctionBody(a64_cc, "void A64Emitter::PopStackpoint()");
  WAVE6_CHECK(push_sp.find("!uses_stackpoints_") != std::string::npos);
  WAVE6_CHECK(pop_sp.find("!uses_stackpoints_") != std::string::npos);

  // 4. STVL/STVR aligned fast path (7441d651 + ef49ec63)
  const std::string vec_cc =
      ReadSourceText(xenia / "cpu/backend/a64/a64_seq_vector.cc");
  const std::string stvl_body =
      ExtractFunctionBody(vec_cc, "struct STVL_V128");
  const std::string stvr_body =
      ExtractFunctionBody(vec_cc, "struct STVR_V128");
  WAVE6_CHECK(stvl_body.find("tst(e.x0, 0xF)") != std::string::npos);
  WAVE6_CHECK(stvl_body.find("str(QReg(0), ptr(e.x0))") != std::string::npos);
  WAVE6_CHECK(stvl_body.find("NewCachedLabel()") != std::string::npos);
  WAVE6_CHECK(stvl_body.find("e.L(") != std::string::npos);
  WAVE6_CHECK(stvl_body.find("/*skip_physical_skew=*/false") !=
              std::string::npos);
  WAVE6_CHECK(stvl_body.find("Xbyak_aarch64::Label ") == std::string::npos);
  WAVE6_CHECK(stvl_body.find("LKeepingRemapBound") == std::string::npos);

  const size_t stvr_tst = stvr_body.find("tst(e.x0, 0xF)");
  const size_t stvr_rev = stvr_body.find("e.rev32(");
  WAVE6_CHECK(stvr_tst != std::string::npos);
  WAVE6_CHECK(stvr_rev != std::string::npos);
  WAVE6_CHECK(stvr_tst < stvr_rev);
  WAVE6_CHECK(stvr_body.find("NewCachedLabel()") != std::string::npos);
  WAVE6_CHECK(stvr_body.find("e.L(") != std::string::npos);
  WAVE6_CHECK(stvr_body.find("/*skip_physical_skew=*/false") !=
              std::string::npos);
  WAVE6_CHECK(stvr_body.find("Xbyak_aarch64::Label ") == std::string::npos);
  WAVE6_CHECK(stvr_body.find("LKeepingRemapBound") == std::string::npos);

  // 5. XEvent pulse_epoch_ ordering inside global_lock (bbe33a08)
  const std::string xevent_cc = ReadSourceText(xenia / "kernel/xevent.cc");
  for (const char* fn : {"int32_t XEvent::Set(", "int32_t XEvent::Pulse("}) {
    const std::string body = ExtractFunctionBody(xevent_cc, fn);
    const size_t lock_pos = body.find("global_critical_region_.Acquire()");
    const size_t state_pos = body.find("kevent->header.signal_state =");
    const size_t epoch_pos = body.find("pulse_epoch_.fetch_add(1)");
    const size_t wake_pos = body.find("WakeCooperativeWaiters()");
    WAVE6_CHECK(lock_pos != std::string::npos);
    WAVE6_CHECK(state_pos != std::string::npos);
    WAVE6_CHECK(epoch_pos != std::string::npos);
    WAVE6_CHECK(wake_pos != std::string::npos);
    WAVE6_CHECK(lock_pos < state_pos);
    WAVE6_CHECK(state_pos < epoch_pos);
    WAVE6_CHECK(epoch_pos < wake_pos);
  }

  // 6. XamTaskSchedule CPU 5 & priority flags (0bb3875c)
  const std::string xam_task_cc =
      ReadSourceText(xenia / "kernel/xam/xam_task.cc");
  WAVE6_CHECK(xam_task_cc.find("dword_t optional_ptr, lpdword_t handle_ptr") !=
              std::string::npos);
  WAVE6_CHECK(xam_task_cc.find("struct XAM_TASK_ARGS") != std::string::npos);
  WAVE6_CHECK(xam_task_cc.find("0x20000000") != std::string::npos);
  WAVE6_CHECK(xam_task_cc.find("XEX_HEADER_SYSTEM_FLAGS") != std::string::npos);
  WAVE6_CHECK(xam_task_cc.find("XEX_SYSTEM_FOREGROUND_TASKS") !=
              std::string::npos);
  WAVE6_CHECK(xam_task_cc.find("GetTitleProcess()") != std::string::npos);
  WAVE6_CHECK(xam_task_cc.find("SetBasePriority(") != std::string::npos);

  // 7 & 8. threading_posix.cc SignalAndWaitOn + ThreadStartRoutine lifetime
  const std::string thr_posix =
      ReadSourceText(xenia / "base/threading_posix.cc");
  const size_t event_class_pos =
      thr_posix.find("class PosixCondition<Event>");
  WAVE6_CHECK(event_class_pos != std::string::npos);
  const std::string saw_body = ExtractFunctionBody(
      thr_posix.substr(event_class_pos), "WaitResult SignalAndWaitOn(");
  const size_t push_pos = saw_body.find("PushBack(&node)");
  const size_t sig_pos = saw_body.find("to_signal->Signal()");
  WAVE6_CHECK(push_pos != std::string::npos);
  WAVE6_CHECK(sig_pos != std::string::npos);
  // Ensure enqueue happens before signal on the handoff path.
  const size_t sig_after_push = saw_body.find("to_signal->Signal()", push_pos);
  WAVE6_CHECK(sig_after_push != std::string::npos);
  WAVE6_CHECK(saw_body.find("AutoResetEventHandoff()") != std::string::npos);
  WAVE6_CHECK(saw_body.find("ScopedParked") != std::string::npos);
  WAVE6_CHECK(saw_body.find("ReleaseOneWaiterLocked(") != std::string::npos);

  const std::string tsr_body = ExtractFunctionBody(
      thr_posix, "void* PosixCondition<Thread>::ThreadStartRoutine(");
  const size_t cleanup_push = tsr_body.find("pthread_cleanup_push(");
  const size_t notify_pos = tsr_body.find("NotifyAll(&lock)");
  const size_t cleanup_pop = tsr_body.find("pthread_cleanup_pop(1)");
  WAVE6_CHECK(cleanup_push != std::string::npos);
  WAVE6_CHECK(notify_pos != std::string::npos);
  WAVE6_CHECK(cleanup_pop != std::string::npos);
  WAVE6_CHECK(cleanup_push < notify_pos);
  WAVE6_CHECK(notify_pos < cleanup_pop);
  // ThreadStartRoutine must not dereference a potentially freed PosixThread*.
  WAVE6_CHECK(tsr_body.find("dynamic_cast<PosixThread*>") == std::string::npos);
  WAVE6_CHECK(tsr_body.find("current_thread_condition_ = cond;") !=
              std::string::npos);
  const std::string thr_exit_body =
      ExtractFunctionBody(thr_posix, "void Thread::Exit(");
  WAVE6_CHECK(thr_exit_body.find("current_thread_condition_") !=
              std::string::npos);
  const std::string sig_handler_body = ExtractFunctionBody(
      thr_posix,
      "static void signal_handler(int signal, siginfo_t* info, void* /*context*/) {");
  WAVE6_CHECK(sig_handler_body.find("current_thread_condition_") !=
              std::string::npos);

  const std::string xthread_cc = ReadSourceText(xenia / "kernel/xthread.cc");
  WAVE6_CHECK(xthread_cc.find("HostThreadLifetimeGuard") == std::string::npos);

  // 9. clock_posix.cc / clock_arm64.cc + clock.cc default preserved
  const std::string clock_posix =
      ReadSourceText(xenia / "base/clock_posix.cc");
  const std::string clock_arm64 =
      ReadSourceText(xenia / "base/clock_arm64.cc");
  const std::string clock_cc = ReadSourceText(xenia / "base/clock.cc");
  const std::string freq_plat = ExtractFunctionBody(
      clock_posix, "uint64_t Clock::host_tick_frequency_platform()");
  WAVE6_CHECK(freq_plat.find("static const uint64_t") != std::string::npos);
  WAVE6_CHECK(clock_posix.find("__SIZEOF_INT128__") != std::string::npos);
  const std::string sys_time = ExtractFunctionBody(
      clock_posix, "uint64_t Clock::QueryHostSystemTime()");
  WAVE6_CHECK(sys_time.find("gettimeofday(") != std::string::npos);
  WAVE6_CHECK(clock_arm64.find("host_tick_frequency_platform()") !=
              std::string::npos);
  WAVE6_CHECK(clock_arm64.find("host_tick_count_platform()") !=
              std::string::npos);
  WAVE6_CHECK(clock_cc.find("DEFINE_bool(clock_source_raw, false,") !=
              std::string::npos);

  // Regra Zero forbidden-ports guard
  WAVE6_CHECK(hir_cc.find("TO_SINGLE_KEEP_NAN") == std::string::npos);
  WAVE6_CHECK(altivec_cc.find("PACK_TYPE_SET_SAT") == std::string::npos);
}

}  // namespace

int main(int argc, char** argv) {
  TestMfvscrOpcodeMatrix();
  TestMakesCallOpcodeMatrix();
  TestXamTaskPolicyMatrix();
  TestClock128Math();
  if (argc >= 2 && argv[1] && argv[1][0] != '\0') {
    TestSourceContracts(fs::path(argv[1]));
  }
  std::printf("wave6_cpu_kernel_edge_test: PASS\n");
  return 0;
}
