/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/cpu/backend/a64/a64_emitter.h"

#include <atomic>
#include <bit>
#include <cstring>
#include <unordered_set>

#include "xenia/base/debugging.h"
#include "xenia/base/logging.h"
#include "xenia/base/math.h"
#include "xenia/base/memory.h"
#include "xenia/base/profiling.h"
#include "xenia/cpu/backend/a64/a64_backend.h"
#include "xenia/cpu/backend/a64/a64_code_cache.h"
#include "xenia/cpu/backend/a64/a64_function.h"
#include "xenia/cpu/backend/a64/a64_hle_sync_layout.h"
#include "xenia/cpu/backend/a64/a64_sequences.h"
#include "xenia/cpu/backend/a64/a64_stack_layout.h"
#include "xenia/cpu/backend/a64/a64_stack_skew_analysis.h"
#include "xenia/cpu/backend/a64/a64_tracers.h"
#include "xenia/cpu/cpu_flags.h"
#include "xenia/cpu/export_resolver.h"
#include "xenia/cpu/hir/hir_builder.h"
#include "xenia/cpu/hir/label.h"
#include "xenia/cpu/ppc/ppc_context.h"
#include "xenia/cpu/ppc/ppc_stack_skew_guard.h"
#include "xenia/cpu/processor.h"

DECLARE_int64(a64_max_stackpoints);
DECLARE_bool(a64_enable_host_guest_stack_synchronization);
DECLARE_bool(a64_near_branches);
DECLARE_bool(a64_direct_calls);
DECLARE_bool(a64_skip_stack_skew_fixup);

DEFINE_bool(
    log_safepoint_pc, false,
    "Record the guest address of every JIT safepoint a fiber passes, so the "
    "cooperative scheduler's no-progress report can name where a wedged "
    "fiber last checked in rather than only its link register. Costs a "
    "materialize and a store on every loop back-edge; diagnostic only.",
    "CPU");

DEFINE_bool(
    a64_guest_memory_tso, false,
    "Order guest memory accesses like a TSO machine (x86): a load-load "
    "barrier after every guest load and a store-store one before every guest "
    "store, so no load passes a load and no store passes a load or a store. "
    "ARM reorders all of those, which a guest race the console's or an x86 "
    "host's ordering hides can need. Slow; diagnostic only. Takes effect for "
    "code translated afterwards.",
    "CPU");

DEFINE_bool(
    a64_inline_hle_sync, true,
    "Emit inline AArch64 fast-paths for uncontended high-frequency HLE sync "
    "exports (IRQL, CriticalSection, SpinLock), falling back to C++ handlers "
    "on contention/validation edge cases or when hle_census is enabled.",
    "CPU");

namespace xe {
namespace cpu {
namespace backend {
namespace a64 {

using namespace Xbyak_aarch64;

// Defined in a64_backend.cc.
extern uint64_t ResolveFunction(void* raw_context, uint64_t target_address);
extern uint64_t ResolveDynamicCall(void* raw_context, uint64_t target_address);
extern uint64_t ResolveDirectDynamicCall(void* raw_context,
                                         uint64_t target_address);
extern uint64_t ResolveCachedReturnSiteCall(void* raw_context,
                                            uint64_t target_address);
extern uint64_t ResolveCachedReturnSiteTail(void* raw_context,
                                            uint64_t target_address);
extern uint64_t ResolveDynamicTailCall(void* raw_context,
                                       uint64_t target_address);

static uint64_t UndefinedCallExtern(void* raw_context, uint64_t function_ptr) {
  auto function = reinterpret_cast<Function*>(function_ptr);
  XELOGE("undefined extern call to {:08X} {}", function->address(),
         function->name());
  return 0;
}

static bool MakesCall(const hir::HIRBuilder* builder) {
  for (const auto* block = builder->first_block(); block; block = block->next) {
    for (const auto* instr = block->instr_head; instr; instr = instr->next) {
      switch (instr->GetOpcodeNum()) {
        case hir::OPCODE_CALL:
        case hir::OPCODE_CALL_TRUE:
        case hir::OPCODE_CALL_INDIRECT:
        case hir::OPCODE_CALL_INDIRECT_TRUE:
        case hir::OPCODE_CALL_EXTERN:
        case hir::OPCODE_TRAP:
        case hir::OPCODE_TRAP_TRUE:
        case hir::OPCODE_DEBUG_BREAK:
        case hir::OPCODE_DEBUG_BREAK_TRUE:
          return true;
        default:
          break;
      }
    }
  }
  return false;
}

// Max host code per translated guest function. 1 MiB was too small for large,
// FP-heavy guest functions (each PowerPC FP op emits NaN-handling overhead),
// which overflowed the Xbyak buffer -> uncaught Xbyak_aarch64::Error "code is
// too big" -> terminate. The buffer is a single per-emitter allocation reused
// (reset) across functions, so enlarging it is cheap.
static constexpr size_t kMaxCodeSize = 4_MiB;

// Register maps:
// GPR allocatable registers:
//   0-6:  x22-x28 (callee-saved)
//   7-15: x5-x13  (caller-saved volatile, gated by RegisterAllocationPass)
// (x19=backend context, x20=context, x21=membase are reserved;
//  x0-x4 and x14-x17 remain dedicated emitter/sequence scratch)
const uint32_t A64Emitter::gpr_reg_map_[GPR_COUNT] = {
    22, 23, 24, 25, 26, 27, 28, 5, 6, 7, 8, 9, 10, 11, 12, 13,
};

// VEC allocatable registers: v4-v15, v16-v31
// (v0-v3 are scratch)
const uint32_t A64Emitter::vec_reg_map_[VEC_COUNT] = {
    4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
};

A64Emitter::A64Emitter(A64Backend* backend, XbyakA64Allocator* allocator)
    : CodeGenerator(kMaxCodeSize, Xbyak_aarch64::AutoGrow, allocator),
      processor_(backend->processor()),
      backend_(backend),
      code_cache_(backend->code_cache()),
      allocator_(allocator),
      feature_flags_(arm64::GetFeatureFlags()) {}

A64Emitter::~A64Emitter() = default;

bool A64Emitter::Emit(GuestFunction* function, hir::HIRBuilder* builder,
                      uint32_t debug_info_flags, FunctionDebugInfo* debug_info,
                      void** out_code_address, size_t* out_code_size,
                      std::vector<SourceMapEntry>* out_source_map) {
  SCOPE_profile_cpu_f("cpu");

  guest_module_ = dynamic_cast<XexModule*>(function->module());

  debug_info_ = debug_info;
  debug_info_flags_ = debug_info_flags;
  trace_data_ = &function->trace_data();

  current_guest_function_ = function->address();
  log_guest_call_ = !cvars::log_guest_calls_at.empty() &&
                    GuestAddressInList(cvars::log_guest_calls_at,
                                       current_guest_function_);

  // The prolog, epilog and helpers emit outside the per-opcode guard below, so
  // an unencodable operand needs catching here too. Conditional branches are
  // emitted direct first; a function too large for them to reach across is
  // emitted again with the long-range form.
  bool uses_no_skew = false;
  bool has_unsafe_r1_write = false;
  AnalyzeStackSkewSafety(builder, &uses_no_skew, &has_unsafe_r1_write);
  auto* a64_fn = static_cast<A64Function*>(function);
  const bool forbid_skew_skip =
      a64_fn->forbid_stack_skew_skip() ||
      backend_->IsStackSkewSkipForbidden(function->address());
  if (forbid_skew_skip) {
    a64_fn->set_forbid_stack_skew_skip(true);
  }
  const bool allow_skew_skip =
      xe::memory::allocation_granularity() > 0x1000 &&
      cvars::a64_skip_stack_skew_fixup && !forbid_skew_skip &&
      !has_unsafe_r1_write;
  if (uses_no_skew && !allow_skew_skip) {
    StripNoPhysicalSkewFlags(builder);
    uses_no_skew = false;
  }
  uses_stack_skew_skip_ = allow_skew_skip && uses_no_skew;
  uses_stackpoints_ = MakesCall(builder);

  EmitFunctionInfo func_info = {};
  bool emitted = false;
  // Once: it turns every local into its stack offset (a constant of another
  // type), so a second emission must reuse the layout, not redo it.
  const size_t stack_size = LayOutLocals(builder);
  for (bool near_branches : {bool(cvars::a64_near_branches), false}) {
    // Reset state.
    near_branches_ = near_branches;
    stack_size_ = StackLayout::GUEST_STACK_SIZE;
    source_map_arena_.Reset();
    tail_code_.clear();
    pending_direct_calls_.clear();
    fpcr_mode_ = FPCRMode::Unknown;
    func_info = {};
    try {
      emitted = Emit(builder, stack_size, func_info);
    } catch (const Xbyak_aarch64::Error& e) {
      if (IsNearBranchOutOfRange(e)) {
        XELOGI(
            "A64: guest function {:08X} too large for direct branches, "
            "emitted again with long-range ones",
            current_guest_function_);
        ResetPerFunctionState();
        continue;
      }
      XELOGE("A64: assembler error while emitting guest function {:08X}: {}",
             current_guest_function_, e.what());
      emitted = false;
    }
    break;
  }
  near_branches_ = false;
  uses_stack_skew_skip_ = false;
  uses_stackpoints_ = true;
  if (!emitted) {
    // Emplace only runs on success, so a failed compile has to reset too.
    ResetPerFunctionState();
    return false;
  }

  // Emplace the code into the code cache.
  *out_code_address = Emplace(func_info, function);
  *out_code_size = func_info.code_size.total;

  // Copy source map.
  source_map_arena_.CloneContents(out_source_map);

  return *out_code_address != nullptr;
}

bool A64Emitter::IsNearBranchOutOfRange(const Xbyak_aarch64::Error& e) const {
  return near_branches_ && int(e) == Xbyak_aarch64::ERR_LABEL_IS_TOO_FAR;
}

size_t A64Emitter::LayOutLocals(hir::HIRBuilder* builder) {
  // Calculate local variable stack offsets.
  auto locals = builder->locals();
  size_t stack_offset = StackLayout::GUEST_STACK_SIZE;
  for (auto it = locals.begin(); it != locals.end(); ++it) {
    auto slot = *it;
    size_t type_size = hir::GetTypeSize(slot->type);
    // Align to natural size (at least 4 bytes for ARM64 alignment).
    size_t align_size = xe::round_up(type_size, static_cast<size_t>(4));
    stack_offset = xe::align(stack_offset, align_size);
    slot->set_constant(static_cast<uint32_t>(stack_offset));
    stack_offset += type_size;
  }
  // Align total stack offset to 16 bytes (ARM64 ABI requirement).
  stack_offset -= StackLayout::GUEST_STACK_SIZE;
  stack_offset = xe::align(stack_offset, static_cast<size_t>(16));
  return StackLayout::GUEST_STACK_SIZE + stack_offset;
}

bool A64Emitter::Emit(hir::HIRBuilder* builder, size_t stack_size,
                      EmitFunctionInfo& func_info) {
  // ARM64 ABI: SP must always be 16-byte aligned.
  assert_true(stack_size % 16 == 0);
  func_info.stack_size = stack_size;
  func_info.lr_save_offset = StackLayout::HOST_RET_ADDR;
  stack_size_ = stack_size;

  struct {
    size_t prolog;
    size_t body;
    size_t epilog;
    size_t tail;
    size_t prolog_stack_alloc;
  } code_offsets = {};

  // ========================================================================
  // PROLOG
  // ========================================================================
  code_offsets.prolog = getSize();

  // sub sp, sp, #stack_size
  if (stack_size <= 4095) {
    sub(sp, sp, static_cast<uint32_t>(stack_size));
  } else {
    mov(x17, static_cast<uint64_t>(stack_size));
    sub(sp, sp, x17, UXTX);
  }
  code_offsets.prolog_stack_alloc = getSize();

  // Store host return address (x30/LR) so the epilog can restore it.
  str(x30, ptr(sp, static_cast<uint32_t>(StackLayout::HOST_RET_ADDR)));
  // Store guest PPC return address (passed in x0 by convention).
  str(x0, ptr(sp, static_cast<uint32_t>(StackLayout::GUEST_RET_ADDR)));
  // Store zero for call return address (we haven't made a call yet).
  str(xzr, ptr(sp, static_cast<uint32_t>(StackLayout::GUEST_CALL_RET_ADDR)));

  // Record stackpoint for longjmp recovery.
  PushStackpoint();

  if (uses_stack_skew_skip_) {
    // Guard entry r1 against [0x70000000, 0x7F000000). On a 16 KB host, any
    // call entering with r1 outside the guest stack window pops the frame and
    // re-emits this function without LOAD_STORE_NO_PHYSICAL_SKEW.
    static_assert(kR1ContextOffset < 16380,
                  "r[1] offset must fit A64 ldr unsigned imm12");
    auto& fallback_label = AddToTail([](A64Emitter& e, Label&) {
      e.PopStackpoint();
      e.ldr(e.x0,
            ptr(e.sp, static_cast<uint32_t>(StackLayout::GUEST_RET_ADDR)));
      e.ldr(e.x30,
            ptr(e.sp, static_cast<uint32_t>(StackLayout::HOST_RET_ADDR)));
      if (e.stack_size() <= 4095) {
        e.add(e.sp, e.sp, static_cast<uint32_t>(e.stack_size()));
      } else {
        e.mov(e.x17, static_cast<uint64_t>(e.stack_size()));
        e.add(e.sp, e.sp, e.x17, UXTX);
      }
      e.mov(e.w16, e.current_guest_function());
      e.mov(e.x9, reinterpret_cast<uint64_t>(
                      e.backend()->reemit_without_stack_skew_thunk()));
      e.br(e.x9);
    });
    ldr(w17, ptr(x20, static_cast<uint32_t>(kR1ContextOffset)));
    mov(w16, ppc::kGuestStackRangeBegin);
    sub(w17, w17, w16);
    mov(w16, ppc::kGuestStackRangeEnd - ppc::kGuestStackRangeBegin);
    cmp(w17, w16);
    b(HS, fallback_label);
  }

  // ========================================================================
  // BODY
  // ========================================================================
  code_offsets.body = getSize();

  // FTrace: log guest function entry when the backend was built with
  // function tracing available (gated at runtime by the trace_func flag).
  if (IsTracingFunc()) {
    mov(x1, static_cast<uint64_t>(current_guest_function_));
    CallNative(reinterpret_cast<void*>(TraceFunctionEntry));
  }
  if (log_guest_call_) {
    mov(x1, static_cast<uint64_t>(current_guest_function_));
    CallNative(reinterpret_cast<void*>(LogGuestCallEntry));
  }

  // Allocate the epilog label (owned by label_cache_ for cleanup).
  auto epilog_label_ptr = new Label();
  label_cache_.push_back(epilog_label_ptr);
  epilog_label_ = epilog_label_ptr;

  // Prepare read-only loop promotion lookup tables for this emission attempt.
  const auto& loop_plan = builder->loop_promotion_plan();
  std::vector<const hir::LoopPromotionRegion*> pre_first_block_entries;
  std::unordered_map<const hir::Block*, const hir::LoopPromotionRegion*>
      preheader_entries;
  std::unordered_map<const hir::Block*,
                     std::pair<const hir::LoopPromotionRegion*, uint32_t>>
      exit_blocks;
  std::unordered_map<const hir::Instr*,
                     std::pair<const hir::LoopPromotionRegion*, uint32_t>>
      after_instr_exits;
  std::unordered_map<const hir::Instr*,
                     std::pair<const hir::LoopPromotionRegion*, uint32_t>>
      flush_before_instrs;
  std::unordered_map<const hir::Block*,
                     std::pair<const hir::LoopPromotionRegion*, const hir::Instr*>>
      loop_body_blocks;

  if (!loop_plan.empty()) {
    for (const auto& region : loop_plan.regions) {
      for (const auto& entry : region.entry_points) {
        if (entry.before_first_block) {
          pre_first_block_entries.push_back(&region);
        } else if (entry.preheader_block) {
          preheader_entries[entry.preheader_block] = &region;
        }
      }
      for (const auto& exit_edge : region.exit_points) {
        if (exit_edge.exit_block) {
          exit_blocks[exit_edge.exit_block] = {&region,
                                               exit_edge.store_slot_mask};
        } else if (exit_edge.after_instr) {
          after_instr_exits[exit_edge.after_instr] = {
              &region, exit_edge.store_slot_mask};
        }
      }
      for (const auto& fp : region.flush_points) {
        if (fp.before_instr) {
          flush_before_instrs[fp.before_instr] = {&region, fp.flush_slot_mask};
        }
      }
      for (const auto& span : region.block_spans) {
        if (span.block && span.loop_end_instr) {
          loop_body_blocks[span.block] = {&region, span.loop_end_instr};
        }
      }
    }
  }

  for (const auto* region : pre_first_block_entries) {
    EmitLoadPinnedSlots(*region);
  }

  // Walk HIR blocks and emit ARM64 instructions.
  auto block = builder->first_block();
  synchronize_stack_on_next_instruction_ = false;
  while (block) {
    // Reset FPCR tracking on each block entry (we don't know which
    // predecessor ran, so mode is unknown).
    ForgetFpcrMode();

    // Bind all labels targeting this block.
    auto label = block->label_head;
    while (label) {
      L(GetLabel(label->id));
      label = label->next;
    }
    // A compare hands its flags only to a branch in its own block.
    ResetSequenceHandoffs();

    if (!preheader_entries.empty()) {
      auto pre_it = preheader_entries.find(block);
      if (pre_it != preheader_entries.end()) {
        EmitLoadPinnedSlots(*pre_it->second);
      }
    }
    if (!exit_blocks.empty()) {
      auto exit_it = exit_blocks.find(block);
      if (exit_it != exit_blocks.end()) {
        EmitStorePinnedSlots(*exit_it->second.first, exit_it->second.second);
      }
    }

    const hir::LoopPromotionRegion* active_loop_region = nullptr;
    const hir::Instr* active_loop_end = nullptr;
    if (!loop_body_blocks.empty()) {
      auto body_it = loop_body_blocks.find(block);
      if (body_it != loop_body_blocks.end()) {
        active_loop_region = body_it->second.first;
        active_loop_end = body_it->second.second;
      }
    }

    // Process each instruction in the block.
    const hir::Instr* instr = block->instr_head;
    while (instr) {
      // After a guest call, check for longjmp on the next real instruction.
      // Skip SOURCE_OFFSET because the return address from the call would
      // point past the check, so it would never execute. With debug info on a
      // COMMENT precedes the SOURCE_OFFSET, so skip that too.
      if (synchronize_stack_on_next_instruction_) {
        const hir::Opcode opcode_num = instr->GetOpcodeNum();
        if (opcode_num != hir::OPCODE_SOURCE_OFFSET &&
            opcode_num != hir::OPCODE_COMMENT) {
          synchronize_stack_on_next_instruction_ = false;
          EnsureSynchronizedGuestAndHostStack();
        }
      }
      if (!flush_before_instrs.empty()) {
        auto flush_it = flush_before_instrs.find(instr);
        if (flush_it != flush_before_instrs.end()) {
          EmitStorePinnedSlots(*flush_it->second.first,
                               flush_it->second.second);
        }
      }
      const hir::Instr* new_tail = instr;
      bool selected = false;
      const bool handoff_pending = sequence_handoff_pending();
      const hir::Opcode opcode = instr->GetOpcodeNum();
      const bool tso_load =
          cvars::a64_guest_memory_tso &&
          (opcode == hir::OPCODE_LOAD || opcode == hir::OPCODE_LOAD_OFFSET ||
           opcode == hir::OPCODE_LVL || opcode == hir::OPCODE_LVR ||
           opcode == hir::OPCODE_RESERVED_LOAD);
      if (cvars::a64_guest_memory_tso &&
          (opcode == hir::OPCODE_STORE || opcode == hir::OPCODE_STORE_OFFSET ||
           opcode == hir::OPCODE_STVL || opcode == hir::OPCODE_STVR ||
           opcode == hir::OPCODE_MEMSET)) {
        dmb(ISHST);
      }
      try {
        if (active_loop_region &&
            (opcode == hir::OPCODE_LOAD_CONTEXT ||
             opcode == hir::OPCODE_STORE_CONTEXT) &&
            TryEmitPinnedContextInstr(instr, *active_loop_region)) {
          selected = true;
          new_tail = instr->next;
        } else {
          selected = SelectSequence(this, instr, &new_tail);
        }
      } catch (const Xbyak_aarch64::Error& e) {
        if (IsNearBranchOutOfRange(e)) {
          throw;  // Emitted again with long-range branches.
        }
        // Uncaught this aborts the process with no context, so name the opcode
        // and the guest function and fail just this compile.
        XELOGE(
            "A64: assembler rejected HIR opcode {} in guest function {:08X}: "
            "{}",
            hir::GetOpcodeName(instr->GetOpcodeInfo()), current_guest_function_,
            e.what());
        return false;
      }
      if (!selected) {
        // No sequence matched — this is expected in Phase 1 before
        // sequences are implemented.
        XELOGE("A64: Unable to process HIR opcode {}",
               hir::GetOpcodeName(instr->GetOpcodeInfo()));
        return false;
      }
      if (handoff_pending && sequence_handoff_pending() &&
          !PassesHandoffs(instr)) {
        // The previous sequence emitted nothing and left its work to this
        // one, which did not do it: the code would compute a wrong result.
        XELOGE(
            "A64: HIR opcode {} did not take the handoff of the sequence "
            "before it in guest function {:08X}",
            hir::GetOpcodeName(instr->GetOpcodeInfo()),
            current_guest_function_);
        return false;
      }
      if (tso_load) {
        dmb(ISHLD);
      }
      if (!after_instr_exits.empty()) {
        auto after_it = after_instr_exits.find(instr);
        if (after_it != after_instr_exits.end()) {
          EmitStorePinnedSlots(*after_it->second.first,
                               after_it->second.second);
        }
      }
      if (active_loop_region && instr == active_loop_end) {
        active_loop_region = nullptr;
        active_loop_end = nullptr;
      }
      instr = new_tail;
    }

    if (!MaybeFlushV128ConstPool()) {
      return false;
    }

    block = block->next;
  }

  // ========================================================================
  // EPILOG
  // ========================================================================
  L(*epilog_label_);
  epilog_label_ = nullptr;
  // A call as the last instruction leaves the check unconsumed.
  if (synchronize_stack_on_next_instruction_) {
    synchronize_stack_on_next_instruction_ = false;
    EnsureSynchronizedGuestAndHostStack();
  }
  // FTrace: log the guest return value (r3) on normal return.
  if (IsTracingFunc()) {
    mov(x1, static_cast<uint64_t>(current_guest_function_));
    CallNative(reinterpret_cast<void*>(TraceFunctionReturn));
  }
  if (log_guest_call_) {
    mov(x1, static_cast<uint64_t>(current_guest_function_));
    CallNative(reinterpret_cast<void*>(LogGuestCallReturn));
  }
  code_offsets.epilog = getSize();

  // Pop stackpoint before leaving.
  PopStackpoint();

  // Restore host return address and deallocate stack.
  ldr(x30, ptr(sp, static_cast<uint32_t>(StackLayout::HOST_RET_ADDR)));
  if (stack_size <= 4095) {
    add(sp, sp, static_cast<uint32_t>(stack_size));
  } else {
    mov(x17, static_cast<uint64_t>(stack_size));
    add(sp, sp, x17, UXTX);
  }
  ret();

  // ========================================================================
  // TAIL CODE
  // ========================================================================
  for (auto& tail_item : tail_code_) {
    // ARM64 instructions are always 4-byte aligned, so alignment is mostly
    // a no-op unless we want cache-line alignment for hot paths.
    L(tail_item.label);
    try {
      tail_item.func(*this, tail_item.label);
    } catch (const Xbyak_aarch64::Error& e) {
      if (IsNearBranchOutOfRange(e)) {
        throw;  // Emitted again with long-range branches.
      }
      XELOGE("A64: assembler rejected tail code in guest function {:08X}: {}",
             current_guest_function_, e.what());
      return false;
    }
    if (!MaybeFlushV128ConstPool()) {
      return false;
    }
  }
  code_offsets.tail = getSize();

  // ========================================================================
  // LITERAL POOL
  // ========================================================================
  if (!FlushV128ConstPool(false)) {
    return false;
  }

  // Fill in EmitFunctionInfo metrics.
  assert_zero(code_offsets.prolog);
  func_info.code_size.total = getSize();
  func_info.code_size.prolog = code_offsets.body - code_offsets.prolog;
  func_info.code_size.body = code_offsets.epilog - code_offsets.body;
  func_info.code_size.epilog = code_offsets.tail - code_offsets.epilog;
  func_info.code_size.tail = getSize() - code_offsets.tail;
  func_info.prolog_stack_alloc_offset =
      code_offsets.prolog_stack_alloc - code_offsets.prolog;

  return true;
}

void* A64Emitter::Emplace(const EmitFunctionInfo& func_info,
                          GuestFunction* function) {
  assert_true(func_info.code_size.total == getSize());

  void* new_execute_address;
  void* new_write_address;

  if (function) {
    code_cache_->PlaceGuestCode(
        function->address(),
        const_cast<void*>(static_cast<const void*>(getCode())), func_info,
        function, new_execute_address, new_write_address);
  } else {
    code_cache_->PlaceHostCode(
        0, const_cast<void*>(static_cast<const void*>(getCode())), func_info,
        new_execute_address, new_write_address);
  }

  if (function && !pending_direct_calls_.empty()) {
    const uintptr_t base_rx = reinterpret_cast<uintptr_t>(new_execute_address);
    for (const auto& entry : pending_direct_calls_) {
      const uintptr_t call_site_rx = base_rx + entry.branch_code_offset;
      const uintptr_t tail_stub_rx =
          entry.stub_code_offset != SIZE_MAX
              ? (base_rx + entry.stub_code_offset)
              : 0;
      uint32_t* rw_slot = code_cache_->WriteSlotForExecuteAddress(call_site_rx);
      if (!rw_slot) {
        continue;
      }
      const uint32_t original_stub_insn = *rw_slot;
      code_cache_->RegisterDirectCallSite(entry.target_guest_addr, call_site_rx,
                                          original_stub_insn, entry.is_tail,
                                          tail_stub_rx);
      if (entry.target_fn) {
        uint8_t* target_code =
            static_cast<A64Function*>(entry.target_fn)->machine_code();
        if (target_code) {
          code_cache_->BackpatchDirectCallSites(entry.target_guest_addr,
                                                target_code);
        }
      }
    }
  }

  // In xbyak_aarch64, labels are resolved at define time (backpatching),
  // so all relative offsets are already correct. We just need to reset
  // the codegen state for the next function.
  ResetPerFunctionState();

  return new_execute_address;
}

void A64Emitter::ResetPerFunctionState() {
  reset();
  tail_code_.clear();
  pending_direct_calls_.clear();

  // Clean up cached labels.
  epilog_label_ = nullptr;
  for (auto* cached_label : label_cache_) {
    delete cached_label;
  }
  label_cache_.clear();
  v128_consts_.clear();
  v128_consts_first_use_ = 0;
  ResetSequenceHandoffs();

  // Clean up HIR->xbyak label map. HIR label ids restart at each function, so
  // stale entries would hand the next function this one's labels.
  for (auto& pair : label_map_) {
    delete pair.second;
  }
  label_map_.clear();
}

void A64Emitter::MarkSourceOffset(const hir::Instr* i) {
  auto entry = source_map_arena_.Alloc<SourceMapEntry>();
  entry->guest_address = static_cast<uint32_t>(i->src1.offset);
  entry->hir_offset = uint32_t(i->block->ordinal << 16) | i->ordinal;
  entry->code_offset = static_cast<uint32_t>(getSize());
}

void A64Emitter::DebugBreak() { brk(0xF000); }

void A64Emitter::Trap(uint16_t trap_type) { brk(trap_type); }

void A64Emitter::b(const Xbyak_aarch64::Cond cond,
                   const Xbyak_aarch64::Label& label) {
  if (near_branches_) {
    CodeGenerator::b(cond, label);
    return;
  }
  Xbyak_aarch64::Label skip;
  CodeGenerator::b(static_cast<Xbyak_aarch64::Cond>(cond ^ 1), skip);
  CodeGenerator::b(label);
  L(skip);
}

void A64Emitter::cbz(const Xbyak_aarch64::WReg& rt,
                     const Xbyak_aarch64::Label& label) {
  if (near_branches_) {
    CodeGenerator::cbz(rt, label);
    return;
  }
  Xbyak_aarch64::Label skip;
  CodeGenerator::cbnz(rt, skip);
  CodeGenerator::b(label);
  L(skip);
}

void A64Emitter::cbz(const Xbyak_aarch64::XReg& rt,
                     const Xbyak_aarch64::Label& label) {
  if (near_branches_) {
    CodeGenerator::cbz(rt, label);
    return;
  }
  Xbyak_aarch64::Label skip;
  CodeGenerator::cbnz(rt, skip);
  CodeGenerator::b(label);
  L(skip);
}

void A64Emitter::cbnz(const Xbyak_aarch64::WReg& rt,
                      const Xbyak_aarch64::Label& label) {
  if (near_branches_) {
    CodeGenerator::cbnz(rt, label);
    return;
  }
  Xbyak_aarch64::Label skip;
  CodeGenerator::cbz(rt, skip);
  CodeGenerator::b(label);
  L(skip);
}

void A64Emitter::cbnz(const Xbyak_aarch64::XReg& rt,
                      const Xbyak_aarch64::Label& label) {
  if (near_branches_) {
    CodeGenerator::cbnz(rt, label);
    return;
  }
  Xbyak_aarch64::Label skip;
  CodeGenerator::cbz(rt, skip);
  CodeGenerator::b(label);
  L(skip);
}

void A64Emitter::tbz(const Xbyak_aarch64::WReg& rt, uint32_t imm,
                     const Xbyak_aarch64::Label& label) {
  Xbyak_aarch64::Label skip;
  CodeGenerator::tbnz(rt, imm, skip);
  CodeGenerator::b(label);
  L(skip);
}

void A64Emitter::tbz(const Xbyak_aarch64::XReg& rt, uint32_t imm,
                     const Xbyak_aarch64::Label& label) {
  Xbyak_aarch64::Label skip;
  CodeGenerator::tbnz(rt, imm, skip);
  CodeGenerator::b(label);
  L(skip);
}

void A64Emitter::tbnz(const Xbyak_aarch64::WReg& rt, uint32_t imm,
                      const Xbyak_aarch64::Label& label) {
  Xbyak_aarch64::Label skip;
  CodeGenerator::tbz(rt, imm, skip);
  CodeGenerator::b(label);
  L(skip);
}

void A64Emitter::tbnz(const Xbyak_aarch64::XReg& rt, uint32_t imm,
                      const Xbyak_aarch64::Label& label) {
  Xbyak_aarch64::Label skip;
  CodeGenerator::tbz(rt, imm, skip);
  CodeGenerator::b(label);
  L(skip);
}

void A64Emitter::UnimplementedInstr(const hir::Instr* i) {
  XELOGE("A64: Unimplemented HIR instruction: {}",
         hir::GetOpcodeName(i->GetOpcodeInfo()));
  DebugBreak();
}

void A64Emitter::Call(const hir::Instr* instr, GuestFunction* function) {
  assert_not_null(function);
  ForgetFpcrMode();
  auto fn = static_cast<A64Function*>(function);

  // Calls to dynamic code look the target up every time, since a direct call
  // would keep reaching the old copy once the code is rewritten.
  const bool rewritable = processor()->dynamic_code_enabled() &&
                          code_cache_->has_indirection_table() &&
                          !code_cache_->HasIndirectionSlot(function->address());
  if (cvars::a64_direct_calls && !rewritable &&
      code_cache_->has_indirection_table() &&
      code_cache_->HasIndirectionSlot(function->address())) {
    const uint32_t target_guest_addr = function->address();
    const bool is_tail = (instr->flags & hir::CALL_TAIL) != 0;
    const bool encoded_indirection = code_cache_->encoded_indirection();
    if (!is_tail) {
      const uint64_t patch_thunk =
          reinterpret_cast<uint64_t>(backend()->patch_direct_call_thunk());
      const size_t call_idx = pending_direct_calls_.size();
      Label& tail_stub_label = AddToTail(
          [call_idx, target_guest_addr, patch_thunk, encoded_indirection](
              A64Emitter& e, Label&) {
            e.SetPendingDirectCallStubOffset(call_idx, e.getSize());
            Label do_patch;
            // Entry instruction is patched to NOP if target is > +-128 MiB so
            // subsequent calls bypass PatchDirectCallThunk via indirection.
            e.CodeGenerator::b(do_patch);

            e.mov(e.w16, target_guest_addr);
            if (!encoded_indirection) {
              e.ldr(e.w9, ptr(e.x16, static_cast<uint32_t>(0)));
            } else {
              Label external_target;
              Label indirection_ready;

              e.ldr(e.x14,
                    ptr(e.x19, static_cast<uint32_t>(offsetof(
                                   A64BackendContext,
                                   indirection_table_base_bias))));
              e.add(e.x14, e.x14, e.w16, UXTW);
              e.ldr(e.w9, ptr(e.x14, static_cast<uint32_t>(0)));
              e.tbnz(e.w9, 31, external_target);

              e.ldr(e.x14,
                    ptr(e.x19, static_cast<uint32_t>(offsetof(
                                   A64BackendContext,
                                   code_cache_execute_base))));
              e.add(e.x9, e.x14, e.w9, UXTW);
              e.CodeGenerator::b(indirection_ready);

              e.L(external_target);
              e.and_(e.w15, e.w9, A64CodeCache::kIndirectionExternalIndexMask);
              e.ldr(e.x14,
                    ptr(e.x19, static_cast<uint32_t>(offsetof(
                                   A64BackendContext,
                                   external_indirection_table_base))));
              e.lsl(e.x15, e.x15, 3);
              e.add(e.x14, e.x14, e.x15);
              e.ldr(e.x9, ptr(e.x14, static_cast<uint32_t>(0)));

              e.L(indirection_ready);
            }
            e.br(e.x9);

            e.L(do_patch);
            e.mov(e.w16, target_guest_addr);
            e.mov(e.x9, patch_thunk);
            e.br(e.x9);
          });
      ldr(x0, ptr(sp, static_cast<uint32_t>(StackLayout::GUEST_CALL_RET_ADDR)));
      const size_t branch_code_offset = getSize();
      bl(tail_stub_label);
      synchronize_stack_on_next_instruction_ = true;
      pending_direct_calls_.push_back(
          {branch_code_offset, SIZE_MAX, target_guest_addr, function, false});
    } else {
      Label& tail_stub_label = AddToTail(
          [target_guest_addr, encoded_indirection](A64Emitter& e, Label&) {
            e.mov(e.w16, target_guest_addr);
            if (!encoded_indirection) {
              e.ldr(e.w9, ptr(e.x16, static_cast<uint32_t>(0)));
            } else {
              Label external_target;
              Label indirection_ready;

              e.ldr(e.x14,
                    ptr(e.x19, static_cast<uint32_t>(offsetof(
                                   A64BackendContext,
                                   indirection_table_base_bias))));
              e.add(e.x14, e.x14, e.w16, UXTW);
              e.ldr(e.w9, ptr(e.x14, static_cast<uint32_t>(0)));
              e.tbnz(e.w9, 31, external_target);

              e.ldr(e.x14,
                    ptr(e.x19, static_cast<uint32_t>(offsetof(
                                   A64BackendContext,
                                   code_cache_execute_base))));
              e.add(e.x9, e.x14, e.w9, UXTW);
              e.CodeGenerator::b(indirection_ready);

              e.L(external_target);
              e.and_(e.w15, e.w9, A64CodeCache::kIndirectionExternalIndexMask);
              e.ldr(e.x14,
                    ptr(e.x19, static_cast<uint32_t>(offsetof(
                                   A64BackendContext,
                                   external_indirection_table_base))));
              e.lsl(e.x15, e.x15, 3);
              e.add(e.x14, e.x14, e.x15);
              e.ldr(e.x9, ptr(e.x14, static_cast<uint32_t>(0)));

              e.L(indirection_ready);
            }
            e.br(e.x9);
          });
      PopStackpoint();
      ldr(x0, ptr(sp, static_cast<uint32_t>(StackLayout::GUEST_RET_ADDR)));
      ldr(x30, ptr(sp, static_cast<uint32_t>(StackLayout::HOST_RET_ADDR)));
      if (stack_size() <= 4095) {
        add(sp, sp, static_cast<uint32_t>(stack_size()));
      } else {
        mov(x17, static_cast<uint64_t>(stack_size()));
        add(sp, sp, x17, UXTX);
      }
      const size_t branch_code_offset = getSize();
      b(tail_stub_label);
      pending_direct_calls_.push_back(
          {branch_code_offset, SIZE_MAX, target_guest_addr, function, true});
    }
    return;
  }

  if (fn->machine_code() && !rewritable) {
    // Direct call — function is already compiled.
    mov(x9, reinterpret_cast<uint64_t>(fn->machine_code()));
    if (!(instr->flags & hir::CALL_TAIL)) {
      // Pass the next call's guest return address in x0.
      ldr(x0, ptr(sp, static_cast<uint32_t>(StackLayout::GUEST_CALL_RET_ADDR)));
      blr(x9);
      synchronize_stack_on_next_instruction_ = true;
    } else {
      // Tail call: pass our return address to the callee.
      PopStackpoint();
      ldr(x0, ptr(sp, static_cast<uint32_t>(StackLayout::GUEST_RET_ADDR)));
      ldr(x30, ptr(sp, static_cast<uint32_t>(StackLayout::HOST_RET_ADDR)));
      if (stack_size() <= 4095) {
        add(sp, sp, static_cast<uint32_t>(stack_size()));
      } else {
        mov(x17, static_cast<uint64_t>(stack_size()));
        add(sp, sp, x17, UXTX);
      }
      br(x9);
    }
    return;
  }

  if (code_cache_->has_indirection_table() &&
      code_cache_->HasIndirectionSlot(function->address())) {
    // Must leave the guest address in w16 for the resolve thunk to read.
    mov(w16, function->address());
    if (!code_cache_->encoded_indirection()) {
      // Fast path: table mapped at host VA == guest addr; slot holds raw
      // 32-bit host target.
      ldr(w9, ptr(x16, static_cast<uint32_t>(0)));
    } else {
      // Encoded path: see A64CodeCache for the entry format.
      Label external_target;
      Label indirection_ready;

      ldr(x14, ptr(x19, static_cast<uint32_t>(offsetof(
                            A64BackendContext, indirection_table_base_bias))));
      add(x14, x14, w16, UXTW);
      ldr(w9, ptr(x14, static_cast<uint32_t>(0)));
      tbnz(w9, 31, external_target);

      // Internal: rel32 from code cache base.
      ldr(x14, ptr(x19, static_cast<uint32_t>(offsetof(
                            A64BackendContext, code_cache_execute_base))));
      add(x9, x14, w9, UXTW);
      b(indirection_ready);

      // External: tagged index into the side table.
      L(external_target);
      and_(w15, w9, A64CodeCache::kIndirectionExternalIndexMask);
      ldr(x14,
          ptr(x19, static_cast<uint32_t>(offsetof(
                       A64BackendContext, external_indirection_table_base))));
      lsl(x15, x15, 3);
      add(x14, x14, x15);
      ldr(x9, ptr(x14, static_cast<uint32_t>(0)));

      L(indirection_ready);
    }
  } else if (code_cache_->has_indirection_table()) {
    mov(w16, function->address());
    EmitDynamicCallLookup((instr->flags & hir::CALL_TAIL) != 0, true);
  } else {
    // No indirection table: resolve at runtime.
    mov(x0, x20);  // context
    mov(x1, static_cast<uint64_t>(function->address()));
    mov(x9, reinterpret_cast<uint64_t>(&ResolveFunction));
    blr(x9);
    mov(x9, x0);  // resolved address in x9
  }

  if (instr->flags & hir::CALL_TAIL) {
    PopStackpoint();
    ldr(x0, ptr(sp, static_cast<uint32_t>(StackLayout::GUEST_RET_ADDR)));
    ldr(x30, ptr(sp, static_cast<uint32_t>(StackLayout::HOST_RET_ADDR)));
    if (stack_size() <= 4095) {
      add(sp, sp, static_cast<uint32_t>(stack_size()));
    } else {
      mov(x17, static_cast<uint64_t>(stack_size()));
      add(sp, sp, x17, UXTX);
    }
    br(x9);
  } else {
    ldr(x0, ptr(sp, static_cast<uint32_t>(StackLayout::GUEST_CALL_RET_ADDR)));
    blr(x9);
    synchronize_stack_on_next_instruction_ = true;
  }
}

void A64Emitter::CallIndirect(const hir::Instr* instr, int reg_index) {
  ForgetFpcrMode();
  auto target_w = WReg(reg_index);

  // Check if this is a possible return (e.g., PPC blr).
  if (instr->flags & hir::CALL_POSSIBLE_RETURN) {
    // Compare target guest address with our function's return address.
    ldr(w0, ptr(sp, static_cast<uint32_t>(StackLayout::GUEST_RET_ADDR)));
    cmp(target_w, w0);
    b(EQ, epilog_label());
  }

  // Load host code address from indirection table.
  if (code_cache_->has_indirection_table()) {
    // Must leave the guest address in w16 for the resolve thunk to read.
    if (target_w.getIdx() != w16.getIdx()) {
      mov(w16, target_w);
    }
    // A target without an indirection slot goes through the dynamic call cache.
    // Only dynamic code has such targets, and the check costs code cache space
    // at every indirect call site, so titles without it do not pay for it.
    Label* target_ready = nullptr;
    if (processor()->dynamic_code_enabled()) {
      target_ready = &NewCachedLabel();
      // TODO(has207): while the base is 0x80000000, eor with it gives the same
      // branch outcome in one instruction. Needs a static_assert on the base.
      mov(w14, code_cache_->indirection_guest_base());
      sub(w14, w16, w14);
      mov(w15, code_cache_->indirection_guest_size());
      cmp(w14, w15);
      const bool tail_call = (instr->flags & hir::CALL_TAIL) != 0;
      const bool possible_return =
          tail_call && (instr->flags & hir::CALL_POSSIBLE_RETURN) &&
          cvars::a64_enable_host_guest_stack_synchronization;
      Label& resolve_natively = AddToTail(
          [target_ready, tail_call, possible_return](A64Emitter& e, Label&) {
            if (possible_return) {
              e.str(e.wzr, e.BackendCtxPtr(offsetof(A64BackendContext,
                                                    dynamic_target_in_body)));
            }
            e.EmitDynamicCallLookup(tail_call);
            if (possible_return) {
              e.EmitDropCallingFrame();
            }
            e.b(*target_ready);
          });
      b(HS, resolve_natively);
    }
    // Without dynamic code, a target the table doesn't cover goes to the
    // resolve thunk, since its slot would be read from outside the table.
    // Returns, the most common sites, aren't checked.
    Label* slot_ready = nullptr;
    if (!processor()->dynamic_code_enabled() &&
        !(instr->flags & hir::CALL_POSSIBLE_RETURN)) {
      slot_ready = &NewCachedLabel();
      const uint64_t resolve_thunk =
          reinterpret_cast<uint64_t>(backend()->resolve_function_thunk());
      Label& uncovered =
          AddToTail([slot_ready, resolve_thunk](A64Emitter& e, Label&) {
            e.mov(e.x9, resolve_thunk);
            e.b(*slot_ready);
          });
      if (code_cache_->indirection_guest_base() == 0x80000000u &&
          code_cache_->indirection_guest_size() == 0x1FFFFFFFu) {
        // The table covers the addresses whose top three bits are 100.
        eor(w14, w16, 0x80000000u);
        tst(w14, 0xE0000000u);
        b(NE, uncovered);
      } else {
        mov(w14, code_cache_->indirection_guest_base());
        sub(w14, w16, w14);
        mov(w15, code_cache_->indirection_guest_size());
        cmp(w14, w15);
        b(HS, uncovered);
      }
    }
    if (!code_cache_->encoded_indirection()) {
      // Fast path: table mapped at host VA == guest addr; slot holds raw
      // 32-bit host target.
      ldr(w9, ptr(x16, static_cast<uint32_t>(0)));
    } else {
      // Encoded path: see A64CodeCache for the entry format.
      Label external_target;
      Label indirection_ready;

      ldr(x14, ptr(x19, static_cast<uint32_t>(offsetof(
                            A64BackendContext, indirection_table_base_bias))));
      add(x14, x14, w16, UXTW);
      ldr(w9, ptr(x14, static_cast<uint32_t>(0)));
      tbnz(w9, 31, external_target);

      // Internal: rel32 from code cache base.
      ldr(x14, ptr(x19, static_cast<uint32_t>(offsetof(
                            A64BackendContext, code_cache_execute_base))));
      add(x9, x14, w9, UXTW);
      b(indirection_ready);

      // External: tagged index into the side table.
      L(external_target);
      and_(w15, w9, A64CodeCache::kIndirectionExternalIndexMask);
      ldr(x14,
          ptr(x19, static_cast<uint32_t>(offsetof(
                       A64BackendContext, external_indirection_table_base))));
      lsl(x15, x15, 3);
      add(x14, x14, x15);
      ldr(x9, ptr(x14, static_cast<uint32_t>(0)));

      L(indirection_ready);
    }
    if (slot_ready) {
      L(*slot_ready);
    }
    if (target_ready) {
      L(*target_ready);
    }
  } else {
    // No indirection table: resolve at runtime.
    mov(w16, target_w);
    mov(x0, x20);  // context
    mov(x1, x16);  // guest address
    mov(x9, reinterpret_cast<uint64_t>(&ResolveFunction));
    blr(x9);
    mov(x9, x0);  // resolved address
  }

  if (instr->flags & hir::CALL_TAIL) {
    // Tail call: pass our return address to the callee.
    PopStackpoint();
    ldr(x0, ptr(sp, static_cast<uint32_t>(StackLayout::GUEST_RET_ADDR)));
    ldr(x30, ptr(sp, static_cast<uint32_t>(StackLayout::HOST_RET_ADDR)));
    if (stack_size() <= 4095) {
      add(sp, sp, static_cast<uint32_t>(stack_size()));
    } else {
      mov(x17, static_cast<uint64_t>(stack_size()));
      add(sp, sp, x17, UXTX);
    }
    br(x9);
  } else {
    // Regular call: pass the next call's return address.
    ldr(x0, ptr(sp, static_cast<uint32_t>(StackLayout::GUEST_CALL_RET_ADDR)));
    blr(x9);
    synchronize_stack_on_next_instruction_ = true;
  }
}

// Whether a float or vector value defined before `call` is read after it.
// Registers hold values only within a block, so only this block can. False
// for CALL_EXTERN as it ends its block; this guards one that wouldn't.
static bool FpValueLiveAcross(const hir::Instr* call) {
  const auto is_fp = [](const hir::Value* v) {
    return v->type == hir::FLOAT32_TYPE || v->type == hir::FLOAT64_TYPE ||
           v->type == hir::VEC128_TYPE;
  };
  std::unordered_set<const hir::Value*> before;
  for (const hir::Instr* i = call->block->instr_head; i != call; i = i->next) {
    if (i->dest && is_fp(i->dest)) {
      before.insert(i->dest);
    }
  }
  if (before.empty()) {
    return false;
  }
  bool live = false;
  for (hir::Instr* i = call->next; i && !live; i = i->next) {
    i->VisitValueOperands(
        [&](hir::Value* v, uint32_t) { live |= before.count(v) != 0; });
  }
  return live;
}

void A64Emitter::CallExtern(const hir::Instr* instr, const Function* function) {
  ForgetFpcrMode();
  // The no-vec thunk skips the q4-q31 save, which is right as long as no value
  // lives in a vector register across the call (the thunk addresses are in
  // the backend context: one ldr each).
  const uint32_t thunk_offset = static_cast<uint32_t>(
      FpValueLiveAcross(instr)
          ? offsetof(A64BackendContext, guest_to_host_thunk_address)
          : offsetof(A64BackendContext, guest_to_host_thunk_no_vec_address));
  bool undefined = true;
  if (function->behavior() == Function::Behavior::kBuiltin) {
    auto builtin_function = static_cast<const BuiltinFunction*>(function);
    if (builtin_function->handler()) {
      undefined = false;
      // GuestToHostThunk: x0=target, x1=arg0, x2=arg1
      // Thunk rearranges to: x0=context, x1=arg0, x2=arg1, calls target
      mov(x0, reinterpret_cast<uint64_t>(builtin_function->handler()));
      mov(x1, reinterpret_cast<uint64_t>(builtin_function->arg0()));
      mov(x2, reinterpret_cast<uint64_t>(builtin_function->arg1()));
      ldr(x9, ptr(x19, thunk_offset));
      blr(x9);
    }
  } else if (function->behavior() == Function::Behavior::kExtern) {
    auto extern_function = static_cast<const GuestFunction*>(function);
    if (extern_function->extern_handler()) {
      undefined = false;
      if (TryEmitInlineHleSync(extern_function, thunk_offset)) {
        return;
      }
      // GuestToHostThunk: x0=target, x1=arg0
      mov(x0, reinterpret_cast<uint64_t>(extern_function->extern_handler()));
      ldr(x1, ptr(GetContextReg(), static_cast<int32_t>(offsetof(
                                       ppc::PPCContext, kernel_state))));
      ldr(x9, ptr(x19, thunk_offset));
      blr(x9);
    }
  }
  if (undefined) {
    // Set arg0 = function pointer, then call UndefinedCallExtern via thunk.
    mov(x1, reinterpret_cast<uint64_t>(function));
    CallNativeSafe(reinterpret_cast<void*>(&UndefinedCallExtern));
  }
}

bool A64Emitter::TryEmitInlineHleSync(const GuestFunction* extern_function,
                                      uint32_t thunk_offset) {
  const Export* exp = extern_function->export_data();
  if (!exp || !exp->name) {
    return false;
  }
  ExportResolver* resolver = processor_ ? processor_->export_resolver() : nullptr;
  const InlineHleSyncKind kind = ClassifyInlineHleSync(exp->name);
  const bool census_active = resolver && resolver->hle_census_active();
  const bool log_hf =
      resolver && resolver->log_high_frequency_kernel_calls();
  const bool logging_active = IsInlineHleSyncLoggingActive(
      exp->tags, ExportTag::kLog, ExportTag::kLogResult,
      ExportTag::kHighFrequency, log_hf);
  const ExportTrampoline wake_fn =
      resolver ? resolver->rtl_leave_cs_wake_trampoline() : nullptr;
  if (!ShouldEmitInlineHleSync(cvars::a64_inline_hle_sync, census_active,
                               logging_active, wake_fn != nullptr, kind)) {
    return false;
  }
  static std::atomic<bool> logged_first_inline_hle_sync{false};
  if (!logged_first_inline_hle_sync.exchange(true, std::memory_order_relaxed)) {
    XELOGI("A64Emitter: inline HLE sync fast-path active (first export={})",
           exp->name);
  }

  constexpr int32_t r3_off = static_cast<int32_t>(offsetof(ppc::PPCContext, r[3]));
  constexpr int32_t r4_off = static_cast<int32_t>(offsetof(ppc::PPCContext, r[4]));
  constexpr int32_t r13_off = static_cast<int32_t>(offsetof(ppc::PPCContext, r[13]));
  const bool gran_shift_guard = xe::memory::allocation_granularity() > 0x1000;

  Label slow;
  Label done;

  auto emit_guest_ptr_guard = [&](const WReg& w_addr, bool require_align4,
                                  Label& target_slow) {
    cbz_near(w_addr, target_slow);
    if (require_align4) {
      tst(w_addr, 3);
      b_near(NE, target_slow);
    }
    if (gran_shift_guard) {
      mov(w17, kGuestPhysicalShiftBoundary);
      cmp(w_addr, w17);
      b_near(HS, target_slow);
    }
  };

  auto emit_fallback_call = [&](void* target_fn) {
    mov(x0, reinterpret_cast<uint64_t>(target_fn));
    ldr(x1, ptr(GetContextReg(), static_cast<int32_t>(offsetof(
                                     ppc::PPCContext, kernel_state))));
    ldr(x9, ptr(x19, thunk_offset));
    blr(x9);
  };

  switch (kind) {
    case InlineHleSyncKind::kKeRaiseIrqlToDpcLevel: {
      ldr(w1, ptr(x20, r13_off));
      emit_guest_ptr_guard(w1, false, slow);
      add(x3, x21, w1, UXTW);
      ldrb(w0, ptr(x3, kKpcrCurrentIrqlOffset));
      cmp(w0, 2);
      b_near(HI, slow);
      mov(w2, 2);
      strb(w2, ptr(x3, kKpcrCurrentIrqlOffset));
      str(x0, ptr(x20, r3_off));
      b(done);
      break;
    }
    case InlineHleSyncKind::kKfLowerIrql: {
      ldrb(w0, ptr(x20, r3_off));
      ldr(w1, ptr(x20, r13_off));
      emit_guest_ptr_guard(w1, false, slow);
      add(x3, x21, w1, UXTW);
      ldrb(w2, ptr(x3, kKpcrCurrentIrqlOffset));
      cmp(w0, w2);
      b_near(HI, slow);
      strb(w0, ptr(x3, kKpcrCurrentIrqlOffset));
      b(done);
      break;
    }
    case InlineHleSyncKind::kRtlEnterCriticalSection:
    case InlineHleSyncKind::kRtlTryEnterCriticalSection: {
      const bool is_try =
          kind == InlineHleSyncKind::kRtlTryEnterCriticalSection;
      Label cas_retry;
      Label check_recursive;
      Label rec_retry;
      Label contended;

      ldr(w0, ptr(x20, r3_off));
      emit_guest_ptr_guard(w0, true, slow);
      ldr(w1, ptr(x20, r13_off));
      emit_guest_ptr_guard(w1, true, slow);
      add(x2, x21, w0, UXTW);
      add(x3, x21, w1, UXTW);
      ldr(w4, ptr(x3, kKpcrCurrentThreadOffset));
      cbz_near(w4, slow);

      add(x14, x2, kRtlCsLockCountOffset);
      mov(w15, static_cast<uint32_t>(-1));
      L(cas_retry);
      ldaxr(w16, ptr(x14));
      cmp(w16, w15);
      b_near(NE, check_recursive);
      stlxr(w17, wzr, ptr(x14));
      cbnz_near(w17, cas_retry);

      mov(w16, 0x01000000u);
      str(w16, ptr(x2, kRtlCsRecursionCountOffset));
      str(w4, ptr(x2, kRtlCsOwningThreadOffset));
      if (is_try) {
        mov(x16, 1);
        str(x16, ptr(x20, r3_off));
      }
      b(done);

      L(check_recursive);
      clrex(15);
      ldr(w16, ptr(x2, kRtlCsOwningThreadOffset));
      cmp(w16, w4);
      b_near(NE, is_try ? contended : slow);

      L(rec_retry);
      ldaxr(w16, ptr(x14));
      add(w16, w16, 1);
      stlxr(w17, w16, ptr(x14));
      cbnz_near(w17, rec_retry);
      ldr(w16, ptr(x2, kRtlCsRecursionCountOffset));
      rev(w16, w16);
      add(w16, w16, 1);
      rev(w16, w16);
      str(w16, ptr(x2, kRtlCsRecursionCountOffset));
      if (is_try) {
        mov(x16, 1);
        str(x16, ptr(x20, r3_off));
      }
      b(done);

      if (is_try) {
        L(contended);
        str(xzr, ptr(x20, r3_off));
        b(done);
      }
      break;
    }
    case InlineHleSyncKind::kRtlLeaveCriticalSection: {
      Label recursive_leave;
      Label dec_retry;
      Label rec_dec_retry;

      ldr(w0, ptr(x20, r3_off));
      emit_guest_ptr_guard(w0, true, slow);
      ldr(w1, ptr(x20, r13_off));
      emit_guest_ptr_guard(w1, true, slow);
      add(x2, x21, w0, UXTW);
      add(x3, x21, w1, UXTW);
      ldr(w4, ptr(x3, kKpcrCurrentThreadOffset));
      cbz_near(w4, slow);

      ldr(w15, ptr(x2, kRtlCsOwningThreadOffset));
      cmp(w15, w4);
      b_near(NE, slow);

      ldr(w15, ptr(x2, kRtlCsRecursionCountOffset));
      add(x14, x2, kRtlCsLockCountOffset);
      mov(w16, 0x01000000u);
      cmp(w15, w16);
      b_near(NE, recursive_leave);

      str(wzr, ptr(x2, kRtlCsRecursionCountOffset));
      str(wzr, ptr(x2, kRtlCsOwningThreadOffset));
      L(dec_retry);
      ldaxr(w15, ptr(x14));
      sub(w15, w15, 1);
      stlxr(w17, w15, ptr(x14));
      cbnz_near(w17, dec_retry);
      cmn(w15, 1);
      b_near(EQ, done);

      emit_fallback_call(reinterpret_cast<void*>(wake_fn));
      b(done);

      L(recursive_leave);
      rev(w15, w15);
      cmp(w15, 1);
      b_near(LE, slow);
      sub(w15, w15, 1);
      rev(w15, w15);
      str(w15, ptr(x2, kRtlCsRecursionCountOffset));
      L(rec_dec_retry);
      ldaxr(w15, ptr(x14));
      sub(w15, w15, 1);
      stlxr(w17, w15, ptr(x14));
      cbnz_near(w17, rec_dec_retry);
      b(done);
      break;
    }
    case InlineHleSyncKind::kKeAcquireSpinLockAtRaisedIrql:
    case InlineHleSyncKind::kKfAcquireSpinLock: {
      const bool change_irql = kind == InlineHleSyncKind::kKfAcquireSpinLock;
      Label spin_retry;
      Label spin_contended;

      ldr(w0, ptr(x20, r3_off));
      emit_guest_ptr_guard(w0, true, slow);
      ldr(w1, ptr(x20, r13_off));
      emit_guest_ptr_guard(w1, false, slow);
      add(x2, x21, w0, UXTW);
      if (change_irql) {
        add(x3, x21, w1, UXTW);
        ldrb(w4, ptr(x3, kKpcrCurrentIrqlOffset));
        cmp(w4, 2);
        b_near(HI, slow);
      }
      rev(w14, w1);
      L(spin_retry);
      ldaxr(w15, ptr(x2));
      cbnz_near(w15, spin_contended);
      stlxr(w17, w14, ptr(x2));
      cbnz_near(w17, spin_retry);

      if (change_irql) {
        mov(w16, 2);
        strb(w16, ptr(x3, kKpcrCurrentIrqlOffset));
        str(x4, ptr(x20, r3_off));
      }
      b(done);

      L(spin_contended);
      clrex(15);
      break;
    }
    case InlineHleSyncKind::kKeReleaseSpinLockFromRaisedIrql:
    case InlineHleSyncKind::kKfReleaseSpinLock: {
      const bool change_irql = kind == InlineHleSyncKind::kKfReleaseSpinLock;
      Label skip_irql;

      ldr(w0, ptr(x20, r3_off));
      emit_guest_ptr_guard(w0, true, slow);
      ldr(w1, ptr(x20, r13_off));
      emit_guest_ptr_guard(w1, false, slow);
      add(x2, x21, w0, UXTW);
      ldr(w15, ptr(x2));
      rev(w16, w1);
      cmp(w15, w16);
      b_near(NE, slow);

      if (change_irql) {
        ldr(w4, ptr(x20, r4_off));
        cmp(w4, 2);
        b_near(HS, skip_irql);
        add(x3, x21, w1, UXTW);
        ldrb(w14, ptr(x3, kKpcrCurrentIrqlOffset));
        cmp(w4, w14);
        b_near(HI, slow);
        stlr(wzr, ptr(x2));
        strb(w4, ptr(x3, kKpcrCurrentIrqlOffset));
        b(done);
        L(skip_irql);
      }
      stlr(wzr, ptr(x2));
      b(done);
      break;
    }
    case InlineHleSyncKind::kNone:
      return false;
  }

  L(slow);
  emit_fallback_call(reinterpret_cast<void*>(extern_function->extern_handler()));
  L(done);
  return true;
}

void A64Emitter::EmitDynamicCallLookup(bool tail, bool direct) {
  const bool direct_call = direct && !tail;
  Label miss;
  Label done;
  Label check_kind;
  ldr(x14, ptr(x19, static_cast<uint32_t>(
                        offsetof(A64BackendContext, dynamic_call_cache))));
  cbz(x14, miss);
  // Mirrors DynamicCallCacheIndex, which writes the entries this probes.
  lsr(w15, w16, 2);
  eor(w15, w15, w16, LSR, 14);
  and_(w15, w15, kA64DynamicCallCacheSize - 1);
  add(x14, x14, w15, UXTW, 4);
  ldr(w15, ptr(x14, static_cast<uint32_t>(
                        offsetof(A64DynamicCallCacheEntry, guest_address))));
  cmp(w15, w16);
  b(NE, miss);
  if (!direct_call) {
    static_assert(kA64DynamicCallAny == 0);
    ldr(w15, ptr(x14, static_cast<uint32_t>(
                          offsetof(A64DynamicCallCacheEntry, kind))));
    cbnz(w15, check_kind);
  }
  ldr(x9, ptr(x14, static_cast<uint32_t>(
                       offsetof(A64DynamicCallCacheEntry, host_address))));
  // An entry the cache was initialized with holds no address.
  cbz(x9, miss);
  b(done);
  if (!direct_call) {
    L(check_kind);
    cmp(w15, uint32_t(kA64DynamicCallReturnSite));
    b(NE, miss);
    mov(w1, w16);
    CallNativeSafe(reinterpret_cast<void*>(tail ? ResolveCachedReturnSiteTail
                                                : ResolveCachedReturnSiteCall));
    mov(x9, x0);
    b(done);
  }

  L(miss);
  // The host call can clobber x16, so hand the address over as its argument.
  mov(w1, w16);
  CallNativeSafe(reinterpret_cast<void*>(direct_call ? ResolveDirectDynamicCall
                                         : tail      ? ResolveDynamicTailCall
                                                     : ResolveDynamicCall));
  mov(x9, x0);

  L(done);
}

void A64Emitter::EmitDropCallingFrame() {
  // A blr that returns into neither this frame nor its caller enters the target
  // in the caller's frame, as if the caller had tail-called it.
  Label keep_frame;
  // Only a function's entry can take a frame.
  ldr(w10, BackendCtxPtr(offsetof(A64BackendContext, dynamic_target_in_body)));
  cbnz(w10, keep_frame);
  ldr(w10,
      BackendCtxPtr(offsetof(A64BackendContext, current_stackpoint_depth)));
  // This frame, its caller, and the frame the caller returns into.
  cmp(w10, 3);
  b(LO, keep_frame);
  static_assert(sizeof(A64BackendStackpoint) == 16);
  static_assert(offsetof(A64BackendStackpoint, host_stack_) == 0);
  // x11 = &stackpoints[depth - 3], the frame the caller returns into. The
  // caller's record follows it.
  ldr(x11, BackendCtxPtr(offsetof(A64BackendContext, stackpoints)));
  sub(w12, w10, 3);
  add(x11, x11, w12, UXTW, 4);
  ldr(x12, ptr(x11, static_cast<uint32_t>(sizeof(A64BackendStackpoint))));
  // The caller called this frame directly, so its stack pointer is just above
  // this frame.
  if (stack_size() <= 4095) {
    add(x13, sp, static_cast<uint32_t>(stack_size()));
  } else {
    mov(x13, static_cast<uint64_t>(stack_size()));
    add(x13, sp, x13, UXTX);
  }
  cmp(x12, x13);
  b(NE, keep_frame);
  ldr(x13, ptr(x11, static_cast<uint32_t>(0)));
  cmp(x13, x12);
  b(LS, keep_frame);
  // A caller entered from host code returns into host code.
  ldr(w14, ptr(x12, static_cast<uint32_t>(StackLayout::GUEST_RET_ADDR)));
  mov(w15, kHostReturnAddress);
  cmp(w14, w15);
  b(EQ, keep_frame);
  // Pass the caller's return addresses over and return where it would.
  ldr(x0, ptr(x12, static_cast<uint32_t>(StackLayout::GUEST_RET_ADDR)));
  ldr(x30, ptr(x12, static_cast<uint32_t>(StackLayout::HOST_RET_ADDR)));
  mov(sp, x13);
  sub(w10, w10, 2);
  str(w10,
      BackendCtxPtr(offsetof(A64BackendContext, current_stackpoint_depth)));
  br(x9);
  L(keep_frame);
}

void A64Emitter::CallNative(void* fn) { CallNativeSafe(fn); }

void A64Emitter::CallNativeSafe(void* fn) {
  ForgetFpcrMode();
  // GuestToHostThunk: x0=target function, x1/x2=args (set by caller).
  // The thunk rearranges: saves x0 in x9, sets x0=context, calls x9.
  mov(x0, reinterpret_cast<uint64_t>(fn));
  ldr(x9, ptr(x19, static_cast<uint32_t>(offsetof(
                       A64BackendContext, guest_to_host_thunk_address))));
  blr(x9);
}

void A64Emitter::SetReturnAddress(uint64_t value) {
  mov(x0, value);
  str(x0, ptr(sp, static_cast<uint32_t>(StackLayout::GUEST_CALL_RET_ADDR)));
}

void A64Emitter::ReloadMembase() {
  // Reload x21 from context->virtual_membase.
  ldr(x21, ptr(x20, static_cast<int32_t>(
                        offsetof(ppc::PPCContext, virtual_membase))));
}

bool A64Emitter::ChangeFpcrMode(FPCRMode new_mode, bool already_set) {
  if (fpcr_mode_ == new_mode) {
    return false;
  }
  fpcr_mode_ = new_mode;
  if (!already_set) {
    // Load the pre-computed FPCR value from the backend context.
    // This avoids an expensive MRS + read-modify-write cycle.
    auto bctx = GetBackendCtxReg();
    if (new_mode == FPCRMode::Vmx) {
      ldr(w0, Xbyak_aarch64::ptr(bctx, static_cast<uint32_t>(offsetof(
                                           A64BackendContext, fpcr_vmx))));
    } else {
      ldr(w0, Xbyak_aarch64::ptr(bctx, static_cast<uint32_t>(offsetof(
                                           A64BackendContext, fpcr_fpu))));
    }
    msr(3, 3, 4, 4, 0, x0);  // msr FPCR, x0
  }
  return true;
}

Label& A64Emitter::AddToTail(TailEmitCallback callback, uint32_t alignment) {
  TailEmitter tail;
  tail.alignment = alignment;
  tail.func = std::move(callback);
  tail_code_.push_back(std::move(tail));
  return tail_code_.back().label;
}

Label& A64Emitter::NewCachedLabel() {
  auto* label = new Label();
  label_cache_.push_back(label);
  return *label;
}

uint32_t A64Emitter::MapReg(const hir::Value* v, const uint32_t* map, int count,
                            const char* set_name) {
  // reg.index is a signed int32 and is -1 while unassigned, so the unsigned
  // compare catches both "never allocated" and "past the end of the set".
  const uint32_t index = static_cast<uint32_t>(v->reg.index);
  if (index >= static_cast<uint32_t>(count)) {
    XELOGE(
        "A64: value v{} (type {}, def opcode {}) has no {} register assignment "
        "(index {} of {}); codegen would emit a bogus register",
        v->ordinal, static_cast<uint32_t>(v->type),
        v->def ? hir::GetOpcodeName(v->def->GetOpcodeInfo()) : "<none>",
        set_name, index, count);
    assert_always("register allocation missed a value");
    // Clamp so the assembler still produces a decodable instruction.
    return map[0];
  }
  return map[index];
}

void A64Emitter::EmitPreemptCheck(uint32_t guest_address) {
  // Only safe at a block head, where the per-block register allocator leaves no
  // guest value live and ForgetFpcrMode has already run, so the unannounced
  // guest->host call cannot lose a register or desync the mode tracking.
  //
  // Tests the preempt flag other threads raise. The cold path clears it, a
  // deferred yield re-sets it.
  Label& after = NewCachedLabel();
  // ldrb/strb unsigned-offset encoding caps at 4095.
  static_assert(offsetof(ppc::PPCContext, preempt_requested) < 4096);
  const uint32_t flag_offset =
      static_cast<uint32_t>(offsetof(ppc::PPCContext, preempt_requested));
  Label& do_yield = AddToTail([&after, flag_offset](A64Emitter& e, Label&) {
    e.strb(e.wzr, ptr(e.x20, flag_offset));
    // Null until the scheduler starts, and a stale flag can reach here after
    // it shuts down, so check before calling.
    e.mov(e.x0,
          reinterpret_cast<uint64_t>(&xe::cpu::backend::preempt_yield_handler));
    e.ldr(e.x0, ptr(e.x0));
    e.cbz(e.x0, after);
    e.ldr(e.x9, ptr(e.x19, static_cast<uint32_t>(offsetof(
                               A64BackendContext, guest_to_host_thunk_address))));
    e.blr(e.x9);
    e.b(after);
  });
  if (cvars::log_safepoint_pc && guest_address) {
    // Diagnostic only: costs a materialize + store on every loop back-edge, so
    // it stays off unless a wedge is being chased.
    static_assert(offsetof(ppc::PPCContext, last_safepoint_pc) < 16384);
    mov(w17, guest_address);
    str(w17, ptr(x20, static_cast<uint32_t>(
                          offsetof(ppc::PPCContext, last_safepoint_pc))));
  }
  ldrb(w16, ptr(x20, flag_offset));
  cbnz(w16, do_yield);
  L(after);
}

Label& A64Emitter::GetLabel(uint32_t label_id) {
  auto it = label_map_.find(label_id);
  if (it != label_map_.end()) {
    return *it->second;
  }
  auto* label = new Label();
  label_map_[label_id] = label;
  return *label;
}

// Half of LDR (literal)'s +-1MB reach, leaving room for the item in flight.
static constexpr size_t kV128ConstPoolReach = 512 * 1024;

Label& A64Emitter::GetV128ConstLabel(const vec128_t& value) {
  for (auto& entry : v128_consts_) {
    if (entry.first == value) {
      return *entry.second;
    }
  }
  if (v128_consts_.empty()) {
    v128_consts_first_use_ = getSize();
  }
  auto* label = new Label();
  label_cache_.push_back(label);
  v128_consts_.emplace_back(value, label);
  return *label;
}

bool A64Emitter::FlushV128ConstPool(bool branch_over) {
  if (v128_consts_.empty()) {
    return true;
  }
  Label* skip = nullptr;
  if (branch_over) {
    skip = new Label();
    label_cache_.push_back(skip);
    b(*skip);
  }
  // Functions start 16-byte aligned in the code cache.
  while (getSize() % 16) {
    dd(0);
  }
  try {
    for (auto& entry : v128_consts_) {
      L(*entry.second);
      for (int word = 0; word < 4; ++word) {
        dd(entry.first.u32[word]);
      }
    }
  } catch (const Xbyak_aarch64::Error& e) {
    XELOGE(
        "A64: v128 literal pool out of range in guest function {:08X} "
        "({} constants, {} bytes since the first use): {}",
        current_guest_function_, v128_consts_.size(),
        getSize() - v128_consts_first_use_, e.what());
    return false;
  }
  v128_consts_.clear();
  if (skip) {
    L(*skip);
  }
  return true;
}

bool A64Emitter::MaybeFlushV128ConstPool() {
  if (v128_consts_.empty() ||
      getSize() - v128_consts_first_use_ < kV128ConstPoolReach) {
    return true;
  }
  return FlushV128ConstPool(true);
}

void A64Emitter::HandleStackpointOverflowError(ppc::PPCContext* context) {
  if (debugging::IsDebuggerAttached()) {
    debugging::Break();
  }
  xe::FatalError(
      "Overflowed stackpoints! Please report this error for this title to "
      "Xenia developers.");
}

void A64Emitter::PushStackpoint() {
  if (!cvars::a64_enable_host_guest_stack_synchronization ||
      !uses_stackpoints_) {
    return;
  }
  // x8 = stackpoints array, w9 = current depth
  ldr(x8, ptr(x19,
              static_cast<uint32_t>(offsetof(A64BackendContext, stackpoints))));
  ldr(w9, ptr(x19, static_cast<uint32_t>(
                       offsetof(A64BackendContext, current_stackpoint_depth))));

  // x8 += w9 * sizeof(A64BackendStackpoint) via scaled extended-register add.
  static_assert(sizeof(A64BackendStackpoint) == 16,
                "stackpoint indexing relies on a 16-byte element size");
  add(x8, x8, w9, UXTW, 4);

  // Store host SP.
  mov(x10, sp);
  str(x10, ptr(x8, static_cast<uint32_t>(
                       offsetof(A64BackendStackpoint, host_stack_))));
  // Store guest r1 (32-bit).
  ldr(w10, ptr(x20, static_cast<int32_t>(offsetof(ppc::PPCContext, r[1]))));
  str(w10, ptr(x8, static_cast<uint32_t>(
                       offsetof(A64BackendStackpoint, guest_stack_))));
  // Store guest LR (32-bit).
  ldr(w10, ptr(x20, static_cast<int32_t>(offsetof(ppc::PPCContext, lr))));
  str(w10, ptr(x8, static_cast<uint32_t>(
                       offsetof(A64BackendStackpoint, guest_return_address_))));

  // Increment depth.
  add(w9, w9, 1);
  str(w9, ptr(x19, static_cast<uint32_t>(
                       offsetof(A64BackendContext, current_stackpoint_depth))));

  // Check for overflow.
  mov(w10, static_cast<uint32_t>(cvars::a64_max_stackpoints));
  cmp(w9, w10);
  auto& overflow_label = AddToTail([](A64Emitter& e, Label& lbl) {
    e.CallNativeSafe(
        reinterpret_cast<void*>(A64Emitter::HandleStackpointOverflowError));
  });
  // overflow_label is emitted at the function end, which can exceed B.cond's
  // +/-1 MiB range in large functions - the b(cond) shadow takes care of that.
  b(GE, overflow_label);
}

void A64Emitter::PopStackpoint() {
  if (!cvars::a64_enable_host_guest_stack_synchronization ||
      !uses_stackpoints_) {
    return;
  }
  // Decrement current_stackpoint_depth.
  ldr(w8, ptr(x19, static_cast<uint32_t>(
                       offsetof(A64BackendContext, current_stackpoint_depth))));
  sub(w8, w8, 1);
  str(w8, ptr(x19, static_cast<uint32_t>(
                       offsetof(A64BackendContext, current_stackpoint_depth))));
}

void A64Emitter::EnsureSynchronizedGuestAndHostStack() {
  if (!cvars::a64_enable_host_guest_stack_synchronization) {
    return;
  }
  // ResolveFunction marks this when it returns a return-site address inside an
  // existing frame. The marker lives in backend context because native SP can
  // still point at a skipped frame here.
  auto& return_from_sync = NewCachedLabel();

  ldr(w16, ptr(x19, static_cast<uint32_t>(offsetof(
                        A64BackendContext, pending_stackpoint_sync_depth))));
  if (near_branches_) {
    // The tail is within the cbnz's reach, so within adr's (both +/-1 MiB):
    // the common path just falls through.
    auto& sync_label =
        AddToTail([&return_from_sync](A64Emitter& e, Label& lbl) {
          //   x8 = return address (where to resume after fixup)
          e.adr(e.x8, return_from_sync);
          e.mov(e.x10,
                reinterpret_cast<uint64_t>(
                    e.backend()->synchronize_guest_and_host_stack_helper()));
          e.br(e.x10);
        });
    cbnz(w16, sync_label);
    L(return_from_sync);
    return;
  }
  cbz(w16, return_from_sync);

  auto& sync_label = AddToTail([](A64Emitter& e, Label& lbl) {
    // x8 was set up in the body to point at return_from_sync; do that there
    // instead of here because adr's ±1 MiB range can't span body+tail in
    // large functions.
    //   x8 = return address (where to resume after fixup)
    e.mov(e.x10, reinterpret_cast<uint64_t>(
                     e.backend()->synchronize_guest_and_host_stack_helper()));
    e.br(e.x10);
  });
  adr(x8, return_from_sync);
  b(sync_label);

  L(return_from_sync);
}

void A64Emitter::EmitLoadPinnedSlots(const hir::LoopPromotionRegion& region) {
  for (const auto& s : region.slots) {
    if (s.is_vec) {
      ldr(QReg(vec_reg_map_[s.host_reg_index]), ptr(x20, s.offset));
    } else if (s.is_float) {
      ldr(DReg(vec_reg_map_[s.host_reg_index]), ptr(x20, s.offset));
    } else {
      ldr(XReg(gpr_reg_map_[s.host_reg_index]), ptr(x20, s.offset));
    }
  }
}

void A64Emitter::EmitStorePinnedSlots(const hir::LoopPromotionRegion& region,
                                      uint32_t slot_mask) {
  for (size_t idx = 0; idx < region.slots.size(); ++idx) {
    if (!(slot_mask & (1u << idx))) {
      continue;
    }
    const auto& s = region.slots[idx];
    if (s.is_vec) {
      str(QReg(vec_reg_map_[s.host_reg_index]), ptr(x20, s.offset));
    } else if (s.is_float) {
      str(DReg(vec_reg_map_[s.host_reg_index]), ptr(x20, s.offset));
    } else {
      str(XReg(gpr_reg_map_[s.host_reg_index]), ptr(x20, s.offset));
    }
  }
}

namespace {

bool IsMovi64ImmLocal(uint64_t value) {
  if (value == 0 || value == ~uint64_t(0)) {
    return true;
  }
  for (int shift = 0; shift < 8; ++shift) {
    const uint8_t shift_u8 = static_cast<uint8_t>(value >> (shift * 8));
    if (shift_u8 != 0 && shift_u8 != 0xFF) {
      return false;
    }
  }
  return true;
}

constexpr bool IsFmov32ImmLocal(float f32) {
  const uint32_t u32 = std::bit_cast<uint32_t, float>(f32);
  int32_t exp = ((u32 >> 23) & 0xff) - 127;
  int64_t mantissa = u32 & 0x7fffff;
  if (mantissa & 0x7ffff) {
    return false;
  }
  if (exp < -3 || exp > 4) {
    return false;
  }
  mantissa >>= 19;
  return (mantissa & 0b1111) == mantissa;
}

constexpr bool IsFmov64ImmLocal(double f64) {
  const uint64_t u64 = std::bit_cast<uint64_t, double>(f64);
  int32_t exp = ((u64 >> 52) & 0x7ff) - 1023;
  int64_t mantissa = u64 & 0xfffffffffffffULL;
  if (mantissa & 0xffffffffffffULL) {
    return false;
  }
  if (exp < -3 || exp > 4) {
    return false;
  }
  mantissa >>= 48;
  return (mantissa & 0b1111) == mantissa;
}

void LoadV128ConstLocal(A64Emitter& e, int vreg_idx, const vec128_t& val) {
  if (!val.low && !val.high) {
    e.movi(VReg2D(vreg_idx), 0);
    return;
  } else if (val.low == ~uint64_t(0) && val.high == ~uint64_t(0)) {
    e.movi(VReg2D(vreg_idx), ~0ULL);
    return;
  }

  bool all_equal_u8 = true;
  const uint8_t splat_u8 = val.u8[0];
  for (unsigned i = 1; i < 16; ++i) {
    if (val.u8[i] != splat_u8) {
      all_equal_u8 = false;
      break;
    }
  }
  if (all_equal_u8) {
    e.movi(VReg(vreg_idx).b16, static_cast<uint8_t>(splat_u8));
    return;
  }

  bool all_equal_u16 = true;
  const uint16_t splat_u16 = val.u16[0];
  for (unsigned i = 1; i < 8; ++i) {
    if (val.u16[i] != splat_u16) {
      all_equal_u16 = false;
      break;
    }
  }
  if (all_equal_u16) {
    if ((splat_u16 & 0xFF'00) == 0) {
      e.movi(VReg(vreg_idx).h8, static_cast<uint8_t>(splat_u16 >> 0), LSL, 0);
    } else if ((splat_u16 & 0x00'FF) == 0) {
      e.movi(VReg(vreg_idx).h8, static_cast<uint8_t>(splat_u16 >> 8), LSL, 8);
    } else if ((splat_u16 & 0xFF'00) == 0xFF'00) {
      e.mvni(VReg(vreg_idx).h8, ~static_cast<uint8_t>(splat_u16 >> 0) & 0xFF,
             LSL, 0);
    } else if ((splat_u16 & 0x00'FF) == 0x00'FF) {
      e.mvni(VReg(vreg_idx).h8, ~static_cast<uint8_t>(splat_u16 >> 8) & 0xFF,
             LSL, 8);
    } else {
      e.ldr(QReg(vreg_idx), e.GetV128ConstLabel(val));
    }
    return;
  }

  bool all_equal_u32 = true;
  const uint32_t splat_u32 = val.u32[0];
  const float splat_f32 = val.f32[0];
  for (unsigned i = 1; i < 4; ++i) {
    if (val.u32[i] != splat_u32) {
      all_equal_u32 = false;
      break;
    }
  }
  if (all_equal_u32) {
    if ((splat_u32 & 0xFF'FF'FF'00) == 0) {
      e.movi(VReg(vreg_idx).s4, static_cast<uint8_t>(splat_u32 >> 0), LSL, 0);
    } else if ((splat_u32 & 0xFF'FF'00'FF) == 0) {
      e.movi(VReg(vreg_idx).s4, static_cast<uint8_t>(splat_u32 >> 8), LSL, 8);
    } else if ((splat_u32 & 0xFF'00'FF'FF) == 0) {
      e.movi(VReg(vreg_idx).s4, static_cast<uint8_t>(splat_u32 >> 16), LSL, 16);
    } else if ((splat_u32 & 0x00'FF'FF'FF) == 0) {
      e.movi(VReg(vreg_idx).s4, static_cast<uint8_t>(splat_u32 >> 24), LSL, 24);
    } else if ((splat_u32 & 0xFF'FF'FF'00) == 0xFF'FF'FF'00) {
      e.mvni(VReg(vreg_idx).s4, ~static_cast<uint8_t>(splat_u32 >> 0) & 0xFF,
             LSL, 0);
    } else if ((splat_u32 & 0xFF'FF'00'FF) == 0xFF'FF'00'FF) {
      e.mvni(VReg(vreg_idx).s4, ~static_cast<uint8_t>(splat_u32 >> 8) & 0xFF,
             LSL, 8);
    } else if ((splat_u32 & 0xFF'00'FF'FF) == 0xFF'00'FF'FF) {
      e.mvni(VReg(vreg_idx).s4, ~static_cast<uint8_t>(splat_u32 >> 16) & 0xFF,
             LSL, 16);
    } else if ((splat_u32 & 0x00'FF'FF'FF) == 0x00'FF'FF'FF) {
      e.mvni(VReg(vreg_idx).s4, ~static_cast<uint8_t>(splat_u32 >> 24) & 0xFF,
             LSL, 24);
    } else if (IsFmov32ImmLocal(splat_f32)) {
      e.fmov(VReg(vreg_idx).s4, splat_f32);
    } else {
      e.ldr(QReg(vreg_idx), e.GetV128ConstLabel(val));
    }
    return;
  }

  const bool all_equal_u64 = val.low == val.high;
  const double splat_f64 = val.f64[0];
  if (all_equal_u64) {
    if (IsMovi64ImmLocal(val.low)) {
      e.movi(VReg2D(vreg_idx), val.low);
    } else if (IsFmov64ImmLocal(splat_f64)) {
      e.fmov(VReg(vreg_idx).d2, splat_f64);
    } else {
      e.ldr(QReg(vreg_idx), e.GetV128ConstLabel(val));
    }
    return;
  }

  e.ldr(QReg(vreg_idx), e.GetV128ConstLabel(val));
}

}  // namespace

bool A64Emitter::TryEmitPinnedContextInstr(
    const hir::Instr* instr, const hir::LoopPromotionRegion& region) {
  const hir::Opcode opcode = instr->GetOpcodeNum();
  if (opcode == hir::OPCODE_LOAD_CONTEXT) {
    const uint32_t offset = static_cast<uint32_t>(instr->src1.offset);
    const uint32_t size =
        static_cast<uint32_t>(hir::GetTypeSize(instr->dest->type));
    for (const auto& s : region.slots) {
      if (s.offset != offset || s.size != size) {
        continue;
      }
      if (s.is_vec) {
        if (instr->dest->type != hir::VEC128_TYPE) {
          return false;
        }
        const uint32_t dst_idx =
            MapReg(instr->dest, vec_reg_map_, VEC_COUNT, "vec");
        const uint32_t host_idx = vec_reg_map_[s.host_reg_index];
        if (dst_idx != host_idx) {
          orr(VReg(dst_idx).b16, VReg(host_idx).b16, VReg(host_idx).b16);
        }
        return true;
      } else if (s.is_float) {
        if (instr->dest->type != hir::FLOAT64_TYPE) {
          return false;
        }
        const uint32_t dst_idx =
            MapReg(instr->dest, vec_reg_map_, VEC_COUNT, "vec");
        const uint32_t host_idx = vec_reg_map_[s.host_reg_index];
        if (dst_idx != host_idx) {
          fmov(DReg(dst_idx), DReg(host_idx));
        }
        return true;
      } else {
        if (instr->dest->type != hir::INT64_TYPE) {
          return false;
        }
        const uint32_t dst_idx =
            MapReg(instr->dest, gpr_reg_map_, GPR_COUNT, "gpr");
        const uint32_t host_idx = gpr_reg_map_[s.host_reg_index];
        if (dst_idx != host_idx) {
          mov(XReg(dst_idx), XReg(host_idx));
        }
        return true;
      }
    }
    return false;
  }

  if (opcode == hir::OPCODE_STORE_CONTEXT) {
    const uint32_t offset = static_cast<uint32_t>(instr->src1.offset);
    const hir::Value* val = instr->src2.value;
    const uint32_t size = static_cast<uint32_t>(hir::GetTypeSize(val->type));
    for (const auto& s : region.slots) {
      if (s.offset != offset || s.size != size) {
        continue;
      }
      if (s.is_vec) {
        if (val->type != hir::VEC128_TYPE) {
          return false;
        }
        const uint32_t host_idx = vec_reg_map_[s.host_reg_index];
        if (val->IsConstant()) {
          LoadV128ConstLocal(*this, static_cast<int>(host_idx),
                             val->constant.v128);
        } else {
          const uint32_t src_idx = MapReg(val, vec_reg_map_, VEC_COUNT, "vec");
          if (src_idx != host_idx) {
            orr(VReg(host_idx).b16, VReg(src_idx).b16, VReg(src_idx).b16);
          }
        }
        return true;
      } else if (s.is_float) {
        if (val->type != hir::FLOAT64_TYPE) {
          return false;
        }
        const uint32_t host_idx = vec_reg_map_[s.host_reg_index];
        if (val->IsConstant()) {
          union {
            double d;
            uint64_t u;
          } c;
          c.d = val->constant.f64;
          mov(x0, c.u);
          fmov(DReg(host_idx), x0);
        } else {
          const uint32_t src_idx = MapReg(val, vec_reg_map_, VEC_COUNT, "vec");
          if (src_idx != host_idx) {
            fmov(DReg(host_idx), DReg(src_idx));
          }
        }
        return true;
      } else {
        if (val->type != hir::INT64_TYPE) {
          return false;
        }
        const uint32_t host_idx = gpr_reg_map_[s.host_reg_index];
        if (val->IsConstant()) {
          mov(XReg(host_idx), val->constant.u64);
        } else {
          const uint32_t src_idx = MapReg(val, gpr_reg_map_, GPR_COUNT, "gpr");
          if (src_idx != host_idx) {
            mov(XReg(host_idx), XReg(src_idx));
          }
        }
        return true;
      }
    }
    return false;
  }

  return false;
}

}  // namespace a64
}  // namespace backend
}  // namespace cpu
}  // namespace xe
