/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/cpu/compiler/passes/dead_context_store_elimination_pass.h"

#include <algorithm>
#include <vector>

#include "xenia/base/cvar.h"
#include "xenia/base/profiling.h"
#include "xenia/cpu/compiler/passes/cross_block_context_analysis.h"
#include "xenia/cpu/ppc/ppc_context.h"

DECLARE_bool(debug);
DECLARE_bool(disable_context_promotion);
DECLARE_bool(store_all_context_values);
DECLARE_bool(full_optimization_even_with_debug);
DECLARE_bool(context_promote_vec128);

DEFINE_bool(
    dse_cross_block_context, true,
    "Eliminate dead cross-block guest context stores (GPR/FPR/CTR/LR/XER/"
    "VEC128) in addition to CR stores.",
    "CPU");

DEFINE_bool(
    dse_cross_block_vec128, true,
    "Include VMX VEC128 registers in cross-block dead context store "
    "elimination when context_promote_vec128 is enabled.",
    "CPU");

namespace xe {
namespace cpu {
namespace compiler {
namespace passes {

using namespace xe::cpu::hir;

using xe::cpu::hir::Block;
using xe::cpu::hir::HIRBuilder;
using xe::cpu::hir::Instr;

namespace {

constexpr cross_block::ContextLayout kContextLayout{
    static_cast<uint32_t>(offsetof(ppc::PPCContext, cr0)),
    static_cast<uint32_t>(offsetof(ppc::PPCContext, r)),
    static_cast<uint32_t>(offsetof(ppc::PPCContext, ctr)),
    static_cast<uint32_t>(offsetof(ppc::PPCContext, lr)),
    static_cast<uint32_t>(offsetof(ppc::PPCContext, f)),
    static_cast<uint32_t>(offsetof(ppc::PPCContext, v)),
    static_cast<uint32_t>(offsetof(ppc::PPCContext, xer_ca)),
    static_cast<uint32_t>(offsetof(ppc::PPCContext, xer_ov)),
    static_cast<uint32_t>(offsetof(ppc::PPCContext, xer_so)),
};

static_assert(kContextLayout.cr_offset == 0);
static_assert(kContextLayout.r_offset == 40);
static_assert(kContextLayout.ctr_offset == 296);
static_assert(kContextLayout.lr_offset == 304);
static_assert(kContextLayout.f_offset == 320);
static_assert(kContextLayout.v_offset == 576);
static_assert(kContextLayout.xer_ca_offset == 2640);
static_assert(kContextLayout.xer_ov_offset == 2641);
static_assert(kContextLayout.xer_so_offset == 2642);

}  // namespace

DeadContextStoreEliminationPass::DeadContextStoreEliminationPass()
    : CompilerPass() {}

DeadContextStoreEliminationPass::~DeadContextStoreEliminationPass() = default;

bool DeadContextStoreEliminationPass::Run(HIRBuilder* builder) {
  SCOPE_profile_cpu_f("cpu");

  if (cvars::disable_context_promotion ||
      !(cvars::full_optimization_even_with_debug ||
        (!cvars::debug && !cvars::store_all_context_values)) ||
      (builder->attributes() & FUNCTION_ATTRIB_KEEP_CONTEXT_STORES)) {
    return true;
  }

  size_t block_count = 0;
  for (Block* block = builder->first_block(); block; block = block->next) {
    if (block_count > UINT16_MAX) {
      return true;
    }
    block->ordinal = static_cast<uint16_t>(block_count++);
  }
  if (!block_count) {
    return true;
  }

  std::vector<cross_block::LivenessBlock> liveness_blocks(block_count);
  std::vector<std::vector<Instr*>> instr_ptrs(block_count);

  for (Block* block = builder->first_block(); block; block = block->next) {
    const size_t b = block->ordinal;
    auto& lb = liveness_blocks[b];
    auto& ptrs = instr_ptrs[b];

    bool ended_unconditionally = false;
    for (Instr* i = block->instr_head; i; i = i->next) {
      cross_block::LivenessInstr li;
      const Opcode num = i->GetOpcodeNum();

      if (const Label* target = i->BranchLabel()) {
        const uint32_t target_idx =
            (target->block && target->block->ordinal < block_count)
                ? static_cast<uint32_t>(target->block->ordinal)
                : UINT32_MAX;
        li.target_block = target_idx;
        if (num == OPCODE_BRANCH) {
          li.kind = cross_block::LivenessInstrKind::kBranch;
          if (!ended_unconditionally && target_idx < block_count &&
              std::find(lb.successors.begin(), lb.successors.end(),
                        target_idx) == lb.successors.end()) {
            lb.successors.push_back(target_idx);
          }
          ended_unconditionally = true;
        } else {
          li.kind = cross_block::LivenessInstrKind::kBranchCond;
          if (!ended_unconditionally && target_idx < block_count &&
              std::find(lb.successors.begin(), lb.successors.end(),
                        target_idx) == lb.successors.end()) {
            lb.successors.push_back(target_idx);
          }
        }
      } else if (num == OPCODE_LOAD_CONTEXT) {
        li.kind = cross_block::LivenessInstrKind::kLoadContext;
        li.offset = static_cast<uint32_t>(i->src1.offset);
        li.size = static_cast<uint32_t>(GetTypeSize(i->dest->type));
      } else if (num == OPCODE_STORE_CONTEXT) {
        li.kind = cross_block::LivenessInstrKind::kStoreContext;
        li.offset = static_cast<uint32_t>(i->src1.offset);
        li.size = static_cast<uint32_t>(GetTypeSize(i->src2.value->type));
      } else if (num == OPCODE_CHECK_PREEMPT) {
        li.kind = cross_block::LivenessInstrKind::kCheckPreempt;
      } else if (num == OPCODE_CONTEXT_BARRIER) {
        li.kind = cross_block::LivenessInstrKind::kContextBarrier;
      } else if (i->opcode->flags & OPCODE_FLAG_VOLATILE) {
        li.kind = cross_block::LivenessInstrKind::kVolatile;
        if (num == OPCODE_RETURN ||
            ((num == OPCODE_CALL || num == OPCODE_CALL_INDIRECT) &&
             (i->flags & CALL_TAIL))) {
          ended_unconditionally = true;
        }
      } else if (i->opcode->flags & OPCODE_FLAG_MEMORY) {
        li.kind = cross_block::LivenessInstrKind::kMemory;
      } else {
        li.kind = cross_block::LivenessInstrKind::kOther;
      }

      lb.instrs.push_back(li);
      ptrs.push_back(i);
    }

    if (!ended_unconditionally && block->next) {
      const uint32_t next_idx = static_cast<uint32_t>(block->next->ordinal);
      if (std::find(lb.successors.begin(), lb.successors.end(), next_idx) ==
          lb.successors.end()) {
        lb.successors.push_back(next_idx);
      }
    }
  }

  const bool track_all = cvars::dse_cross_block_context;
  const bool track_vec =
      track_all && cvars::dse_cross_block_vec128 && cvars::context_promote_vec128;
  const auto res = cross_block::SolveCrossBlockContextLiveness(
      liveness_blocks, kContextLayout, track_all, track_vec);

  for (size_t b = 0; b < block_count; ++b) {
    const auto& dead = res.dead_store[b];
    auto& ptrs = instr_ptrs[b];
    for (size_t idx = 0; idx < dead.size(); ++idx) {
      if (dead[idx]) {
        ptrs[idx]->UnlinkAndNOP();
      }
    }
  }

  return true;
}

}  // namespace passes
}  // namespace compiler
}  // namespace cpu
}  // namespace xe
