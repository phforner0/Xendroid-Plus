/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/cpu/compiler/passes/loop_context_promotion_pass.h"

#include <algorithm>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "xenia/base/cvar.h"
#include "xenia/base/profiling.h"
#include "xenia/cpu/backend/backend.h"
#include "xenia/cpu/backend/machine_info.h"
#include "xenia/cpu/compiler/passes/cross_block_context_analysis.h"
#include "xenia/cpu/compiler/passes/register_allocation_pass.h"
#include "xenia/cpu/ppc/ppc_context.h"
#include "xenia/cpu/processor.h"

DECLARE_bool(debug);
DECLARE_bool(disable_context_promotion);
DECLARE_bool(store_all_context_values);
DECLARE_bool(full_optimization_even_with_debug);
DECLARE_bool(context_promote_vec128);

DEFINE_bool(
    a64_loop_context_promotion, true,
    "Promote hot guest context registers (GPR/FPR/VEC/CTR/LR) to pinned "
    "ARM64 host registers across small (1-4 block) call-free natural loops.",
    "CPU");

DEFINE_int32(
    a64_loop_promote_max_regs, 6,
    "Maximum number of guest context registers promoted per loop "
    "(clamped to [1, 6]).",
    "CPU");

DEFINE_bool(
    a64_loop_promote_defer_stores_across_memory, false,
    "Defer flushing dirty promoted context stores across guest memory "
    "accesses inside promoted loops (faster, but guest data-fault handlers "
    "will not observe mid-loop updates).",
    "CPU");

namespace xe {
namespace cpu {
namespace compiler {
namespace passes {

using namespace xe::cpu::hir;

using xe::cpu::hir::Block;
using xe::cpu::hir::HIRBuilder;
using xe::cpu::hir::Instr;
using xe::cpu::hir::Label;

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

bool IsUnconditionalJumpInstr(const Instr* i) {
  const Opcode num = i->GetOpcodeNum();
  if (num == OPCODE_BRANCH || num == OPCODE_RETURN) {
    return true;
  }
  if ((num == OPCODE_CALL || num == OPCODE_CALL_INDIRECT) &&
      (i->flags & CALL_TAIL)) {
    return true;
  }
  return false;
}

Block* AppendDetachedBlock(HIRBuilder* builder) {
  Block* tail_blk = builder->last_block();
  if (tail_blk) {
    Instr* tail_ins = tail_blk->instr_tail;
    if (!tail_ins || (!IsUnconditionalJumpInstr(tail_ins) &&
                      tail_ins->GetOpcodeNum() != OPCODE_TRAP)) {
      builder->AppendReturn(tail_blk);
    }
  }
  return builder->InsertBlockAfter(builder->last_block());
}

int FindPromotedSlotIndex(const std::vector<LoopPromotedSlot>& slots,
                          uint32_t offset, uint32_t size) {
  for (size_t idx = 0; idx < slots.size(); ++idx) {
    if (slots[idx].offset == offset && slots[idx].size == size) {
      return static_cast<int>(idx);
    }
  }
  return -1;
}

}  // namespace

LoopContextPromotionPass::LoopContextPromotionPass() : CompilerPass() {}

LoopContextPromotionPass::~LoopContextPromotionPass() = default;

bool LoopContextPromotionPass::Run(HIRBuilder* builder) {
  SCOPE_profile_cpu_f("cpu");
  builder->loop_promotion_plan().clear();

  if (!cvars::a64_loop_context_promotion || cvars::disable_context_promotion ||
      cvars::store_all_context_values ||
      (cvars::debug && !cvars::full_optimization_even_with_debug) ||
      (builder->attributes() & FUNCTION_ATTRIB_KEEP_CONTEXT_STORES)) {
    return true;
  }

  if (processor_ && processor_->backend()) {
    const auto* backend = processor_->backend();
    if (backend->trace_instr_available() || backend->trace_data_available() ||
        backend->trace_func_available()) {
      return true;
    }
    const auto* mi = backend->machine_info();
    if (mi) {
      uint32_t gpr_count = 0;
      uint32_t vec_count = 0;
      for (uint32_t n = 0; mi->register_sets[n].count != 0; ++n) {
        if (mi->register_sets[n].types &
            backend::MachineInfo::RegisterSet::INT_TYPES) {
          gpr_count = mi->register_sets[n].count;
        }
        if (mi->register_sets[n].types &
            backend::MachineInfo::RegisterSet::VEC_TYPES) {
          vec_count = mi->register_sets[n].count;
        }
      }
      if (gpr_count < 7 || vec_count < 28) {
        return true;
      }
    }
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

  std::vector<Block*> orig_blocks(block_count, nullptr);
  std::vector<cross_block::LoopBlock> loop_blocks(block_count);
  std::vector<std::vector<Instr*>> instr_ptrs(block_count);

  for (Block* block = builder->first_block(); block; block = block->next) {
    const size_t b = block->ordinal;
    orig_blocks[b] = block;
    auto& lb = loop_blocks[b];
    auto& ptrs = instr_ptrs[b];
    lb.fallthrough_block =
        block->next ? static_cast<uint32_t>(block->next->ordinal) : UINT32_MAX;

    for (Instr* i = block->instr_head; i; i = i->next) {
      cross_block::LoopInstr li;
      const Opcode num = i->GetOpcodeNum();
      li.clobbers_volatile_gprs =
          RegisterAllocationPass::ClobbersVolatileGprs(i);
      li.is_unconditional_jump = IsUnconditionalJumpInstr(i);

      if (const Label* target = i->BranchLabel()) {
        li.target_block =
            (target->block && target->block->ordinal < block_count)
                ? static_cast<uint32_t>(target->block->ordinal)
                : UINT32_MAX;
        li.kind = (num == OPCODE_BRANCH) ? cross_block::LoopInstrKind::kBranch
                                         : cross_block::LoopInstrKind::kBranchCond;
      } else if (num == OPCODE_CALL || num == OPCODE_CALL_TRUE ||
                 num == OPCODE_CALL_INDIRECT ||
                 num == OPCODE_CALL_INDIRECT_TRUE ||
                 num == OPCODE_CALL_EXTERN) {
        li.kind = cross_block::LoopInstrKind::kCall;
      } else if (num == OPCODE_RETURN || num == OPCODE_RETURN_TRUE) {
        li.kind = cross_block::LoopInstrKind::kReturn;
      } else if (num == OPCODE_LOAD_CONTEXT) {
        li.kind = cross_block::LoopInstrKind::kLoadContext;
        li.offset = static_cast<uint32_t>(i->src1.offset);
        li.size = static_cast<uint32_t>(GetTypeSize(i->dest->type));
      } else if (num == OPCODE_STORE_CONTEXT) {
        li.kind = cross_block::LoopInstrKind::kStoreContext;
        li.offset = static_cast<uint32_t>(i->src1.offset);
        li.size = static_cast<uint32_t>(GetTypeSize(i->src2.value->type));
      } else if (num == OPCODE_CHECK_PREEMPT) {
        li.kind = cross_block::LoopInstrKind::kCheckPreempt;
      } else if (num == OPCODE_CONTEXT_BARRIER) {
        li.kind = cross_block::LoopInstrKind::kContextBarrier;
      } else if (num == OPCODE_TRAP || num == OPCODE_TRAP_TRUE) {
        li.kind = cross_block::LoopInstrKind::kTrap;
      } else if (i->opcode->flags & OPCODE_FLAG_VOLATILE) {
        li.kind = cross_block::LoopInstrKind::kVolatile;
      } else if (i->opcode->flags & OPCODE_FLAG_MEMORY) {
        li.kind = cross_block::LoopInstrKind::kMemory;
      } else {
        li.kind = cross_block::LoopInstrKind::kOther;
      }

      lb.instrs.push_back(li);
      ptrs.push_back(i);
    }
  }

  cross_block::LoopAnalysisOptions opts;
  opts.promote_vec128 = cvars::context_promote_vec128;
  opts.defer_stores_across_memory =
      cvars::a64_loop_promote_defer_stores_across_memory;
  opts.max_regs = static_cast<uint32_t>(
      std::clamp<int32_t>(cvars::a64_loop_promote_max_regs, 1, 6));

  const auto plans =
      cross_block::SelectPromotableLoops(loop_blocks, kContextLayout, opts);
  if (plans.empty()) {
    return true;
  }

  auto& out_plan = builder->loop_promotion_plan();
  uint32_t next_region_id = 0;

  for (const auto& plan : plans) {
    LoopPromotionRegion region;
    region.id = next_region_id++;
    region.header = orig_blocks[plan.header];

    uint32_t written_mask = 0;
    for (size_t s = 0; s < plan.promoted.size(); ++s) {
      const auto& ps = plan.promoted[s];
      LoopPromotedSlot slot;
      slot.offset = ps.offset;
      slot.size = ps.size;
      slot.host_reg_index = ps.host_reg_index;
      slot.is_vec = ps.is_vec;
      slot.is_float = ps.is_float;
      slot.written = (ps.writes > 0);
      if (slot.written) {
        written_mask |= (1u << s);
      }
      if (slot.is_vec || slot.is_float) {
        region.reserved_vec_mask |= (1u << slot.host_reg_index);
      } else {
        region.reserved_gpr_mask |= (1u << slot.host_reg_index);
      }
      region.slots.push_back(slot);
    }

    std::unordered_set<Block*> in_loop_blocks;
    std::unordered_map<Block*, size_t> block_to_span_idx;
    for (size_t i = 0; i < plan.blocks.size(); ++i) {
      Block* b = orig_blocks[plan.blocks[i]];
      in_loop_blocks.insert(b);
      block_to_span_idx[b] = i;
      Instr* end_instr =
          (plan.loop_end_instr_index[i] < instr_ptrs[plan.blocks[i]].size())
              ? instr_ptrs[plan.blocks[i]][plan.loop_end_instr_index[i]]
              : nullptr;
      region.block_spans.push_back({b, end_instr});
    }

    // Fault-exact dirty tracking across [1..4] blocks of L when memory ops exist.
    const bool fault_exact =
        plan.has_memory_ops && !opts.defer_stores_across_memory &&
        written_mask != 0;
    const size_t kL = region.block_spans.size();
    std::vector<uint32_t> dirty_in(kL, 0);

    if (fault_exact) {
      std::vector<std::vector<size_t>> internal_succs(kL);
      for (size_t u = 0; u < kL; ++u) {
        Block* b = region.block_spans[u].block;
        Instr* end_i = region.block_spans[u].loop_end_instr;
        bool ended_uncond = false;
        for (Instr* ins = b->instr_head; ins && end_i; ins = ins->next) {
          if (const Label* tgt = ins->BranchLabel()) {
            auto it = block_to_span_idx.find(tgt->block);
            if (it != block_to_span_idx.end()) {
              internal_succs[u].push_back(it->second);
            }
          }
          if (IsUnconditionalJumpInstr(ins)) {
            ended_uncond = true;
          }
          if (ins == end_i) {
            break;
          }
        }
        if (!ended_uncond && end_i == b->instr_tail && b->next) {
          auto it = block_to_span_idx.find(b->next);
          if (it != block_to_span_idx.end()) {
            internal_succs[u].push_back(it->second);
          }
        }
      }

      bool changed = true;
      while (changed) {
        changed = false;
        for (size_t u = 0; u < kL; ++u) {
          uint32_t d = dirty_in[u];
          Block* b = region.block_spans[u].block;
          Instr* end_i = region.block_spans[u].loop_end_instr;
          for (Instr* ins = b->instr_head; ins && end_i; ins = ins->next) {
            if (ins->GetOpcodeNum() == OPCODE_STORE_CONTEXT) {
              const int s = FindPromotedSlotIndex(
                  region.slots, static_cast<uint32_t>(ins->src1.offset),
                  static_cast<uint32_t>(GetTypeSize(ins->src2.value->type)));
              if (s >= 0) {
                d |= (1u << s);
              }
            } else if (ins->opcode->flags & OPCODE_FLAG_MEMORY) {
              d = 0;
            }
            if (ins == end_i) {
              break;
            }
          }
          for (size_t v : internal_succs[u]) {
            if ((dirty_in[v] | d) != dirty_in[v]) {
              dirty_in[v] |= d;
              changed = true;
            }
          }
        }
      }
    }

    // 1. Preheader / entry setup.
    Label* header_lbl = region.header->label_head;
    if (!header_lbl) {
      header_lbl = builder->NewLabel();
      builder->MarkLabel(header_lbl, region.header);
    }

    if (region.header == builder->first_block()) {
      region.entry_points.push_back({nullptr, true});
      // Redirect any external blocks that explicitly branch to first_block().
      std::vector<Instr*> ext_branches;
      for (Block* p = builder->first_block(); p; p = p->next) {
        if (in_loop_blocks.count(p)) {
          continue;
        }
        for (Instr* ins = p->instr_head; ins; ins = ins->next) {
          if (const Label* tgt = ins->BranchLabel()) {
            if (tgt->block == region.header) {
              ext_branches.push_back(ins);
            }
          }
        }
      }
      if (!ext_branches.empty()) {
        Block* b_entry = AppendDetachedBlock(builder);
        Label* entry_lbl = builder->NewLabel();
        builder->MarkLabel(entry_lbl, b_entry);
        builder->AppendBranchToLabel(b_entry, header_lbl);
        for (Instr* ins : ext_branches) {
          if (ins->GetOpcodeNum() == OPCODE_BRANCH) {
            ins->src1.label = entry_lbl;
          } else {
            ins->src2.label = entry_lbl;
          }
        }
        region.entry_points.push_back({b_entry, false});
      }
    } else {
      // Never place the preheader between an in-loop predecessor and
      // region.header: an in-loop fallthrough back-edge would otherwise fall
      // through into the preheader on every iteration and reload stale context!
      Block* b_preheader =
          (region.header->prev && in_loop_blocks.count(region.header->prev))
              ? AppendDetachedBlock(builder)
              : builder->InsertBlockBefore(region.header);
      Label* preheader_lbl = builder->NewLabel();
      builder->MarkLabel(preheader_lbl, b_preheader);
      builder->AppendBranchToLabel(b_preheader, header_lbl);
      for (Block* p = builder->first_block(); p; p = p->next) {
        if (p == b_preheader || in_loop_blocks.count(p)) {
          continue;
        }
        for (Instr* ins = p->instr_head; ins; ins = ins->next) {
          if (const Label* tgt = ins->BranchLabel()) {
            if (tgt->block == region.header) {
              if (ins->GetOpcodeNum() == OPCODE_BRANCH) {
                ins->src1.label = preheader_lbl;
              } else {
                ins->src2.label = preheader_lbl;
              }
            }
          }
        }
      }
      region.entry_points.push_back({b_preheader, false});
    }

    // 2. Exit trampolines, extended-block fall-through exits, and fault-exact
    //    memory flushes.
    for (size_t u = 0; u < kL; ++u) {
      Block* b = region.block_spans[u].block;
      Instr* end_i = region.block_spans[u].loop_end_instr;
      if (!end_i) {
        continue;
      }
      uint32_t d = fault_exact ? dirty_in[u] : written_mask;
      for (Instr* ins = b->instr_head; ins; ins = ins->next) {
        const Opcode num = ins->GetOpcodeNum();
        if (fault_exact) {
          if (num == OPCODE_STORE_CONTEXT) {
            const int s = FindPromotedSlotIndex(
                region.slots, static_cast<uint32_t>(ins->src1.offset),
                static_cast<uint32_t>(GetTypeSize(ins->src2.value->type)));
            if (s >= 0) {
              d |= (1u << s);
            }
          } else if (ins->opcode->flags & OPCODE_FLAG_MEMORY) {
            if (d != 0) {
              region.flush_points.push_back({ins, d});
              d = 0;
            }
          }
        }

        if (const Label* tgt = ins->BranchLabel()) {
          if (!in_loop_blocks.count(tgt->block) && d != 0) {
            if (num == OPCODE_BRANCH) {
              // Unconditional exit branch: flush right before the branch.
              region.flush_points.push_back({ins, d});
            } else {
              // Conditional exit branch: use a trampoline block so fused
              // compare-branch flags are never disturbed.
              Block* b_exit = AppendDetachedBlock(builder);
              Label* exit_lbl = builder->NewLabel();
              builder->MarkLabel(exit_lbl, b_exit);
              builder->AppendBranchToLabel(b_exit, ins->src2.label);
              ins->src2.label = exit_lbl;
              region.exit_points.push_back({b_exit, nullptr, d});
            }
          }
        }

        if (ins == end_i) {
          break;
        }
      }

      if (d != 0) {
        if (end_i->next != nullptr) {
          // Extended block fall-through to post-loop code in the same block.
          region.exit_points.push_back({nullptr, end_i, d});
        } else if (end_i == b->instr_tail && !IsUnconditionalJumpInstr(end_i) &&
                   (!b->next || !in_loop_blocks.count(b->next))) {
          // Direct fall-through out of the loop (either to b->next or epilog).
          region.exit_points.push_back({nullptr, end_i, d});
        }
      }
    }

    out_plan.regions.push_back(std::move(region));
  }

  return true;
}

}  // namespace passes
}  // namespace compiler
}  // namespace cpu
}  // namespace xe
