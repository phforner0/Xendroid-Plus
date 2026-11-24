/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_CPU_BACKEND_A64_A64_STACK_SKEW_ANALYSIS_H_
#define XENIA_CPU_BACKEND_A64_A64_STACK_SKEW_ANALYSIS_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

#include "xenia/cpu/hir/hir_builder.h"
#include "xenia/cpu/ppc/ppc_context.h"

namespace xe {
namespace cpu {
namespace backend {
namespace a64 {

inline constexpr size_t kR1ContextOffset =
    offsetof(ppc::PPCContext, r) + sizeof(uint64_t);
inline constexpr size_t kR1ContextEnd = kR1ContextOffset + sizeof(uint64_t);
inline constexpr int64_t kMaxCumulativeStackDelta = 0x100000LL;  // 1 MiB

struct StackR1State {
  bool visited = false;
  bool tainted = false;
  int64_t min_delta = 0;
  int64_t max_delta = 0;

  bool operator==(const StackR1State& other) const {
    return visited == other.visited && tainted == other.tainted &&
           min_delta == other.min_delta && max_delta == other.max_delta;
  }
  bool operator!=(const StackR1State& other) const { return !(*this == other); }
};

struct StackR1Interval {
  bool valid = false;
  int64_t min_delta = 0;
  int64_t max_delta = 0;
};

inline bool EvalStackR1Value(
    const hir::Value* value, const hir::Block* current_block,
    const std::unordered_map<const hir::Value*, StackR1Interval>&
        load_intervals,
    StackR1Interval* out_interval, int depth = 0) {
  if (!value || value->type != hir::INT64_TYPE || depth > 16) {
    return false;
  }
  const hir::Instr* def = value->def;
  if (!def || def->block != current_block) {
    return false;
  }
  switch (def->GetOpcodeNum()) {
    case hir::OPCODE_ASSIGN:
      return EvalStackR1Value(def->src1.value, current_block, load_intervals,
                              out_interval, depth + 1);
    case hir::OPCODE_LOAD_CONTEXT: {
      if (def->src1.offset != kR1ContextOffset) {
        return false;
      }
      auto it = load_intervals.find(value);
      if (it == load_intervals.end() || !it->second.valid) {
        return false;
      }
      *out_interval = it->second;
      return true;
    }
    case hir::OPCODE_ADD: {
      const hir::Value* lhs = def->src1.value;
      const hir::Value* rhs = def->src2.value;
      const hir::Value* base = nullptr;
      const hir::Value* offset_val = nullptr;
      if (rhs && rhs->IsConstant() && rhs->type == hir::INT64_TYPE) {
        base = lhs;
        offset_val = rhs;
      } else if (lhs && lhs->IsConstant() && lhs->type == hir::INT64_TYPE) {
        base = rhs;
        offset_val = lhs;
      } else {
        return false;
      }
      StackR1Interval base_interval;
      if (!EvalStackR1Value(base, current_block, load_intervals, &base_interval,
                            depth + 1)) {
        return false;
      }
      const int64_t c = offset_val->constant.i64;
      if (c < -kMaxCumulativeStackDelta || c > kMaxCumulativeStackDelta) {
        return false;
      }
      const int64_t min_d = base_interval.min_delta + c;
      const int64_t max_d = base_interval.max_delta + c;
      if (min_d < -kMaxCumulativeStackDelta ||
          max_d > kMaxCumulativeStackDelta) {
        return false;
      }
      out_interval->valid = true;
      out_interval->min_delta = min_d;
      out_interval->max_delta = max_d;
      return true;
    }
    case hir::OPCODE_SUB: {
      const hir::Value* lhs = def->src1.value;
      const hir::Value* rhs = def->src2.value;
      if (!rhs || !rhs->IsConstant() || rhs->type != hir::INT64_TYPE) {
        return false;
      }
      StackR1Interval base_interval;
      if (!EvalStackR1Value(lhs, current_block, load_intervals, &base_interval,
                            depth + 1)) {
        return false;
      }
      const int64_t c = rhs->constant.i64;
      if (c < -kMaxCumulativeStackDelta || c > kMaxCumulativeStackDelta) {
        return false;
      }
      const int64_t min_d = base_interval.min_delta - c;
      const int64_t max_d = base_interval.max_delta - c;
      if (min_d < -kMaxCumulativeStackDelta ||
          max_d > kMaxCumulativeStackDelta) {
        return false;
      }
      out_interval->valid = true;
      out_interval->min_delta = min_d;
      out_interval->max_delta = max_d;
      return true;
    }
    default:
      return false;
  }
}

inline void AnalyzeStackSkewSafety(hir::HIRBuilder* builder,
                                   bool* out_uses_no_skew,
                                   bool* out_has_unsafe_r1_write) {
  bool uses_no_skew = false;
  bool has_unsafe_r1_write = false;
  if (!builder || !builder->first_block()) {
    *out_uses_no_skew = false;
    *out_has_unsafe_r1_write = false;
    return;
  }

  std::vector<const hir::Block*> blocks;
  std::unordered_map<const hir::Block*, size_t> block_index;
  for (const hir::Block* b = builder->first_block(); b; b = b->next) {
    block_index.emplace(b, blocks.size());
    blocks.push_back(b);
  }
  const size_t n = blocks.size();

  auto collect_succs = [&](size_t u, std::vector<size_t>& out_succs) {
    out_succs.clear();
    auto add_succ = [&](const hir::Block* target) {
      if (!target) {
        return;
      }
      auto it = block_index.find(target);
      if (it == block_index.end()) {
        return;
      }
      if (std::find(out_succs.begin(), out_succs.end(), it->second) ==
          out_succs.end()) {
        out_succs.push_back(it->second);
      }
    };
    const hir::Block* blk = blocks[u];
    for (const hir::Edge* e = blk->outgoing_edge_head; e;
         e = e->outgoing_next) {
      add_succ(e->dest);
    }
    bool has_unconditional_stop = false;
    for (const hir::Instr* instr = blk->instr_tail; instr;
         instr = instr->prev) {
      const hir::Opcode op = instr->GetOpcodeNum();
      if (op == hir::OPCODE_BRANCH) {
        add_succ(instr->src1.label ? instr->src1.label->block : nullptr);
        has_unconditional_stop = true;
      } else if (op == hir::OPCODE_BRANCH_TRUE ||
                 op == hir::OPCODE_BRANCH_FALSE) {
        add_succ(instr->src2.label ? instr->src2.label->block : nullptr);
      } else if (op == hir::OPCODE_RETURN ||
                 ((op == hir::OPCODE_CALL ||
                   op == hir::OPCODE_CALL_INDIRECT) &&
                  (instr->flags & hir::CALL_TAIL) != 0)) {
        has_unconditional_stop = true;
      } else if (op != hir::OPCODE_COMMENT &&
                 op != hir::OPCODE_SOURCE_OFFSET) {
        break;
      }
    }
    if (!has_unconditional_stop && blk->next) {
      add_succ(blk->next);
    }
  };

  std::vector<uint8_t> dfs_state(n, 0);
  std::vector<std::vector<size_t>> succs(n);
  std::vector<size_t> postorder;
  postorder.reserve(n);
  struct DfsFrame {
    size_t u;
    size_t next_child;
  };
  std::vector<DfsFrame> stack;
  dfs_state[0] = 1;
  collect_succs(0, succs[0]);
  stack.push_back({0, 0});
  std::vector<std::pair<size_t, size_t>> back_edges;
  while (!stack.empty()) {
    auto& top = stack.back();
    const size_t u = top.u;
    if (top.next_child < succs[u].size()) {
      const size_t v = succs[u][top.next_child++];
      if (dfs_state[v] == 0) {
        dfs_state[v] = 1;
        collect_succs(v, succs[v]);
        stack.push_back({v, 0});
      } else if (dfs_state[v] == 1) {
        back_edges.emplace_back(u, v);
      }
    } else {
      dfs_state[u] = 2;
      postorder.push_back(u);
      stack.pop_back();
    }
  }

  auto is_back_edge = [&](size_t u, size_t v) {
    for (const auto& be : back_edges) {
      if (be.first == u && be.second == v) {
        return true;
      }
    }
    return false;
  };

  std::vector<StackR1State> in_states(n);
  std::vector<uint32_t> visits(n, 0);
  in_states[0] = StackR1State{true, false, 0, 0};

  std::vector<size_t> worklist;
  std::vector<uint8_t> in_worklist(n, 0);
  worklist.reserve(postorder.size());
  for (size_t idx : postorder) {
    worklist.push_back(idx);
    in_worklist[idx] = 1;
  }

  std::unordered_map<const hir::Value*, StackR1Interval> load_intervals;
  while (!worklist.empty()) {
    const size_t u = worklist.back();
    worklist.pop_back();
    in_worklist[u] = 0;

    if (!in_states[u].visited) {
      continue;
    }
    ++visits[u];
    StackR1State cur = in_states[u];
    if (visits[u] > 4) {
      cur.tainted = true;
    }

    load_intervals.clear();
    const hir::Block* block = blocks[u];
    for (const hir::Instr* instr = block->instr_head; instr;
         instr = instr->next) {
      const hir::Opcode op = instr->GetOpcodeNum();
      if (op == hir::OPCODE_LOAD_CONTEXT &&
          instr->src1.offset == kR1ContextOffset && instr->dest) {
        if (!cur.tainted && instr->dest->type == hir::INT64_TYPE) {
          load_intervals[instr->dest] =
              StackR1Interval{true, cur.min_delta, cur.max_delta};
        } else {
          load_intervals[instr->dest] = StackR1Interval{false, 0, 0};
        }
      } else if (op == hir::OPCODE_CALL || op == hir::OPCODE_CALL_TRUE ||
                 op == hir::OPCODE_CALL_INDIRECT ||
                 op == hir::OPCODE_CALL_INDIRECT_TRUE ||
                 op == hir::OPCODE_CALL_EXTERN) {
        // A callee may mutate r1 (alloca/longjmp/stack swap); entry r1 guard
        // cannot be trusted after any call returns.
        cur.tainted = true;
      } else if (op == hir::OPCODE_STORE_CONTEXT) {
        const size_t offset = static_cast<size_t>(instr->src1.offset);
        const hir::Value* val = instr->src2.value;
        const size_t size = val ? hir::GetTypeSize(val->type) : 0;
        if (offset < kR1ContextEnd && offset + size > kR1ContextOffset) {
          StackR1Interval new_iv;
          if (offset != kR1ContextOffset ||
              !EvalStackR1Value(val, block, load_intervals, &new_iv)) {
            cur.tainted = true;
            has_unsafe_r1_write = true;
          } else {
            cur.min_delta = new_iv.min_delta;
            cur.max_delta = new_iv.max_delta;
          }
        }
      } else if ((op == hir::OPCODE_LOAD || op == hir::OPCODE_LOAD_OFFSET ||
                  op == hir::OPCODE_STORE || op == hir::OPCODE_STORE_OFFSET) &&
                 (instr->flags & hir::LOAD_STORE_NO_PHYSICAL_SKEW) != 0) {
        uses_no_skew = true;
        if (cur.tainted) {
          has_unsafe_r1_write = true;
        } else {
          StackR1Interval addr_iv;
          if (!EvalStackR1Value(instr->src1.value, block, load_intervals,
                                &addr_iv)) {
            has_unsafe_r1_write = true;
          } else if (op == hir::OPCODE_LOAD_OFFSET ||
                     op == hir::OPCODE_STORE_OFFSET) {
            const hir::Value* off_val = instr->src2.value;
            if (!off_val || !off_val->IsConstant() ||
                off_val->type != hir::INT64_TYPE ||
                off_val->constant.i64 < -kMaxCumulativeStackDelta ||
                off_val->constant.i64 > kMaxCumulativeStackDelta ||
                addr_iv.min_delta + off_val->constant.i64 <
                    -2 * kMaxCumulativeStackDelta ||
                addr_iv.max_delta + off_val->constant.i64 >
                    2 * kMaxCumulativeStackDelta) {
              has_unsafe_r1_write = true;
            }
          }
        }
      }
    }

    for (size_t v : succs[u]) {
      StackR1State& dst = in_states[v];
      if (!dst.visited) {
        dst = cur;
        dst.visited = true;
        if (!in_worklist[v]) {
          worklist.push_back(v);
          in_worklist[v] = 1;
        }
      } else {
        StackR1State merged = dst;
        if (dst.tainted || cur.tainted) {
          merged.tainted = true;
        } else if (is_back_edge(u, v) &&
                   (cur.min_delta != dst.min_delta ||
                    cur.max_delta != dst.max_delta)) {
          merged.tainted = true;
        } else {
          merged.min_delta = std::min(dst.min_delta, cur.min_delta);
          merged.max_delta = std::max(dst.max_delta, cur.max_delta);
          if (merged.min_delta < -kMaxCumulativeStackDelta ||
              merged.max_delta > kMaxCumulativeStackDelta) {
            merged.tainted = true;
          }
        }
        if (merged != dst) {
          dst = merged;
          if (!in_worklist[v]) {
            worklist.push_back(v);
            in_worklist[v] = 1;
          }
        }
      }
    }
  }

  for (size_t i = 0; i < n; ++i) {
    if (dfs_state[i] != 0) {
      continue;
    }
    for (const hir::Instr* instr = blocks[i]->instr_head; instr;
         instr = instr->next) {
      const hir::Opcode op = instr->GetOpcodeNum();
      if ((op == hir::OPCODE_LOAD || op == hir::OPCODE_LOAD_OFFSET ||
           op == hir::OPCODE_STORE || op == hir::OPCODE_STORE_OFFSET) &&
          (instr->flags & hir::LOAD_STORE_NO_PHYSICAL_SKEW) != 0) {
        uses_no_skew = true;
        has_unsafe_r1_write = true;
      }
    }
  }

  *out_uses_no_skew = uses_no_skew;
  *out_has_unsafe_r1_write = has_unsafe_r1_write;
}

inline void StripNoPhysicalSkewFlags(hir::HIRBuilder* builder) {
  for (hir::Block* block = builder->first_block(); block; block = block->next) {
    for (hir::Instr* instr = block->instr_head; instr; instr = instr->next) {
      if (instr->flags & hir::LOAD_STORE_NO_PHYSICAL_SKEW) {
        instr->flags &=
            ~static_cast<uint16_t>(hir::LOAD_STORE_NO_PHYSICAL_SKEW);
      }
    }
  }
}

}  // namespace a64
}  // namespace backend
}  // namespace cpu
}  // namespace xe

#endif  // XENIA_CPU_BACKEND_A64_A64_STACK_SKEW_ANALYSIS_H_
