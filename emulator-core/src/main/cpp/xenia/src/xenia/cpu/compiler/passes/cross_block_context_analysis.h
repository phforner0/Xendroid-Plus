/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_CPU_COMPILER_PASSES_CROSS_BLOCK_CONTEXT_ANALYSIS_H_
#define XENIA_CPU_COMPILER_PASSES_CROSS_BLOCK_CONTEXT_ANALYSIS_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace xe {
namespace cpu {
namespace compiler {
namespace passes {
namespace cross_block {

// Fixed 229-slot context register universe:
//   0..31   : cr0..cr7 (1 byte per slot, 32 B total)
//   32..63  : r[0..31] (8 bytes per slot)
//   64..95  : f[0..31] (8 bytes per slot)
//   96      : ctr      (8 bytes)
//   97      : lr       (8 bytes)
//   98..100 : xer_ca, xer_ov, xer_so (1 byte per slot)
//   101..228: v[0..127] (16 bytes per slot, gated by track_vec128)
inline constexpr size_t kCrSlotStart = 0;
inline constexpr size_t kCrSlotCount = 32;
inline constexpr size_t kGprSlotStart = 32;
inline constexpr size_t kGprSlotCount = 32;
inline constexpr size_t kFprSlotStart = 64;
inline constexpr size_t kFprSlotCount = 32;
inline constexpr size_t kCtrSlot = 96;
inline constexpr size_t kLrSlot = 97;
inline constexpr size_t kXerCaSlot = 98;
inline constexpr size_t kXerOvSlot = 99;
inline constexpr size_t kXerSoSlot = 100;
inline constexpr size_t kVecSlotStart = 101;
inline constexpr size_t kVecSlotCount = 128;
inline constexpr size_t kTotalSlots = 229;

struct ContextBitset {
  static constexpr size_t kWords = 4;
  static constexpr uint64_t kTopWordMask =
      (uint64_t(1) << (kTotalSlots - 192)) - 1;

  std::array<uint64_t, kWords> words = {};

  constexpr bool Test(size_t idx) const {
    return idx < kTotalSlots &&
           ((words[idx >> 6] >> (idx & 63)) & uint64_t(1)) != 0;
  }

  constexpr void Set(size_t idx) {
    if (idx < kTotalSlots) {
      words[idx >> 6] |= (uint64_t(1) << (idx & 63));
    }
  }

  constexpr void Reset(size_t idx) {
    if (idx < kTotalSlots) {
      words[idx >> 6] &= ~(uint64_t(1) << (idx & 63));
    }
  }

  constexpr void Clear() { words = {}; }

  constexpr bool Any() const {
    return (words[0] | words[1] | words[2] | (words[3] & kTopWordMask)) != 0;
  }

  constexpr bool operator==(const ContextBitset& rhs) const {
    return words == rhs.words;
  }

  constexpr bool operator!=(const ContextBitset& rhs) const {
    return words != rhs.words;
  }

  constexpr ContextBitset operator|(const ContextBitset& rhs) const {
    ContextBitset out;
    for (size_t i = 0; i < kWords; ++i) {
      out.words[i] = words[i] | rhs.words[i];
    }
    return out;
  }

  constexpr ContextBitset operator&(const ContextBitset& rhs) const {
    ContextBitset out;
    for (size_t i = 0; i < kWords; ++i) {
      out.words[i] = words[i] & rhs.words[i];
    }
    return out;
  }

  constexpr ContextBitset operator~() const {
    ContextBitset out;
    out.words[0] = ~words[0];
    out.words[1] = ~words[1];
    out.words[2] = ~words[2];
    out.words[3] = (~words[3]) & kTopWordMask;
    return out;
  }

  constexpr ContextBitset& operator|=(const ContextBitset& rhs) {
    for (size_t i = 0; i < kWords; ++i) {
      words[i] |= rhs.words[i];
    }
    return *this;
  }

  constexpr ContextBitset& operator&=(const ContextBitset& rhs) {
    for (size_t i = 0; i < kWords; ++i) {
      words[i] &= rhs.words[i];
    }
    return *this;
  }
};

// Byte offsets into PPCContext (defaults match ppc_context.h; verified by
// static_assert in dead_context_store_elimination_pass.cc).
struct ContextLayout {
  uint32_t cr_offset = 0;
  uint32_t r_offset = 40;
  uint32_t ctr_offset = 296;
  uint32_t lr_offset = 304;
  uint32_t f_offset = 320;
  uint32_t v_offset = 576;
  uint32_t xer_ca_offset = 2640;
  uint32_t xer_ov_offset = 2641;
  uint32_t xer_so_offset = 2642;
};

inline ContextBitset MakeAllTrackedMask(bool track_all_context,
                                        bool track_vec128) {
  ContextBitset mask;
  // cr0..cr7 (slots 0..31) are always tracked.
  mask.words[0] = 0xFFFFFFFFull;
  if (!track_all_context) {
    return mask;
  }
  // r[0..31] (32..63) + f[0..31] (64..95) + ctr(96) + lr(97) + xer(98..100).
  mask.words[0] |= 0xFFFFFFFF00000000ull;
  mask.words[1] = 0xFFFFFFFFull | (uint64_t(0x1Full) << 32);
  if (track_vec128) {
    // v[0..127] (101..228):
    // word 1 bits 37..63 (27 slots: 101..127)
    mask.words[1] |= (~uint64_t(0)) << 37;
    // word 2 bits 0..63 (64 slots: 128..191)
    mask.words[2] = ~uint64_t(0);
    // word 3 bits 0..36 (37 slots: 192..228)
    mask.words[3] = ContextBitset::kTopWordMask;
  }
  return mask;
}

struct ContextAccessMasks {
  ContextBitset touch_mask;
  ContextBitset kill_mask;
  uint32_t covered_bytes = 0;
};

inline ContextAccessMasks ComputeContextAccessMasks(const ContextLayout& layout,
                                                    uint32_t offset,
                                                    uint32_t size,
                                                    bool track_all_context,
                                                    bool track_vec128) {
  ContextAccessMasks out;
  if (size == 0) {
    return out;
  }
  const uint64_t req_start = offset;
  const uint64_t req_end = req_start + size;

  auto record_slot = [&](size_t slot, uint64_t s_start, uint64_t s_size) {
    const uint64_t s_end = s_start + s_size;
    if (req_start < s_end && s_start < req_end) {
      out.touch_mask.Set(slot);
      const uint64_t ov_start = std::max(req_start, s_start);
      const uint64_t ov_end = std::min(req_end, s_end);
      out.covered_bytes += static_cast<uint32_t>(ov_end - ov_start);
      if (req_start <= s_start && s_end <= req_end) {
        out.kill_mask.Set(slot);
      }
    }
  };

  auto scan_uniform = [&](size_t slot_base, size_t count, uint64_t base_off,
                          uint64_t elem_size) {
    const uint64_t region_end = base_off + count * elem_size;
    if (req_end <= base_off || req_start >= region_end) {
      return;
    }
    const size_t first =
        req_start <= base_off
            ? 0
            : static_cast<size_t>((req_start - base_off) / elem_size);
    const size_t last =
        std::min(count - 1,
                 static_cast<size_t>((req_end - 1 - base_off) / elem_size));
    for (size_t idx = first; idx <= last; ++idx) {
      record_slot(slot_base + idx, base_off + idx * elem_size, elem_size);
    }
  };

  scan_uniform(kCrSlotStart, kCrSlotCount, layout.cr_offset, 1);
  if (!track_all_context) {
    return out;
  }

  scan_uniform(kGprSlotStart, kGprSlotCount, layout.r_offset, 8);
  scan_uniform(kFprSlotStart, kFprSlotCount, layout.f_offset, 8);
  record_slot(kCtrSlot, layout.ctr_offset, 8);
  record_slot(kLrSlot, layout.lr_offset, 8);
  record_slot(kXerCaSlot, layout.xer_ca_offset, 1);
  record_slot(kXerOvSlot, layout.xer_ov_offset, 1);
  record_slot(kXerSoSlot, layout.xer_so_offset, 1);

  if (track_vec128) {
    scan_uniform(kVecSlotStart, kVecSlotCount, layout.v_offset, 16);
  }
  return out;
}

// =============================================================================
// Cross-Block Backward Context Liveness Solver (Extended Blocks)
// =============================================================================

enum class LivenessInstrKind {
  kOther,
  kLoadContext,
  kStoreContext,
  kBranch,
  kBranchCond,
  kCheckPreempt,
  kContextBarrier,
  kVolatile,
  kMemory,
};

struct LivenessInstr {
  LivenessInstrKind kind = LivenessInstrKind::kOther;
  uint32_t offset = 0;
  uint32_t size = 0;
  uint32_t target_block = UINT32_MAX;
};

struct LivenessBlock {
  std::vector<LivenessInstr> instrs;
  std::vector<uint32_t> successors;
};

struct LivenessResult {
  std::vector<ContextBitset> live_in;
  std::vector<std::vector<bool>> dead_store;
};

inline LivenessResult SolveCrossBlockContextLiveness(
    const std::vector<LivenessBlock>& blocks, const ContextLayout& layout,
    bool track_all_context, bool track_vec128) {
  LivenessResult result;
  const size_t n = blocks.size();
  if (n == 0) {
    return result;
  }

  const ContextBitset all_tracked =
      MakeAllTrackedMask(track_all_context, track_vec128);
  result.live_in.resize(n);
  result.dead_store.resize(n);
  for (size_t b = 0; b < n; ++b) {
    result.dead_store[b].assign(blocks[b].instrs.size(), false);
  }

  auto live_out_of = [&](size_t b) -> ContextBitset {
    const auto& succs = blocks[b].successors;
    if (succs.empty()) {
      return all_tracked;
    }
    ContextBitset out;
    for (uint32_t succ : succs) {
      if (succ < n) {
        out |= result.live_in[succ];
      } else {
        out |= all_tracked;
      }
    }
    return out;
  };

  auto transfer_block = [&](size_t b, bool apply) -> ContextBitset {
    ContextBitset live = live_out_of(b);
    const auto& instrs = blocks[b].instrs;
    for (size_t idx = instrs.size(); idx-- > 0;) {
      const auto& ins = instrs[idx];
      switch (ins.kind) {
        case LivenessInstrKind::kBranch: {
          live = (ins.target_block < n) ? result.live_in[ins.target_block]
                                        : all_tracked;
          break;
        }
        case LivenessInstrKind::kBranchCond: {
          live |= (ins.target_block < n) ? result.live_in[ins.target_block]
                                         : all_tracked;
          break;
        }
        case LivenessInstrKind::kLoadContext: {
          const auto masks = ComputeContextAccessMasks(
              layout, ins.offset, ins.size, track_all_context, track_vec128);
          live |= masks.touch_mask;
          break;
        }
        case LivenessInstrKind::kStoreContext: {
          const auto masks = ComputeContextAccessMasks(
              layout, ins.offset, ins.size, track_all_context, track_vec128);
          if (ins.size > 0 && masks.covered_bytes == ins.size &&
              !(masks.touch_mask & live).Any()) {
            if (apply) {
              result.dead_store[b][idx] = true;
            }
          } else {
            live &= ~masks.kill_mask;
          }
          break;
        }
        case LivenessInstrKind::kCheckPreempt: {
          // PreemptCurrentFiber reads PPCContext::r[13] (KPCR) and, in user
          // mode, reads/restores cr0..cr7 via SaveInterruptedUserCr().
          for (size_t s = kCrSlotStart; s < kCrSlotStart + kCrSlotCount; ++s) {
            live.Set(s);
          }
          if (track_all_context) {
            live.Set(kGprSlotStart + 13u);
          }
          break;
        }
        case LivenessInstrKind::kOther:
          break;
        case LivenessInstrKind::kContextBarrier:
        case LivenessInstrKind::kVolatile:
        case LivenessInstrKind::kMemory:
          live = all_tracked;
          break;
      }
    }
    return live;
  };

  const size_t max_iters = std::max<size_t>(256, n * 4 + 16);
  bool converged = false;
  for (size_t iter = 0; iter < max_iters; ++iter) {
    bool changed = false;
    for (size_t b = n; b-- > 0;) {
      const ContextBitset new_live_in = transfer_block(b, false);
      if (new_live_in != result.live_in[b]) {
        result.live_in[b] = new_live_in;
        changed = true;
      }
    }
    if (!changed) {
      converged = true;
      break;
    }
  }

  if (converged) {
    for (size_t b = 0; b < n; ++b) {
      transfer_block(b, true);
    }
  } else {
    std::fill(result.live_in.begin(), result.live_in.end(), all_tracked);
  }
  return result;
}

// =============================================================================
// Natural Loop Detection & Register Budgeter (1-4 Blocks, Call-Free)
// =============================================================================

enum class LoopInstrKind {
  kOther,
  kLoadContext,
  kStoreContext,
  kBranch,
  kBranchCond,
  kCheckPreempt,
  kCall,
  kTrap,
  kReturn,
  kContextBarrier,
  kVolatile,
  kMemory,
};

struct LoopInstr {
  LoopInstrKind kind = LoopInstrKind::kOther;
  uint32_t offset = 0;
  uint32_t size = 0;
  uint32_t target_block = UINT32_MAX;
  bool clobbers_volatile_gprs = false;
  bool is_unconditional_jump = false;
};

struct LoopBlock {
  std::vector<LoopInstr> instrs;
  uint32_t fallthrough_block = UINT32_MAX;
};

struct LoopAnalysisOptions {
  bool promote_vec128 = false;
  bool defer_stores_across_memory = false;
  uint32_t max_regs = 6;
};

struct LoopPromotedSlot {
  uint32_t slot = 0;
  uint32_t offset = 0;
  uint32_t size = 0;
  uint32_t reads = 0;
  uint32_t writes = 0;
  uint32_t host_reg_index = 0;
  bool is_vec = false;
  bool is_float = false;
};

struct LoopPlan {
  uint32_t header = UINT32_MAX;
  std::vector<uint32_t> blocks;
  std::vector<size_t> loop_end_instr_index;
  std::vector<LoopPromotedSlot> promoted;
  bool clobbers_volatile_gprs = false;
  bool has_memory_ops = false;
  uint32_t score = 0;
};

inline std::vector<LoopPlan> SelectPromotableLoops(
    const std::vector<LoopBlock>& blocks, const ContextLayout& layout,
    const LoopAnalysisOptions& opts) {
  std::vector<LoopPlan> result;
  const size_t n = blocks.size();
  if (n == 0 || opts.max_regs == 0) {
    return result;
  }

  // 1. Reconstruct CFG from surviving instructions (ignoring stale Edge ptrs).
  constexpr size_t kNoInstr = std::numeric_limits<size_t>::max();
  std::vector<size_t> exec_end(n, kNoInstr);
  std::vector<bool> falls_through(n, false);
  std::vector<std::vector<uint32_t>> succs(n);
  std::vector<std::vector<uint32_t>> preds(n);

  auto add_edge = [&](uint32_t src, uint32_t dst) {
    if (dst >= n) {
      return;
    }
    auto& s = succs[src];
    if (std::find(s.begin(), s.end(), dst) == s.end()) {
      s.push_back(dst);
      preds[dst].push_back(src);
    }
  };

  for (size_t b = 0; b < n; ++b) {
    const auto& instrs = blocks[b].instrs;
    bool ended_unconditionally = false;
    for (size_t i = 0; i < instrs.size(); ++i) {
      const auto& ins = instrs[i];
      exec_end[b] = i;
      if (ins.kind == LoopInstrKind::kBranch) {
        add_edge(static_cast<uint32_t>(b), ins.target_block);
        ended_unconditionally = true;
        break;
      } else if (ins.kind == LoopInstrKind::kBranchCond) {
        add_edge(static_cast<uint32_t>(b), ins.target_block);
      } else if (ins.kind == LoopInstrKind::kReturn ||
                 ins.kind == LoopInstrKind::kTrap ||
                 ins.is_unconditional_jump) {
        ended_unconditionally = true;
        break;
      }
    }
    if (!ended_unconditionally && blocks[b].fallthrough_block < n) {
      falls_through[b] = true;
      add_edge(static_cast<uint32_t>(b), blocks[b].fallthrough_block);
    }
  }

  // 2. Reachability and RPO from entry block 0.
  std::vector<bool> reachable(n, false);
  std::vector<uint32_t> postorder;
  postorder.reserve(n);
  {
    struct Frame {
      uint32_t b;
      size_t succ_idx;
    };
    std::vector<Frame> stack;
    reachable[0] = true;
    stack.push_back({0u, 0u});
    while (!stack.empty()) {
      auto& top = stack.back();
      if (top.succ_idx < succs[top.b].size()) {
        const uint32_t nxt = succs[top.b][top.succ_idx++];
        if (!reachable[nxt]) {
          reachable[nxt] = true;
          stack.push_back({nxt, 0u});
        }
      } else {
        postorder.push_back(top.b);
        stack.pop_back();
      }
    }
  }
  std::vector<uint32_t> rpo(postorder.rbegin(), postorder.rend());

  // 3. Dominator sets using dynamic bitsets (works for any block count <= 64K).
  const size_t dom_words = (n + 63) / 64;
  std::vector<std::vector<uint64_t>> dom(
      n, std::vector<uint64_t>(dom_words, 0));
  auto set_dom = [&](uint32_t b, uint32_t d) {
    dom[b][d >> 6] |= (uint64_t(1) << (d & 63));
  };
  auto is_dom = [&](uint32_t d, uint32_t b) -> bool {
    return ((dom[b][d >> 6] >> (d & 63)) & uint64_t(1)) != 0;
  };
  std::vector<uint64_t> all_reach_bits(dom_words, 0);
  for (uint32_t b : rpo) {
    all_reach_bits[b >> 6] |= (uint64_t(1) << (b & 63));
  }
  set_dom(0, 0);
  for (uint32_t b : rpo) {
    if (b != 0) {
      dom[b] = all_reach_bits;
    }
  }

  std::vector<uint64_t> tmp_dom(dom_words, 0);
  bool dom_changed = true;
  while (dom_changed) {
    dom_changed = false;
    for (uint32_t b : rpo) {
      if (b == 0) {
        continue;
      }
      bool first_pred = true;
      for (uint32_t p : preds[b]) {
        if (!reachable[p]) {
          continue;
        }
        if (first_pred) {
          tmp_dom = dom[p];
          first_pred = false;
        } else {
          for (size_t w = 0; w < dom_words; ++w) {
            tmp_dom[w] &= dom[p][w];
          }
        }
      }
      if (first_pred) {
        std::fill(tmp_dom.begin(), tmp_dom.end(), 0);
      }
      tmp_dom[b >> 6] |= (uint64_t(1) << (b & 63));
      if (tmp_dom != dom[b]) {
        dom[b] = tmp_dom;
        dom_changed = true;
      }
    }
  }

  // 4. Collect back-edges grouped by header h.
  std::vector<std::vector<uint32_t>> latches_for_header(n);
  for (uint32_t u : rpo) {
    for (uint32_t h : succs[u]) {
      if (is_dom(h, u)) {
        latches_for_header[h].push_back(u);
      }
    }
  }

  struct SlotCandidate {
    uint32_t slot = 0;
    uint32_t offset = 0;
    uint32_t size = 0;
    uint32_t reads = 0;
    uint32_t writes = 0;
    bool is_vec = false;
    bool is_float = false;
    bool aliased = false;
    uint32_t score = 0;
  };

  auto slot_info_for = [&](uint32_t offset, uint32_t size,
                           SlotCandidate& out) -> bool {
    if (size == 8 && offset >= layout.r_offset &&
        offset < layout.r_offset + kGprSlotCount * 8 &&
        ((offset - layout.r_offset) & 7u) == 0) {
      const uint32_t k = (offset - layout.r_offset) / 8u;
      out.slot = static_cast<uint32_t>(kGprSlotStart + k);
      out.offset = offset;
      out.size = 8;
      out.is_vec = false;
      out.is_float = false;
      return true;
    }
    if (size == 8 && offset == layout.ctr_offset) {
      out.slot = static_cast<uint32_t>(kCtrSlot);
      out.offset = offset;
      out.size = 8;
      out.is_vec = false;
      out.is_float = false;
      return true;
    }
    if (size == 8 && offset == layout.lr_offset) {
      out.slot = static_cast<uint32_t>(kLrSlot);
      out.offset = offset;
      out.size = 8;
      out.is_vec = false;
      out.is_float = false;
      return true;
    }
    if (size == 8 && offset >= layout.f_offset &&
        offset < layout.f_offset + kFprSlotCount * 8 &&
        ((offset - layout.f_offset) & 7u) == 0) {
      const uint32_t k = (offset - layout.f_offset) / 8u;
      out.slot = static_cast<uint32_t>(kFprSlotStart + k);
      out.offset = offset;
      out.size = 8;
      out.is_vec = false;
      out.is_float = true;
      return true;
    }
    if (opts.promote_vec128 && size == 16 && offset >= layout.v_offset &&
        offset < layout.v_offset + kVecSlotCount * 16 &&
        ((offset - layout.v_offset) & 15u) == 0) {
      const uint32_t k = (offset - layout.v_offset) / 16u;
      out.slot = static_cast<uint32_t>(kVecSlotStart + k);
      out.offset = offset;
      out.size = 16;
      out.is_vec = true;
      out.is_float = false;
      return true;
    }
    return false;
  };

  std::vector<LoopPlan> candidates;
  std::vector<bool> in_loop(n, false);

  for (uint32_t h = 0; h < static_cast<uint32_t>(n); ++h) {
    if (latches_for_header[h].empty()) {
      continue;
    }

    // Build natural loop L(h) by backward predecessor closure from latches.
    std::fill(in_loop.begin(), in_loop.end(), false);
    std::vector<uint32_t> loop_blocks;
    std::vector<uint32_t> worklist;
    in_loop[h] = true;
    loop_blocks.push_back(h);
    for (uint32_t latch : latches_for_header[h]) {
      if (!in_loop[latch]) {
        in_loop[latch] = true;
        loop_blocks.push_back(latch);
        worklist.push_back(latch);
      }
    }
    bool too_large = loop_blocks.size() > 4;
    while (!worklist.empty() && !too_large) {
      const uint32_t cur = worklist.back();
      worklist.pop_back();
      for (uint32_t p : preds[cur]) {
        if (!reachable[p]) {
          continue;
        }
        if (!in_loop[p]) {
          in_loop[p] = true;
          loop_blocks.push_back(p);
          if (loop_blocks.size() > 4) {
            too_large = true;
            break;
          }
          worklist.push_back(p);
        }
      }
    }
    if (too_large || loop_blocks.empty() || loop_blocks.size() > 4) {
      continue;
    }

    // Deterministic order: header first, then ascending block index.
    std::sort(loop_blocks.begin(), loop_blocks.end());
    auto hdr_it = std::find(loop_blocks.begin(), loop_blocks.end(), h);
    if (hdr_it != loop_blocks.begin() && hdr_it != loop_blocks.end()) {
      std::rotate(loop_blocks.begin(), hdr_it, hdr_it + 1);
    }

    // Strict single-entry: no block b in L \ {h} has a reachable pred outside L.
    bool single_entry = true;
    for (uint32_t b : loop_blocks) {
      if (b == h) {
        continue;
      }
      for (uint32_t p : preds[b]) {
        if (reachable[p] && !in_loop[p]) {
          single_entry = false;
          break;
        }
      }
      if (!single_entry) {
        break;
      }
    }
    if (!single_entry) {
      continue;
    }

    // Compute loop_end[i] for each b = loop_blocks[i], enforcing extended-block
    // safety: post-loop instructions after loop_end must not branch/fall into L.
    std::vector<size_t> loop_ends(loop_blocks.size(), kNoInstr);
    bool ext_safe = true;
    for (size_t idx = 0; idx < loop_blocks.size(); ++idx) {
      const uint32_t b = loop_blocks[idx];
      const size_t end_i = exec_end[b];
      if (falls_through[b] && in_loop[blocks[b].fallthrough_block]) {
        loop_ends[idx] = end_i;
        continue;
      }
      if (end_i == kNoInstr) {
        ext_safe = false;
        break;
      }
      size_t last_loop_br = kNoInstr;
      for (size_t i = 0; i <= end_i; ++i) {
        const auto& ins = blocks[b].instrs[i];
        if ((ins.kind == LoopInstrKind::kBranch ||
             ins.kind == LoopInstrKind::kBranchCond) &&
            ins.target_block < n && in_loop[ins.target_block]) {
          last_loop_br = i;
        }
      }
      if (last_loop_br == kNoInstr) {
        ext_safe = false;
        break;
      }
      if (last_loop_br < end_i &&
          blocks[b].instrs[last_loop_br].kind != LoopInstrKind::kBranchCond) {
        ext_safe = false;
        break;
      }
      loop_ends[idx] = last_loop_br;
    }
    if (!ext_safe) {
      continue;
    }

    // Check disqualifiers & collect context accesses inside [0 .. loop_end].
    bool disqualified = false;
    bool clobbers_volatile_gprs = false;
    bool has_memory_ops = false;
    bool has_check_preempt = false;
    std::vector<SlotCandidate> slot_cands;
    std::vector<std::pair<uint32_t, uint32_t>> all_ctx_ranges;

    for (size_t b_idx = 0; b_idx < loop_blocks.size() && !disqualified; ++b_idx) {
      const uint32_t b = loop_blocks[b_idx];
      const size_t end_i = loop_ends[b_idx];
      if (end_i == kNoInstr) {
        continue;
      }
      for (size_t i = 0; i <= end_i; ++i) {
        const auto& ins = blocks[b].instrs[i];
        switch (ins.kind) {
          case LoopInstrKind::kCall:
          case LoopInstrKind::kTrap:
          case LoopInstrKind::kReturn:
          case LoopInstrKind::kContextBarrier:
          case LoopInstrKind::kVolatile:
            disqualified = true;
            break;
          case LoopInstrKind::kMemory:
            has_memory_ops = true;
            break;
          case LoopInstrKind::kLoadContext:
          case LoopInstrKind::kStoreContext: {
            all_ctx_ranges.emplace_back(ins.offset, ins.size);
            SlotCandidate info;
            if (slot_info_for(ins.offset, ins.size, info)) {
              auto it = std::find_if(
                  slot_cands.begin(), slot_cands.end(),
                  [&](const SlotCandidate& c) { return c.slot == info.slot; });
              if (it == slot_cands.end()) {
                slot_cands.push_back(info);
                it = slot_cands.end() - 1;
              }
              if (ins.kind == LoopInstrKind::kLoadContext) {
                ++it->reads;
              } else {
                ++it->writes;
              }
            }
            break;
          }
          case LoopInstrKind::kCheckPreempt:
            // PreemptCurrentFiber reads PPCContext::r[13] (X_KPCR*); do not
            // promote r13 across a loop that contains a preemption check.
            has_check_preempt = true;
            break;
          case LoopInstrKind::kBranch:
          case LoopInstrKind::kBranchCond:
          case LoopInstrKind::kOther:
            break;
        }
        if (disqualified) {
          break;
        }
        if (ins.clobbers_volatile_gprs) {
          clobbers_volatile_gprs = true;
        }
      }
    }
    if (disqualified || slot_cands.empty()) {
      continue;
    }

    // Strict aliasing guard: exclude any candidate overlapped by a mismatched
    // context load/store anywhere in the loop body.
    for (auto& cand : slot_cands) {
      const uint64_t c_start = cand.offset;
      const uint64_t c_end = c_start + cand.size;
      for (const auto& r : all_ctx_ranges) {
        const uint64_t r_start = r.first;
        const uint64_t r_end = r_start + r.second;
        if (r_start < c_end && c_start < r_end &&
            (r.first != cand.offset || r.second != cand.size)) {
          cand.aliased = true;
          break;
        }
      }
    }

    std::vector<SlotCandidate> eligible;
    const uint32_t write_weight =
        (!has_memory_ops || opts.defer_stores_across_memory) ? 2u : 1u;
    for (auto& cand : slot_cands) {
      if (cand.aliased) {
        continue;
      }
      if (has_check_preempt && cand.slot == kGprSlotStart + 13u) {
        continue;
      }
      if (cand.reads >= 1u && (cand.reads + cand.writes) >= 2u) {
        cand.score = cand.reads * 2u + cand.writes * write_weight;
        eligible.push_back(cand);
      }
    }
    if (eligible.empty()) {
      continue;
    }

    std::sort(eligible.begin(), eligible.end(),
              [](const SlotCandidate& a, const SlotCandidate& b) {
                if (a.score != b.score) {
                  return a.score > b.score;
                }
                return a.slot < b.slot;
              });

    // ARM64 physical register budget:
    // - GPR callee-saved pool (indices 0..6 = x22..x28):
    //   use x28(6), x27(5), x26(4) first (<= 3 if ClobbersVolatileGprs),
    //   then x25(3), x24(2) if !ClobbersVolatileGprs (<= 5).
    // - VEC/FPR pool (indices 0..27 = v4..v31):
    //   use v31(27), v30(26), v29(25), v28(24) (<= 4).
    // - Global cap: min(6, max_regs).
    static constexpr std::array<uint32_t, 5> kGprRegPool = {6u, 5u, 4u, 3u, 2u};
    static constexpr std::array<uint32_t, 4> kVecRegPool = {27u, 26u, 25u, 24u};

    const uint32_t global_cap = std::min<uint32_t>(6u, opts.max_regs);
    const uint32_t gpr_cap = clobbers_volatile_gprs ? 3u : 5u;
    const uint32_t vec_cap = 4u;

    uint32_t used_gpr = 0;
    uint32_t used_vec = 0;
    LoopPlan plan;
    plan.header = h;
    plan.blocks = std::move(loop_blocks);
    plan.loop_end_instr_index = std::move(loop_ends);
    plan.clobbers_volatile_gprs = clobbers_volatile_gprs;
    plan.has_memory_ops = has_memory_ops;

    for (const auto& cand : eligible) {
      if (plan.promoted.size() >= global_cap) {
        break;
      }
      const bool uses_vec_file = cand.is_vec || cand.is_float;
      if (uses_vec_file) {
        if (used_vec >= vec_cap) {
          continue;
        }
        LoopPromotedSlot ps;
        ps.slot = cand.slot;
        ps.offset = cand.offset;
        ps.size = cand.size;
        ps.reads = cand.reads;
        ps.writes = cand.writes;
        ps.host_reg_index = kVecRegPool[used_vec++];
        ps.is_vec = cand.is_vec;
        ps.is_float = cand.is_float;
        plan.promoted.push_back(ps);
        plan.score += cand.score;
      } else {
        if (used_gpr >= gpr_cap) {
          continue;
        }
        LoopPromotedSlot ps;
        ps.slot = cand.slot;
        ps.offset = cand.offset;
        ps.size = cand.size;
        ps.reads = cand.reads;
        ps.writes = cand.writes;
        ps.host_reg_index = kGprRegPool[used_gpr++];
        ps.is_vec = false;
        ps.is_float = false;
        plan.promoted.push_back(ps);
        plan.score += cand.score;
      }
    }

    if (!plan.promoted.empty()) {
      candidates.push_back(std::move(plan));
    }
  }

  // 5. Select strictly disjoint loops, preferring innermost (fewer blocks),
  //    then higher total access score, then smaller header.
  std::sort(candidates.begin(), candidates.end(),
            [](const LoopPlan& a, const LoopPlan& b) {
              if (a.blocks.size() != b.blocks.size()) {
                return a.blocks.size() < b.blocks.size();
              }
              if (a.score != b.score) {
                return a.score > b.score;
              }
              return a.header < b.header;
            });

  std::vector<bool> block_claimed(n, false);
  for (auto& cand : candidates) {
    bool overlaps = false;
    for (uint32_t b : cand.blocks) {
      if (block_claimed[b]) {
        overlaps = true;
        break;
      }
    }
    if (overlaps) {
      continue;
    }
    for (uint32_t b : cand.blocks) {
      block_claimed[b] = true;
    }
    result.push_back(std::move(cand));
  }

  return result;
}

}  // namespace cross_block
}  // namespace passes
}  // namespace compiler
}  // namespace cpu
}  // namespace xe

#endif  // XENIA_CPU_COMPILER_PASSES_CROSS_BLOCK_CONTEXT_ANALYSIS_H_
