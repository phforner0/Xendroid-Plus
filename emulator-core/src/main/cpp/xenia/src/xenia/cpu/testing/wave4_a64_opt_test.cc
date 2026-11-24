/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/cpu/testing/util.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "xenia/base/byte_order.h"
#include "xenia/base/cvar.h"
#include "xenia/cpu/compiler/compiler.h"
#include "xenia/cpu/compiler/compiler_passes.h"
#include "xenia/cpu/compiler/passes/dead_context_store_elimination_pass.h"
#include "xenia/cpu/compiler/passes/loop_context_promotion_pass.h"
#include "xenia/cpu/hir/hir_builder.h"
#include "xenia/cpu/hir/opcodes.h"
#include "xenia/cpu/ppc/ppc_context.h"
#include "xenia/cpu/raw_module.h"

DECLARE_bool(context_promote_vec128);
DECLARE_bool(disable_context_promotion);
DECLARE_bool(debug);
DECLARE_bool(store_all_context_values);
DECLARE_bool(dse_cross_block_context);
DECLARE_bool(dse_cross_block_vec128);
DECLARE_bool(a64_loop_context_promotion);
DECLARE_int32(a64_loop_promote_max_regs);
DECLARE_bool(a64_loop_promote_defer_stores_across_memory);
DECLARE_bool(guest_scheduler);

using namespace xe;
using namespace xe::cpu;
using namespace xe::cpu::hir;
using namespace xe::cpu::testing;
using xe::cpu::ppc::PPCContext;
namespace passes = xe::cpu::compiler::passes;

namespace {

template <typename T>
struct ScopedCvar {
  T& ref;
  T saved;
  ScopedCvar(T& var, T val) : ref(var), saved(var) { ref = val; }
  ~ScopedCvar() { ref = saved; }
  ScopedCvar(const ScopedCvar&) = delete;
  ScopedCvar& operator=(const ScopedCvar&) = delete;
};

size_t CountLiveContextStoresAtOffset(HIRBuilder& b, uint32_t target_offset) {
  size_t count = 0;
  for (Block* blk = b.first_block(); blk; blk = blk->next) {
    for (Instr* i = blk->instr_head; i; i = i->next) {
      if (i->opcode == &OPCODE_STORE_CONTEXT_info &&
          static_cast<uint32_t>(i->src1.offset) == target_offset) {
        ++count;
      }
    }
  }
  return count;
}

size_t CountLiveContextLoadsAtOffset(HIRBuilder& b, uint32_t target_offset) {
  size_t count = 0;
  for (Block* blk = b.first_block(); blk; blk = blk->next) {
    for (Instr* i = blk->instr_head; i; i = i->next) {
      if (i->opcode == &OPCODE_LOAD_CONTEXT_info &&
          static_cast<uint32_t>(i->src1.offset) == target_offset) {
        ++count;
      }
    }
  }
  return count;
}

void BuiltinVecClobber(PPCContext* ctx, void* arg0, void* /*arg1*/) {
  auto* observed_v4 = static_cast<vec128_t*>(arg0);
  if (observed_v4) {
    *observed_v4 = ctx->v[4];
  }
  ctx->v[31] = vec128i(0x55555555u, 0x66666666u, 0x77777777u, 0x88888888u);
  ctx->r[3] = 0x123456789ABCDEF0ULL;
}

}  // namespace

// =============================================================================
// Pillar 1: Dedicated Vector Regression Suite (context_promote_vec128)
// =============================================================================

TEST_CASE("WAVE4_CTX_PROMOTE_VEC128_MEMORY_FAULT_PRESERVATION",
          "[wave4][vec128]") {
  ScopedCvar<bool> guard_vec(cvars::context_promote_vec128, true);
  ScopedCvar<bool> guard_disable(cvars::disable_context_promotion, false);
  ScopedCvar<bool> guard_debug(cvars::debug, false);
  ScopedCvar<bool> guard_store_all(cvars::store_all_context_values, false);

  const uint32_t v4_off = static_cast<uint32_t>(offsetof(PPCContext, v[4]));

  // Pass-level HIR test: StoreVR(4, sentinel) -> Load(addr, INT32_TYPE) ->
  // StoreVR(4, valB) -> Return.
  // Before the Wave 4 fix, ContextPromotionPass::RemoveDeadStoresBlock
  // excludes PPCContext::v from OPCODE_FLAG_MEMORY reset, eliminating the first
  // StoreVR(4) so a guest DataFaultHook during Load sees stale v[4]!
  {
    HIRBuilder b;
    auto addr = LoadGPR(b, 4);
    StoreVR(b, 4, b.LoadConstantVec128(vec128i(0x11111111u, 0x22222222u,
                                                0x33333333u, 0x44444444u)));
    auto loaded = b.Load(addr, INT32_TYPE);
    StoreGPR(b, 3, b.ZeroExtend(loaded, INT64_TYPE));
    StoreVR(b, 4, b.LoadConstantVec128(vec128i(0xAAAAAAAAu, 0xBBBBBBBBu,
                                                0xCCCCCCCCu, 0xDDDDDDDDu)));
    b.Return();
    REQUIRE(b.Finalize());

    auto memory = std::make_unique<Memory>();
    REQUIRE(memory->Initialize());
    auto processor = std::make_unique<Processor>(memory.get(), nullptr);
    compiler::Compiler compiler(processor.get());
    compiler.AddPass(std::make_unique<passes::ControlFlowAnalysisPass>());
    compiler.AddPass(std::make_unique<passes::ContextPromotionPass>());
    REQUIRE(compiler.Compile(&b));

    REQUIRE(CountLiveContextStoresAtOffset(b, v4_off) == 2u);
  }

  // Same check for OPCODE_LVL (also OPCODE_FLAG_MEMORY).
  {
    HIRBuilder b;
    auto addr = LoadGPR(b, 4);
    StoreVR(b, 4, b.LoadConstantVec128(vec128i(1, 2, 3, 4)));
    auto lvl = b.LoadVectorLeft(addr);
    StoreVR(b, 5, lvl);
    StoreVR(b, 4, b.LoadConstantVec128(vec128i(5, 6, 7, 8)));
    b.Return();
    REQUIRE(b.Finalize());

    auto memory = std::make_unique<Memory>();
    REQUIRE(memory->Initialize());
    auto processor = std::make_unique<Processor>(memory.get(), nullptr);
    compiler::Compiler compiler(processor.get());
    compiler.AddPass(std::make_unique<passes::ControlFlowAnalysisPass>());
    compiler.AddPass(std::make_unique<passes::ContextPromotionPass>());
    REQUIRE(compiler.Compile(&b));

    REQUIRE(CountLiveContextStoresAtOffset(b, v4_off) == 2u);
  }

  // Processor + DataFaultHook API + differential execution with valid memory.
  {
    auto build_fn = [](HIRBuilder& b) {
      auto addr = LoadGPR(b, 4);
      StoreVR(b, 4, b.LoadConstantVec128(vec128i(0xDEAD0001u, 0xDEAD0002u,
                                                  0xDEAD0003u, 0xDEAD0004u)));
      auto loaded = b.Load(addr, INT32_TYPE);
      StoreGPR(b, 3, b.ZeroExtend(loaded, INT64_TYPE));
      StoreVR(b, 4, b.LoadConstantVec128(vec128i(0xBEEF0001u, 0xBEEF0002u,
                                                  0xBEEF0003u, 0xBEEF0004u)));
      b.Return();
    };
    cvars::context_promote_vec128 = false;
    TestFunction ref_test(build_fn);
    cvars::context_promote_vec128 = true;
    TestFunction opt_test(build_fn);

    opt_test.processors[0]->set_data_fault_hook(
        [](PPCContext* /*ctx*/, uint32_t /*pc*/, bool /*is_store*/,
           uint32_t /*addr*/) -> uint32_t { return 0u; });

    uint32_t guest_va_ref = ref_test.memory->SystemHeapAlloc(64u, 16u);
    uint32_t guest_va_opt = opt_test.memory->SystemHeapAlloc(64u, 16u);
    REQUIRE(guest_va_ref != 0u);
    REQUIRE(guest_va_opt != 0u);
    *reinterpret_cast<uint32_t*>(ref_test.memory->TranslateVirtual(guest_va_ref)) =
        0x13572468u;
    *reinterpret_cast<uint32_t*>(opt_test.memory->TranslateVirtual(guest_va_opt)) =
        0x13572468u;

    vec128_t ref_v4 = {}, opt_v4 = {};
    uint64_t ref_r3 = 0, opt_r3 = 0;
    ref_test.Run(
        [&](PPCContext* ctx) { ctx->r[4] = guest_va_ref; },
        [&](PPCContext* ctx) {
          ref_v4 = ctx->v[4];
          ref_r3 = ctx->r[3];
        });
    opt_test.Run(
        [&](PPCContext* ctx) { ctx->r[4] = guest_va_opt; },
        [&](PPCContext* ctx) {
          opt_v4 = ctx->v[4];
          opt_r3 = ctx->r[3];
        });
    REQUIRE(ref_r3 == opt_r3);
    REQUIRE(ref_v4 == opt_v4);
    ref_test.memory->SystemHeapFree(guest_va_ref);
    opt_test.memory->SystemHeapFree(guest_va_opt);
  }
}

TEST_CASE("WAVE4_CTX_PROMOTE_CONTEXT_BARRIER_SYNC", "[wave4][vec128]") {
  ScopedCvar<bool> guard_vec(cvars::context_promote_vec128, true);
  ScopedCvar<bool> guard_disable(cvars::disable_context_promotion, false);
  ScopedCvar<bool> guard_debug(cvars::debug, false);
  ScopedCvar<bool> guard_store_all(cvars::store_all_context_values, false);

  const uint32_t msr_off = static_cast<uint32_t>(offsetof(PPCContext, msr));
  const uint32_t v4_off = static_cast<uint32_t>(offsetof(PPCContext, v[4]));

  HIRBuilder b;
  b.StoreContext(msr_off, b.LoadConstantUint64(0x8000ULL));
  StoreVR(b, 4, b.LoadConstantVec128(vec128i(10, 20, 30, 40)));
  b.ContextBarrier();
  b.StoreContext(msr_off, b.LoadConstantUint64(0ULL));
  auto reloaded_v4 = LoadVR(b, 4);
  StoreVR(b, 5, reloaded_v4);
  b.Return();
  REQUIRE(b.Finalize());

  auto memory = std::make_unique<Memory>();
  REQUIRE(memory->Initialize());
  auto processor = std::make_unique<Processor>(memory.get(), nullptr);
  compiler::Compiler compiler(processor.get());
  compiler.AddPass(std::make_unique<passes::ControlFlowAnalysisPass>());
  compiler.AddPass(std::make_unique<passes::ContextPromotionPass>());
  REQUIRE(compiler.Compile(&b));

  // First StoreContext(msr) before ContextBarrier must NOT be eliminated,
  // StoreVR(4) before ContextBarrier must NOT be eliminated, and LoadVR(4)
  // after ContextBarrier must NOT be replaced by OPCODE_ASSIGN.
  REQUIRE(CountLiveContextStoresAtOffset(b, msr_off) == 2u);
  REQUIRE(CountLiveContextStoresAtOffset(b, v4_off) == 1u);
  REQUIRE(CountLiveContextLoadsAtOffset(b, v4_off) == 1u);
}

TEST_CASE("WAVE4_VEC128_PROMOTED_EXTRACT_INSERT_PERMUTE", "[wave4][vec128]") {
  ScopedCvar<bool> guard_vec(cvars::context_promote_vec128, false);
  ScopedCvar<bool> guard_disable(cvars::disable_context_promotion, false);
  ScopedCvar<bool> guard_debug(cvars::debug, false);
  ScopedCvar<bool> guard_store_all(cvars::store_all_context_values, false);

  // Direct Value::Insert out-of-range index masking invariant check.
  {
    HIRBuilder b;
    Value* v_i8 = b.LoadConstantVec128(
        vec128b(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15));
    Value* idx_19 = b.LoadConstantUint8(19);  // 19 & 0xF == 3 -> 3 ^ 3 == 0
    Value* part_i8 = b.LoadConstantUint8(0xAAu);
    v_i8->Insert(idx_19, part_i8, INT8_TYPE);
    REQUIRE(v_i8->constant.v128.u8[(19u & 0xFu) ^ 3u] == 0xAAu);

    Value* v_i16 = b.LoadConstantVec128(vec128s(0, 1, 2, 3, 4, 5, 6, 7));
    Value* idx_11 = b.LoadConstantUint8(11);  // 11 & 0x7 == 3 -> 3 ^ 1 == 2
    Value* part_i16 = b.LoadConstantUint16(0xBEEFu);
    v_i16->Insert(idx_11, part_i16, INT16_TYPE);
    REQUIRE(v_i16->constant.v128.u16[(11u & 0x7u) ^ 1u] == 0xBEEFu);

    Value* v_i32 = b.LoadConstantVec128(vec128i(0, 1, 2, 3));
    Value* idx_7 = b.LoadConstantUint8(7);  // 7 & 0x3 == 3
    Value* part_i32 = b.LoadConstantUint32(0xCAFEBABEu);
    v_i32->Insert(idx_7, part_i32, INT32_TYPE);
    REQUIRE(v_i32->constant.v128.u32[7u & 0x3u] == 0xCAFEBABEu);
  }

  auto build_fn = [](HIRBuilder& b) {
    const vec128_t src_bytes = vec128b(
        0x80, 0x01, 0x82, 0x03, 0x84, 0x05, 0x86, 0x07,
        0x88, 0x09, 0x8A, 0x0B, 0x8C, 0x0D, 0x8E, 0x0F);
    const vec128_t alt_vec = vec128b(
        0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
        0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F);
    StoreVR(b, 4, b.LoadConstantVec128(src_bytes));
    StoreVR(b, 5, b.LoadConstantVec128(alt_vec));

    auto v4 = LoadVR(b, 4);
    auto v5 = LoadVR(b, 5);

    uint64_t acc_i8 = 0;
    Value* sum8 = b.LoadConstantUint64(0u);
    for (uint8_t idx = 0; idx < 16; ++idx) {
      auto ex = b.ZeroExtend(b.Extract(v4, idx, INT8_TYPE), INT64_TYPE);
      sum8 = b.Add(sum8, b.Shl(ex, static_cast<int8_t>(idx & 7)));
    }
    (void)acc_i8;
    Value* sum16 = b.LoadConstantUint64(0u);
    for (uint8_t idx = 0; idx < 8; ++idx) {
      auto ex = b.ZeroExtend(b.Extract(v4, idx, INT16_TYPE), INT64_TYPE);
      sum16 = b.Add(sum16, ex);
    }
    StoreGPR(b, 3, sum8);
    StoreGPR(b, 4, sum16);

    auto ins8 = b.Insert(v4, uint64_t(5), b.LoadConstantUint8(0x77u));
    auto ins16 = b.Insert(ins8, uint64_t(3), b.LoadConstantUint16(0x4321u));
    auto ins32 = b.Insert(ins16, uint64_t(2), b.LoadConstantUint32(0x11223344u));
    StoreVR(b, 6, ins32);

    auto perm8_ctrl = b.LoadConstantVec128(
        vec128b(0x00, 0x10, 0x03, 0x13, 0x07, 0x17, 0x0F, 0x1F,
                0x01, 0x11, 0x05, 0x15, 0x09, 0x19, 0x0D, 0x1D));
    StoreVR(b, 7, b.Permute(perm8_ctrl, v4, v5, INT8_TYPE));

    auto perm16_ctrl = b.LoadConstantVec128(vec128s(0, 8, 1, 9, 2, 10, 3, 11));
    StoreVR(b, 8, b.Permute(perm16_ctrl, v4, v5, INT16_TYPE));

    StoreVR(b, 9, b.Swizzle(ins32, INT32_TYPE, MakeSwizzleMask(3, 1, 2, 0)));
    b.Return();
  };

  cvars::context_promote_vec128 = false;
  TestFunction ref_test(build_fn);
  cvars::context_promote_vec128 = true;
  TestFunction opt_test(build_fn);

  std::array<vec128_t, 10> ref_v = {}, opt_v = {};
  uint64_t ref_r3 = 0, ref_r4 = 0, opt_r3 = 0, opt_r4 = 0;

  ref_test.Run(
      [](PPCContext*) {},
      [&](PPCContext* ctx) {
        ref_r3 = ctx->r[3];
        ref_r4 = ctx->r[4];
        for (size_t i = 0; i < 10; ++i) ref_v[i] = ctx->v[i];
      });
  opt_test.Run(
      [](PPCContext*) {},
      [&](PPCContext* ctx) {
        opt_r3 = ctx->r[3];
        opt_r4 = ctx->r[4];
        for (size_t i = 0; i < 10; ++i) opt_v[i] = ctx->v[i];
      });

  REQUIRE(ref_r3 == opt_r3);
  REQUIRE(ref_r4 == opt_r4);
  for (size_t i = 0; i < 10; ++i) {
    REQUIRE(ref_v[i] == opt_v[i]);
  }
}

TEST_CASE("WAVE4_VEC128_PROMOTED_ALU_SCRATCH_AND_SELECT", "[wave4][vec128]") {
  ScopedCvar<bool> guard_vec(cvars::context_promote_vec128, false);
  ScopedCvar<bool> guard_disable(cvars::disable_context_promotion, false);
  ScopedCvar<bool> guard_debug(cvars::debug, false);
  ScopedCvar<bool> guard_store_all(cvars::store_all_context_values, false);

  auto build_fn = [](HIRBuilder& b) {
    StoreVR(b, 4, b.LoadConstantVec128(vec128f(1.5f, -2.25f, 3.75f, 0.5f)));
    StoreVR(b, 5, b.LoadConstantVec128(vec128f(2.0f, 4.0f, -1.0f, 8.0f)));
    StoreVR(b, 6, b.LoadConstantVec128(vec128f(0.25f, 0.5f, 0.75f, 1.25f)));
    StoreVR(b, 7, b.LoadConstantVec128(vec128i(0x00000001u, 0x80000001u,
                                                0x3F800000u, 0xBF800000u)));
    StoreVR(b, 8, b.LoadConstantVec128(vec128i(0xFFFFFFFFu, 0u,
                                                0xFFFFFFFFu, 0u)));

    auto v4 = LoadVR(b, 4);
    auto v5 = LoadVR(b, 5);
    auto v6 = LoadVR(b, 6);
    auto v7 = LoadVR(b, 7);
    auto v8 = LoadVR(b, 8);

    auto dp3 = b.DotProduct3(v4, v5);
    auto dp4 = b.DotProduct4(v4, v5);
    StoreFPR(b, 3, b.Convert(dp3, FLOAT64_TYPE));
    StoreFPR(b, 4, b.Convert(dp4, FLOAT64_TYPE));

    auto sh_amt = b.LoadConstantVec128(vec128i(1, 2, 3, 4));
    StoreVR(b, 9, b.VectorShl(v4, sh_amt, INT32_TYPE));
    StoreVR(b, 10, b.VectorShr(v5, sh_amt, INT32_TYPE));
    StoreVR(b, 11, b.VectorRotateLeft(v6, sh_amt, INT32_TYPE));
    StoreVR(b, 12, b.VectorDenormFlush(v7));

    StoreVR(b, 13, b.MulAdd(v4, v5, v6));
    StoreVR(b, 14, b.MulSub(v4, v5, v6));
    StoreVR(b, 15, b.Select(v8, v4, v5));

    // Distinct non-zero float32/vec128 constants in same block (SameValueOrEqualConstant).
    auto cmp_eq = b.VectorCompareEQ(v4, b.LoadConstantVec128(vec128f(1.5f, 0.0f, 3.75f, 0.0f)),
                                    FLOAT32_TYPE);
    auto cmp_sgt = b.VectorCompareSGT(v4, b.LoadConstantVec128(vec128f(1.0f, -3.0f, 4.0f, 0.0f)),
                                      FLOAT32_TYPE);
    StoreVR(b, 16, b.Xor(cmp_eq, cmp_sgt));
    b.Return();
  };

  cvars::context_promote_vec128 = false;
  TestFunction ref_test(build_fn);
  cvars::context_promote_vec128 = true;
  TestFunction opt_test(build_fn);

  std::array<vec128_t, 17> ref_v = {}, opt_v = {};
  double ref_f3 = 0.0, ref_f4 = 0.0, opt_f3 = 0.0, opt_f4 = 0.0;

  ref_test.Run(
      [](PPCContext*) {},
      [&](PPCContext* ctx) {
        ref_f3 = ctx->f[3];
        ref_f4 = ctx->f[4];
        for (size_t i = 0; i < 17; ++i) ref_v[i] = ctx->v[i];
      });
  opt_test.Run(
      [](PPCContext*) {},
      [&](PPCContext* ctx) {
        opt_f3 = ctx->f[3];
        opt_f4 = ctx->f[4];
        for (size_t i = 0; i < 17; ++i) opt_v[i] = ctx->v[i];
      });

  uint64_t ref_f3_bits = 0, opt_f3_bits = 0, ref_f4_bits = 0, opt_f4_bits = 0;
  std::memcpy(&ref_f3_bits, &ref_f3, 8);
  std::memcpy(&opt_f3_bits, &opt_f3, 8);
  std::memcpy(&ref_f4_bits, &ref_f4, 8);
  std::memcpy(&opt_f4_bits, &opt_f4, 8);
  REQUIRE(ref_f3_bits == opt_f3_bits);
  REQUIRE(ref_f4_bits == opt_f4_bits);
  for (size_t i = 0; i < 17; ++i) {
    REQUIRE(ref_v[i] == opt_v[i]);
  }
}

TEST_CASE("WAVE4_VEC128_SPILL_REMAT_AND_CALL_EXTERN", "[wave4][vec128]") {
  ScopedCvar<bool> guard_vec(cvars::context_promote_vec128, false);
  ScopedCvar<bool> guard_disable(cvars::disable_context_promotion, false);
  ScopedCvar<bool> guard_debug(cvars::debug, false);
  ScopedCvar<bool> guard_store_all(cvars::store_all_context_values, false);

  auto run_case = [&](bool promote_vec, vec128_t& out_v3, vec128_t& out_v4,
                      vec128_t& out_observed, uint64_t& out_r3) {
    cvars::context_promote_vec128 = promote_vec;
    vec128_t observed_in_builtin = {};
    auto memory = std::make_unique<Memory>();
    REQUIRE(memory->Initialize());
    auto backend = CreateBackend();
    if (!backend) {
      return;
    }
    auto processor = std::make_unique<Processor>(memory.get(), nullptr);
    REQUIRE(processor->Setup(std::move(backend)));

    Function* builtin_fn = processor->DefineBuiltin(
        "Wave4VecSpillBuiltin", &BuiltinVecClobber, &observed_in_builtin,
        nullptr);
    REQUIRE(builtin_fn != nullptr);

    auto mod = std::make_unique<TestModule>(
        processor.get(), "Wave4SpillTest",
        [](uint32_t addr) { return addr == 0x80000000u; },
        [builtin_fn](HIRBuilder& b) {
          // 34 simultaneously live VEC128 values (> 28 A64 VEC regs) -> forces
          // SpillOneRegister + FindContextResidence rematerialization.
          std::array<Value*, 34> vals = {};
          for (size_t i = 0; i < vals.size(); ++i) {
            const uint32_t off =
                static_cast<uint32_t>(offsetof(PPCContext, v) + (i % 16) * 16);
            Value* base = b.LoadContext(off, VEC128_TYPE);
            vals[i] = b.VectorAdd(
                base,
                b.LoadConstantVec128(vec128i(static_cast<uint32_t>(i + 1),
                                             static_cast<uint32_t>(i + 2),
                                             static_cast<uint32_t>(i + 3),
                                             static_cast<uint32_t>(i + 4))),
                INT32_TYPE);
            if (i < 8) {
              StoreVR(b, static_cast<int>(16 + i), vals[i]);
            }
          }
          Value* acc = b.LoadConstantVec128(vec128i(0, 0, 0, 0));
          for (size_t i = 0; i < vals.size(); ++i) {
            acc = b.VectorAdd(acc, vals[i], INT32_TYPE);
          }
          StoreVR(b, 4, acc);
          b.ContextBarrier();
          b.CallExtern(builtin_fn);
          StoreVR(b, 3, b.VectorAdd(LoadVR(b, 4), LoadVR(b, 31), INT32_TYPE));
          b.Return();
          return true;
        });
    REQUIRE(processor->AddModule(std::move(mod)));
    processor->backend()->CommitExecutableRange(0x80000000u, 0x80010000u);

    Function* fn = processor->ResolveFunction(0x80000000u);
    REQUIRE(fn != nullptr);

    uint32_t stack_addr = memory->SystemHeapAlloc(64u * 1024u);
    auto ts = std::make_unique<ThreadState>(processor.get(), 0x400u,
                                            stack_addr + 64u * 1024u);
    PPCContext* ctx = ts->context();
    ctx->lr = 0xBCBCBCBCu;
    for (int i = 0; i < 16; ++i) {
      ctx->v[i] = vec128i(i * 3 + 1, i * 5 + 2, i * 7 + 3, i * 11 + 4);
    }
    fn->Call(ts.get(), static_cast<uint32_t>(ctx->lr));
    out_v3 = ctx->v[3];
    out_v4 = ctx->v[4];
    out_observed = observed_in_builtin;
    out_r3 = ctx->r[3];

    ts.reset();
    memory->SystemHeapFree(stack_addr);
  };

  vec128_t ref_v3 = {}, ref_v4 = {}, ref_obs = {};
  vec128_t opt_v3 = {}, opt_v4 = {}, opt_obs = {};
  uint64_t ref_r3 = 0, opt_r3 = 0;
  run_case(false, ref_v3, ref_v4, ref_obs, ref_r3);
  run_case(true, opt_v3, opt_v4, opt_obs, opt_r3);

  REQUIRE(ref_r3 == opt_r3);
  REQUIRE(ref_v3 == opt_v3);
  REQUIRE(ref_v4 == opt_v4);
  REQUIRE(ref_obs == opt_obs);
}

// =============================================================================
// Pillar 2: Cross-Block DSE & Loop-Local Context Promotion
// =============================================================================

TEST_CASE("WAVE4_DSE_CROSS_BLOCK_EXTENDED_AND_BARRIER", "[wave4][crossblock]") {
  ScopedCvar<bool> guard_dse(cvars::dse_cross_block_context, true);
  ScopedCvar<bool> guard_vec(cvars::dse_cross_block_vec128, true);
  ScopedCvar<bool> guard_promote_vec(cvars::context_promote_vec128, true);
  ScopedCvar<bool> guard_disable(cvars::disable_context_promotion, false);
  ScopedCvar<bool> guard_debug(cvars::debug, false);
  ScopedCvar<bool> guard_store_all(cvars::store_all_context_values, false);

  const uint32_t r5_off = static_cast<uint32_t>(offsetof(PPCContext, r[5]));
  const uint32_t r6_off = static_cast<uint32_t>(offsetof(PPCContext, r[6]));
  const uint32_t r7_off = static_cast<uint32_t>(offsetof(PPCContext, r[7]));
  const uint32_t cr0_off = static_cast<uint32_t>(offsetof(PPCContext, cr0));

  HIRBuilder b;
  auto* target_lbl = b.NewLabel();

  // r5: 8B store overwritten in target block (dead across block!),
  //     plus 4B partial write on r6 does NOT kill earlier 8B store on r6.
  StoreGPR(b, 5, b.LoadConstantUint64(0x1111ULL));
  StoreGPR(b, 6, b.LoadConstantUint64(0x2222333344445555ULL));
  b.StoreContext(r6_off, b.LoadConstantUint32(0x99999999u));  // partial 4B
  // r7: CHECK_PREEMPT is transparent -> earlier store is still dead when overwritten.
  StoreGPR(b, 7, b.LoadConstantUint64(0x7777ULL));
  b.CheckPreempt();
  StoreGPR(b, 7, b.LoadConstantUint64(0x8888ULL));
  // cr0_lt (1B) followed by cr0.value (4B) -> 1B store must be eliminated!
  b.StoreContext(cr0_off, b.LoadConstantUint8(1u));
  b.StoreContext(cr0_off, b.LoadConstantUint32(0x02000000u));
  // ContextBarrier preserves prior r6 stores, but r5 goes straight to Branch(target).
  b.Branch(target_lbl);
  b.MarkLabel(target_lbl);
  StoreGPR(b, 5, b.LoadConstantUint64(0x9999ULL));
  b.ContextBarrier();
  StoreGPR(b, 6, b.LoadConstantUint64(0xAAAAULL));
  b.Return();
  REQUIRE(b.Finalize());

  auto memory = std::make_unique<Memory>();
  REQUIRE(memory->Initialize());
  auto processor = std::make_unique<Processor>(memory.get(), nullptr);
  compiler::Compiler compiler(processor.get());
  compiler.AddPass(std::make_unique<passes::ControlFlowAnalysisPass>());
  compiler.AddPass(std::make_unique<passes::DeadContextStoreEliminationPass>());
  compiler.AddPass(std::make_unique<passes::DeadCodeEliminationPass>());
  REQUIRE(compiler.Compile(&b));

  // Dead cross-block r5 store eliminated -> only 1 StoreGPR(5) remains.
  REQUIRE(CountLiveContextStoresAtOffset(b, r5_off) == 1u);
  // CheckPreempt was transparent -> dead r7 store eliminated -> only 1 remains.
  REQUIRE(CountLiveContextStoresAtOffset(b, r7_off) == 1u);
  // Partial 4B write on r6 + ContextBarrier preserved both r6 8B & 4B stores -> 3 total.
  REQUIRE(CountLiveContextStoresAtOffset(b, r6_off) == 3u);
  // cr0_lt 1B store was killed by cr0.value 4B store -> only 1 cr0 store remains.
  REQUIRE(CountLiveContextStoresAtOffset(b, cr0_off) == 1u);
}

TEST_CASE("WAVE4_DSE_FIND_CONTEXT_RESIDENCE_FALLBACK", "[wave4][crossblock]") {
  ScopedCvar<bool> guard_dse(cvars::dse_cross_block_context, true);
  ScopedCvar<bool> guard_disable(cvars::disable_context_promotion, false);
  ScopedCvar<bool> guard_debug(cvars::debug, false);
  ScopedCvar<bool> guard_store_all(cvars::store_all_context_values, false);

  auto run_pressure = [&](bool enable_dse, uint64_t& out_r3, uint64_t& out_r4) {
    cvars::dse_cross_block_context = enable_dse;
    auto memory = std::make_unique<Memory>();
    REQUIRE(memory->Initialize());
    auto backend = CreateBackend();
    if (!backend) {
      return;
    }
    auto processor = std::make_unique<Processor>(memory.get(), nullptr);
    REQUIRE(processor->Setup(std::move(backend)));

    // Write a RawModule PPC function with >18 live GPRs where an early store
    // is eliminated by DSE across a branch while values spill in regalloc.
    const uint32_t code_base = 0x80000000u;
    const uint32_t code_size = 0x10000u;
    auto* vheap = memory->LookupHeap(code_base);
    REQUIRE(vheap->AllocFixed(code_base, code_size, 0x1000u,
                              kMemoryAllocationReserve | kMemoryAllocationCommit,
                              kMemoryProtectRead | kMemoryProtectWrite));
    auto w32 = [&](uint32_t addr, uint32_t val) {
      xe::store_and_swap<uint32_t>(memory->TranslateVirtual(addr), val);
    };
    const uint32_t fn_addr = code_base + 0x100u;
    // addi r5, r3, 1; b +8; nop; addi r5, r3, 2; add r3, r5, r4; blr
    uint32_t p = fn_addr;
    w32(p, 0x38A30001u); p += 4;  // addi r5, r3, 1
    w32(p, 0x48000008u); p += 4;  // b +8
    w32(p, 0x60000000u); p += 4;  // nop
    w32(p, 0x38A30002u); p += 4;  // addi r5, r3, 2
    w32(p, 0x7C652214u); p += 4;  // add r3, r5, r4
    w32(p, 0x4E800020u);          // blr

    auto raw = std::make_unique<RawModule>(processor.get());
    raw->set_name("Wave4DseResidenceTest");
    raw->set_executable(true);
    raw->SetAddressRange(code_base, code_size);
    REQUIRE(processor->AddModule(std::move(raw)));
    processor->backend()->CommitExecutableRange(code_base, code_base + code_size);

    uint32_t stack_addr = memory->SystemHeapAlloc(64u * 1024u);
    auto ts = std::make_unique<ThreadState>(processor.get(), 0x500u,
                                            stack_addr + 64u * 1024u);
    uint64_t args[2] = {10u, 20u};
    out_r3 = processor->Execute(ts.get(), fn_addr, args, 2);
    out_r4 = ts->context()->r[5];

    ts.reset();
    memory->SystemHeapFree(stack_addr);
    vheap->Release(code_base);
  };

  uint64_t ref_r3 = 0, ref_r5 = 0, opt_r3 = 0, opt_r5 = 0;
  run_pressure(false, ref_r3, ref_r5);
  run_pressure(true, opt_r3, opt_r5);
  REQUIRE(ref_r3 == 32u);
  REQUIRE(ref_r3 == opt_r3);
  REQUIRE(ref_r5 == opt_r5);
}

TEST_CASE("WAVE4_LOOP_PROMOTION_1_TO_4_BLOCKS_AND_CALL", "[wave4][loop]") {
  ScopedCvar<bool> guard_loop(cvars::a64_loop_context_promotion, false);
  ScopedCvar<int32_t> guard_max(cvars::a64_loop_promote_max_regs, 6);
  ScopedCvar<bool> guard_sched(cvars::guest_scheduler, false);
  ScopedCvar<bool> guard_disable(cvars::disable_context_promotion, false);
  ScopedCvar<bool> guard_debug(cvars::debug, false);
  ScopedCvar<bool> guard_store_all(cvars::store_all_context_values, false);

  auto run_ppc_suite = [&](bool enable_loop_promo, bool enable_sched,
                           uint64_t& out_loop1, uint64_t& out_loop3,
                           uint64_t& out_loop_call) {
    cvars::a64_loop_context_promotion = enable_loop_promo;
    cvars::guest_scheduler = enable_sched;

    auto memory = std::make_unique<Memory>();
    REQUIRE(memory->Initialize());
    auto backend = CreateBackend();
    if (!backend) {
      return;
    }
    auto processor = std::make_unique<Processor>(memory.get(), nullptr);
    REQUIRE(processor->Setup(std::move(backend)));

    const uint32_t code_base = 0x80000000u;
    const uint32_t code_size = 0x10000u;
    auto* vheap = memory->LookupHeap(code_base);
    REQUIRE(vheap->AllocFixed(code_base, code_size, 0x1000u,
                              kMemoryAllocationReserve | kMemoryAllocationCommit,
                              kMemoryProtectRead | kMemoryProtectWrite));
    auto w32 = [&](uint32_t addr, uint32_t val) {
      xe::store_and_swap<uint32_t>(memory->TranslateVirtual(addr), val);
    };

    // 1. 1-block countdown loop at 0x80000100:
    //    r3 = sum(i=0..r3-1), using r5/r6/ctr across iterations.
    const uint32_t fn1_addr = 0x80000100u;
    {
      uint32_t p = fn1_addr;
      w32(p, 0x7C6903A6u); p += 4;  // mtctr r3
      w32(p, 0x38600000u); p += 4;  // li r3, 0
      w32(p, 0x38A00001u); p += 4;  // li r5, 1
      // loop (fn1_addr + 0x0C):
      w32(p, 0x7C632A14u); p += 4;  // add r3, r3, r5
      w32(p, 0x38A50002u); p += 4;  // addi r5, r5, 2
      w32(p, 0x4200FFF8u); p += 4;  // bdnz -8 (to fn1_addr + 0x0C)
      w32(p, 0x4E800020u);          // blr
    }

    // 2. 3-block diamond loop at 0x80000200:
    //    r3 = trip count, r4 = initial acc; adds 3 on even steps, 7 on odd steps.
    const uint32_t fn3_addr = 0x80000200u;
    {
      uint32_t p = fn3_addr;
      w32(p, 0x7C6903A6u); p += 4;  // mtctr r3
      w32(p, 0x7C832378u); p += 4;  // mr r3, r4
      w32(p, 0x38A00000u); p += 4;  // li r5, 0
      // head (0x0C):
      w32(p, 0x70A60001u); p += 4;  // andi. r6, r5, 1
      w32(p, 0x4182000Cu); p += 4;  // beq +0x0C (to even)
      // odd:
      w32(p, 0x38630007u); p += 4;  // addi r3, r3, 7
      w32(p, 0x48000008u); p += 4;  // b +8 (to latch)
      // even:
      w32(p, 0x38630003u); p += 4;  // addi r3, r3, 3
      // latch:
      w32(p, 0x38A50001u); p += 4;  // addi r5, r5, 1
      w32(p, 0x4200FFE8u); p += 4;  // bdnz -0x18 (to head)
      w32(p, 0x4E800020u);          // blr
    }

    // 3. Loop containing a bl call at 0x80000300 (must be rejected safely).
    const uint32_t helper_addr = 0x80000400u;
    w32(helper_addr + 0, 0x38630005u);  // addi r3, r3, 5
    w32(helper_addr + 4, 0x4E800020u);  // blr

    const uint32_t fn_call_addr = 0x80000300u;
    {
      uint32_t p = fn_call_addr;
      w32(p, 0x7C0802A6u); p += 4;  // mflr r0
      w32(p, 0x9001FFF8u); p += 4;  // stw r0, -8(r1)
      w32(p, 0x7C8903A6u); p += 4;  // mtctr r4
      // loop (0x0C):
      w32(p, 0x480000F5u); p += 4;  // bl helper_addr (0x8000030C + 0xF4 = 0x80000400)
      w32(p, 0x4200FFFCu); p += 4;  // bdnz -4
      w32(p, 0x8001FFF8u); p += 4;  // lwz r0, -8(r1)
      w32(p, 0x7C0803A6u); p += 4;  // mtlr r0
      w32(p, 0x4E800020u);          // blr
    }

    auto raw = std::make_unique<RawModule>(processor.get());
    raw->set_name("Wave4LoopPromoTest");
    raw->set_executable(true);
    raw->SetAddressRange(code_base, code_size);
    REQUIRE(processor->AddModule(std::move(raw)));
    processor->backend()->CommitExecutableRange(code_base, code_base + code_size);

    uint32_t stack_addr = memory->SystemHeapAlloc(64u * 1024u);
    auto ts = std::make_unique<ThreadState>(processor.get(), 0x600u,
                                            stack_addr + 64u * 1024u);
    {
      uint64_t args[1] = {10u};
      out_loop1 = processor->Execute(ts.get(), fn1_addr, args, 1);
    }
    {
      uint64_t args[2] = {10u, 100u};
      out_loop3 = processor->Execute(ts.get(), fn3_addr, args, 2);
    }
    {
      uint64_t args[2] = {2u, 4u};
      out_loop_call = processor->Execute(ts.get(), fn_call_addr, args, 2);
    }

    ts.reset();
    memory->SystemHeapFree(stack_addr);
    vheap->Release(code_base);
  };

  uint64_t ref1 = 0, ref3 = 0, refc = 0;
  uint64_t opt1 = 0, opt3 = 0, optc = 0;
  uint64_t sched1 = 0, sched3 = 0, schedc = 0;

  run_ppc_suite(false, false, ref1, ref3, refc);
  run_ppc_suite(true, false, opt1, opt3, optc);
  run_ppc_suite(true, true, sched1, sched3, schedc);

  REQUIRE(ref1 == 100u);  // sum of first 10 odd numbers = 10^2 = 100
  REQUIRE(ref3 == 150u);  // 100 + 5*3 + 5*7 = 150
  REQUIRE(refc == 22u);   // 2 + 4*5 = 22

  REQUIRE(opt1 == ref1);
  REQUIRE(opt3 == ref3);
  REQUIRE(optc == refc);

  REQUIRE(sched1 == ref1);
  REQUIRE(sched3 == ref3);
  REQUIRE(schedc == refc);
}

TEST_CASE("WAVE4_LOOP_PROMOTION_FAULT_EXACT_MEMORY_AND_EXTENDED_BLOCK",
          "[wave4][loop]") {
  ScopedCvar<bool> guard_loop(cvars::a64_loop_context_promotion, false);
  ScopedCvar<bool> guard_defer(cvars::a64_loop_promote_defer_stores_across_memory,
                               false);
  ScopedCvar<bool> guard_disable(cvars::disable_context_promotion, false);
  ScopedCvar<bool> guard_debug(cvars::debug, false);
  ScopedCvar<bool> guard_store_all(cvars::store_all_context_values, false);

  // Part A: Loop with memory load/store (fault-exact mode flushes dirty promoted
  // slots before OPCODE_FLAG_MEMORY).
  auto run_mem_loop = [&](bool enable_loop_promo, uint64_t& out_r3,
                          uint64_t& out_r5) {
    cvars::a64_loop_context_promotion = enable_loop_promo;
    cvars::a64_loop_promote_defer_stores_across_memory = false;

    auto memory = std::make_unique<Memory>();
    REQUIRE(memory->Initialize());
    auto backend = CreateBackend();
    if (!backend) {
      return;
    }
    auto processor = std::make_unique<Processor>(memory.get(), nullptr);
    REQUIRE(processor->Setup(std::move(backend)));

    const uint32_t code_base = 0x80000000u;
    const uint32_t code_size = 0x10000u;
    auto* vheap = memory->LookupHeap(code_base);
    REQUIRE(vheap->AllocFixed(code_base, code_size, 0x1000u,
                              kMemoryAllocationReserve | kMemoryAllocationCommit,
                              kMemoryProtectRead | kMemoryProtectWrite));
    auto w32 = [&](uint32_t addr, uint32_t val) {
      xe::store_and_swap<uint32_t>(memory->TranslateVirtual(addr), val);
    };

    uint32_t buf_va = memory->SystemHeapAlloc(64u, 16u);
    for (uint32_t i = 0; i < 4u; ++i) {
      xe::store_and_swap<uint32_t>(memory->TranslateVirtual(buf_va + i * 4u),
                                   10u + i);
    }

    // fn(buf_va, count): sums count uint32_t words from [r3] into r5, mutating r5 before lwz.
    const uint32_t fn_addr = 0x80000100u;
    uint32_t p = fn_addr;
    w32(p, 0x7C8903A6u); p += 4;  // mtctr r4
    w32(p, 0x38A00001u); p += 4;  // li r5, 1
    // loop:
    w32(p, 0x38A50002u); p += 4;  // addi r5, r5, 2 (dirty promoted r5 before lwz!)
    w32(p, 0x80C30000u); p += 4;  // lwz r6, 0(r3)
    w32(p, 0x7CA53214u); p += 4;  // add r5, r5, r6
    w32(p, 0x38630004u); p += 4;  // addi r3, r3, 4
    w32(p, 0x4200FFF0u); p += 4;  // bdnz -0x10
    w32(p, 0x7CA32B78u); p += 4;  // mr r3, r5
    w32(p, 0x4E800020u);          // blr

    auto raw = std::make_unique<RawModule>(processor.get());
    raw->set_name("Wave4LoopMemTest");
    raw->set_executable(true);
    raw->SetAddressRange(code_base, code_size);
    REQUIRE(processor->AddModule(std::move(raw)));
    processor->backend()->CommitExecutableRange(code_base, code_base + code_size);

    uint32_t stack_addr = memory->SystemHeapAlloc(64u * 1024u);
    auto ts = std::make_unique<ThreadState>(processor.get(), 0x700u,
                                            stack_addr + 64u * 1024u);
    uint64_t args[2] = {buf_va, 4u};
    out_r3 = processor->Execute(ts.get(), fn_addr, args, 2);
    out_r5 = ts->context()->r[5];

    ts.reset();
    memory->SystemHeapFree(stack_addr);
    memory->SystemHeapFree(buf_va);
    vheap->Release(code_base);
  };

  uint64_t mem_ref_r3 = 0, mem_ref_r5 = 0, mem_opt_r3 = 0, mem_opt_r5 = 0;
  run_mem_loop(false, mem_ref_r3, mem_ref_r5);
  run_mem_loop(true, mem_opt_r3, mem_opt_r5);
  REQUIRE(mem_ref_r3 == 55u);  // 1 + 4*2 + (10+11+12+13) = 55
  REQUIRE(mem_opt_r3 == mem_ref_r3);
  REQUIRE(mem_opt_r5 == mem_ref_r5);

  // Part B: Extended-block fall-through SSA preservation + WAR Load->Store->Use
  // via HIRBuilder + LoopContextPromotionPass + RegisterAllocationPass.
  auto run_extended_war = [&](bool enable_loop_promo, uint64_t& out_r3,
                              uint64_t& out_r4, uint64_t& out_r6) {
    cvars::a64_loop_context_promotion = enable_loop_promo;
    auto memory = std::make_unique<Memory>();
    REQUIRE(memory->Initialize());
    auto backend = CreateBackend();
    if (!backend) {
      return;
    }
    auto processor = std::make_unique<Processor>(memory.get(), nullptr);
    REQUIRE(processor->Setup(std::move(backend)));

    auto raw = std::make_unique<RawModule>(processor.get());
    raw->set_name("Wave4ExtWarTest");
    raw->set_executable(true);
    raw->SetAddressRange(0x80000000u, 0x10000u);
    Module* raw_ptr = raw.get();
    REQUIRE(processor->AddModule(std::move(raw)));
    processor->backend()->CommitExecutableRange(0x80000000u, 0x80010000u);

    auto assembler = processor->backend()->CreateAssembler();
    assembler->Initialize();
    auto fn = std::unique_ptr<GuestFunction>(
        processor->backend()->CreateGuestFunction(raw_ptr, 0x80001000u));

    HIRBuilder b;
    auto* loop_lbl = b.NewLabel();
    auto* post_lbl = b.NewLabel();
    b.Branch(loop_lbl);
    b.MarkLabel(loop_lbl);
    // WAR hazard: load r4, overwrite r4, then still use `old_r4`!
    auto old_r4 = LoadGPR(b, 4);
    auto new_r4 = b.Add(old_r4, b.LoadConstantUint64(3u));
    StoreGPR(b, 4, new_r4);
    auto combined = b.Add(old_r4, b.LoadConstantUint64(100u));
    StoreGPR(b, 6, combined);

    auto cnt = LoadGPR(b, 5);
    auto next_cnt = b.Sub(cnt, b.LoadConstantUint64(1u));
    StoreGPR(b, 5, next_cnt);
    auto cond = b.CompareNE(next_cnt, b.LoadConstantUint64(0u));
    b.BranchTrue(cond, loop_lbl);
    // Extended block fall-through: `combined` and `new_r4` remain live after merge!
    b.Branch(post_lbl);
    b.MarkLabel(post_lbl);
    StoreGPR(b, 3, b.Add(combined, new_r4));
    b.Return();
    REQUIRE(b.Finalize());

    compiler::Compiler compiler(processor.get());
    compiler.AddPass(std::make_unique<passes::ControlFlowAnalysisPass>());
    compiler.AddPass(std::make_unique<passes::ControlFlowSimplificationPass>());
    compiler.AddPass(std::make_unique<passes::ControlFlowAnalysisPass>());
    compiler.AddPass(std::make_unique<passes::DeadContextStoreEliminationPass>());
    compiler.AddPass(std::make_unique<passes::DeadCodeEliminationPass>());
    compiler.AddPass(std::make_unique<passes::LoopContextPromotionPass>());
    compiler.AddPass(std::make_unique<passes::RegisterAllocationPass>(
        processor->backend()->machine_info()));
    compiler.AddPass(std::make_unique<passes::FinalizationPass>());
    REQUIRE(compiler.Compile(&b));

    REQUIRE(assembler->Assemble(fn.get(), &b, 0, nullptr));

    uint32_t stack_addr = memory->SystemHeapAlloc(64u * 1024u);
    auto ts = std::make_unique<ThreadState>(processor.get(), 0x800u,
                                            stack_addr + 64u * 1024u);
    PPCContext* ctx = ts->context();
    ctx->lr = 0xBCBCBCBCu;
    ctx->r[4] = 10u;
    ctx->r[5] = 4u;
    fn->Call(ts.get(), static_cast<uint32_t>(ctx->lr));
    out_r3 = ctx->r[3];
    out_r4 = ctx->r[4];
    out_r6 = ctx->r[6];

    ts.reset();
    memory->SystemHeapFree(stack_addr);
  };

  uint64_t ext_ref_r3 = 0, ext_ref_r4 = 0, ext_ref_r6 = 0;
  uint64_t ext_opt_r3 = 0, ext_opt_r4 = 0, ext_opt_r6 = 0;
  run_extended_war(false, ext_ref_r3, ext_ref_r4, ext_ref_r6);
  run_extended_war(true, ext_opt_r3, ext_opt_r4, ext_opt_r6);
  // 4 iterations starting r4=10: final iter has old_r4=19, new_r4=22, combined=119, r3=141.
  REQUIRE(ext_ref_r4 == 22u);
  REQUIRE(ext_ref_r6 == 119u);
  REQUIRE(ext_ref_r3 == 141u);
  REQUIRE(ext_opt_r3 == ext_ref_r3);
  REQUIRE(ext_opt_r4 == ext_ref_r4);
  REQUIRE(ext_opt_r6 == ext_ref_r6);
}

TEST_CASE("WAVE4_LOOP_PROMO_IN_LOOP_FALLTHROUGH_BACKEDGE",
          "[wave4][crossblock][loop]") {
  ScopedCvar<bool> guard_loop(cvars::a64_loop_context_promotion, false);
  ScopedCvar<bool> guard_dse(cvars::dse_cross_block_context, true);
  ScopedCvar<bool> guard_disable(cvars::disable_context_promotion, false);
  ScopedCvar<bool> guard_debug(cvars::debug, false);
  ScopedCvar<bool> guard_store_all(cvars::store_all_context_values, false);

  auto run_fallthrough = [&](bool enable_loop_promo, uint64_t& out_r4,
                             uint64_t& out_r5) {
    cvars::a64_loop_context_promotion = enable_loop_promo;

    auto memory = std::make_unique<Memory>();
    REQUIRE(memory->Initialize());
    auto backend = CreateBackend();
    if (!backend) {
      return;
    }
    auto processor = std::make_unique<Processor>(memory.get(), nullptr);
    REQUIRE(processor->Setup(std::move(backend)));

    auto raw = std::make_unique<RawModule>(processor.get());
    raw->set_name("Wave4FallthroughBackedgeTest");
    raw->set_executable(true);
    raw->SetAddressRange(0x80000000u, 0x10000u);
    Module* raw_ptr = raw.get();
    REQUIRE(processor->AddModule(std::move(raw)));
    processor->backend()->CommitExecutableRange(0x80000000u, 0x80010000u);

    auto assembler = processor->backend()->CreateAssembler();
    assembler->Initialize();
    auto fn = std::unique_ptr<GuestFunction>(
        processor->backend()->CreateGuestFunction(raw_ptr, 0x80001000u));

    // Layout: B0 -> Branch(hdr_lbl)
    //         B1(latch_lbl): r4 += 7; (falls through into B2, no explicit branch!)
    //         B2(hdr_lbl):   r5 -= 1; BranchTrue(r5 != 0, latch_lbl); Return()
    // Preheader must NOT be inserted between B1 and B2.
    HIRBuilder b;
    auto* latch_lbl = b.NewLabel();
    auto* hdr_lbl = b.NewLabel();
    b.Branch(hdr_lbl);

    b.MarkLabel(latch_lbl);
    auto cur_r4 = LoadGPR(b, 4);
    StoreGPR(b, 4, b.Add(cur_r4, b.LoadConstantUint64(7u)));

    b.MarkLabel(hdr_lbl);
    auto cnt = LoadGPR(b, 5);
    auto next_cnt = b.Sub(cnt, b.LoadConstantUint64(1u));
    StoreGPR(b, 5, next_cnt);
    auto cond = b.CompareNE(next_cnt, b.LoadConstantUint64(0u));
    b.BranchTrue(cond, latch_lbl);
    b.Return();

    compiler::Compiler compiler(processor.get());
    compiler.AddPass(std::make_unique<passes::ControlFlowAnalysisPass>());
    compiler.AddPass(std::make_unique<passes::DeadContextStoreEliminationPass>());
    compiler.AddPass(std::make_unique<passes::DeadCodeEliminationPass>());
    compiler.AddPass(std::make_unique<passes::LoopContextPromotionPass>());
    compiler.AddPass(std::make_unique<passes::RegisterAllocationPass>(
        processor->backend()->machine_info()));
    compiler.AddPass(std::make_unique<passes::FinalizationPass>());
    REQUIRE(compiler.Compile(&b));

    REQUIRE(latch_lbl->block->next == hdr_lbl->block);
    if (enable_loop_promo) {
      REQUIRE_FALSE(b.loop_promotion_plan().empty());
      REQUIRE_FALSE(b.loop_promotion_plan().regions[0].entry_points.empty());
      Block* preheader =
          b.loop_promotion_plan().regions[0].entry_points[0].preheader_block;
      REQUIRE(preheader != nullptr);
      REQUIRE(preheader != latch_lbl->block);
      REQUIRE(preheader->prev != latch_lbl->block);
    }

    REQUIRE(assembler->Assemble(fn.get(), &b, 0, nullptr));

    uint32_t stack_addr = memory->SystemHeapAlloc(64u * 1024u);
    auto ts = std::make_unique<ThreadState>(processor.get(), 0x900u,
                                            stack_addr + 64u * 1024u);
    PPCContext* ctx = ts->context();
    ctx->lr = 0xBCBCBCBCu;
    ctx->r[4] = 10u;
    ctx->r[5] = 4u;
    fn->Call(ts.get(), static_cast<uint32_t>(ctx->lr));
    out_r4 = ctx->r[4];
    out_r5 = ctx->r[5];

    ts.reset();
    memory->SystemHeapFree(stack_addr);
  };

  uint64_t ref_r4 = 0, ref_r5 = 0, opt_r4 = 0, opt_r5 = 0;
  run_fallthrough(false, ref_r4, ref_r5);
  run_fallthrough(true, opt_r4, opt_r5);
  REQUIRE(ref_r4 == 31u);  // 10 + 3 * 7
  REQUIRE(ref_r5 == 0u);
  REQUIRE(opt_r4 == ref_r4);
  REQUIRE(opt_r5 == ref_r5);
}

TEST_CASE("WAVE4_LOOP_PROMO_FINALIZATION_PRESERVES_FLUSH_BRANCH",
          "[wave4][crossblock][loop]") {
  ScopedCvar<bool> guard_loop(cvars::a64_loop_context_promotion, false);
  ScopedCvar<bool> guard_dse(cvars::dse_cross_block_context, true);
  ScopedCvar<bool> guard_disable(cvars::disable_context_promotion, false);
  ScopedCvar<bool> guard_debug(cvars::debug, false);
  ScopedCvar<bool> guard_store_all(cvars::store_all_context_values, false);

  // Part A: FinalizationPass must keep a tail OPCODE_BRANCH(block->next) when
  // LoopPromotionPlan anchors a flush point on it, while still eliding
  // unanchored fallthrough branches.
  {
    HIRBuilder b;
    auto* b1_lbl = b.NewLabel();
    auto* b2_lbl = b.NewLabel();
    b.Branch(b1_lbl);
    Instr* b0_branch = b.last_instr();

    b.MarkLabel(b1_lbl);
    StoreGPR(b, 4, b.LoadConstantUint64(42u));
    b.Branch(b2_lbl);
    Instr* b1_branch = b.last_instr();

    b.MarkLabel(b2_lbl);
    b.Return();
    REQUIRE(b.Finalize());

    hir::LoopPromotionRegion reg;
    reg.flush_points.push_back({b1_branch, 1u});
    b.loop_promotion_plan().regions.push_back(std::move(reg));

    REQUIRE_FALSE(b.loop_promotion_plan().ReferencesInstr(b0_branch));
    REQUIRE(b.loop_promotion_plan().ReferencesInstr(b1_branch));

    auto memory = std::make_unique<Memory>();
    REQUIRE(memory->Initialize());
    auto processor = std::make_unique<Processor>(memory.get(), nullptr);
    compiler::Compiler compiler(processor.get());
    compiler.AddPass(std::make_unique<passes::FinalizationPass>());
    REQUIRE(compiler.Compile(&b));

    REQUIRE(b0_branch->opcode == &OPCODE_NOP_info);
    REQUIRE(b1_branch->opcode == &OPCODE_BRANCH_info);
    REQUIRE(b1_lbl->block->instr_tail == b1_branch);
  }

  // Part B: End-to-end execution of a 2-block loop whose header exits via an
  // unconditional OPCODE_BRANCH targeting header->next.
  auto run_exit_branch = [&](bool enable_loop_promo, uint64_t& out_r4,
                             uint64_t& out_r5, uint64_t& out_r6) {
    cvars::a64_loop_context_promotion = enable_loop_promo;

    auto memory = std::make_unique<Memory>();
    REQUIRE(memory->Initialize());
    auto backend = CreateBackend();
    if (!backend) {
      return;
    }
    auto processor = std::make_unique<Processor>(memory.get(), nullptr);
    REQUIRE(processor->Setup(std::move(backend)));

    auto raw = std::make_unique<RawModule>(processor.get());
    raw->set_name("Wave4ExitBranchFlushTest");
    raw->set_executable(true);
    raw->SetAddressRange(0x80000000u, 0x10000u);
    Module* raw_ptr = raw.get();
    REQUIRE(processor->AddModule(std::move(raw)));
    processor->backend()->CommitExecutableRange(0x80000000u, 0x80010000u);

    auto assembler = processor->backend()->CreateAssembler();
    assembler->Initialize();
    auto fn = std::unique_ptr<GuestFunction>(
        processor->backend()->CreateGuestFunction(raw_ptr, 0x80001000u));

    HIRBuilder b;
    auto* hdr_lbl = b.NewLabel();
    auto* exit_lbl = b.NewLabel();
    auto* cont_lbl = b.NewLabel();
    b.Branch(hdr_lbl);

    b.MarkLabel(hdr_lbl);
    StoreGPR(b, 4, b.Add(LoadGPR(b, 4), b.LoadConstantUint64(5u)));
    auto next_cnt = b.Sub(LoadGPR(b, 5), b.LoadConstantUint64(1u));
    StoreGPR(b, 5, next_cnt);
    b.BranchTrue(b.CompareNE(next_cnt, b.LoadConstantUint64(0u)), cont_lbl);
    b.Branch(exit_lbl);

    b.MarkLabel(exit_lbl);
    b.Return();

    b.MarkLabel(cont_lbl);
    StoreGPR(b, 6, b.Add(LoadGPR(b, 6), b.LoadConstantUint64(11u)));
    b.Branch(hdr_lbl);
    REQUIRE(b.Finalize());

    compiler::Compiler compiler(processor.get());
    compiler.AddPass(std::make_unique<passes::ControlFlowAnalysisPass>());
    compiler.AddPass(std::make_unique<passes::DeadContextStoreEliminationPass>());
    compiler.AddPass(std::make_unique<passes::DeadCodeEliminationPass>());
    compiler.AddPass(std::make_unique<passes::LoopContextPromotionPass>());
    compiler.AddPass(std::make_unique<passes::RegisterAllocationPass>(
        processor->backend()->machine_info()));
    compiler.AddPass(std::make_unique<passes::FinalizationPass>());
    REQUIRE(compiler.Compile(&b));
    REQUIRE(assembler->Assemble(fn.get(), &b, 0, nullptr));

    uint32_t stack_addr = memory->SystemHeapAlloc(64u * 1024u);
    auto ts = std::make_unique<ThreadState>(processor.get(), 0xA00u,
                                            stack_addr + 64u * 1024u);
    PPCContext* ctx = ts->context();
    ctx->lr = 0xBCBCBCBCu;
    ctx->r[4] = 10u;
    ctx->r[5] = 4u;
    ctx->r[6] = 1u;
    fn->Call(ts.get(), static_cast<uint32_t>(ctx->lr));
    out_r4 = ctx->r[4];
    out_r5 = ctx->r[5];
    out_r6 = ctx->r[6];

    ts.reset();
    memory->SystemHeapFree(stack_addr);
  };

  uint64_t ref_r4 = 0, ref_r5 = 0, ref_r6 = 0;
  uint64_t opt_r4 = 0, opt_r5 = 0, opt_r6 = 0;
  run_exit_branch(false, ref_r4, ref_r5, ref_r6);
  run_exit_branch(true, opt_r4, opt_r5, opt_r6);
  REQUIRE(ref_r4 == 30u);  // 10 + 4*5
  REQUIRE(ref_r5 == 0u);
  REQUIRE(ref_r6 == 34u);  // 1 + 3*11
  REQUIRE(opt_r4 == ref_r4);
  REQUIRE(opt_r5 == ref_r5);
  REQUIRE(opt_r6 == ref_r6);
}
