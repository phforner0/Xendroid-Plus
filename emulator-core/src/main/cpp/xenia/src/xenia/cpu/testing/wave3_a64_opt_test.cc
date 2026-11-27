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
#include <atomic>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

#include "xenia/base/byte_order.h"
#include "xenia/base/cvar.h"
#include "xenia/base/memory.h"
#include "xenia/cpu/ppc/ppc_hle_memory.h"
#include "xenia/cpu/ppc/ppc_stack_skew_guard.h"
#include "xenia/cpu/raw_module.h"

#if XE_ARCH_ARM64
#include "xenia/cpu/backend/a64/a64_code_cache.h"
#include "xenia/cpu/backend/a64/a64_direct_call.h"
#include "xenia/cpu/backend/a64/a64_stack_skew_analysis.h"
#endif

DECLARE_bool(shared_memory_preserve_gpu_writes);
DECLARE_bool(vscr_sat_census);
#if XE_ARCH_ARM64
DECLARE_bool(a64_indirect_call_pic);
#endif

using namespace xe;
using namespace xe::cpu;
using namespace xe::cpu::hir;
using namespace xe::cpu::testing;
using xe::cpu::ppc::HleMemcpy;
using xe::cpu::ppc::HleMemmove;
using xe::cpu::ppc::HleMemset;
using xe::cpu::ppc::PPCContext;

// =============================================================================
// 1. WAVE3_A64_CODE_CACHE_DUAL_MAP_PATCH
// =============================================================================
#if XE_ARCH_ARM64
TEST_CASE("WAVE3_A64_CODE_CACHE_DUAL_MAP_PATCH", "[wave3][a64]") {
  using xe::cpu::backend::EmitFunctionInfo;
  using xe::cpu::backend::a64::A64CodeCache;
  using xe::cpu::backend::a64::DecodeA64BranchRel26;
  using xe::cpu::backend::a64::EncodeA64BranchRel26;
  using xe::cpu::backend::a64::IsA64Rel26InRange;
  using xe::cpu::backend::a64::PatchA64InstructionAtomic;

  auto cache = A64CodeCache::Create();
  REQUIRE(cache != nullptr);
  REQUIRE(cache->Initialize());

  // Place two minimal functions (NOP + RET) into the code cache.
  const std::array<uint32_t, 4> stub_fn = {
      0xD503201Fu,  // nop (placeholder call site at offset 0)
      0xD65F03C0u,  // ret
      0xD503201Fu,  // nop
      0xD503201Fu,  // nop
  };
  const std::array<uint32_t, 2> target_fn = {
      0xD503201Fu,  // nop
      0xD65F03C0u,  // ret
  };

  EmitFunctionInfo caller_info = {};
  caller_info.code_size.total = stub_fn.size() * sizeof(uint32_t);
  void* caller_rx = nullptr;
  void* caller_rw = nullptr;
  cache->PlaceGuestCode(0x82010000u, const_cast<uint32_t*>(stub_fn.data()),
                        caller_info, nullptr, caller_rx, caller_rw);
  REQUIRE(caller_rx != nullptr);
  REQUIRE(caller_rw != nullptr);

  EmitFunctionInfo target_info = {};
  target_info.code_size.total = target_fn.size() * sizeof(uint32_t);
  void* target_rx = nullptr;
  void* target_rw = nullptr;
  cache->PlaceGuestCode(0x82010100u, const_cast<uint32_t*>(target_fn.data()),
                        target_info, nullptr, target_rx, target_rw);
  REQUIRE(target_rx != nullptr);

  const uintptr_t call_site_rx = reinterpret_cast<uintptr_t>(caller_rx);
  const uintptr_t target_addr_rx = reinterpret_cast<uintptr_t>(target_rx);
  REQUIRE(cache->IsInExecuteRange(call_site_rx));
  REQUIRE(cache->IsInExecuteRange(target_addr_rx));

  uint32_t* mapped_rw = cache->WriteSlotForExecuteAddress(call_site_rx);
  REQUIRE(mapped_rw == reinterpret_cast<uint32_t*>(caller_rw));

  // Register call site, back-patch to target, verify RX reflects patched bl.
  const uint32_t original_stub_insn = 0x94000003u;  // bl +12
  cache->RegisterDirectCallSite(0x82010100u, call_site_rx, original_stub_insn,
                                false);
  cache->BackpatchDirectCallSites(0x82010100u, target_rx);

  REQUIRE(IsA64Rel26InRange(call_site_rx, target_addr_rx));
  const int64_t expected_diff =
      static_cast<int64_t>(target_addr_rx) - static_cast<int64_t>(call_site_rx);
  const uint32_t expected_bl = EncodeA64BranchRel26(expected_diff, true);
  uint32_t observed_rx = 0;
  std::memcpy(&observed_rx, reinterpret_cast<const void*>(call_site_rx), 4);
  REQUIRE(observed_rx == expected_bl);

  int64_t decoded_diff = 0;
  bool decoded_link = false;
  REQUIRE(DecodeA64BranchRel26(observed_rx, &decoded_diff, &decoded_link));
  REQUIRE(decoded_diff == expected_diff);
  REQUIRE(decoded_link == true);

  // Revert via InvalidateCodeRange hook and confirm stub instruction returns.
  cache->RevertDirectCallSitesInRange(0x82010100u, 0x82010110u);
  std::memcpy(&observed_rx, reinterpret_cast<const void*>(call_site_rx), 4);
  REQUIRE(observed_rx == original_stub_insn);

  // Out-of-range (> +128 MiB) target must be rejected cleanly.
  const uintptr_t far_target_rx = call_site_rx + 0x08000000ULL;
  REQUIRE(!IsA64Rel26InRange(call_site_rx, far_target_rx));

  // Concurrent multi-threaded BackpatchDirectCallSites / Revert stress.
  std::array<void*, 4> site_rx_list = {};
  for (size_t i = 0; i < site_rx_list.size(); ++i) {
    void* rx = nullptr;
    void* rw = nullptr;
    cache->PlaceGuestCode(0x82020000u + static_cast<uint32_t>(i * 0x20u),
                          const_cast<uint32_t*>(stub_fn.data()), caller_info,
                          nullptr, rx, rw);
    site_rx_list[i] = rx;
    cache->RegisterDirectCallSite(0x82030000u, reinterpret_cast<uintptr_t>(rx),
                                  original_stub_insn, false);
  }
  std::vector<std::thread> workers;
  for (int t = 0; t < 4; ++t) {
    workers.emplace_back([&, t]() {
      if (t % 2 == 0) {
        cache->BackpatchDirectCallSites(0x82030000u, target_rx);
      } else {
        cache->RevertDirectCallSitesInRange(0x82030000u, 0x82030010u);
      }
    });
  }
  for (auto& th : workers) {
    th.join();
  }
}
#endif  // XE_ARCH_ARM64

// =============================================================================
// 2. WAVE3_STACK_NO_PHYSICAL_SKEW
// =============================================================================
TEST_CASE("WAVE3_STACK_NO_PHYSICAL_SKEW", "[wave3][stack]") {
  TestFunction test([](HIRBuilder& b) {
    auto r1 = LoadGPR(b, 1);
    auto loaded = b.LoadOffset(r1, b.LoadConstantInt64(0x10), INT32_TYPE,
                               LOAD_STORE_NO_PHYSICAL_SKEW);
    b.StoreOffset(r1, b.LoadConstantInt64(0x18),
                  b.Add(loaded, b.LoadConstantUint32(1u)),
                  LOAD_STORE_NO_PHYSICAL_SKEW);
    StoreGPR(b, 3, b.ZeroExtend(loaded, INT64_TYPE));
    b.Return();
  });

  // Case A: r1 inside [0x70000000, 0x7F000000) stack window.
  uint32_t stack_addr = 0;
  auto* vheap = test.memory->LookupHeap(0x70000000u);
  if (vheap &&
      vheap->AllocRange(0x70010000u, 0x70020000u, 0x10000u, 0x1000u,
                        kMemoryAllocationReserve | kMemoryAllocationCommit,
                        kMemoryProtectRead | kMemoryProtectWrite, false,
                        &stack_addr)) {
    test.Run(
        [&](PPCContext* ctx) {
          ctx->r[1] = stack_addr;
          uint32_t* slot =
              reinterpret_cast<uint32_t*>(test.memory->TranslateVirtual(stack_addr + 0x10u));
          *slot = 0x12345678u;
        },
        [&](PPCContext* ctx) {
          REQUIRE(static_cast<uint32_t>(ctx->r[3]) == 0x12345678u);
          uint32_t* out_slot =
              reinterpret_cast<uint32_t*>(test.memory->TranslateVirtual(stack_addr + 0x18u));
          REQUIRE(*out_slot == 0x12345679u);
          vheap->Release(stack_addr);
        });
  }

  // Case B: r1 outside stack range (0xE0000000 physical window).
  // Must fall back cleanly on 16 KB hosts so TranslateVirtual and JIT agree.
  uint32_t phys_addr = 0;
  auto* pheap = test.memory->LookupHeap(0xE0000000u);
  REQUIRE(pheap != nullptr);
  REQUIRE(pheap->AllocRange(0xE0000000u, 0xE1000000u, 0x10000u, 0x10000u,
                            kMemoryAllocationReserve | kMemoryAllocationCommit,
                            kMemoryProtectRead | kMemoryProtectWrite, false,
                            &phys_addr));
  test.Run(
      [&](PPCContext* ctx) {
        ctx->r[1] = phys_addr;
        uint32_t* in_slot = reinterpret_cast<uint32_t*>(
            test.memory->TranslateVirtual(phys_addr + 0x10u));
        *in_slot = 0xCAFEBABEu;
      },
      [&](PPCContext* ctx) {
        REQUIRE(static_cast<uint32_t>(ctx->r[3]) == 0xCAFEBABEu);
        uint32_t* out_slot = reinterpret_cast<uint32_t*>(
            test.memory->TranslateVirtual(phys_addr + 0x18u));
        REQUIRE(*out_slot == 0xCAFEBABFu);
        pheap->Release(phys_addr);
      });
}

// =============================================================================
// 3. WAVE3_HLE_MEMCPY_WRITEWATCH
// =============================================================================
TEST_CASE("WAVE3_HLE_MEMCPY_WRITEWATCH", "[wave3][hle]") {
  auto memory = std::make_unique<Memory>();
  REQUIRE(memory->Initialize());
  auto backend = CreateBackend();
  if (!backend) {
    return;
  }
  auto processor = std::make_unique<Processor>(memory.get(), nullptr);
  REQUIRE(processor->Setup(std::move(backend)));

  const uint32_t alloc_size = 0x10000u;  // 64 KiB physical block
  uint32_t dst_va = memory->SystemHeapAlloc(alloc_size, 0x10000u,
                                            kSystemHeapPhysical);
  uint32_t src_va = memory->SystemHeapAlloc(alloc_size, 0x10000u,
                                            kSystemHeapPhysical);
  REQUIRE(dst_va != 0u);
  REQUIRE(src_va != 0u);

  const uint32_t dst_phys = memory->GetPhysicalAddress(dst_va);
  const uint32_t src_phys = memory->GetPhysicalAddress(src_va);
  REQUIRE(dst_phys != UINT32_MAX);
  REQUIRE(src_phys != UINT32_MAX);

  struct WatchLog {
    uint32_t calls = 0;
    uint32_t last_start = 0;
    uint32_t last_len = 0;
    bool saw_exact_range = false;
  } watch_log;

  void* cb_handle = memory->RegisterPhysicalMemoryInvalidationCallback(
      [](void* ctx_ptr, uint32_t physical_address_start, uint32_t length,
         bool exact_range) -> std::pair<uint32_t, uint32_t> {
        auto* log = static_cast<WatchLog*>(ctx_ptr);
        log->calls++;
        log->last_start = physical_address_start;
        log->last_len = length;
        if (exact_range) {
          log->saw_exact_range = true;
        }
        return {0u, UINT32_MAX};
      },
      &watch_log);

  // Arm physical writewatch over dst.
  memory->EnablePhysicalMemoryAccessCallbacks(dst_phys, alloc_size, true,
                                              false);

  // Fill src via physical view.
  uint8_t* src_host = memory->TranslatePhysical(src_phys);
  for (uint32_t i = 0; i < 256u; ++i) {
    src_host[i] = static_cast<uint8_t>((i * 7u + 3u) & 0xFFu);
  }

  uint32_t stack_addr = memory->SystemHeapAlloc(64u * 1024u);
  auto thread_state = std::make_unique<ThreadState>(processor.get(), 0x200u,
                                                    stack_addr + 64u * 1024u);
  PPCContext* ctx = thread_state->context();

  // Execute HLE memcpy: triggers writewatch cleanly and copies bytes.
  ctx->r[3] = dst_va;
  ctx->r[4] = src_va;
  ctx->r[5] = 256u;
  ctx->scratch = 0;
  HleMemcpy(ctx, nullptr, nullptr);
  REQUIRE(ctx->scratch == 1u);
  REQUIRE(ctx->r[3] == dst_va);
  REQUIRE(watch_log.calls >= 1u);
  REQUIRE(watch_log.saw_exact_range == !cvars::shared_memory_preserve_gpu_writes);

  uint8_t* dst_host = memory->TranslatePhysical(dst_phys);
  REQUIRE(std::memcmp(dst_host, src_host, 256u) == 0);

  // Execute HLE memset over [dst_va + 64, dst_va + 192).
  ctx->r[3] = dst_va + 64u;
  ctx->r[4] = 0x5Au;
  ctx->r[5] = 128u;
  ctx->scratch = 0;
  HleMemset(ctx, nullptr, nullptr);
  REQUIRE(ctx->scratch == 1u);
  for (uint32_t i = 64u; i < 192u; ++i) {
    REQUIRE(dst_host[i] == 0x5Au);
  }

  // Overlapping virtual aliases (0xA0000000 + phys vs 0xE0000000 + phys):
  // When shared_memory_preserve_gpu_writes is false, memmove handles alias overlap safely.
  const bool prev_preserve = cvars::shared_memory_preserve_gpu_writes;
  cvars::shared_memory_preserve_gpu_writes = false;
  for (uint32_t i = 0; i < 128u; ++i) {
    src_host[i] = static_cast<uint8_t>(i);
  }
  const uint32_t alias_a = 0xA0000000u + src_phys;
  const uint32_t alias_e = 0xE0000000u + src_phys;
  std::array<uint8_t, 64> expected_overlap = {};
  std::memcpy(expected_overlap.data(), src_host, 64u);
  ctx->r[3] = alias_e + 16u;
  ctx->r[4] = alias_a;
  ctx->r[5] = 64u;
  ctx->scratch = 0;
  HleMemmove(ctx, nullptr, nullptr);
  REQUIRE(ctx->scratch == 1u);
  REQUIRE(std::memcmp(src_host + 16u, expected_overlap.data(), 64u) == 0);

  // When shared_memory_preserve_gpu_writes is true, overlapping physical ranges fall back.
  cvars::shared_memory_preserve_gpu_writes = true;
  ctx->r[3] = alias_e + 16u;
  ctx->r[4] = alias_a;
  ctx->r[5] = 64u;
  ctx->scratch = 1u;
  HleMemcpy(ctx, nullptr, nullptr);
  REQUIRE(ctx->scratch == 0u);

  // Cross-alias HLE memcpy: allocate in vA0000000, arm writewatch, write via
  // 0xE0000000 alias -> invalidation callback must still fire.
  uint32_t alias_va_a = 0;
  auto* heap_a = memory->LookupHeap(0xA0000000u);
  REQUIRE(heap_a != nullptr);
  if (heap_a->Alloc(alloc_size, 0x10000u,
                    kMemoryAllocationReserve | kMemoryAllocationCommit,
                    kMemoryProtectRead | kMemoryProtectWrite, false,
                    &alias_va_a)) {
    const uint32_t alias_phys = memory->GetPhysicalAddress(alias_va_a);
    REQUIRE(alias_phys >= 0x1000u);
    const uint32_t alias_va_e = 0xE0000000u + (alias_phys - 0x1000u);
    watch_log = {};
    memory->EnablePhysicalMemoryAccessCallbacks(alias_phys, alloc_size, true,
                                                false);
    ctx->r[3] = alias_va_e;
    ctx->r[4] = src_va;
    ctx->r[5] = 256u;
    ctx->scratch = 0;
    HleMemcpy(ctx, nullptr, nullptr);
    REQUIRE(ctx->scratch == 1u);
    REQUIRE(watch_log.calls >= 1u);
    REQUIRE(!watch_log.saw_exact_range);
    REQUIRE(std::memcmp(memory->TranslatePhysical(alias_phys), src_host,
                        256u) == 0);

    // Full system-page aligned write with preserve_gpu_writes=true may pass
    // exact_range=true safely.
    const uint32_t sys_page = static_cast<uint32_t>(xe::memory::page_size());
    if (sys_page <= alloc_size && (alias_phys & (sys_page - 1u)) == 0u) {
      watch_log = {};
      memory->EnablePhysicalMemoryAccessCallbacks(alias_phys, alloc_size, true,
                                                  false);
      ctx->r[3] = alias_va_a;
      ctx->r[4] = 0x7Au;
      ctx->r[5] = sys_page;
      ctx->scratch = 0;
      HleMemset(ctx, nullptr, nullptr);
      REQUIRE(ctx->scratch == 1u);
      REQUIRE(watch_log.calls >= 1u);
      REQUIRE(watch_log.saw_exact_range);
    }
    heap_a->Release(alias_va_a);
  }

  // Virtual membase mismatch (ctx->virtual_membase != memory->virtual_membase())
  // must fall back cleanly instead of writing through the wrong view.
  uint32_t virt_buf = memory->SystemHeapAlloc(4096u, 4096u, 0);
  if (virt_buf) {
    uint8_t* saved_membase = ctx->virtual_membase;
    ctx->virtual_membase = nullptr;
    ctx->r[3] = virt_buf;
    ctx->r[4] = 0x22u;
    ctx->r[5] = 256u;
    ctx->scratch = 1u;
    HleMemset(ctx, nullptr, nullptr);
    REQUIRE(ctx->scratch == 0u);
    ctx->virtual_membase = saved_membase;
    memory->SystemHeapFree(virt_buf);
  }

  cvars::shared_memory_preserve_gpu_writes = prev_preserve;

  memory->UnregisterPhysicalMemoryInvalidationCallback(cb_handle);
  thread_state.reset();
  memory->SystemHeapFree(stack_addr);
  memory->SystemHeapFree(src_va);
  memory->SystemHeapFree(dst_va);
}

// =============================================================================
// 4. WAVE3_SCANNER_JUMP_TABLE_RECOVERY
// =============================================================================
TEST_CASE("WAVE3_SCANNER_JUMP_TABLE_RECOVERY", "[wave3][ppc]") {
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
  REQUIRE(vheap != nullptr);
  uint32_t alloc_base = 0;
  REQUIRE(vheap->AllocFixed(code_base, code_size, 0x1000u,
                            kMemoryAllocationReserve | kMemoryAllocationCommit,
                            kMemoryProtectRead | kMemoryProtectWrite));
  alloc_base = code_base;

  // Synthetic PPC function at 0x80000100 with 4-entry SWITCH_ABSOLUTE table at 0x80000200.
  const uint32_t func_addr = 0x80000100u;
  const uint32_t table_addr = 0x80000200u;
  const uint32_t case0_addr = func_addr + 0x28u;
  const uint32_t case1_addr = func_addr + 0x30u;
  const uint32_t case2_addr = func_addr + 0x38u;
  const uint32_t case3_addr = func_addr + 0x40u;
  const uint32_t default_addr = func_addr + 0x20u;

  auto write_be32 = [&](uint32_t addr, uint32_t val) {
    xe::store_and_swap<uint32_t>(memory->TranslateVirtual(addr), val);
  };

  // Switch dispatch at 0x80000100..0x8000011C (8 insns = 0x20 bytes):
  // +0x00: cmplwi cr0, r3, 3
  write_be32(func_addr + 0x00u, 0x28030003u);
  // +0x04: bgt cr0, +0x1C (default_addr = func_addr + 0x20)
  write_be32(func_addr + 0x04u, 0x4181001Cu);
  // +0x08: rlwinm r11, r3, 2, 0, 29
  write_be32(func_addr + 0x08u, 0x546B103Au);
  // +0x0C: lis r10, 0x8000
  write_be32(func_addr + 0x0Cu, 0x3D408000u);
  // +0x10: addi r10, r10, 0x0200 (table_addr = 0x80000200)
  write_be32(func_addr + 0x10u, 0x394A0200u);
  // +0x14: lwzx r11, r10, r11
  write_be32(func_addr + 0x14u, 0x7D6A582Eu);
  // +0x18: mtctr r11
  write_be32(func_addr + 0x18u, 0x7D6903A6u);
  // +0x1C: bctr
  const uint32_t bctr_addr = func_addr + 0x1Cu;
  write_be32(bctr_addr, 0x4E800420u);

  // +0x20 (default): li r3, 999; blr
  write_be32(default_addr + 0x00u, 0x386003E7u);
  write_be32(default_addr + 0x04u, 0x4E800020u);
  // +0x28 (case 0): li r3, 100; blr
  write_be32(case0_addr + 0x00u, 0x38600064u);
  write_be32(case0_addr + 0x04u, 0x4E800020u);
  // +0x30 (case 1): li r3, 101; blr
  write_be32(case1_addr + 0x00u, 0x38600065u);
  write_be32(case1_addr + 0x04u, 0x4E800020u);
  // +0x38 (case 2): li r3, 102; blr
  write_be32(case2_addr + 0x00u, 0x38600066u);
  write_be32(case2_addr + 0x04u, 0x4E800020u);
  // +0x40 (case 3): li r3, 103; blr
  write_be32(case3_addr + 0x00u, 0x38600067u);
  write_be32(case3_addr + 0x04u, 0x4E800020u);

  // Table entries at 0x80000200:
  write_be32(table_addr + 0x00u, case0_addr);
  write_be32(table_addr + 0x04u, case1_addr);
  write_be32(table_addr + 0x08u, case2_addr);
  write_be32(table_addr + 0x0Cu, case3_addr);

  auto raw_module = std::make_unique<RawModule>(processor.get());
  raw_module->set_name("Wave3JumpTableTest");
  raw_module->set_executable(true);
  raw_module->SetAddressRange(code_base, code_size);
  REQUIRE(processor->AddModule(std::move(raw_module)));
  processor->backend()->CommitExecutableRange(code_base, code_base + code_size);

  Function* fn = processor->ResolveFunction(func_addr);
  REQUIRE(fn != nullptr);
  auto* guest_fn = static_cast<GuestFunction*>(fn);
  const auto* recovered = guest_fn->LookupJumpTableTargets(bctr_addr);
  REQUIRE(recovered != nullptr);
  REQUIRE(recovered->size() == 4u);

  uint32_t stack_addr = memory->SystemHeapAlloc(64u * 1024u);
  auto thread_state = std::make_unique<ThreadState>(processor.get(), 0x300u,
                                                    stack_addr + 64u * 1024u);
  for (uint32_t i = 0; i < 4u; ++i) {
    uint64_t args[1] = {i};
    uint64_t ret = processor->Execute(thread_state.get(), func_addr, args, 1);
    REQUIRE(ret == 100u + i);
  }
  {
    uint64_t args[1] = {5u};
    uint64_t ret = processor->Execute(thread_state.get(), func_addr, args, 1);
    REQUIRE(ret == 999u);
  }

  thread_state.reset();
  memory->SystemHeapFree(stack_addr);
  vheap->Release(alloc_base);
}

#if XE_ARCH_ARM64
// =============================================================================
// 5. WAVE3_STACK_SKEW_ANALYSIS_ADVERSARIAL
// =============================================================================
TEST_CASE("WAVE3_STACK_SKEW_ANALYSIS_ADVERSARIAL", "[wave3][stack]") {
  using xe::cpu::backend::a64::AnalyzeStackSkewSafety;
  using xe::cpu::backend::a64::kR1ContextOffset;

  auto make_r1_sub = [](HIRBuilder& b, int64_t delta) {
    auto r1 = b.LoadContext(kR1ContextOffset, INT64_TYPE);
    auto next_r1 = b.Sub(r1, b.LoadConstantInt64(delta));
    b.StoreContext(kR1ContextOffset, next_r1);
    return next_r1;
  };

  // Safe single-block frame stays inside +-1 MiB and has no calls.
  {
    HIRBuilder b;
    auto* r1_adj = make_r1_sub(b, 0x80);
    b.StoreOffset(r1_adj, b.LoadConstantInt64(0x10), b.LoadConstantUint32(1u),
                  LOAD_STORE_NO_PHYSICAL_SKEW);
    auto r1_reload = b.LoadContext(kR1ContextOffset, INT64_TYPE);
    b.StoreContext(kR1ContextOffset, b.Add(r1_reload, b.LoadConstantInt64(0x80)));
    b.Return();
    REQUIRE(b.Finalize());
    bool uses_no_skew = false;
    bool has_unsafe = false;
    AnalyzeStackSkewSafety(&b, &uses_no_skew, &has_unsafe);
    REQUIRE(uses_no_skew);
    REQUIRE(!has_unsafe);
  }

  // P4-1: Loop subtracting r1 must be flagged unsafe.
  {
    HIRBuilder b;
    auto* loop_lbl = b.NewLabel();
    auto* exit_lbl = b.NewLabel();
    b.Branch(loop_lbl);
    b.MarkLabel(loop_lbl);
    auto* r1_cur = make_r1_sub(b, 0x10000);
    b.LoadOffset(r1_cur, b.LoadConstantInt64(0x10), INT32_TYPE,
                 LOAD_STORE_NO_PHYSICAL_SKEW);
    auto cond = b.CompareNE(b.Truncate(r1_cur, INT32_TYPE),
                            b.LoadConstantUint32(0u));
    b.BranchTrue(cond, loop_lbl);
    b.Branch(exit_lbl);
    b.MarkLabel(exit_lbl);
    b.Return();
    REQUIRE(b.Finalize());
    bool uses_no_skew = false;
    bool has_unsafe = false;
    AnalyzeStackSkewSafety(&b, &uses_no_skew, &has_unsafe);
    REQUIRE(uses_no_skew);
    REQUIRE(has_unsafe);
  }

  // P4-2: Multi-block acyclic drift exceeding 1 MiB cumulative delta -> unsafe.
  {
    HIRBuilder b;
    for (int i = 0; i < 4; ++i) {
      auto* next_lbl = b.NewLabel();
      make_r1_sub(b, 0x60000);  // 4 * 0x60000 = 0x180000 > 1 MiB
      b.Branch(next_lbl);
      b.MarkLabel(next_lbl);
    }
    auto r1_cur = b.LoadContext(kR1ContextOffset, INT64_TYPE);
    b.LoadOffset(r1_cur, b.LoadConstantInt64(0x10), INT32_TYPE,
                 LOAD_STORE_NO_PHYSICAL_SKEW);
    b.Return();
    REQUIRE(b.Finalize());
    bool uses_no_skew = false;
    bool has_unsafe = false;
    AnalyzeStackSkewSafety(&b, &uses_no_skew, &has_unsafe);
    REQUIRE(uses_no_skew);
    REQUIRE(has_unsafe);
  }

  // P4-3: CALL_EXTERN before LOAD_STORE_NO_PHYSICAL_SKEW -> unsafe.
  {
    HIRBuilder b;
    BuiltinFunction fn(nullptr, 0x82001000u);
    b.CallExtern(&fn);
    auto r1_cur = b.LoadContext(kR1ContextOffset, INT64_TYPE);
    b.LoadOffset(r1_cur, b.LoadConstantInt64(0x10), INT32_TYPE,
                 LOAD_STORE_NO_PHYSICAL_SKEW);
    b.Return();
    REQUIRE(b.Finalize());
    bool uses_no_skew = false;
    bool has_unsafe = false;
    AnalyzeStackSkewSafety(&b, &uses_no_skew, &has_unsafe);
    REQUIRE(uses_no_skew);
    REQUIRE(has_unsafe);
  }
}

TEST_CASE("WAVE3_STRIP_NO_PHYSICAL_SKEW_PRESERVES_NON_LOAD_STORE_FLAGS",
          "[wave3][stack]") {
  using xe::cpu::backend::a64::kR1ContextOffset;
  using xe::cpu::backend::a64::StripNoPhysicalSkewFlags;

  HIRBuilder b;
  auto* next_lbl = b.NewLabel();
  auto* r1 = b.LoadContext(kR1ContextOffset, INT64_TYPE);
  auto* loaded = b.LoadOffset(r1, b.LoadConstantInt64(0x10), INT32_TYPE,
                              LOAD_STORE_NO_PHYSICAL_SKEW | LOAD_STORE_BYTE_SWAP);
  b.StoreOffset(r1, b.LoadConstantInt64(0x18), loaded,
                LOAD_STORE_NO_PHYSICAL_SKEW);
  auto* vec = b.LoadConstantVec128(vec128q(0, 1));
  b.Swizzle(vec, INT32_TYPE, MakeSwizzleMask(3, 2, 1, 0));
  b.Extract(vec, uint8_t{0}, INT64_TYPE);
  b.Branch(next_lbl, BRANCH_LIKELY);
  b.MarkLabel(next_lbl);
  b.CallIndirect(b.LoadConstantUint64(0x82001000u),
                 CALL_TAIL | CALL_POSSIBLE_RETURN);
  REQUIRE(b.Finalize());

  StripNoPhysicalSkewFlags(&b);

  int load_count = 0;
  int store_count = 0;
  int swizzle_count = 0;
  int extract_count = 0;
  int branch_count = 0;
  int call_count = 0;
  for (hir::Block* block = b.first_block(); block; block = block->next) {
    for (hir::Instr* instr = block->instr_head; instr; instr = instr->next) {
      switch (instr->GetOpcodeNum()) {
        case OPCODE_LOAD_OFFSET:
          ++load_count;
          REQUIRE(instr->flags == LOAD_STORE_BYTE_SWAP);
          break;
        case OPCODE_STORE_OFFSET:
          ++store_count;
          REQUIRE(instr->flags == 0);
          break;
        case OPCODE_SWIZZLE:
          ++swizzle_count;
          REQUIRE(instr->flags == INT32_TYPE);
          break;
        case OPCODE_EXTRACT:
          ++extract_count;
          REQUIRE(instr->flags == INT64_TYPE);
          break;
        case OPCODE_BRANCH:
          ++branch_count;
          REQUIRE(instr->flags == BRANCH_LIKELY);
          break;
        case OPCODE_CALL_INDIRECT:
          ++call_count;
          REQUIRE(instr->flags == (CALL_TAIL | CALL_POSSIBLE_RETURN));
          break;
        default:
          break;
      }
    }
  }
  REQUIRE(load_count == 1);
  REQUIRE(store_count == 1);
  REQUIRE(swizzle_count == 1);
  REQUIRE(extract_count == 1);
  REQUIRE(branch_count == 1);
  REQUIRE(call_count == 1);
}

// =============================================================================
// Wave 7 (Onda 1B): A64 PIC site eviction, unified FlushCodeRange,
// ResetIndirection, and 16 KB/parallel loop alignment.
// =============================================================================
TEST_CASE("WAVE7_A64_PIC_EVICTION_RESET_AND_ALIGNMENT", "[wave7][a64]") {
  using xe::cpu::backend::EmitFunctionInfo;
  using xe::cpu::backend::a64::A64CodeCache;
  using xe::cpu::backend::a64::ComputeLoopHeaderAlignmentPaddingForPage;
  using xe::cpu::backend::a64::DecodeA64BranchRel26;
  using xe::cpu::backend::a64::kA64NopInsn;

  auto cache = A64CodeCache::Create();
  REQUIRE(cache != nullptr);
  REQUIRE(cache->Initialize());
  cache->CommitExecutableRange(0x82010000u, 0x82030000u);

  // Place a 20-word synthetic PIC caller buffer and a 2-word target function.
  std::array<uint32_t, 20> pic_buf{};
  pic_buf.fill(kA64NopInsn);
  pic_buf[19] = 0xD65F03C0u;  // ret
  const std::array<uint32_t, 2> target_fn = {kA64NopInsn, 0xD65F03C0u};

  EmitFunctionInfo pic_info = {};
  pic_info.code_size.total = pic_buf.size() * sizeof(uint32_t);
  EmitFunctionInfo target_info = {};
  target_info.code_size.total = target_fn.size() * sizeof(uint32_t);

  void* caller1_rx = nullptr;
  void* caller1_rw = nullptr;
  void* caller2_rx = nullptr;
  void* caller2_rw = nullptr;
  void* caller3_rx = nullptr;
  void* caller3_rw = nullptr;
  void* target_rx = nullptr;
  void* target_rw = nullptr;

  // First placement into an empty code cache with code_size >= 64 B MUST land
  // at execute_base + 0 (no leading 64 B BRK #0 gap from round_up(0, 64)).
  REQUIRE(cache->predicted_next_code_offset(64u) == 0u);
  cache->PlaceGuestCode(0x82010000u, pic_buf.data(), pic_info, nullptr,
                        caller1_rx, caller1_rw);
  REQUIRE(reinterpret_cast<uintptr_t>(caller1_rx) ==
          cache->execute_base_address());
  cache->PlaceGuestCode(0x82010100u, pic_buf.data(), pic_info, nullptr,
                        caller2_rx, caller2_rw);
  cache->PlaceGuestCode(0x82010200u, pic_buf.data(), pic_info, nullptr,
                        caller3_rx, caller3_rw);
  cache->PlaceGuestCode(0x82020000u, const_cast<uint32_t*>(target_fn.data()),
                        target_info, nullptr, target_rx, target_rw);
  REQUIRE(caller1_rx != nullptr);
  REQUIRE(caller2_rx != nullptr);
  REQUIRE(caller3_rx != nullptr);
  REQUIRE(target_rx != nullptr);

  auto reg_site = [&](void* rx, uint32_t caller_addr) {
    const uintptr_t base = reinterpret_cast<uintptr_t>(rx);
    cache->RegisterIndirectCallPicSite(base + 0, base + 4, base + 8, base + 20,
                                       base + 64, false, caller_addr);
    return base;
  };
  const uintptr_t g1 = reg_site(caller1_rx, 0x82010000u);
  const uintptr_t g2 = reg_site(caller2_rx, 0x82010100u);
  const uintptr_t g3 = reg_site(caller3_rx, 0x82010200u);
  REQUIRE(cache->TrackedPicSiteCount() == 3u);

  // Patch g1 to target_rx -> gate becomes NOP.
  REQUIRE(cache->TryPatchIndirectCallPicSite(g1, 0x82020000u, target_rx));
  uint32_t gate1_word = 0;
  std::memcpy(&gate1_word, reinterpret_cast<const void*>(g1), 4);
  REQUIRE(gate1_word == kA64NopInsn);

  // Point-invalidation (guest_start == guest_end) on target -> g1 gate switches
  // to `b fallback_rx` AND g1 is evicted in O(1) without sweeping unrelated sites.
  cache->RevertDirectCallSitesInRange(0x82020000u, 0x82020000u);
  std::memcpy(&gate1_word, reinterpret_cast<const void*>(g1), 4);
  int64_t fb_off = 0;
  bool fb_link = true;
  REQUIRE(DecodeA64BranchRel26(gate1_word, &fb_off, &fb_link));
  REQUIRE(!fb_link);
  REQUIRE(fb_off == 64);
  REQUIRE(cache->TrackedPicSiteCount() == 2u);

  // Direct fallback on g2 -> evicted immediately; repeat fallback returns false.
  REQUIRE(cache->FallbackIndirectCallPicSite(g2));
  REQUIRE(!cache->FallbackIndirectCallPicSite(g2));
  REQUIRE(cache->TrackedPicSiteCount() == 1u);

  // Point-invalidation (guest_start == guest_end) on caller for uninitialized
  // g3 evicts g3 in O(1) -> 0 tracked sites.
  cache->RevertDirectCallSitesInRange(0x82010200u, 0x82010200u);
  REQUIRE(cache->TrackedPicSiteCount() == 0u);

  // ResetIndirection resets committed slot safely and ignores uncommitted ones.
  cache->ResetIndirection(0x82020000u);
  cache->ResetIndirection(0x88000000u);  // uncommitted -> safe no-op

  // ScopedCodeEmitter active counter + 16 KB / parallel alignment check.
  REQUIRE(cache->active_emitters() == 0u);
  {
    auto g_outer = cache->EnterCodeEmitter();
    REQUIRE(cache->active_emitters() == 1u);
    {
      auto g_inner = cache->EnterCodeEmitter();
      REQUIRE(cache->active_emitters() == 2u);
    }
    REQUIRE(cache->active_emitters() == 1u);
  }
  REQUIRE(cache->active_emitters() == 0u);
  REQUIRE(ComputeLoopHeaderAlignmentPaddingForPage(0x3FC0u, 0u, 16384u, true) ==
          64u);
  REQUIRE(ComputeLoopHeaderAlignmentPaddingForPage(0x0FC0u, 0u, 16384u, true) ==
          0u);
  REQUIRE(ComputeLoopHeaderAlignmentPaddingForPage(0x3FC4u, 0u, 16384u, false) ==
          28u);
}
#endif  // XE_ARCH_ARM64

// =============================================================================
// Wave 7 (Onda 1B): Indirect tail-call (bctr) + EnableDynamicCode() safety
// and VSCR[SAT] census / kill-switch / saturating AltiVec execution.
// =============================================================================
TEST_CASE("WAVE7_INDIRECT_TAIL_CALL_AND_VSCR_CENSUS_EXEC",
          "[wave7][a64][vscr]") {
  const bool saved_vscr_census = cvars::vscr_sat_census;
#if XE_ARCH_ARM64
  const bool saved_pic = cvars::a64_indirect_call_pic;
  cvars::a64_indirect_call_pic = true;
#endif
  auto restore_cvars = [&]() {
    cvars::vscr_sat_census = saved_vscr_census;
#if XE_ARCH_ARM64
    cvars::a64_indirect_call_pic = saved_pic;
#endif
  };

  auto memory = std::make_unique<Memory>();
  REQUIRE(memory->Initialize());
  auto backend = CreateBackend();
  if (!backend) {
    restore_cvars();
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

  // Target A (0x80000200): r3 = r3 + 7; blr
  const uint32_t target_a = 0x80000200u;
  w32(target_a + 0, 0x38630007u);  // addi r3, r3, 7
  w32(target_a + 4, 0x4E800020u);  // blr

  // Target B (0x80000300): r3 = r3 + 11; blr
  const uint32_t target_b = 0x80000300u;
  w32(target_b + 0, 0x3863000Bu);  // addi r3, r3, 11
  w32(target_b + 4, 0x4E800020u);  // blr

  // Indirect non-tail caller (bctrl) at 0x80000100: calls target in r4.
  const uint32_t caller_bctrl = 0x80000100u;
  {
    uint32_t p = caller_bctrl;
    w32(p, 0x7C0802A6u); p += 4;  // mflr r0
    w32(p, 0x9001FFF8u); p += 4;  // stw r0, -8(r1)
    w32(p, 0x7C8903A6u); p += 4;  // mtctr r4
    w32(p, 0x4E800421u); p += 4;  // bctrl
    w32(p, 0x8001FFF8u); p += 4;  // lwz r0, -8(r1)
    w32(p, 0x7C0803A6u); p += 4;  // mtlr r0
    w32(p, 0x4E800020u);          // blr
  }

  // Indirect tail caller (bctr) at 0x80000140: tail-calls target in r4.
  const uint32_t caller_bctr = 0x80000140u;
  w32(caller_bctr + 0, 0x7C8903A6u);  // mtctr r4
  w32(caller_bctr + 4, 0x4E800420u);  // bctr

  // AltiVec saturating add + mfvscr at 0x80000400:
  // mtvscr v0 (clear SAT); vaddsbs v3, v4, v5; mfvscr v6; blr
  const uint32_t vscr_fn = 0x80000400u;
  w32(vscr_fn + 0, 0x10000644u);  // mtvscr v0
  w32(vscr_fn + 4, 0x10642B00u);  // vaddsbs v3, v4, v5
  w32(vscr_fn + 8, 0x10C00604u);  // mfvscr v6
  w32(vscr_fn + 12, 0x4E800020u);  // blr

  auto raw = std::make_unique<RawModule>(processor.get());
  raw->set_name("Wave7IndirectVscrTest");
  raw->set_executable(true);
  raw->SetAddressRange(code_base, code_size);
  REQUIRE(processor->AddModule(std::move(raw)));
  processor->backend()->CommitExecutableRange(code_base, code_base + code_size);

  // Non-XEX RawModule must keep vscr_sat_observable() == true after census.
  processor->BeginVscrSatCensus();
  processor->FinishVscrSatCensus();
  REQUIRE(processor->vscr_sat_observable());

  // Kill-switch vscr_sat_census=false must keep vscr_sat_observable() == true.
  cvars::vscr_sat_census = false;
  REQUIRE(processor->vscr_sat_observable());
  cvars::vscr_sat_census = true;

  uint32_t stack_addr = memory->SystemHeapAlloc(64u * 1024u);
  auto ts = std::make_unique<ThreadState>(processor.get(), 0x777u,
                                          stack_addr + 64u * 1024u);

  // Execute bctrl (PIC) & bctr (tail) before EnableDynamicCode().
  {
    uint64_t args_a[2] = {10u, target_a};
    uint64_t args_b[2] = {10u, target_b};
    REQUIRE(processor->Execute(ts.get(), caller_bctrl, args_a, 2) == 17u);
    REQUIRE(processor->Execute(ts.get(), caller_bctrl, args_a, 2) == 17u);
    REQUIRE(processor->Execute(ts.get(), caller_bctrl, args_b, 2) == 21u);
    REQUIRE(processor->Execute(ts.get(), caller_bctr, args_a, 2) == 17u);
    REQUIRE(processor->Execute(ts.get(), caller_bctr, args_b, 2) == 21u);
  }

  // Enable dynamic code AFTER callers were compiled and place a dynamic target
  // outside the indirection table range (0x00100000) to verify no frame/PIC
  // mismatch on either bctrl or tail bctr.
  const uint32_t dyn_target = 0x00100000u;
  auto* low_heap = memory->LookupHeap(dyn_target);
  REQUIRE(low_heap->AllocFixed(
      dyn_target, 0x1000u, 0x1000u,
      kMemoryAllocationReserve | kMemoryAllocationCommit,
      kMemoryProtectRead | kMemoryProtectWrite));
  w32(dyn_target + 0, 0x3863001Du);  // addi r3, r3, 29
  w32(dyn_target + 4, 0x4E800020u);  // blr
  processor->EnableDynamicCode();
  REQUIRE(processor->vscr_sat_observable());
  {
    uint64_t args_dyn[2] = {100u, dyn_target};
    REQUIRE(processor->Execute(ts.get(), caller_bctrl, args_dyn, 2) == 129u);
    REQUIRE(processor->Execute(ts.get(), caller_bctr, args_dyn, 2) == 129u);
  }

  // Verify AltiVec vaddsbs sets PPCContext::vscr_sat when saturating and
  // saturates signed byte 120 + 120 -> 127.
  {
    PPCContext* ctx = ts->context();
    ctx->v[0] = vec128i(0u, 0u, 0u, 0u);
    ctx->v[3] = vec128i(0u, 0u, 0u, 0u);
    ctx->v[4] = vec128b(120, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    ctx->v[5] = vec128b(120, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    ctx->vscr_sat = 0u;
    REQUIRE(processor->Execute(ts.get(), vscr_fn));
    REQUIRE(ctx->v[3].i8[0] == 127);
    REQUIRE(ctx->vscr_sat != 0u);
  }

  ts.reset();
  memory->SystemHeapFree(stack_addr);
  low_heap->Release(dyn_target);
  vheap->Release(code_base);
  restore_cvars();
}
