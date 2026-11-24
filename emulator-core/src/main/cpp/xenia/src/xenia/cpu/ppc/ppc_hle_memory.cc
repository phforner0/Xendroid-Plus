/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/cpu/ppc/ppc_hle_memory.h"

#include <algorithm>
#include <cstring>

#include "xenia/base/cvar.h"
#include "xenia/base/memory.h"
#include "xenia/base/mutex.h"
#include "xenia/cpu/ppc/ppc_context.h"
#include "xenia/cpu/processor.h"
#include "xenia/memory.h"

DEFINE_bool(
    ppc_hle_memory_ops, true,
    "Intercept canonical CRT/XDK memcpy/memset/memmove with validated host "
    "Bionic/NEON memory ops.",
    "CPU");

DEFINE_string(
    ppc_hle_memory_functions, "",
    "Comma-separated <hex_addr>:<memcpy|memset|memmove>:<xxh3_hex> overrides.",
    "CPU");

DEFINE_uint32(
    ppc_hle_memory_min_bytes, 64,
    "Minimum byte count to dispatch HLE memory op instead of inline PPC loop.",
    "CPU");

DECLARE_bool(shared_memory_preserve_gpu_writes);

namespace xe {
namespace cpu {
namespace ppc {

namespace {

bool HasVirtualMappedRangeOverlap(xe::Memory* memory, uint32_t start,
                                  uint32_t len) {
  const uint32_t last = start + (len - 1u);
  if (memory->LookupVirtualMappedRange(start) ||
      memory->LookupVirtualMappedRange(last)) {
    return true;
  }
  constexpr uint32_t kStep = 4096u;
  uint32_t cur = start & ~(kStep - 1u);
  if (cur <= UINT32_MAX - kStep) {
    cur += kStep;
    while (cur < last) {
      if (memory->LookupVirtualMappedRange(cur)) {
        return true;
      }
      if (cur > UINT32_MAX - kStep) {
        break;
      }
      cur += kStep;
    }
  }
  return false;
}

bool TriggerPhysicalCallbacksAllAliases(xe::Memory* memory, uint32_t phys,
                                        uint32_t len, bool is_write,
                                        bool unwatch_exact_range) {
  if (!memory || len == 0u || phys >= 0x20000000u ||
      len > 0x20000000u - phys) {
    return false;
  }
  bool any_triggered = false;
  if (memory->TriggerPhysicalMemoryCallbacks(
          xe::global_critical_region::AcquireDirect(), 0xA0000000u + phys, len,
          is_write, unwatch_exact_range, /*unprotect=*/true)) {
    any_triggered = true;
  }
  if (memory->TriggerPhysicalMemoryCallbacks(
          xe::global_critical_region::AcquireDirect(), 0xC0000000u + phys, len,
          is_write, unwatch_exact_range, /*unprotect=*/true)) {
    any_triggered = true;
  }
  const uint64_t phys_start = phys;
  const uint64_t phys_end = phys_start + len;
  const uint64_t e_start = std::max<uint64_t>(phys_start, 0x00001000ull);
  const uint64_t e_end = std::min<uint64_t>(phys_end, 0x1FD01000ull);
  if (e_start < e_end) {
    const uint32_t e_va =
        0xE0000000u + static_cast<uint32_t>(e_start - 0x00001000ull);
    const uint32_t e_len = static_cast<uint32_t>(e_end - e_start);
    if (memory->TriggerPhysicalMemoryCallbacks(
            xe::global_critical_region::AcquireDirect(), e_va, e_len, is_write,
            unwatch_exact_range, /*unprotect=*/true)) {
      any_triggered = true;
    }
  }
  return any_triggered;
}

void ExecuteHleMemoryOp(PPCContext* ctx, HleMemoryOp op) {
  if (!ctx || !ctx->processor || !ctx->processor->memory() ||
      op == HleMemoryOp::kNone || !cvars::ppc_hle_memory_ops) {
    if (ctx) {
      ctx->scratch = 0;
    }
    return;
  }

  xe::Memory* memory = ctx->processor->memory();
  if (memory->user_virtual_membase() &&
      ctx->virtual_membase == memory->user_virtual_membase()) {
    ctx->scratch = 0;
    return;
  }

  const uint32_t dst = static_cast<uint32_t>(ctx->r[3]);
  const uint32_t arg = static_cast<uint32_t>(ctx->r[4]);
  const uint32_t len = static_cast<uint32_t>(ctx->r[5]);
  if (len == 0u) {
    ctx->r[3] = dst;
    ctx->scratch = 1u;
    return;
  }

  if (!CheckGuestRange32NoWrap(dst, len)) {
    ctx->scratch = 0;
    return;
  }
  const uint32_t alloc_granularity =
      static_cast<uint32_t>(xe::memory::allocation_granularity());
  if (CrossesSkew16KbBoundary(dst, len, alloc_granularity)) {
    ctx->scratch = 0;
    return;
  }

  const uint32_t dst_last = dst + (len - 1u);
  if (dst < 0x80000000u && dst_last >= 0x7FC00000u) {
    ctx->scratch = 0;
    return;
  }

  if (HasVirtualMappedRangeOverlap(memory, dst, len)) {
    ctx->scratch = 0;
    return;
  }

  xe::BaseHeap* dst_start_heap = memory->LookupHeap(dst);
  xe::BaseHeap* dst_end_heap = memory->LookupHeap(dst_last);
  if (!dst_start_heap || !dst_end_heap ||
      (dst_start_heap->heap_type() == xe::HeapType::kGuestPhysical) !=
          (dst_end_heap->heap_type() == xe::HeapType::kGuestPhysical) ||
      dst_start_heap != dst_end_heap) {
    ctx->scratch = 0;
    return;
  }
  xe::PhysicalHeap* dst_phys_heap =
      dst_start_heap->heap_type() == xe::HeapType::kGuestPhysical
          ? static_cast<xe::PhysicalHeap*>(dst_start_heap)
          : nullptr;
  uint32_t dst_phys = 0;
  uint8_t* host_dst = nullptr;
  xe::memory::PageAccess dst_access =
      dst_start_heap->QueryRangeAccess(dst, dst_last);
  if (dst_phys_heap) {
    dst_phys = dst_phys_heap->GetPhysicalAddress(dst);
    if (dst_phys >= 0x20000000u || len > 0x20000000u - dst_phys) {
      ctx->scratch = 0;
      return;
    }
    if (dst_access == xe::memory::PageAccess::kNoAccess &&
        dst_phys_heap->parent_heap()) {
      // Physical memory allocated via another alias window (e.g. 0xE0000000 vs
      // 0xA0000000) still has its authoritative page access in parent_heap_.
      dst_access = dst_phys_heap->parent_heap()->QueryRangeAccess(
          dst_phys, dst_phys + (len - 1u));
    }
    if (dst_access != xe::memory::PageAccess::kReadWrite) {
      ctx->scratch = 0;
      return;
    }
    host_dst = memory->TranslatePhysical(dst_phys);
  } else {
    if (ctx->virtual_membase != memory->virtual_membase()) {
      ctx->scratch = 0;
      return;
    }
    if (dst_access != xe::memory::PageAccess::kReadWrite) {
      ctx->scratch = 0;
      return;
    }
    host_dst = memory->TranslateVirtual(dst);
  }

  const uint32_t src = arg;
  const uint8_t* host_src = nullptr;
  xe::PhysicalHeap* src_phys_heap = nullptr;
  uint32_t src_phys = 0;
  if (op != HleMemoryOp::kMemset) {
    if (!CheckGuestRange32NoWrap(src, len)) {
      ctx->scratch = 0;
      return;
    }
    if (CrossesSkew16KbBoundary(src, len, alloc_granularity)) {
      ctx->scratch = 0;
      return;
    }
    const uint32_t src_last = src + (len - 1u);
    if (src < 0x80000000u && src_last >= 0x7FC00000u) {
      ctx->scratch = 0;
      return;
    }
    if (HasVirtualMappedRangeOverlap(memory, src, len)) {
      ctx->scratch = 0;
      return;
    }
    xe::BaseHeap* src_start_heap = memory->LookupHeap(src);
    xe::BaseHeap* src_end_heap = memory->LookupHeap(src_last);
    if (!src_start_heap || !src_end_heap ||
        (src_start_heap->heap_type() == xe::HeapType::kGuestPhysical) !=
            (src_end_heap->heap_type() == xe::HeapType::kGuestPhysical) ||
        src_start_heap != src_end_heap) {
      ctx->scratch = 0;
      return;
    }
    xe::memory::PageAccess src_access =
        src_start_heap->QueryRangeAccess(src, src_last);
    if (src_start_heap->heap_type() == xe::HeapType::kGuestPhysical) {
      src_phys_heap = static_cast<xe::PhysicalHeap*>(src_start_heap);
      src_phys = src_phys_heap->GetPhysicalAddress(src);
      if (src_phys >= 0x20000000u || len > 0x20000000u - src_phys) {
        ctx->scratch = 0;
        return;
      }
      if (src_access == xe::memory::PageAccess::kNoAccess &&
          src_phys_heap->parent_heap()) {
        src_access = src_phys_heap->parent_heap()->QueryRangeAccess(
            src_phys, src_phys + (len - 1u));
      }
      if (src_access != xe::memory::PageAccess::kReadOnly &&
          src_access != xe::memory::PageAccess::kReadWrite) {
        ctx->scratch = 0;
        return;
      }
      host_src = memory->TranslatePhysical(src_phys);
    } else {
      if (ctx->virtual_membase != memory->virtual_membase()) {
        ctx->scratch = 0;
        return;
      }
      if (src_access != xe::memory::PageAccess::kReadOnly &&
          src_access != xe::memory::PageAccess::kReadWrite) {
        ctx->scratch = 0;
        return;
      }
      host_src = memory->TranslateVirtual(src);
    }

    if (dst_phys_heap && src_phys_heap &&
        cvars::shared_memory_preserve_gpu_writes &&
        PhysicalRangesOverlap(dst_phys, src_phys, len)) {
      ctx->scratch = 0;
      return;
    }
  }

  // Trigger readwatch/writewatch pre-invalidation across all physical alias
  // heaps (0xA0/0xC0/0xE0) holding no other mutexes.
  if (op != HleMemoryOp::kMemset && src_phys_heap) {
    if (!TriggerPhysicalCallbacksAllAliases(memory, src_phys, len,
                                            /*is_write=*/false,
                                            /*unwatch_exact_range=*/true)) {
      ctx->scratch = 0;
      return;
    }
  }
  if (dst_phys_heap) {
    const size_t page_size = xe::memory::page_size();
    const bool dst_full_pages =
        page_size > 0 &&
        ((static_cast<size_t>(dst_phys) | static_cast<size_t>(len)) &
         (page_size - 1u)) == 0u;
    const bool unwatch_exact_range =
        !cvars::shared_memory_preserve_gpu_writes || dst_full_pages;
    if (!TriggerPhysicalCallbacksAllAliases(memory, dst_phys, len,
                                            /*is_write=*/true,
                                            unwatch_exact_range)) {
      ctx->scratch = 0;
      return;
    }
  }

  if (op == HleMemoryOp::kMemset) {
    std::memset(host_dst, static_cast<int>(arg & 0xFFu), len);
  } else {
    // Always use memmove so physical virtual aliases (0xA0000000 / 0xE0000000)
    // remain overlap-safe.
    std::memmove(host_dst, host_src, len);
  }

  ctx->r[3] = dst;
  ctx->scratch = 1u;
}

}  // namespace

void HleMemcpy(PPCContext* ppc_context, void* arg0, void* arg1) {
  (void)arg0;
  (void)arg1;
  ExecuteHleMemoryOp(ppc_context, HleMemoryOp::kMemcpy);
}

void HleMemset(PPCContext* ppc_context, void* arg0, void* arg1) {
  (void)arg0;
  (void)arg1;
  ExecuteHleMemoryOp(ppc_context, HleMemoryOp::kMemset);
}

void HleMemmove(PPCContext* ppc_context, void* arg0, void* arg1) {
  (void)arg0;
  (void)arg1;
  ExecuteHleMemoryOp(ppc_context, HleMemoryOp::kMemmove);
}

}  // namespace ppc
}  // namespace cpu
}  // namespace xe
