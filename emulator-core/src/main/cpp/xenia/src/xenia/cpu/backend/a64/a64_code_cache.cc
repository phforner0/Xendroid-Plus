/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/cpu/backend/a64/a64_code_cache.h"

#include <cstdint>

#include "xenia/base/cvar.h"
#include "xenia/base/platform.h"
#include "xenia/cpu/backend/a64/a64_direct_call.h"
#include "xenia/cpu/function.h"
#if XE_PLATFORM_WIN32
#include "xenia/base/platform_win.h"
#endif
#if XE_PLATFORM_xendroid
#include "../aarch64_disasm.h"
#endif
#if defined(__linux__)
#include <unistd.h>

#include <cinttypes>
#include <cstring>
#endif

DEFINE_bool(a64_perf_map, true,
            "Write a simpleperf/perf 'generic JIT symbols' map file "
            "(perf-<pid>.map) mapping generated code to guest function "
            "names, so profilers can attribute JIT samples.",
            "CPU");

namespace xe {
namespace cpu {
namespace backend {
namespace a64 {

A64CodeCache::~A64CodeCache() {
  if (perf_map_file_) {
    std::fclose(perf_map_file_);
  }
}

void A64CodeCache::OnCodePlaced(uint32_t guest_address,
                                GuestFunction* function_info,
                                void* code_execute_address, size_t code_size) {
#if defined(__linux__)
  if (!cvars::a64_perf_map) {
    return;
  }
  std::lock_guard<std::mutex> lock(perf_map_mutex_);
  if (!perf_map_file_) {
    if (perf_map_open_failed_) {
      return;
    }
    // simpleperf looks for the map in the app's data directory
    // (/data/data/<package>/perf-<pid>.map); the package name is the process
    // name up to the ':' (this runs in the :emu process). Fall back to the
    // standalone-program locations.
    char process_name[256] = {0};
    if (FILE* cmdline = std::fopen("/proc/self/cmdline", "rb")) {
      std::fread(process_name, 1, sizeof(process_name) - 1, cmdline);
      std::fclose(cmdline);
    }
    if (char* colon = std::strchr(process_name, ':')) {
      *colon = '\0';
    }
    const int pid = getpid();
    char path[512];
    std::snprintf(path, sizeof(path), "/data/data/%s/perf-%d.map",
                  process_name, pid);
    perf_map_file_ = std::fopen(path, "we");
    if (!perf_map_file_) {
      std::snprintf(path, sizeof(path), "/data/local/tmp/perf-%d.map", pid);
      perf_map_file_ = std::fopen(path, "we");
    }
    if (!perf_map_file_) {
      std::snprintf(path, sizeof(path), "/tmp/perf-%d.map", pid);
      perf_map_file_ = std::fopen(path, "we");
    }
    if (!perf_map_file_) {
      perf_map_open_failed_ = true;
      return;
    }
  }
  const uint64_t address = reinterpret_cast<uint64_t>(code_execute_address);
  if (function_info && !function_info->name().empty()) {
    std::fprintf(perf_map_file_, "0x%" PRIx64 " 0x%zx %s\n", address,
                 code_size, function_info->name().c_str());
  } else if (guest_address) {
    std::fprintf(perf_map_file_, "0x%" PRIx64 " 0x%zx sub_%08X\n", address,
                 code_size, guest_address);
  } else {
    std::fprintf(perf_map_file_, "0x%" PRIx64 " 0x%zx a64_host_code\n",
                 address, code_size);
  }
  std::fflush(perf_map_file_);
#endif
}

void A64CodeCache::FillCode(void* write_address, size_t size) {
  // Fill with BRK #0 (0xD4200000), 4-byte aligned.
  constexpr uint32_t kBrk0 = 0xD4200000;
  auto* p = reinterpret_cast<uint32_t*>(write_address);
  auto* end =
      reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(write_address) + size);
  for (; p < end; ++p) {
    *p = kBrk0;
  }
}

void A64CodeCache::FlushCodeRange(void* write_address, void* execute_address,
                                  size_t size) {
  if (!size) {
    return;
  }
#if XE_PLATFORM_WIN32
  void* target = execute_address ? execute_address : write_address;
  FlushInstructionCache(GetCurrentProcess(), target, size);
#elif XE_ARCH_ARM64 && !XE_PLATFORM_MAC
  uint64_t ctr_el0 = 0;
  asm volatile("mrs %0, ctr_el0" : "=r"(ctr_el0));
  const size_t dline = size_t(4) << ((ctr_el0 >> 16) & 0xFu);
  const size_t iline = size_t(4) << (ctr_el0 & 0xFu);

  if (write_address) {
    const uintptr_t d_start =
        reinterpret_cast<uintptr_t>(write_address) & ~(dline - 1);
    const uintptr_t d_end = reinterpret_cast<uintptr_t>(write_address) + size;
    for (uintptr_t p = d_start; p < d_end; p += dline) {
      asm volatile("dc cvau, %0" : : "r"(p) : "memory");
    }
  }
  asm volatile("dsb ish" : : : "memory");

  if (execute_address) {
    const uintptr_t i_start =
        reinterpret_cast<uintptr_t>(execute_address) & ~(iline - 1);
    const uintptr_t i_end =
        reinterpret_cast<uintptr_t>(execute_address) + size;
    for (uintptr_t p = i_start; p < i_end; p += iline) {
      asm volatile("ic ivau, %0" : : "r"(p) : "memory");
    }
  }
  asm volatile("dsb ish\nisb" : : : "memory");
#else
#if XE_PLATFORM_xendroid
  // XELOGI("ASM:\n{}", aarch64_disasm(...));
#endif
  void* target = execute_address ? execute_address : write_address;
  __builtin___clear_cache(
      reinterpret_cast<char*>(target),
      reinterpret_cast<char*>(static_cast<uint8_t*>(target) + size));
#endif
}

void A64CodeCache::FlushPatchedInstruction(void* write_address,
                                           void* execute_address) {
  FlushCodeRange(write_address, execute_address, sizeof(uint32_t));
}

void A64CodeCache::RegisterDirectCallSite(uint32_t target_guest_addr,
                                          uintptr_t call_site_rx,
                                          uint32_t original_stub_insn,
                                          bool is_tail,
                                          uintptr_t tail_stub_rx) {
  if (!IsInExecuteRange(call_site_rx)) {
    return;
  }
  std::lock_guard<std::mutex> lock(direct_call_sites_mutex_);
  auto& sites = pending_direct_call_sites_[target_guest_addr];
  for (const auto& existing : sites) {
    if (existing.call_site_rx == call_site_rx) {
      return;
    }
  }
  sites.push_back(DirectCallSite{call_site_rx, original_stub_insn, is_tail,
                                 tail_stub_rx});
}

void A64CodeCache::BackpatchDirectCallSites(uint32_t target_guest_addr,
                                            void* target_execute_address) {
  if (!target_execute_address) {
    return;
  }
  const uintptr_t target_rx =
      reinterpret_cast<uintptr_t>(target_execute_address);
  if (!IsInExecuteRange(target_rx)) {
    return;
  }
  std::lock_guard<std::mutex> lock(direct_call_sites_mutex_);
  auto it = pending_direct_call_sites_.find(target_guest_addr);
  if (it == pending_direct_call_sites_.end()) {
    return;
  }
  std::vector<DirectCallSite> remaining;
  for (const auto& site : it->second) {
    if (!IsInExecuteRange(site.call_site_rx)) {
      continue;
    }
    int64_t diff = 0;
    if (!ComputeA64Rel26Offset(site.call_site_rx, target_rx, &diff)) {
      // Target is > +-128 MiB away: bypass PatchDirectCallThunk permanently by
      // switching the non-tail stub's entry branch to NOP (falling into the
      // indirection table lookup) and drop this unpatchable site from pending.
      if (!site.is_tail && site.tail_stub_rx != 0 &&
          IsInExecuteRange(site.tail_stub_rx)) {
        if (uint32_t* stub_rw = WriteSlotForExecuteAddress(site.tail_stub_rx)) {
          constexpr uint32_t kA64NopInsn = 0xD503201Fu;
          if (PatchA64InstructionAtomic(stub_rw, kA64NopInsn)) {
            FlushPatchedInstruction(
                stub_rw, reinterpret_cast<void*>(site.tail_stub_rx));
          }
        }
      }
      continue;
    }
    uint32_t* rw_slot = WriteSlotForExecuteAddress(site.call_site_rx);
    if (!rw_slot) {
      remaining.push_back(site);
      continue;
    }
    const uint32_t branch_insn = EncodeA64BranchRel26(diff, !site.is_tail);
    if (PatchA64InstructionAtomic(rw_slot, branch_insn)) {
      FlushPatchedInstruction(rw_slot,
                              reinterpret_cast<void*>(site.call_site_rx));
      patched_direct_call_sites_[target_guest_addr].push_back(site);
    } else {
      remaining.push_back(site);
    }
  }
  if (remaining.empty()) {
    pending_direct_call_sites_.erase(it);
  } else {
    it->second = std::move(remaining);
  }
}

bool A64CodeCache::TryPatchDirectCallSite(uint32_t target_guest_addr,
                                          uintptr_t call_site_rx,
                                          void* target_execute_address) {
  if (!target_execute_address || !IsInExecuteRange(call_site_rx)) {
    return false;
  }
  const uintptr_t target_rx =
      reinterpret_cast<uintptr_t>(target_execute_address);
  if (!IsInExecuteRange(target_rx)) {
    return false;
  }
  std::lock_guard<std::mutex> lock(direct_call_sites_mutex_);
  auto it = pending_direct_call_sites_.find(target_guest_addr);
  if (it == pending_direct_call_sites_.end()) {
    return false;
  }
  auto& pending = it->second;
  for (size_t i = 0; i < pending.size(); ++i) {
    if (pending[i].call_site_rx != call_site_rx) {
      continue;
    }
    int64_t diff = 0;
    if (!ComputeA64Rel26Offset(call_site_rx, target_rx, &diff)) {
      const DirectCallSite site = pending[i];
      if (!site.is_tail && site.tail_stub_rx != 0 &&
          IsInExecuteRange(site.tail_stub_rx)) {
        if (uint32_t* stub_rw = WriteSlotForExecuteAddress(site.tail_stub_rx)) {
          constexpr uint32_t kA64NopInsn = 0xD503201Fu;
          if (PatchA64InstructionAtomic(stub_rw, kA64NopInsn)) {
            FlushPatchedInstruction(
                stub_rw, reinterpret_cast<void*>(site.tail_stub_rx));
          }
        }
      }
      pending[i] = pending.back();
      pending.pop_back();
      if (pending.empty()) {
        pending_direct_call_sites_.erase(it);
      }
      return false;
    }
    uint32_t* rw_slot = WriteSlotForExecuteAddress(call_site_rx);
    if (!rw_slot) {
      return false;
    }
    const DirectCallSite site = pending[i];
    const uint32_t branch_insn = EncodeA64BranchRel26(diff, !site.is_tail);
    if (!PatchA64InstructionAtomic(rw_slot, branch_insn)) {
      return false;
    }
    FlushPatchedInstruction(rw_slot, reinterpret_cast<void*>(call_site_rx));
    patched_direct_call_sites_[target_guest_addr].push_back(site);
    pending[i] = pending.back();
    pending.pop_back();
    if (pending.empty()) {
      pending_direct_call_sites_.erase(it);
    }
    return true;
  }
  return false;
}

void A64CodeCache::RevertDirectCallSitesInRange(uint32_t guest_start,
                                                uint32_t guest_end) {
  if (guest_start > guest_end) {
    return;
  }
  std::lock_guard<std::mutex> lock(direct_call_sites_mutex_);
  for (auto it = patched_direct_call_sites_.begin();
       it != patched_direct_call_sites_.end();) {
    const uint32_t target_addr = it->first;
    if (target_addr < guest_start || target_addr > guest_end) {
      ++it;
      continue;
    }
    auto& pending = pending_direct_call_sites_[target_addr];
    for (const auto& site : it->second) {
      uint32_t* rw_slot = WriteSlotForExecuteAddress(site.call_site_rx);
      if (rw_slot &&
          PatchA64InstructionAtomic(rw_slot, site.original_stub_insn)) {
        FlushPatchedInstruction(rw_slot,
                                reinterpret_cast<void*>(site.call_site_rx));
      }
      pending.push_back(site);
    }
    it = patched_direct_call_sites_.erase(it);
  }
}

}  // namespace a64
}  // namespace backend
}  // namespace cpu
}  // namespace xe
