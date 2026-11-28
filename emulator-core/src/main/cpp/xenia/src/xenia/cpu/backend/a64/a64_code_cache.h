/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_CPU_BACKEND_A64_A64_CODE_CACHE_H_
#define XENIA_CPU_BACKEND_A64_A64_CODE_CACHE_H_

#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "xenia/cpu/backend/a64/a64_direct_call.h"
#include "xenia/cpu/backend/code_cache_base.h"

namespace xe {
namespace cpu {
namespace backend {
namespace a64 {

// Host-page-aware loop header alignment padding (bytes, multiple of 4):
// - Under parallel compilation (allow_page_nudge == false) or non-power-of-two
//   page sizes: align strictly to 32 B relative to the 64 B-aligned entry
//   (<= 28 B NOPs, zero speculative page-split drift).
// - When uncontended (allow_page_nudge == true): nudge headers landing in the
//   final 64 B of a host page (4 KB or 16 KB on Android) across the page
//   boundary (<= 64 B NOPs, also 32 B aligned); otherwise align to 32 B.
inline constexpr size_t ComputeLoopHeaderAlignmentPaddingForPage(
    size_t current_offset, size_t entry_mod_page, size_t host_page_size,
    bool allow_page_nudge) {
  if ((current_offset & 3u) != 0) {
    return 0;
  }
  if (!allow_page_nudge || host_page_size < 4096u ||
      (host_page_size & (host_page_size - 1u)) != 0) {
    return (32u - (current_offset & 31u)) & 31u;
  }
  const size_t page_mask = host_page_size - 1u;
  const size_t page_off =
      ((entry_mod_page & page_mask) + current_offset) & page_mask;
  if (page_off >= host_page_size - 64u) {
    return host_page_size - page_off;
  }
  return (32u - (page_off & 31u)) & 31u;
}

class A64CodeCache : public CodeCacheBase<A64CodeCache> {
 public:
  ~A64CodeCache() override;

  static std::unique_ptr<A64CodeCache> Create();

  void* LookupUnwindInfo(uint64_t host_pc) override { return nullptr; }

  // CRTP hooks for CodeCacheBase.
  void FillCode(void* write_address, size_t size);
  void FlushCodeRange(void* write_address, void* execute_address, size_t size);
  void FlushPatchedInstruction(void* write_address, void* execute_address);
  size_t PreferredEntryAlignment(size_t code_size) const;

  // Direct call site registry for atomic bl/b <rel26> back-patching and
  // rollback on InvalidateCodeRange / ForgetSymbol.
  struct DirectCallSite {
    uintptr_t call_site_rx;
    uint32_t original_stub_insn;
    bool is_tail;
    uintptr_t tail_stub_rx = 0;
  };
  void RegisterDirectCallSite(uint32_t target_guest_addr,
                              uintptr_t call_site_rx,
                              uint32_t original_stub_insn, bool is_tail,
                              uintptr_t tail_stub_rx = 0);
  void BackpatchDirectCallSites(uint32_t target_guest_addr,
                                void* target_execute_address);
  bool TryPatchDirectCallSite(uint32_t target_guest_addr,
                              uintptr_t call_site_rx,
                              void* target_execute_address);
  void RevertDirectCallSitesInRange(uint32_t guest_start, uint32_t guest_end);

  // 1-slot monomorphic PIC registry for CallIndirect (bctrl) with 2-stage
  // 1-word atomic CMODX publication and safe fallback rollback.
  void RegisterIndirectCallPicSite(uintptr_t gate_rx, uintptr_t cmp_lo_rx,
                                   uintptr_t cmp_hi_rx, uintptr_t call_op_rx,
                                   uintptr_t fallback_rx, bool is_tail,
                                   uint32_t caller_guest_addr = 0);
  bool TryPatchIndirectCallPicSite(uintptr_t gate_rx,
                                   uint32_t target_guest_addr,
                                   void* target_execute_address);
  bool FallbackIndirectCallPicSite(uintptr_t gate_rx);
  size_t TrackedPicSiteCount() const;

  // Writes one simpleperf "generic JIT symbols" map entry per placed
  // function (perf-<pid>.map) so profilers can attribute guest code by name
  // instead of showing an anonymous "unknown" DSO. Gated by the a64_perf_map
  // cvar; Linux/Android only.
  void OnCodePlaced(uint32_t guest_address, GuestFunction* function_info,
                    void* code_execute_address, size_t code_size);

  // Virtual for platform-specific overrides (_win.cc / _posix.cc).
  virtual UnwindReservation RequestUnwindReservation(uint8_t* entry_address) {
    return UnwindReservation();
  }
  virtual void PlaceCode(uint32_t guest_address, void* machine_code,
                         const EmitFunctionInfo& func_info,
                         void* code_execute_address,
                         UnwindReservation unwind_reservation) {}

 protected:
  A64CodeCache() = default;

  std::mutex perf_map_mutex_;
  FILE* perf_map_file_ = nullptr;
  bool perf_map_open_failed_ = false;

  std::mutex direct_call_sites_mutex_;
  std::unordered_map<uint32_t, std::vector<DirectCallSite>>
      pending_direct_call_sites_;
  std::unordered_map<uint32_t, std::vector<DirectCallSite>>
      patched_direct_call_sites_;

  void RetirePicSiteLocked(
      std::unordered_map<uintptr_t, IndirectCallPicSite>::iterator site_it);

  mutable std::mutex pic_sites_mutex_;
  std::unordered_map<uintptr_t, IndirectCallPicSite> pic_sites_by_gate_;
  std::unordered_map<uint32_t, std::vector<uintptr_t>> patched_pic_sites_;
  std::unordered_map<uintptr_t, uint32_t> pic_caller_by_gate_;
  std::unordered_map<uint32_t, std::vector<uintptr_t>> pic_sites_by_caller_;
};

}  // namespace a64
}  // namespace backend
}  // namespace cpu
}  // namespace xe

#endif  // XENIA_CPU_BACKEND_A64_A64_CODE_CACHE_H_
