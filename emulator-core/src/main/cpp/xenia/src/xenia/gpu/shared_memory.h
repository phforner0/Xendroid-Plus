/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_SHARED_MEMORY_H_
#define XENIA_GPU_SHARED_MEMORY_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace xe {
namespace gpu {

// Tracks logical 4 KB subpage validity, GPU-written ownership, upload hashes,
// and submission-local invalidations inside larger host pages (e.g. 16 KB
// Android kernels: 4 subpages per host page -> 4-bit mask 0x0F).
class SubpageTracker {
 public:
  static constexpr uint32_t kSubpageSizeLog2 = 12;
  static constexpr uint32_t kSubpageSize = uint32_t(1) << kSubpageSizeLog2;

  bool Initialize(uint32_t buffer_size_log2, uint32_t host_page_size_log2) {
    Reset();
    if (host_page_size_log2 <= kSubpageSizeLog2 ||
        host_page_size_log2 > kSubpageSizeLog2 + 3 ||
        buffer_size_log2 <= host_page_size_log2) {
      return false;
    }
    buffer_size_ = uint32_t(1) << buffer_size_log2;
    host_page_size_log2_ = host_page_size_log2;
    subpages_per_host_page_log2_ = host_page_size_log2 - kSubpageSizeLog2;
    subpages_per_host_page_ = uint32_t(1) << subpages_per_host_page_log2_;
    full_host_page_mask_ =
        uint8_t((uint32_t(1) << subpages_per_host_page_) - 1u);
    host_page_count_ = uint32_t(1) << (buffer_size_log2 - host_page_size_log2);
    subpage_count_ = uint32_t(1) << (buffer_size_log2 - kSubpageSizeLog2);

    valid_mask_.assign(host_page_count_, 0);
    gpu_written_mask_.assign(host_page_count_, 0);
    hash_valid_mask_.assign(host_page_count_, 0);
    invalidated_in_submission_mask_.assign(host_page_count_,
                                           full_host_page_mask_);
    upload_hash_.assign(subpage_count_, 0);
    active_ = true;
    return true;
  }

  void Reset() {
    active_ = false;
    buffer_size_ = 0;
    host_page_size_log2_ = 0;
    subpages_per_host_page_log2_ = 0;
    subpages_per_host_page_ = 0;
    full_host_page_mask_ = 0;
    host_page_count_ = 0;
    subpage_count_ = 0;
    valid_mask_.clear();
    gpu_written_mask_.clear();
    hash_valid_mask_.clear();
    invalidated_in_submission_mask_.clear();
    upload_hash_.clear();
  }

  bool active() const { return active_; }
  uint32_t host_page_size_log2() const { return host_page_size_log2_; }
  uint32_t subpages_per_host_page_log2() const {
    return subpages_per_host_page_log2_;
  }
  uint32_t subpages_per_host_page() const { return subpages_per_host_page_; }
  uint8_t full_host_page_mask() const { return full_host_page_mask_; }
  uint32_t host_page_count() const { return host_page_count_; }
  uint32_t subpage_count() const { return subpage_count_; }

  uint8_t valid_mask(uint32_t host_page) const {
    return host_page < host_page_count_ ? valid_mask_[host_page] : 0;
  }
  uint8_t gpu_written_mask(uint32_t host_page) const {
    return host_page < host_page_count_ ? gpu_written_mask_[host_page] : 0;
  }
  uint8_t hash_valid_mask(uint32_t host_page) const {
    return host_page < host_page_count_ ? hash_valid_mask_[host_page] : 0;
  }
  bool IsHostPageFullyValid(uint32_t host_page) const {
    return host_page < host_page_count_ &&
           valid_mask_[host_page] == full_host_page_mask_;
  }
  bool HasAnyGpuWrittenSubpage(uint32_t host_page) const {
    return host_page < host_page_count_ && gpu_written_mask_[host_page] != 0;
  }

  static uint8_t LocalSubpageRangeMask(uint32_t local_first,
                                       uint32_t local_last) {
    return uint8_t(((uint32_t(1) << (local_last - local_first + 1)) - 1u)
                   << local_first);
  }

  uint8_t SubpageMaskForHostPage(uint32_t host_page, uint32_t sp_first,
                                 uint32_t sp_last) const {
    const uint32_t page_sp_base = host_page << subpages_per_host_page_log2_;
    const uint32_t page_sp_end = page_sp_base + subpages_per_host_page_ - 1;
    const uint32_t local_first =
        (sp_first > page_sp_base) ? (sp_first - page_sp_base) : 0u;
    const uint32_t local_last = (sp_last < page_sp_end)
                                    ? (sp_last - page_sp_base)
                                    : (subpages_per_host_page_ - 1);
    return LocalSubpageRangeMask(local_first, local_last);
  }

  void InvalidateAll() {
    if (!active_) {
      return;
    }
    std::fill(valid_mask_.begin(), valid_mask_.end(), uint8_t(0));
    std::fill(gpu_written_mask_.begin(), gpu_written_mask_.end(), uint8_t(0));
    std::fill(hash_valid_mask_.begin(), hash_valid_mask_.end(), uint8_t(0));
    std::fill(invalidated_in_submission_mask_.begin(),
              invalidated_in_submission_mask_.end(), full_host_page_mask_);
  }

  void OnGpuSubmissionOpened() {
    if (!active_) {
      return;
    }
    std::fill(invalidated_in_submission_mask_.begin(),
              invalidated_in_submission_mask_.end(), uint8_t(0));
  }

  void MarkRangeValid(uint32_t start, uint32_t length, bool written_by_gpu) {
    if (!active_ || !length || start >= buffer_size_) {
      return;
    }
    length = std::min(length, buffer_size_ - start);
    const uint32_t sp_first = start >> kSubpageSizeLog2;
    const uint32_t sp_last = (start + length - 1) >> kSubpageSizeLog2;
    const uint32_t hp_first = sp_first >> subpages_per_host_page_log2_;
    const uint32_t hp_last = sp_last >> subpages_per_host_page_log2_;
    for (uint32_t hp = hp_first; hp <= hp_last; ++hp) {
      const uint8_t mask = SubpageMaskForHostPage(hp, sp_first, sp_last);
      valid_mask_[hp] |= mask;
      if (written_by_gpu) {
        gpu_written_mask_[hp] |= mask;
        hash_valid_mask_[hp] &= uint8_t(~mask);
      } else {
        gpu_written_mask_[hp] &= uint8_t(~mask);
      }
    }
  }

  void SetSubpageUploadHash(uint32_t subpage, uint64_t hash) {
    if (!active_ || subpage >= subpage_count_) {
      return;
    }
    upload_hash_[subpage] = hash;
    const uint32_t hp = subpage >> subpages_per_host_page_log2_;
    const uint32_t local = subpage & (subpages_per_host_page_ - 1);
    hash_valid_mask_[hp] |= uint8_t(uint32_t(1) << local);
  }

  void InvalidateSubpage(uint32_t subpage, bool mark_submission_invalidated) {
    if (!active_ || subpage >= subpage_count_) {
      return;
    }
    const uint32_t hp = subpage >> subpages_per_host_page_log2_;
    const uint32_t local = subpage & (subpages_per_host_page_ - 1);
    const uint8_t bit = uint8_t(uint32_t(1) << local);
    valid_mask_[hp] &= uint8_t(~bit);
    gpu_written_mask_[hp] &= uint8_t(~bit);
    hash_valid_mask_[hp] &= uint8_t(~bit);
    if (mark_submission_invalidated) {
      invalidated_in_submission_mask_[hp] |= bit;
    }
  }

  void InvalidateHostPages(uint32_t hp_first, uint32_t hp_last,
                           bool mark_submission_invalidated) {
    if (!active_ || hp_first >= host_page_count_) {
      return;
    }
    hp_last = std::min(hp_last, host_page_count_ - 1);
    for (uint32_t hp = hp_first; hp <= hp_last; ++hp) {
      const uint8_t gpu_bits = gpu_written_mask_[hp];
      valid_mask_[hp] = 0;
      gpu_written_mask_[hp] = 0;
      hash_valid_mask_[hp] &= uint8_t(~gpu_bits);
      if (mark_submission_invalidated) {
        invalidated_in_submission_mask_[hp] = full_host_page_mask_;
      }
    }
  }

  void InvalidateExactByteRangeOnUnprotectedHostPages(
      uint32_t start, uint32_t length, uint32_t hp_first, uint32_t hp_last,
      bool mark_submission_invalidated) {
    if (!active_ || !length || start >= buffer_size_ ||
        hp_first >= host_page_count_) {
      return;
    }
    length = std::min(length, buffer_size_ - start);
    const uint32_t sp_first = start >> kSubpageSizeLog2;
    const uint32_t sp_last = (start + length - 1) >> kSubpageSizeLog2;
    hp_last = std::min(hp_last, host_page_count_ - 1);
    for (uint32_t hp = hp_first; hp <= hp_last; ++hp) {
      const uint8_t gpu_bits = gpu_written_mask_[hp];
      const uint8_t exact_bits =
          (hp >= (sp_first >> subpages_per_host_page_log2_) &&
           hp <= (sp_last >> subpages_per_host_page_log2_))
              ? SubpageMaskForHostPage(hp, sp_first, sp_last)
              : uint8_t(0);
      valid_mask_[hp] = 0;
      gpu_written_mask_[hp] = 0;
      hash_valid_mask_[hp] &= uint8_t(~(gpu_bits | exact_bits));
      if (mark_submission_invalidated) {
        invalidated_in_submission_mask_[hp] = full_host_page_mask_;
      }
    }
  }

  void SetHostPageValidFromGpuWritten(uint32_t hp) {
    if (!active_ || hp >= host_page_count_) {
      return;
    }
    const uint8_t prev_valid = valid_mask_[hp];
    const uint8_t gpu_mask = gpu_written_mask_[hp];
    invalidated_in_submission_mask_[hp] |= uint8_t(prev_valid & ~gpu_mask);
    valid_mask_[hp] = gpu_mask;
    hash_valid_mask_[hp] &= uint8_t(~gpu_mask);
  }

  bool IsRangeValid(uint32_t start, uint32_t length) const {
    if (!length) {
      return true;
    }
    if (!active_ || start > buffer_size_ || (buffer_size_ - start) < length) {
      return false;
    }
    const uint32_t sp_first = start >> kSubpageSizeLog2;
    const uint32_t sp_last = (start + length - 1) >> kSubpageSizeLog2;
    const uint32_t hp_first = sp_first >> subpages_per_host_page_log2_;
    const uint32_t hp_last = sp_last >> subpages_per_host_page_log2_;
    for (uint32_t hp = hp_first; hp <= hp_last; ++hp) {
      const uint8_t req_mask = SubpageMaskForHostPage(hp, sp_first, sp_last);
      if ((valid_mask_[hp] & req_mask) != req_mask) {
        return false;
      }
    }
    return true;
  }

  bool IsRangeGpuWritten(uint32_t start, uint32_t length) const {
    if (!length || !active_ || start > buffer_size_ ||
        (buffer_size_ - start) < length) {
      return false;
    }
    const uint32_t sp_first = start >> kSubpageSizeLog2;
    const uint32_t sp_last = (start + length - 1) >> kSubpageSizeLog2;
    const uint32_t hp_first = sp_first >> subpages_per_host_page_log2_;
    const uint32_t hp_last = sp_last >> subpages_per_host_page_log2_;
    for (uint32_t hp = hp_first; hp <= hp_last; ++hp) {
      const uint8_t req_mask = SubpageMaskForHostPage(hp, sp_first, sp_last);
      if ((valid_mask_[hp] & gpu_written_mask_[hp] & req_mask) != req_mask) {
        return false;
      }
    }
    return true;
  }

  void CollectDirtySubpageRanges(
      uint32_t start, uint32_t length,
      std::vector<std::pair<uint32_t, uint32_t>>& out_ranges) const {
    out_ranges.clear();
    if (!active_ || !length || start >= buffer_size_) {
      return;
    }
    length = std::min(length, buffer_size_ - start);
    const uint32_t sp_first = start >> kSubpageSizeLog2;
    const uint32_t sp_last = (start + length - 1) >> kSubpageSizeLog2;
    const uint32_t hp_first = sp_first >> subpages_per_host_page_log2_;
    const uint32_t hp_last = sp_last >> subpages_per_host_page_log2_;

    uint32_t run_start = UINT32_MAX;
    for (uint32_t hp = hp_first; hp <= hp_last; ++hp) {
      const uint8_t req_mask = SubpageMaskForHostPage(hp, sp_first, sp_last);
      const uint8_t dirty_mask = uint8_t(req_mask & ~valid_mask_[hp]);
      if (dirty_mask == 0) {
        if (run_start != UINT32_MAX) {
          const uint32_t run_end = hp << subpages_per_host_page_log2_;
          out_ranges.emplace_back(run_start, run_end - run_start);
          run_start = UINT32_MAX;
        }
        continue;
      }
      const uint32_t page_sp_base = hp << subpages_per_host_page_log2_;
      const uint32_t local_first =
          (sp_first > page_sp_base) ? (sp_first - page_sp_base) : 0u;
      const uint32_t local_last =
          (sp_last < page_sp_base + subpages_per_host_page_ - 1)
              ? (sp_last - page_sp_base)
              : (subpages_per_host_page_ - 1);
      for (uint32_t local = local_first; local <= local_last; ++local) {
        const uint32_t sp = page_sp_base + local;
        if (dirty_mask & uint8_t(uint32_t(1) << local)) {
          if (run_start == UINT32_MAX) {
            run_start = sp;
          }
        } else if (run_start != UINT32_MAX) {
          out_ranges.emplace_back(run_start, sp - run_start);
          run_start = UINT32_MAX;
        }
      }
    }
    if (run_start != UINT32_MAX) {
      out_ranges.emplace_back(run_start, sp_last + 1 - run_start);
    }
  }

  template <typename HashFn>
  void FilterUnchangedDirtySubpages(
      const std::vector<std::pair<uint32_t, uint32_t>>& dirty_ranges,
      HashFn&& try_hash_subpage,
      std::vector<std::pair<uint32_t, uint32_t>>& out_upload_ranges,
      std::vector<uint32_t>& out_newly_full_host_pages) {
    out_upload_ranges.clear();
    out_newly_full_host_pages.clear();
    if (!active_) {
      return;
    }
    uint32_t run_start = UINT32_MAX;
    uint32_t run_end = 0;
    auto flush_run = [&]() {
      if (run_start != UINT32_MAX) {
        out_upload_ranges.emplace_back(run_start, run_end - run_start);
        run_start = UINT32_MAX;
        run_end = 0;
      }
    };

    for (const auto& range : dirty_ranges) {
      const uint32_t sp_end = range.first + range.second;
      for (uint32_t sp = range.first; sp < sp_end; ++sp) {
        const uint32_t hp = sp >> subpages_per_host_page_log2_;
        const uint32_t local = sp & (subpages_per_host_page_ - 1);
        const uint8_t bit = uint8_t(uint32_t(1) << local);
        bool unchanged = false;
        if ((hash_valid_mask_[hp] & bit) && !(gpu_written_mask_[hp] & bit)) {
          uint64_t current_hash = 0;
          if (try_hash_subpage(sp, current_hash) &&
              current_hash == upload_hash_[sp]) {
            unchanged = true;
          }
        }
        if (unchanged) {
          flush_run();
          const bool was_full = valid_mask_[hp] == full_host_page_mask_;
          valid_mask_[hp] |= bit;
          if (!was_full && valid_mask_[hp] == full_host_page_mask_) {
            out_newly_full_host_pages.push_back(hp);
          }
        } else {
          if (run_start == UINT32_MAX) {
            run_start = sp;
            run_end = sp + 1;
          } else if (sp == run_end) {
            ++run_end;
          } else {
            flush_run();
            run_start = sp;
            run_end = sp + 1;
          }
        }
      }
      flush_run();
    }
  }

  bool AnySubpageRangeInvalidatedSinceSubmissionOpen(
      const std::pair<uint32_t, uint32_t>* subpage_ranges,
      uint32_t count) const {
    if (!active_) {
      return false;
    }
    for (uint32_t i = 0; i < count; ++i) {
      if (!subpage_ranges[i].second) {
        continue;
      }
      const uint32_t sp_first = subpage_ranges[i].first;
      const uint32_t sp_last = sp_first + subpage_ranges[i].second - 1;
      const uint32_t hp_first = sp_first >> subpages_per_host_page_log2_;
      const uint32_t hp_last =
          std::min(sp_last >> subpages_per_host_page_log2_, host_page_count_ - 1);
      for (uint32_t hp = hp_first; hp <= hp_last; ++hp) {
        const uint8_t mask = SubpageMaskForHostPage(hp, sp_first, sp_last);
        if (invalidated_in_submission_mask_[hp] & mask) {
          return true;
        }
      }
    }
    return false;
  }

  uint64_t upload_hash(uint32_t subpage) const {
    return subpage < subpage_count_ ? upload_hash_[subpage] : 0;
  }

 private:
  bool active_ = false;
  uint32_t buffer_size_ = 0;
  uint32_t host_page_size_log2_ = 0;
  uint32_t subpages_per_host_page_log2_ = 0;
  uint32_t subpages_per_host_page_ = 0;
  uint8_t full_host_page_mask_ = 0;
  uint32_t host_page_count_ = 0;
  uint32_t subpage_count_ = 0;

  std::vector<uint8_t> valid_mask_;
  std::vector<uint8_t> gpu_written_mask_;
  std::vector<uint8_t> hash_valid_mask_;
  std::vector<uint8_t> invalidated_in_submission_mask_;
  std::vector<uint64_t> upload_hash_;
};

}  // namespace gpu
}  // namespace xe

#ifndef XENIA_GPU_SHARED_MEMORY_PURE_LOGIC_ONLY

#include "xenia/memory.h"

namespace xe {
namespace gpu {

// Manages memory for unconverted textures, resolve targets, vertex and index
// buffers that can be accessed from shaders with Xenon physical addresses, with
// system page size granularity.
class SharedMemory {
 public:
  static constexpr uint32_t kBufferSizeLog2 = 29;
  static constexpr uint32_t kBufferSize = 1 << kBufferSizeLog2;

  virtual ~SharedMemory();
  // Call in the implementation-specific ClearCache.
  virtual void ClearCache();
  virtual void SetSystemPageBlocksValidWithGpuDataWritten();

  void InvalidateAllPages();

  // Call when a new GPU submission starts recording; resets the tracking of
  // pages invalidated while the submission was open.
  void OnGpuSubmissionOpened();

  typedef void (*GlobalWatchCallback)(
      const global_unique_lock_type& global_lock, void* context,
      uint32_t address_first, uint32_t address_last, bool invalidated_by_gpu);
  typedef void* GlobalWatchHandle;
  // Registers a callback invoked when something is invalidated in the GPU
  // memory copy by the CPU or (if triggered explicitly - such as by a resolve)
  // by the GPU. It will be fired for writes to pages previously requested, but
  // may also be fired regardless of whether it was used by GPU emulation - for
  // example, if the game changes protection level of a memory range containing
  // the watched range.
  //
  // The callback is called within the global critical region.
  GlobalWatchHandle RegisterGlobalWatch(GlobalWatchCallback callback,
                                        void* callback_context);
  void UnregisterGlobalWatch(GlobalWatchHandle handle);
  typedef void (*WatchCallback)(const global_unique_lock_type& global_lock,
                                void* context, void* data, uint64_t argument,
                                bool invalidated_by_gpu);
  typedef void* WatchHandle;
  // Registers a callback invoked when the specified memory range is invalidated
  // in the GPU memory copy by the CPU or (if triggered explicitly - such as by
  // a resolve) by the GPU. It will be fired for writes to pages previously
  // requested, but may also be fired regardless of whether it was used by GPU
  // emulation - for example, if the game changes protection level of a memory
  // range containing the watched range.
  //
  // Generally the context is the subsystem pointer (for example, the texture
  // cache), the data is the object (such as a texture), and the argument is
  // additional subsystem/object-specific data (such as whether the range
  // belongs to the base mip level or to the rest of the mips).
  //
  // Called with the global critical region locked. Do NOT watch or unwatch
  // ranges from within it! The watch for the callback is cancelled after the
  // callback - the handle becomes invalid.
  WatchHandle WatchMemoryRange(uint32_t start, uint32_t length,
                               WatchCallback callback, void* callback_context,
                               void* callback_data, uint64_t callback_argument);
  // Unregisters previously registered watched memory range.
  void UnwatchMemoryRange(WatchHandle handle);

  // Checks if the range has been updated, uploads new data if needed and
  // ensures the host GPU memory backing the range are resident. Returns true if
  // the range has been fully updated and is usable.
  bool RequestRange(uint32_t start, uint32_t length);

  // shared_memory_page_history (diagnostics for the frame dump): per page, the
  // frame (as last given to SetHistoryFrame, plus 1, 0 for never) of the last
  // upload from guest memory, CPU invalidation, GPU write and copy of the GPU's
  // data to guest memory, with what made that copy.
  enum class GuestCopySource : uint32_t {
    kNone,
    kResolveReadback,
    kPreserveGuestWrite,
    kPreserveAroundHostWrite,
    kMemexportReadback,
  };
  struct PageHistory {
    uint32_t upload = 0;
    uint32_t invalidation = 0;
    uint32_t gpu_write = 0;
    // The range of the last GPU write (a resolve's or memexport's).
    uint32_t gpu_write_start = 0;
    uint32_t gpu_write_length = 0;
    uint32_t guest_copy = 0;
    GuestCopySource guest_copy_source = GuestCopySource::kNone;
  };
  void SetHistoryFrame(uint32_t frame);
  // Diagnostics: the pages CPU writes made invalid since the previous call,
  // and of them the ones invalidated in the same or the previous frame already
  // (counted with shared_memory_page_history only).
  void TakeInvalidationStats(uint64_t& pages, uint64_t& repeated_pages);
  bool HasPageHistory() const { return !page_history_.empty(); }
  PageHistory GetPageHistory(uint32_t address) const;
  // Records a copy of the GPU's data to guest memory made outside the shared
  // memory (resolve and memory export readback).
  void NoteGuestCopy(uint32_t start, uint32_t length, GuestCopySource source);
  // Returns whether every page in the range is currently valid in the host GPU
  // memory copy. Hold the global critical region if relying on this for state
  // transitions such as watch installation.
  bool IsRangeValid(uint32_t start, uint32_t length) const;
  // Returns whether every page in the range is valid AND was last written by
  // the GPU (not the CPU). Necessary, not sufficient, for treating a resolve
  // destination as still GPU-authoritative: page-granular, so it over-claims
  // on sub-page tails, and sticky across GPU writes so it cannot order one
  // resolve against another.
  bool IsRangeGpuWritten(uint32_t start, uint32_t length) const;

  void TryFindUploadRange(const uint32_t& block_first,
                          const uint32_t& block_last,
                          const uint32_t& page_first, const uint32_t& page_last,
                          uint32_t& range_start,
                          unsigned int& current_upload_range,
                          std::pair<uint32_t, uint32_t>* uploads);

  void TryGetNextUploadRange(uint32_t& range_start, uint64_t& block_valid,
                             const uint32_t& i,
                             unsigned int& current_upload_range,
                             std::pair<uint32_t, uint32_t>* uploads);

  // Marks the range and, if not exact_range, potentially its surroundings
  // (to up to the first GPU-written page, as an access violation exception
  // count optimization) as modified by the CPU, also invalidating GPU-written
  // pages directly in the range.
  std::pair<uint32_t, uint32_t> MemoryInvalidationCallback(
      uint32_t physical_address_start, uint32_t length, bool exact_range);

  // Marks the range as containing GPU-generated data (such as resolves),
  // triggering modification callbacks, making it valid (so pages are not
  // copied from the main memory until they're modified by the CPU) and
  // protecting it. Before writing anything from the GPU side, RequestRange must
  // be called, to make sure, if the GPU writes don't overwrite *everything* in
  // the pages they touch, the CPU data is properly loaded to the unmodified
  // regions in those pages.
  // written_to_buffer false means the GPU wrote guest RAM directly (memexport
  // routed to the host-imported buffer), leaving the shared memory buffer stale
  // for this range rather than up to date.
  // read_back: a memory export whose output the backend reads back into guest
  // memory itself (only the bytes the export changed) - the rest of the bytes
  // of its pages are the CPU's, and the GPU buffer's copy of them may be older
  // than guest memory (an upload or the export recorded but not executed yet),
  // so shared_memory_preserve_gpu_writes doesn't copy those pages over guest
  // memory. Marked with the GPU write itself, so a guest write faulting in
  // between can't see the pages as preserved. Another GPU write of the pages
  // makes them preserved again, and pages that one (a resolve) left valid only
  // in the GPU's copy stay preserved under an export.
  void RangeWrittenByGpu(uint32_t start, uint32_t length,
                         bool written_to_buffer = true, bool read_back = false);

 protected:
  SharedMemory(Memory& memory);
  // Call in implementation-specific initialization.
  bool InitializeCommon();
  void InitializeSparseHostGpuMemory(uint32_t granularity_log2);
  // Call last in implementation-specific shutdown, also callable from the
  // destructor.
  void ShutdownCommon();

  // Sparse allocations are 4 MB, so not too many of them are allocated, but
  // also not to waste too much memory for padding (with 16 MB there's too
  // much).
  static constexpr uint32_t kHostGpuMemoryOptimalSparseAllocationLog2 = 22;
  static_assert(kHostGpuMemoryOptimalSparseAllocationLog2 <= kBufferSizeLog2);

  Memory& memory() const { return memory_; }

  uint32_t page_size_log2() const { return page_size_log2_; }

  // shared_memory_preserve_gpu_writes: copies a range of the host GPU memory
  // copy to guest memory - of pages the GPU wrote last that a CPU write is
  // invalidating, so that their reupload keeps the GPU's data. Called within
  // the global critical region. Returns whether the range was copied.
  virtual bool CopyToGuestMemory(uint32_t start, uint32_t length) {
    return false;
  }

  uint32_t host_gpu_memory_sparse_granularity_log2() const {
    return host_gpu_memory_sparse_granularity_log2_;
  }

  // Allocations in the host buffer are aligned the same way as in the guest
  // physical memory (for instance, if an allocation is 64 KB, it can represent
  // 0-64 KB, 64-128 KB, 128-192 KB in the guest memory, and so on, but not
  // something like 16-80 KB. This is assumed by the rules for texture data
  // access in the texture cache.
  virtual bool AllocateSparseHostGpuMemoryRange(uint32_t offset_allocations,
                                                uint32_t length_allocations);

  // Mark the memory range as updated and protect it.
  void MakeRangeValid(uint32_t start, uint32_t length, bool written_by_gpu,
                      bool read_back = false);

  // Records XXH3 upload hashes for dynamically unwatched pages in
  // [page_first, page_first + page_count) before copying guest RAM.
  void SetPageUploadHashes(uint32_t page_first, uint32_t page_count,
                           const uint8_t* source);
  // Records 4 KB subpage XXH3 hashes for [byte_offset, byte_offset + byte_size)
  // when 4 KB subpage tracking is active on >4 KB host pages.
  void SetSubpageUploadHashes(uint32_t byte_offset, uint32_t byte_size,
                              const uint8_t* source);
  bool subpage_tracking_active() const { return subpage_tracker_.active(); }
  void SetUnwatchSkipVerify(bool skip) { unwatch_skip_verify_ = skip; }

  // Whether any page in the ranges lost validity since the current GPU
  // submission opened. Pages that were never invalidated while the submission
  // was recording can't have been legitimately read by an already-recorded
  // command, so their upload may be reordered to the submission start.
  bool AnyPageInvalidatedSinceSubmissionOpen(
      const std::pair<uint32_t, uint32_t>* page_ranges, uint32_t count) const;
  bool AnyRangeInvalidatedSinceSubmissionOpen(
      const std::pair<uint32_t, uint32_t>* ranges, uint32_t count,
      uint32_t unit_size_log2) const;

  // Uploads a range of host pages - only called if host GPU sparse memory
  // allocation succeeded if needed. While uploading, MakeRangeValid must be
  // called for each successfully uploaded range as early as possible, before
  // the memcpy, to make sure invalidation that happened during the CPU -> GPU
  // memcpy isn't missed (upload_page_ranges is in pages because of this -
  // MakeRangeValid has page granularity). upload_page_ranges are sorted in
  // ascending address order, so front and back can be used to determine the
  // overall bounds of pages to be uploaded.
  virtual bool UploadRanges(
      const std::pair<uint32_t, uint32_t>* upload_page_ranges,
      uint32_t num_upload_ranges) = 0;

  // Optional 4 KB subpage upload hook for backends supporting logical subpage
  // tracking on >4 KB host kernels (e.g. 16 KB Android).
  virtual bool SupportsSubpageUpload() const { return false; }
  virtual bool UploadSubpageRanges(
      const std::pair<uint32_t, uint32_t>* upload_subpage_ranges,
      uint32_t num_upload_ranges) {
    return false;
  }

  // Notification that GPU commands in the current submission may read or write
  // the byte range [start, start + length) in the shared-memory buffer.
  virtual void NoteByteRangeUsedByGpu(uint32_t start, uint32_t length) {}

  const std::vector<std::pair<uint32_t, uint32_t>>& trace_download_ranges() {
    return trace_download_ranges_;
  }
  uint32_t trace_download_page_count() const {
    return trace_download_page_count_;
  }
  // Fills trace_download_ranges() and trace_download_page_count() with
  // GPU-written ranges that need to be downloaded, and also invalidates
  // non-GPU-written ranges so only the needed data - not the all the collected
  // data - will be written in the trace. trace_download_page_count() will be 0
  // if nothing to download.
  void PrepareForTraceDownload();
  // Release memory used for trace download ranges, to be called after
  // downloading or in cases when download is dropped.
  void ReleaseTraceDownloadRanges();

 private:
  Memory& memory_;

  // Log2 of invalidation granularity (the system page size, but the dependency
  // on it is not hard - the access callback takes a range as an argument, and
  // touched pages of the buffer of this size will be invalidated).
  uint32_t page_size_log2_;

  // shared_memory_page_history, empty when off.
  std::vector<PageHistory> page_history_;
  uint32_t history_frame_ = 1;
  // TakeInvalidationStats, with the global critical region held.
  uint64_t invalidated_pages_ = 0;
  uint64_t repeated_invalidated_pages_ = 0;
  // Sets one field of the history of the pages of a range.
  template <uint32_t PageHistory::*field>
  void NotePages(uint32_t start, uint32_t length) {
    if (page_history_.empty() || !length || start >= kBufferSize) {
      return;
    }
    const uint32_t last =
        std::min(start + (length - 1), kBufferSize - 1) >> page_size_log2_;
    for (uint32_t page = start >> page_size_log2_; page <= last; ++page) {
      page_history_[page].*field = history_frame_;
    }
  }

  bool EnsureHostGpuMemoryAllocated(uint32_t start, uint32_t length);
  uint32_t host_gpu_memory_sparse_granularity_log2_ = UINT32_MAX;
  std::vector<uint64_t> host_gpu_memory_sparse_allocated_;
  uint32_t host_gpu_memory_sparse_allocations_ = 0;
  uint32_t host_gpu_memory_sparse_used_bytes_ = 0;

  void* memory_invalidation_callback_handle_ = nullptr;
  void* memory_data_provider_handle_ = nullptr;
  static constexpr unsigned int MAX_UPLOAD_RANGES = 65536;

  // Ranges that need to be uploaded, generated by GetRangesToUpload (a
  // persistently allocated vector).
  // std::vector<std::pair<uint32_t, uint32_t>> upload_ranges_;
  FixedVMemVector<MAX_UPLOAD_RANGES * sizeof(std::pair<uint32_t, uint32_t>)>
      upload_ranges_;

  // Mutex between the guest memory subsystem and the command processor, to be
  // locked when checking or updating validity of pages/ranges and when firing
  // watches.
  static constexpr xe::global_critical_region global_critical_region_{};

  // ***************************************************************************
  // Things below should be fully protected by global_critical_region.
  // ***************************************************************************

  struct SystemPageFlagsBlock {
    // Whether each page is up to date in the GPU buffer.
    uint64_t valid;
    // Subset of valid pages - whether each page in the GPU buffer contains data
    // that was written on the GPU, thus should not be invalidated spuriously.
    uint64_t valid_and_gpu_written;
  };

  // chrispy: todo, systempageflagsblock should be 3 different arrays
  //  Flags for each 64 system pages, interleaved as blocks, so bit scan can be
  //  used to quickly extract ranges.
  // std::vector<SystemPageFlagsBlock> system_page_flags_;

  uint64_t* system_page_flags_valid_ = nullptr;
  uint64_t* system_page_flags_valid_and_gpu_written_ = nullptr;
  // Pages invalidated since the current GPU submission opened - cleared by
  // OnGpuSubmissionOpened.
  uint64_t* system_page_flags_invalidated_in_submission_ = nullptr;
  // Pages of exports read back (RangeWrittenByGpu read_back).
  uint64_t* system_page_flags_preserve_exempt_ = nullptr;
  // Pages accessed by GPU commands in the current open submission.
  std::vector<uint64_t> system_page_flags_used_in_submission_;

  // Dynamic writewatch unwatching for pages rewritten across frames.
  std::vector<uint64_t> page_upload_hash_;
  std::vector<uint64_t> page_upload_hash_valid_;
  std::vector<uint64_t> unwatched_pages_;
  std::vector<uint64_t> texture_watched_pages_;
  std::vector<uint64_t> dirty_this_frame_;
  std::vector<uint64_t> dirty_score_nonzero_;
  std::vector<uint64_t> unwatch_candidates_;
  std::vector<uint8_t> dirty_score_;
  uint32_t unwatched_page_count_ = 0;
  uint32_t unwatch_decay_counter_ = 0;
  bool unwatch_skip_verify_ = false;

  void OnFrameEndUnwatch();
  void EnableWatchesExceptDynamic(uint32_t page_first, uint32_t page_last);
  void VerifyUnwatchedPages(uint32_t page_first, uint32_t page_last);
  void VerifyUnwatchedSubpages(uint32_t start, uint32_t length);
  void RewatchPages(uint32_t block, uint64_t mask);
  bool UnwatchedPageMatchesUpload(uint32_t page);
  void MarkRangeUsedInSubmission(uint32_t start, uint32_t length);
  std::pair<uint32_t, uint32_t> MemoryInvalidationCallbackImpl(
      uint32_t physical_address_start, uint32_t length, bool exact_range,
      bool from_unwatch_verify);

  SubpageTracker subpage_tracker_;
  std::vector<std::pair<uint32_t, uint32_t>> dirty_subpage_ranges_;
  std::vector<std::pair<uint32_t, uint32_t>> filtered_subpage_ranges_;
  std::vector<uint32_t> newly_full_host_pages_;

  // Set when GPU-written flags change, so an unchanged frame skips the copy.
  bool gpu_written_data_dirty_ = false;

  // Bitmap of dirty 64-entry chunks of valid_and_gpu_written, so localized GPU
  // writes only copy the chunks that changed.
  uint32_t dirty_blocks_ = 0;

  unsigned num_system_page_flags_ = 0;

  static std::pair<uint32_t, uint32_t> MemoryInvalidationCallbackThunk(
      void* context_ptr, uint32_t physical_address_start, uint32_t length,
      bool exact_range);

  struct GlobalWatch {
    GlobalWatchCallback callback;
    void* callback_context;
  };
  std::vector<GlobalWatch*> global_watches_;
  struct WatchNode;
  // Watched range placed by other GPU subsystems.
  struct WatchRange {
    union {
      struct {
        WatchCallback callback;
        void* callback_context;
        void* callback_data;
        uint64_t callback_argument;
        WatchNode* node_first;
        uint32_t page_first;
        uint32_t page_last;
      };
      WatchRange* next_free;
    };
  };
  // Node for faster checking of watches when pages have been written to - all
  // 512 MB are split into smaller equally sized buckets, and then ranges are
  // linearly checked.
  struct WatchNode {
    union {
      struct {
        WatchRange* range;
        // Link to another node of this watched range in the next bucket.
        WatchNode* range_node_next;
        // Links to nodes belonging to other watched ranges in the bucket.
        WatchNode* bucket_node_previous;
        WatchNode* bucket_node_next;
      };
      WatchNode* next_free;
    };
  };
  static constexpr uint32_t kWatchBucketSizeLog2 = 22;
  static constexpr uint32_t kWatchBucketCount =
      1 << (kBufferSizeLog2 - kWatchBucketSizeLog2);
  WatchNode* watch_buckets_[kWatchBucketCount] = {};
  // Allocation from pools - taking new WatchRanges and WatchNodes from the free
  // list, and if there are none, creating a pool if the current one is fully
  // used, and linearly allocating from the current pool.
  static constexpr uint32_t kWatchRangePoolSize = 8192;
  static constexpr uint32_t kWatchNodePoolSize = 8192;
  std::vector<WatchRange*> watch_range_pools_;
  std::vector<WatchNode*> watch_node_pools_;
  uint32_t watch_range_current_pool_allocated_ = 0;
  uint32_t watch_node_current_pool_allocated_ = 0;
  WatchRange* watch_range_first_free_ = nullptr;
  WatchNode* watch_node_first_free_ = nullptr;

  // GPU-written memory downloading for traces. <Start address, length>.
  std::vector<std::pair<uint32_t, uint32_t>> trace_download_ranges_;
  uint32_t trace_download_page_count_ = 0;

  // Triggers the watches (global and per-range), removing triggered range
  // watches.
  void FireWatches(uint32_t page_first, uint32_t page_last,
                   bool invalidated_by_gpu);
  // Unlinks and frees the range and its nodes. Call this in the global critical
  // region.
  void UnlinkWatchRange(WatchRange* range);
};

}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_SHARED_MEMORY_PURE_LOGIC_ONLY

#endif  // XENIA_GPU_SHARED_MEMORY_H_
