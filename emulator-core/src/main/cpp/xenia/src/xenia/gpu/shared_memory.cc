/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2020 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/gpu/shared_memory.h"

#include <algorithm>
#include <atomic>

#if defined(__ANDROID__)
#include <sys/system_properties.h>
#endif

#include "xenia/base/assert.h"
#include "xenia/base/bit_range.h"
#include "xenia/base/cvar.h"
#include "xenia/base/logging.h"
#include "xenia/base/profiling.h"
#include "xenia/base/xxhash.h"

DECLARE_bool(shm_coalesce_writewatch_pages);

DEFINE_bool(
    shm_unwatch_release_exclusion, true,
    "Pages watched by a texture or CPU-side cache stop being excluded from "
    "dynamic unwatching once a CPU write fires their watches.",
    "GPU");
DEFINE_bool(
    shared_memory_unwatch_dynamic_pages, true,
    "Leave shared memory pages that the CPU rewrites frame after frame "
    "unwatched and verify them against their XXH3 upload hash instead of "
    "re-protecting with mprotect on every upload.",
    "GPU");
DEFINE_int32(
    shm_unwatch_score, 2,
    "Frames with a CPU rewrite (score +1 each, -1 per frame without) after "
    "which an uploaded shared memory page is left unwatched and hash-verified.",
    "GPU");
DEFINE_int32(
    shm_unwatch_score_max, 8,
    "Ceiling of a page's rewrite score before it decays back to watched.",
    "GPU");
DEFINE_int32(
    shm_unwatch_decay_frames, 1,
    "Frames between decrements of a shared memory page's rewrite score.",
    "GPU");

DEFINE_bool(
    shared_memory_preserve_gpu_writes, false,
    "When the CPU writes to a page the GPU wrote last, copy the page's GPU "
    "data to guest memory before invalidating it, so the reupload keeps it "
    "(the data the guest memory never got: textures a game copies with the "
    "GPU, resolves not read back yet). Without it, the whole page is "
    "reuploaded from guest memory and the GPU's data is lost - Crysis 3 drew "
    "a character with another texture's leftovers that way. Needs a "
    "host-mapped GPU memory copy. The pages of memory exports read back are "
    "not copied (their readback brings the bytes the export changed): Need "
    "for Speed: Most Wanted froze with them copied where its prologue gives "
    "way to the first 3D scene (3 of 3 launches, 2026-10-07; 0 of 4 with them "
    "left out, 2026-10-11).",
    "GPU");
DEFINE_bool(
    shared_memory_page_history, false,
    "Diagnostics: remember, per page of the GPU's memory copy, the frame of its "
    "last upload, CPU invalidation, GPU write and copy of the GPU's data to "
    "guest memory, for the frame dump (debug.xendroid.frame_dump) to report "
    "where broken data came from. 2.5 MB.",
    "GPU");
DEFINE_bool(
    shared_memory_subpage_tracking, false,
    "Track logical 4 KB subpage validity and dirty bitmasks (4 bits per "
    "16 KB host page) on host kernels with >4 KB pages (such as Android "
    "16 KB devices) to reduce upload and invalidation amplification. "
    "debug.xendroid.subpage_tracking overrides at startup on Android.",
    "GPU");

namespace xe {
namespace gpu {

SharedMemory::SharedMemory(Memory& memory) : memory_(memory) {
  page_size_log2_ = xe::log2_ceil(uint32_t(xe::memory::page_size()));
}

SharedMemory::~SharedMemory() { ShutdownCommon(); }

bool SharedMemory::InitializeCommon() {
  size_t num_system_page_flags_entries =
      ((kBufferSize >> page_size_log2_) + 63) / 64;
  num_system_page_flags_ = static_cast<uint32_t>(num_system_page_flags_entries);

  // Four flag arrays (valid, valid_and_gpu_written,
  // invalidated_in_submission, preserve_exempt), 8 bytes per 64-page entry.
  uint64_t* system_page_flags_base = (uint64_t*)memory::AllocFixed(
      nullptr, num_system_page_flags_ * 4 * sizeof(uint64_t),
      memory::AllocationType::kReserveCommit, memory::PageAccess::kReadWrite);

  if (!system_page_flags_base) {
    XELOGE("SharedMemory: Failed to allocate system page flags");
    return false;
  }

  system_page_flags_valid_ = system_page_flags_base;
  system_page_flags_valid_and_gpu_written_ =
      system_page_flags_base + num_system_page_flags_;
  system_page_flags_invalidated_in_submission_ =
      system_page_flags_base + 2 * num_system_page_flags_;
  system_page_flags_preserve_exempt_ =
      system_page_flags_base + 3 * num_system_page_flags_;

  memset(system_page_flags_base, 0, 4 * 8 * num_system_page_flags_entries);

  system_page_flags_used_in_submission_.assign(num_system_page_flags_,
                                               UINT64_MAX);
  page_upload_hash_.assign(size_t(num_system_page_flags_) * 64, 0);
  page_upload_hash_valid_.assign(num_system_page_flags_, 0);
  unwatched_pages_.assign(num_system_page_flags_, 0);
  texture_watched_pages_.assign(num_system_page_flags_, 0);
  dirty_this_frame_.assign(num_system_page_flags_, 0);
  dirty_score_nonzero_.assign(num_system_page_flags_, 0);
  unwatch_candidates_.assign(num_system_page_flags_, 0);
  dirty_score_.assign(size_t(num_system_page_flags_) * 64, 0);
  unwatched_page_count_ = 0;
  unwatch_decay_counter_ = 0;

  if (cvars::shared_memory_page_history) {
    page_history_.resize(kBufferSize >> page_size_log2_);
  }

  bool subpage_tracking = cvars::shared_memory_subpage_tracking;
#if defined(__ANDROID__)
  {
    char value[PROP_VALUE_MAX] = {};
    if (__system_property_get("debug.xendroid.subpage_tracking", value) > 0 &&
        (value[0] == '0' || value[0] == '1') && value[1] == '\0') {
      subpage_tracking = (value[0] == '1');
      if (cvars::shared_memory_subpage_tracking != subpage_tracking) {
        cvars::shared_memory_subpage_tracking = subpage_tracking;
        XELOGI(
            "debug.xendroid.subpage_tracking: shared_memory_subpage_tracking "
            "= {}",
            subpage_tracking);
      }
    }
  }
#endif
  if (subpage_tracking && SupportsSubpageUpload() &&
      subpage_tracker_.Initialize(kBufferSizeLog2, page_size_log2_)) {
    XELOGI(
        "Shared memory: 4 KB subpage tracking enabled ({} subpages per {} KB "
        "host page)",
        subpage_tracker_.subpages_per_host_page(),
        (uint32_t(1) << page_size_log2_) >> 10);
  }

  memory_invalidation_callback_handle_ =
      memory_.RegisterPhysicalMemoryInvalidationCallback(
          MemoryInvalidationCallbackThunk, this);
  return true;
}

void SharedMemory::InitializeSparseHostGpuMemory(uint32_t granularity_log2) {
  assert_true(granularity_log2 <= kBufferSizeLog2);
  assert_true(host_gpu_memory_sparse_granularity_log2_ == UINT32_MAX);
  host_gpu_memory_sparse_granularity_log2_ = granularity_log2;
  host_gpu_memory_sparse_allocated_.resize(
      size_t(1) << (std::max(kBufferSizeLog2 - granularity_log2, uint32_t(6)) -
                    6));
}

void SharedMemory::ShutdownCommon() {
  ReleaseTraceDownloadRanges();

  FireWatches(0, (kBufferSize - 1) >> page_size_log2_, false);
  assert_true(global_watches_.empty());
  // No watches now, so no references to the pools accessible by guest threads -
  // safe not to enter the global critical region.
  watch_node_first_free_ = nullptr;
  watch_node_current_pool_allocated_ = 0;
  for (WatchNode* pool : watch_node_pools_) {
    delete[] pool;
  }
  watch_node_pools_.clear();
  watch_range_first_free_ = nullptr;
  watch_range_current_pool_allocated_ = 0;
  for (WatchRange* pool : watch_range_pools_) {
    delete[] pool;
  }
  watch_range_pools_.clear();

  if (memory_invalidation_callback_handle_ != nullptr) {
    memory_.UnregisterPhysicalMemoryInvalidationCallback(
        memory_invalidation_callback_handle_);
    memory_invalidation_callback_handle_ = nullptr;
  }

  if (host_gpu_memory_sparse_used_bytes_) {
    host_gpu_memory_sparse_used_bytes_ = 0;
    COUNT_profile_set("gpu/shared_memory/host_gpu_memory_sparse_used_mb", 0);
  }
  if (host_gpu_memory_sparse_allocations_) {
    host_gpu_memory_sparse_allocations_ = 0;
    COUNT_profile_set("gpu/shared_memory/host_gpu_memory_sparse_allocations",
                      0);
  }
  host_gpu_memory_sparse_allocated_.clear();
  host_gpu_memory_sparse_allocated_.shrink_to_fit();
  host_gpu_memory_sparse_granularity_log2_ = UINT32_MAX;

  // One AllocFixed backs all flag arrays; valid_ is its base pointer.
  if (system_page_flags_valid_) {
    memory::DeallocFixed(system_page_flags_valid_, 0,
                         memory::DeallocationType::kRelease);
    system_page_flags_valid_ = nullptr;
  }

  system_page_flags_valid_and_gpu_written_ = nullptr;
  system_page_flags_invalidated_in_submission_ = nullptr;
  system_page_flags_preserve_exempt_ = nullptr;
  system_page_flags_used_in_submission_.clear();
  page_upload_hash_.clear();
  page_upload_hash_valid_.clear();
  unwatched_pages_.clear();
  texture_watched_pages_.clear();
  dirty_this_frame_.clear();
  dirty_score_nonzero_.clear();
  unwatch_candidates_.clear();
  dirty_score_.clear();
  unwatched_page_count_ = 0;
  unwatch_decay_counter_ = 0;
  num_system_page_flags_ = 0;
  subpage_tracker_.Reset();
  dirty_subpage_ranges_.clear();
  filtered_subpage_ranges_.clear();
  newly_full_host_pages_.clear();
}

void SharedMemory::InvalidateAllPages() {
  auto global_lock = global_critical_region_.Acquire();

  for (uint32_t b = 0; b < num_system_page_flags_; ++b) {
    __atomic_store_n(&system_page_flags_valid_[b], uint64_t(0),
                     __ATOMIC_RELEASE);
  }
  std::memset(system_page_flags_valid_and_gpu_written_, 0,
              num_system_page_flags_ * sizeof(uint64_t));
  std::memset(system_page_flags_invalidated_in_submission_, 0xFF,
              num_system_page_flags_ * sizeof(uint64_t));
  std::fill(system_page_flags_used_in_submission_.begin(),
            system_page_flags_used_in_submission_.end(), UINT64_MAX);
  std::fill(page_upload_hash_valid_.begin(), page_upload_hash_valid_.end(), 0);
  subpage_tracker_.InvalidateAll();

  // Mark all chunks dirty so the next copy refreshes everything.
  dirty_blocks_ = 0xFFFFFFFF;
  gpu_written_data_dirty_ = true;
}

void SharedMemory::OnGpuSubmissionOpened() {
  if (!system_page_flags_invalidated_in_submission_) {
    return;
  }
  auto global_lock = global_critical_region_.Acquire();
  std::memset(system_page_flags_invalidated_in_submission_, 0,
              num_system_page_flags_ * sizeof(uint64_t));
  std::fill(system_page_flags_used_in_submission_.begin(),
            system_page_flags_used_in_submission_.end(), 0);
  subpage_tracker_.OnGpuSubmissionOpened();
}

void SharedMemory::MarkRangeUsedInSubmission(uint32_t start, uint32_t length) {
  if (!length || start >= kBufferSize) {
    return;
  }
  length = std::min(length, kBufferSize - start);
  NoteByteRangeUsedByGpu(start, length);
  if (system_page_flags_used_in_submission_.empty()) {
    return;
  }
  const uint32_t page_first = start >> page_size_log2_;
  const uint32_t page_last = (start + length - 1) >> page_size_log2_;
  const uint32_t block_first = page_first >> 6;
  const uint32_t block_last = page_last >> 6;
  auto global_lock = global_critical_region_.Acquire();
  for (uint32_t b = block_first; b <= block_last; ++b) {
    uint64_t mask = UINT64_MAX;
    if (b == block_first) {
      mask &= ~((uint64_t(1) << (page_first & 63)) - 1);
    }
    if (b == block_last && (page_last & 63) != 63) {
      mask &= (uint64_t(1) << ((page_last & 63) + 1)) - 1;
    }
    system_page_flags_used_in_submission_[b] |= mask;
  }
}

bool SharedMemory::AnyPageInvalidatedSinceSubmissionOpen(
    const std::pair<uint32_t, uint32_t>* page_ranges, uint32_t count) const {
  auto global_lock = global_critical_region_.Acquire();
  for (uint32_t i = 0; i < count; ++i) {
    uint32_t page_first = page_ranges[i].first;
    uint32_t page_last = page_first + page_ranges[i].second - 1;
    uint32_t block_first = page_first >> 6;
    uint32_t block_last = page_last >> 6;
    for (uint32_t b = block_first; b <= block_last; ++b) {
      uint64_t mask = UINT64_MAX;
      if (b == block_first) {
        mask &= ~((uint64_t(1) << (page_first & 63)) - 1);
      }
      if (b == block_last && (page_last & 63) != 63) {
        mask &= (uint64_t(1) << ((page_last & 63) + 1)) - 1;
      }
      if (system_page_flags_invalidated_in_submission_[b] & mask) {
        return true;
      }
    }
  }
  return false;
}

bool SharedMemory::AnyRangeInvalidatedSinceSubmissionOpen(
    const std::pair<uint32_t, uint32_t>* ranges, uint32_t count,
    uint32_t unit_size_log2) const {
  if (unit_size_log2 == page_size_log2_) {
    return AnyPageInvalidatedSinceSubmissionOpen(ranges, count);
  }
  auto global_lock = global_critical_region_.Acquire();
  if (subpage_tracker_.active() &&
      unit_size_log2 == SubpageTracker::kSubpageSizeLog2) {
    return subpage_tracker_.AnySubpageRangeInvalidatedSinceSubmissionOpen(
        ranges, count);
  }
  for (uint32_t i = 0; i < count; ++i) {
    if (!ranges[i].second) {
      continue;
    }
    const uint32_t byte_first = ranges[i].first << unit_size_log2;
    const uint32_t byte_last =
        ((ranges[i].first + ranges[i].second) << unit_size_log2) - 1;
    const uint32_t page_first = byte_first >> page_size_log2_;
    const uint32_t page_last = byte_last >> page_size_log2_;
    for (uint32_t b = page_first >> 6; b <= page_last >> 6; ++b) {
      uint64_t mask = UINT64_MAX;
      if (b == (page_first >> 6)) {
        mask &= ~((uint64_t(1) << (page_first & 63)) - 1);
      }
      if (b == (page_last >> 6) && (page_last & 63) != 63) {
        mask &= (uint64_t(1) << ((page_last & 63) + 1)) - 1;
      }
      if (system_page_flags_invalidated_in_submission_[b] & mask) {
        return true;
      }
    }
  }
  return false;
}

void SharedMemory::ClearCache() {
  // Keeping GPU-written data, so "invalidated by GPU".
  FireWatches(0, (kBufferSize - 1) >> page_size_log2_, true);
  // No watches now, so no references to the pools accessible by guest threads -
  // safe not to enter the global critical region.
  watch_node_first_free_ = nullptr;
  watch_node_current_pool_allocated_ = 0;
  for (WatchNode* pool : watch_node_pools_) {
    delete[] pool;
  }
  watch_node_pools_.clear();
  watch_range_first_free_ = nullptr;
  watch_range_current_pool_allocated_ = 0;
  for (WatchRange* pool : watch_range_pools_) {
    delete[] pool;
  }
  watch_range_pools_.clear();
  SetSystemPageBlocksValidWithGpuDataWritten();
}

void SharedMemory::SetSystemPageBlocksValidWithGpuDataWritten() {
  auto global_lock = global_critical_region_.Acquire();

  if (gpu_written_data_dirty_) {
    // Only copy the chunks that changed. Read and reset the dirty bitmap.
    uint32_t dirty_mask = dirty_blocks_;
    dirty_blocks_ = 0;
    gpu_written_data_dirty_ = false;

    uint32_t dirty_count = xe::bit_count(dirty_mask);

    auto sync_block_from_gpu_written = [this](uint32_t i) {
      if (subpage_tracker_.active()) {
        uint64_t new_valid_block = 0;
        uint64_t invalidated_block = 0;
        const uint32_t base_hp = i << 6;
        const uint32_t hp_limit =
            std::min(base_hp + 64u, subpage_tracker_.host_page_count());
        for (uint32_t hp = base_hp; hp < hp_limit; ++hp) {
          const uint64_t bit = uint64_t(1) << (hp & 63);
          if (subpage_tracker_.SetHostPageValidFromGpuWritten(hp) != 0) {
            invalidated_block |= bit;
          }
          if (subpage_tracker_.IsHostPageFullyValid(hp)) {
            new_valid_block |= bit;
          }
        }
        system_page_flags_invalidated_in_submission_[i] |= invalidated_block;
        __atomic_store_n(&system_page_flags_valid_[i], new_valid_block,
                         __ATOMIC_RELEASE);
      } else {
        uint64_t gpu_written_block =
            system_page_flags_valid_and_gpu_written_[i];
        system_page_flags_invalidated_in_submission_[i] |=
            system_page_flags_valid_[i] & ~gpu_written_block;
        __atomic_store_n(&system_page_flags_valid_[i], gpu_written_block,
                         __ATOMIC_RELEASE);
      }
    };

    // Past ~50% dirty the per-chunk bookkeeping costs more than a straight
    // copy.
    if (dirty_count == 0 || dirty_count > 16) {
      for (uint32_t i = 0; i < num_system_page_flags_; ++i) {
        sync_block_from_gpu_written(i);
      }
    } else {
      while (dirty_mask) {
        uint32_t block_index;
        xe::bit_scan_forward(dirty_mask, &block_index);
        dirty_mask &= ~(1u << block_index);

        // Each chunk is 64 entries of 8 bytes.
        uint32_t entry_offset = block_index * 64;
        uint32_t entry_end =
            std::min(entry_offset + 64u, num_system_page_flags_);
        for (uint32_t i = entry_offset; i < entry_end; ++i) {
          sync_block_from_gpu_written(i);
        }
      }
    }
  }
}

SharedMemory::GlobalWatchHandle SharedMemory::RegisterGlobalWatch(
    GlobalWatchCallback callback, void* callback_context) {
  GlobalWatch* watch = new GlobalWatch;
  watch->callback = callback;
  watch->callback_context = callback_context;

  auto global_lock = global_critical_region_.Acquire();
  global_watches_.push_back(watch);

  return reinterpret_cast<GlobalWatchHandle>(watch);
}

void SharedMemory::UnregisterGlobalWatch(GlobalWatchHandle handle) {
  auto watch = reinterpret_cast<GlobalWatch*>(handle);

  {
    auto global_lock = global_critical_region_.Acquire();
    auto it = std::find(global_watches_.begin(), global_watches_.end(), watch);
    assert_false(it == global_watches_.end());
    if (it != global_watches_.end()) {
      global_watches_.erase(it);
    }
  }

  delete watch;
}

SharedMemory::WatchHandle SharedMemory::WatchMemoryRange(
    uint32_t start, uint32_t length, WatchCallback callback,
    void* callback_context, void* callback_data, uint64_t callback_argument) {
  if (length == 0 || start >= kBufferSize) {
    return nullptr;
  }
  length = std::min(length, kBufferSize - start);
  uint32_t watch_page_first = start >> page_size_log2_;
  uint32_t watch_page_last = (start + length - 1) >> page_size_log2_;
  uint32_t bucket_first =
      watch_page_first << page_size_log2_ >> kWatchBucketSizeLog2;
  uint32_t bucket_last =
      watch_page_last << page_size_log2_ >> kWatchBucketSizeLog2;
  // Pages with active range watches (e.g., textures) must receive immediate
  // invalidation callbacks and are excluded from dynamic unwatching.
  if (!texture_watched_pages_.empty()) {
    for (uint32_t b = watch_page_first >> 6; b <= watch_page_last >> 6; ++b) {
      uint64_t mask = UINT64_MAX;
      if (b == watch_page_first >> 6) {
        mask &= ~((uint64_t(1) << (watch_page_first & 63)) - 1);
      }
      if (b == watch_page_last >> 6 && (watch_page_last & 63) != 63) {
        mask &= (uint64_t(1) << ((watch_page_last & 63) + 1)) - 1;
      }
      texture_watched_pages_[b] |= mask;
      if (unwatched_pages_[b] & mask) {
        RewatchPages(b, unwatched_pages_[b] & mask);
      }
    }
  }
  // chrispy: Not required the global lock is always held by the caller
  // auto global_lock = global_critical_region_.Acquire();

  // Allocate the range.
  WatchRange* range = watch_range_first_free_;
  if (range != nullptr) {
    watch_range_first_free_ = range->next_free;
  } else {
    if (watch_range_pools_.empty() ||
        watch_range_current_pool_allocated_ >= kWatchRangePoolSize) {
      watch_range_pools_.push_back(new WatchRange[kWatchRangePoolSize]);
      watch_range_current_pool_allocated_ = 0;
    }
    range = &(watch_range_pools_.back()[watch_range_current_pool_allocated_++]);
  }
  range->callback = callback;
  range->callback_context = callback_context;
  range->callback_data = callback_data;
  range->callback_argument = callback_argument;
  range->page_first = watch_page_first;
  range->page_last = watch_page_last;

  // Allocate and link the nodes.
  WatchNode* node_previous = nullptr;
  for (uint32_t i = bucket_first; i <= bucket_last; ++i) {
    WatchNode* node = watch_node_first_free_;
    if (node != nullptr) {
      watch_node_first_free_ = node->next_free;
    } else {
      if (watch_node_pools_.empty() ||
          watch_node_current_pool_allocated_ >= kWatchNodePoolSize) {
        watch_node_pools_.push_back(new WatchNode[kWatchNodePoolSize]);
        watch_node_current_pool_allocated_ = 0;
      }
      node = &(watch_node_pools_.back()[watch_node_current_pool_allocated_++]);
    }
    node->range = range;
    node->range_node_next = nullptr;
    if (node_previous != nullptr) {
      node_previous->range_node_next = node;
    } else {
      range->node_first = node;
    }
    node_previous = node;
    node->bucket_node_previous = nullptr;
    node->bucket_node_next = watch_buckets_[i];
    if (watch_buckets_[i] != nullptr) {
      watch_buckets_[i]->bucket_node_previous = node;
    }
    watch_buckets_[i] = node;
  }

  return reinterpret_cast<WatchHandle>(range);
}

void SharedMemory::UnwatchMemoryRange(WatchHandle handle) {
  auto global_lock = global_critical_region_.Acquire();
  UnlinkWatchRange(reinterpret_cast<WatchRange*>(handle));
}

void SharedMemory::FireWatches(uint32_t page_first, uint32_t page_last,
                               bool invalidated_by_gpu) {
  uint32_t address_first = page_first << page_size_log2_;
  uint32_t address_last =
      (page_last << page_size_log2_) + ((1 << page_size_log2_) - 1);
  uint32_t bucket_first = address_first >> kWatchBucketSizeLog2;
  uint32_t bucket_last = address_last >> kWatchBucketSizeLog2;

  auto global_lock = global_critical_region_.Acquire();

  // Fire global watches.
  for (const auto global_watch : global_watches_) {
    global_watch->callback(global_lock, global_watch->callback_context,
                           address_first, address_last, invalidated_by_gpu);
  }

  // Fire per-range watches.
  for (uint32_t i = bucket_first; i <= bucket_last; ++i) {
    WatchNode* node = watch_buckets_[i];
    if (i + 1 <= bucket_last) {
      WatchNode* nextnode = watch_buckets_[i + 1];
      if (nextnode) {
        swcache::PrefetchL1(nextnode->range);
      }
    }
    while (node != nullptr) {
      WatchRange* range = node->range;
      // Store the next node now since when the callback is triggered, the links
      // will be broken.
      node = node->bucket_node_next;
      if (node) {
        swcache::PrefetchL1(node);
      }
      if (page_first <= range->page_last && page_last >= range->page_first) {
        range->callback(global_lock, range->callback_context,
                        range->callback_data, range->callback_argument,
                        invalidated_by_gpu);
        if (node && node->range) {
          swcache::PrefetchL1(node->range);
        }
        UnlinkWatchRange(range);
      }
    }
  }
}

void SharedMemory::RangeWrittenByGpu(uint32_t start, uint32_t length,
                                     bool written_to_buffer, bool read_back) {
  if (length == 0 || start >= kBufferSize) {
    return;
  }
  length = std::min(length, kBufferSize - start);
  uint32_t end = start + length - 1;
  uint32_t page_first = start >> page_size_log2_;
  uint32_t page_last = end >> page_size_log2_;

  // Trigger modification callbacks so, for instance, resolved data is loaded to
  // the texture.
  FireWatches(page_first, page_last, true);

  // Mark the range as valid (so pages are not reuploaded until modified by the
  // CPU) and watch it so the CPU can reuse it and this will be caught.
  // written_to_buffer also marks it GPU-authoritative, which must not happen
  // when the write went to guest RAM instead - that would keep the buffer's
  // stale copy across cache clears.
  MakeRangeValid(start, length, written_to_buffer, read_back);
}

bool SharedMemory::AllocateSparseHostGpuMemoryRange(
    uint32_t offset_allocations, uint32_t length_allocations) {
  assert_always(
      "Sparse host GPU memory allocation has been initialized, but the "
      "implementation doesn't provide AllocateSparseHostGpuMemoryRange");
  return false;
}

void SharedMemory::SetHistoryFrame(uint32_t frame) {
  history_frame_ = frame + 1;
  OnFrameEndUnwatch();
}

SharedMemory::PageHistory SharedMemory::GetPageHistory(
    uint32_t address) const {
  if (page_history_.empty() || address >= kBufferSize) {
    return PageHistory();
  }
  return page_history_[address >> page_size_log2_];
}

void SharedMemory::NoteGuestCopy(uint32_t start, uint32_t length,
                                 GuestCopySource source) {
  if (page_history_.empty() || !length || start >= kBufferSize) {
    return;
  }
  const uint32_t last =
      std::min(start + (length - 1), kBufferSize - 1) >> page_size_log2_;
  for (uint32_t page = start >> page_size_log2_; page <= last; ++page) {
    page_history_[page].guest_copy = history_frame_;
    page_history_[page].guest_copy_source = source;
  }
}

void SharedMemory::MakeRangeValid(uint32_t start, uint32_t length,
                                  bool written_by_gpu, bool read_back) {
  if (length == 0 || start >= kBufferSize) {
    return;
  }
  length = std::min(length, kBufferSize - start);
  uint32_t last = start + length - 1;
  uint32_t valid_page_first = start >> page_size_log2_;
  uint32_t valid_page_last = last >> page_size_log2_;
  uint32_t valid_block_first = valid_page_first >> 6;
  uint32_t valid_block_last = valid_page_last >> 6;

  {
    auto global_lock = global_critical_region_.Acquire();

    if (written_by_gpu && memory_invalidation_callback_handle_) {
      // Protect before publishing the GPU-written bit so a racing CPU store
      // cannot slip past without faulting while leaving the bit set.
      memory().EnablePhysicalMemoryAccessCallbacks(
          valid_page_first << page_size_log2_,
          (valid_page_last - valid_page_first + 1) << page_size_log2_, true,
          false);
    }

    if (subpage_tracker_.active()) {
      subpage_tracker_.MarkRangeValid(start, length, written_by_gpu);
      for (uint32_t hp = valid_page_first; hp <= valid_page_last; ++hp) {
        const uint32_t blk = hp >> 6;
        const uint64_t bit = uint64_t(1) << (hp & 63);
        if (subpage_tracker_.IsHostPageFullyValid(hp)) {
          __atomic_fetch_or(&system_page_flags_valid_[blk], bit,
                            __ATOMIC_RELEASE);
        } else {
          __atomic_fetch_and(&system_page_flags_valid_[blk], ~bit,
                             __ATOMIC_RELEASE);
        }
        if (!page_upload_hash_valid_.empty()) {
          __atomic_fetch_and(&page_upload_hash_valid_[blk], ~bit,
                             __ATOMIC_RELAXED);
        }
        if (written_by_gpu) {
          if (!unwatched_pages_.empty() && (unwatched_pages_[blk] & bit)) {
            unwatched_pages_[blk] &= ~bit;
            --unwatched_page_count_;
          }
          if (!system_page_flags_used_in_submission_.empty()) {
            system_page_flags_used_in_submission_[blk] |= bit;
          }
          if (read_back) {
            system_page_flags_preserve_exempt_[blk] |=
                bit & ~(system_page_flags_valid_and_gpu_written_[blk] &
                        ~system_page_flags_preserve_exempt_[blk]);
          } else {
            system_page_flags_preserve_exempt_[blk] &= ~bit;
          }
          system_page_flags_valid_and_gpu_written_[blk] |= bit;
          gpu_written_data_dirty_ = true;
          dirty_blocks_ |= 1u << (blk >> 6);
        } else if (!subpage_tracker_.HasAnyGpuWrittenSubpage(hp)) {
          system_page_flags_valid_and_gpu_written_[blk] &= ~bit;
        }
      }
    } else {
      for (uint32_t i = valid_block_first; i <= valid_block_last; ++i) {
        uint64_t valid_bits = UINT64_MAX;
        if (i == valid_block_first) {
          valid_bits &= ~((uint64_t(1) << (valid_page_first & 63)) - 1);
        }
        if (i == valid_block_last && (valid_page_last & 63) != 63) {
          valid_bits &= (uint64_t(1) << ((valid_page_last & 63) + 1)) - 1;
        }
        __atomic_fetch_or(&system_page_flags_valid_[i], valid_bits,
                          __ATOMIC_RELEASE);
        if (!page_upload_hash_valid_.empty()) {
          __atomic_fetch_and(&page_upload_hash_valid_[i], ~valid_bits,
                             __ATOMIC_RELAXED);
        }
        if (written_by_gpu) {
          if (!unwatched_pages_.empty()) {
            uint64_t unwatched_bits = unwatched_pages_[i] & valid_bits;
            if (unwatched_bits) {
              unwatched_pages_[i] &= ~unwatched_bits;
              unwatched_page_count_ -= uint32_t(xe::bit_count(unwatched_bits));
            }
          }
          if (!system_page_flags_used_in_submission_.empty()) {
            system_page_flags_used_in_submission_[i] |= valid_bits;
          }
          // In the same critical region as the GPU write, so a guest write
          // faulting in can't see the pages of an export read back as
          // preserved. A page that a GPU write not read back (a resolve) left
          // only in the GPU's copy stays preserved under the export.
          if (read_back) {
            system_page_flags_preserve_exempt_[i] |=
                valid_bits & ~(system_page_flags_valid_and_gpu_written_[i] &
                               ~system_page_flags_preserve_exempt_[i]);
          } else {
            system_page_flags_preserve_exempt_[i] &= ~valid_bits;
          }
          system_page_flags_valid_and_gpu_written_[i] |= valid_bits;
          // Mark the touched chunk dirty so the next copy picks it up.
          gpu_written_data_dirty_ = true;
          dirty_blocks_ |= 1u << (i >> 6);
        } else {
          system_page_flags_valid_and_gpu_written_[i] &= ~valid_bits;
        }
      }
    }
    if (written_by_gpu) {
      NotePages<&PageHistory::gpu_write>(start, length);
      if (!page_history_.empty()) {
        for (uint32_t page = valid_page_first; page <= valid_page_last;
             ++page) {
          page_history_[page].gpu_write_start = start;
          page_history_[page].gpu_write_length = length;
        }
      }
    } else {
      NotePages<&PageHistory::upload>(start, length);
    }
  }

  if (written_by_gpu) {
    NoteByteRangeUsedByGpu(start, length);
  }

  if (memory_invalidation_callback_handle_ && !written_by_gpu) {
    // A page that isn't writable here gets no watch. A later guest
    // protect-to-writable invalidates it unconditionally, so its writes are
    // still caught.
    if (cvars::shared_memory_unwatch_dynamic_pages &&
        cvars::shm_coalesce_writewatch_pages && !dirty_score_.empty()) {
      EnableWatchesExceptDynamic(valid_page_first, valid_page_last);
    } else {
      memory().EnablePhysicalMemoryAccessCallbacks(
          valid_page_first << page_size_log2_,
          (valid_page_last - valid_page_first + 1) << page_size_log2_, true,
          false);
    }
  }
}

void SharedMemory::SetPageUploadHashes(uint32_t page_first,
                                       uint32_t page_count,
                                       const uint8_t* source) {
  if (page_upload_hash_.empty() || !page_count || !unwatched_page_count_) {
    return;
  }
  const uint32_t last = page_first + page_count - 1;
  const size_t page_size = size_t(1) << page_size_log2_;
  for (uint32_t b = page_first >> 6; b <= last >> 6; ++b) {
    uint64_t mask = UINT64_MAX;
    if (b == page_first >> 6) {
      mask &= ~((uint64_t(1) << (page_first & 63)) - 1);
    }
    if (b == last >> 6 && (last & 63) != 63) {
      mask &= (uint64_t(1) << ((last & 63) + 1)) - 1;
    }
    uint64_t bits = mask & unwatched_pages_[b];
    while (bits) {
      uint32_t bit = 0;
      xe::bit_scan_forward(bits, &bit);
      bits &= bits - 1;
      const uint32_t page = (b << 6) + bit;
      page_upload_hash_[page] = XXH3_64bits(
          source + size_t(page - page_first) * page_size, page_size);
      __atomic_fetch_or(&page_upload_hash_valid_[b], uint64_t(1) << bit,
                        __ATOMIC_RELAXED);
    }
  }
}

void SharedMemory::SetSubpageUploadHashes(uint32_t byte_offset,
                                           uint32_t byte_size,
                                           const uint8_t* source) {
  if (!subpage_tracker_.active() || !byte_size || !source ||
      byte_offset >= kBufferSize) {
    return;
  }
  byte_size = std::min(byte_size, kBufferSize - byte_offset);
  if ((byte_offset & (SubpageTracker::kSubpageSize - 1)) != 0 ||
      (byte_size & (SubpageTracker::kSubpageSize - 1)) != 0) {
    return;
  }
  const uint32_t sp_first = byte_offset >> SubpageTracker::kSubpageSizeLog2;
  const uint32_t sp_count = byte_size >> SubpageTracker::kSubpageSizeLog2;
  const uint32_t shift = subpage_tracker_.subpages_per_host_page_log2();
  const uint32_t local_mask = subpage_tracker_.subpages_per_host_page() - 1;
  for (uint32_t i = 0; i < sp_count; ++i) {
    const uint32_t sp = sp_first + i;
    const uint64_t hash = XXH3_64bits(
        source + size_t(i) * SubpageTracker::kSubpageSize,
        SubpageTracker::kSubpageSize);
    auto global_lock = global_critical_region_.Acquire();
    const uint32_t hp = sp >> shift;
    const uint8_t bit = uint8_t(uint32_t(1) << (sp & local_mask));
    // Only publish hash if the subpage was not invalidated by a racing CPU
    // store between MakeRangeValid and hashing.
    if (subpage_tracker_.valid_mask(hp) & bit) {
      subpage_tracker_.SetSubpageUploadHash(sp, hash);
    }
  }
}

void SharedMemory::EnableWatchesExceptDynamic(uint32_t page_first,
                                              uint32_t page_last) {
  auto global_lock = global_critical_region_.Acquire();
  uint32_t run_start = UINT32_MAX;
  for (uint32_t b = page_first >> 6; b <= page_last >> 6; ++b) {
    uint64_t mask = UINT64_MAX;
    if (b == page_first >> 6) {
      mask &= ~((uint64_t(1) << (page_first & 63)) - 1);
    }
    if (b == page_last >> 6 && (page_last & 63) != 63) {
      mask &= (uint64_t(1) << ((page_last & 63) + 1)) - 1;
    }
    uint64_t dynamic = unwatch_candidates_[b] & mask &
                       ~texture_watched_pages_[b] &
                       ~system_page_flags_valid_and_gpu_written_[b];
    if (subpage_tracker_.active() && dynamic) {
      // Only dynamically unwatch host pages whose 4 KB subpages are all valid.
      uint64_t check = dynamic;
      while (check) {
        uint32_t bit = 0;
        xe::bit_scan_forward(check, &bit);
        check &= check - 1;
        if (!subpage_tracker_.IsHostPageFullyValid((b << 6) + bit)) {
          dynamic &= ~(uint64_t(1) << bit);
        }
      }
    }
    const uint64_t added = dynamic & ~unwatched_pages_[b];
    unwatched_page_count_ += uint32_t(xe::bit_count(added));
    unwatched_pages_[b] |= dynamic;
    const uint64_t to_watch = mask & ~unwatched_pages_[b];

    const uint32_t bit_first = (b == page_first >> 6) ? (page_first & 63) : 0;
    const uint32_t bit_last = (b == page_last >> 6) ? (page_last & 63) : 63;
    for (uint32_t bit = bit_first; bit <= bit_last; ++bit) {
      const uint32_t page = (b << 6) + bit;
      if (to_watch & (uint64_t(1) << bit)) {
        if (run_start == UINT32_MAX) {
          run_start = page;
        }
      } else if (run_start != UINT32_MAX) {
        memory().EnablePhysicalMemoryAccessCallbacks(
            run_start << page_size_log2_,
            (page - run_start) << page_size_log2_, true, false);
        run_start = UINT32_MAX;
      }
    }
  }
  if (run_start != UINT32_MAX) {
    memory().EnablePhysicalMemoryAccessCallbacks(
        run_start << page_size_log2_,
        (page_last + 1 - run_start) << page_size_log2_, true, false);
  }
}

bool SharedMemory::UnwatchedPageMatchesUpload(uint32_t page) {
  const uint32_t address = page << page_size_log2_;
  const uint32_t page_bytes = uint32_t(1) << page_size_log2_;
  auto* phys_heap = memory().GetPhysicalHeap();
  for (uint32_t off = 0; off < page_bytes; off += 4096) {
    uint32_t protect = 0;
    if (!phys_heap->QueryProtectUnlocked(address + off, &protect) ||
        !(protect & kMemoryProtectRead)) {
      return false;
    }
  }
  if (!((__atomic_load_n(&page_upload_hash_valid_[page >> 6],
                         __ATOMIC_RELAXED) >>
         (page & 63)) &
        1)) {
    return false;
  }
  return XXH3_64bits(memory().TranslatePhysical<const uint8_t*>(address),
                     size_t(page_bytes)) == page_upload_hash_[page];
}

void SharedMemory::VerifyUnwatchedPages(uint32_t page_first,
                                        uint32_t page_last) {
  for (uint32_t b = page_first >> 6; b <= page_last >> 6; ++b) {
    uint64_t mask = UINT64_MAX;
    if (b == page_first >> 6) {
      mask &= ~((uint64_t(1) << (page_first & 63)) - 1);
    }
    if (b == page_last >> 6 && (page_last & 63) != 63) {
      mask &= (uint64_t(1) << ((page_last & 63) + 1)) - 1;
    }
    uint64_t bits =
        unwatched_pages_[b] & mask &
        __atomic_load_n(&system_page_flags_valid_[b], __ATOMIC_ACQUIRE);
    while (bits) {
      uint32_t bit = 0;
      xe::bit_scan_forward(bits, &bit);
      bits &= bits - 1;
      const uint32_t page = (b << 6) + bit;
      if (!UnwatchedPageMatchesUpload(page)) {
        MemoryInvalidationCallbackImpl(page << page_size_log2_,
                                       uint32_t(1) << page_size_log2_, true,
                                       true);
      }
    }
  }
}

void SharedMemory::VerifyUnwatchedSubpages(uint32_t start, uint32_t length) {
  if (!subpage_tracker_.active() || !length || start >= kBufferSize) {
    return;
  }
  length = std::min(length, kBufferSize - start);
  const uint32_t sp_first = start >> SubpageTracker::kSubpageSizeLog2;
  const uint32_t sp_last =
      (start + length - 1) >> SubpageTracker::kSubpageSizeLog2;
  const uint32_t page_first = start >> page_size_log2_;
  const uint32_t page_last = (start + length - 1) >> page_size_log2_;
  auto* phys_heap = memory().GetPhysicalHeap();
  for (uint32_t b = page_first >> 6; b <= page_last >> 6; ++b) {
    uint64_t mask = UINT64_MAX;
    if (b == (page_first >> 6)) {
      mask &= ~((uint64_t(1) << (page_first & 63)) - 1);
    }
    if (b == (page_last >> 6) && (page_last & 63) != 63) {
      mask &= (uint64_t(1) << ((page_last & 63) + 1)) - 1;
    }
    uint64_t bits = unwatched_pages_[b] & mask;
    while (bits) {
      uint32_t bit = 0;
      xe::bit_scan_forward(bits, &bit);
      bits &= bits - 1;
      const uint32_t hp = (b << 6) + bit;
      const uint8_t req_mask =
          subpage_tracker_.SubpageMaskForHostPage(hp, sp_first, sp_last);
      uint8_t valid_req = 0;
      uint8_t hash_valid = 0;
      {
        auto global_lock = global_critical_region_.Acquire();
        valid_req = uint8_t(subpage_tracker_.valid_mask(hp) & req_mask);
        hash_valid = subpage_tracker_.hash_valid_mask(hp);
      }
      if (!valid_req) {
        continue;
      }
      const uint32_t page_sp_base =
          hp << subpage_tracker_.subpages_per_host_page_log2();
      bool any_mismatch = false;
      for (uint32_t local = 0;
           local < subpage_tracker_.subpages_per_host_page(); ++local) {
        const uint8_t sbit = uint8_t(uint32_t(1) << local);
        if (!(valid_req & sbit)) {
          continue;
        }
        const uint32_t sp = page_sp_base + local;
        const uint32_t sp_addr = sp << SubpageTracker::kSubpageSizeLog2;
        uint32_t protect = 0;
        bool match = false;
        if ((hash_valid & sbit) &&
            phys_heap->QueryProtectUnlocked(sp_addr, &protect) &&
            (protect & kMemoryProtectRead)) {
          const uint64_t h = XXH3_64bits(
              memory().TranslatePhysical<const uint8_t*>(sp_addr),
              SubpageTracker::kSubpageSize);
          match = (h == subpage_tracker_.upload_hash(sp));
        }
        if (!match) {
          any_mismatch = true;
          auto global_lock = global_critical_region_.Acquire();
          const bool used_in_sub =
              !system_page_flags_used_in_submission_.empty() &&
              ((system_page_flags_used_in_submission_[b] >> bit) & 1u) != 0;
          subpage_tracker_.InvalidateSubpage(sp, used_in_sub);
          __atomic_fetch_and(&system_page_flags_valid_[b],
                             ~(uint64_t(1) << bit), __ATOMIC_RELEASE);
          if (used_in_sub) {
            system_page_flags_invalidated_in_submission_[b] |=
                (uint64_t(1) << bit);
          }
          if (!dirty_this_frame_.empty()) {
            dirty_this_frame_[b] |= (uint64_t(1) << bit);
          }
          if (!page_upload_hash_valid_.empty()) {
            __atomic_fetch_and(&page_upload_hash_valid_[b],
                               ~(uint64_t(1) << bit), __ATOMIC_RELAXED);
          }
        }
      }
      if (any_mismatch) {
        FireWatches(hp, hp, false);
      }
    }
  }
}

void SharedMemory::RewatchPages(uint32_t block, uint64_t mask) {
  auto global_lock = global_critical_region_.Acquire();
  mask &= unwatched_pages_[block];
  if (!mask) {
    return;
  }
  unwatched_pages_[block] &= ~mask;
  unwatched_page_count_ -= uint32_t(xe::bit_count(mask));
  // Arm protection before verifying so any racing CPU store faults cleanly.
  // When subpage tracking is active, a partially valid host page has its
  // host-page valid bit cleared while 1..3 subpages remain valid; include
  // every host page with any valid subpage so writewatch protection is always
  // re-armed and those remaining subpages are verified against their 4 KB
  // upload hashes.
  uint64_t valid =
      subpage_tracker_.active()
          ? subpage_tracker_.FilterBlockMaskWithAnyValidSubpage(block, mask)
          : (mask & system_page_flags_valid_[block]);
  uint64_t runs = valid;
  while (runs) {
    uint32_t first = 0;
    xe::bit_scan_forward(runs, &first);
    const uint64_t shifted = runs >> first;
    const uint32_t count =
        shifted == UINT64_MAX ? 64 - first : uint32_t(xe::tzcnt(~shifted));
    memory().EnablePhysicalMemoryAccessCallbacks(
        ((block << 6) + first) << page_size_log2_, count << page_size_log2_,
        true, false);
    runs &= ~(count >= 64 ? UINT64_MAX
                          : (((uint64_t(1) << count) - 1) << first));
  }
  if (subpage_tracker_.active()) {
    auto* phys_heap = memory().GetPhysicalHeap();
    while (valid) {
      uint32_t bit = 0;
      xe::bit_scan_forward(valid, &bit);
      valid &= valid - 1;
      const uint32_t page = (block << 6) + bit;
      const bool used_in_sub =
          !system_page_flags_used_in_submission_.empty() &&
          ((system_page_flags_used_in_submission_[block] >> bit) & 1u) != 0;
      const uint8_t mismatched =
          subpage_tracker_.VerifyAndInvalidateHostPageSubpages(
              page, used_in_sub,
              [&](uint32_t sp, uint64_t& out_hash) -> bool {
                const uint32_t sp_addr = sp << SubpageTracker::kSubpageSizeLog2;
                uint32_t protect = 0;
                if (!phys_heap->QueryProtectUnlocked(sp_addr, &protect) ||
                    !(protect & kMemoryProtectRead)) {
                  return false;
                }
                out_hash = XXH3_64bits(
                    memory().TranslatePhysical<const uint8_t*>(sp_addr),
                    SubpageTracker::kSubpageSize);
                return true;
              });
      if (mismatched != 0) {
        const uint64_t page_bit = uint64_t(1) << bit;
        __atomic_fetch_and(&system_page_flags_valid_[block], ~page_bit,
                           __ATOMIC_RELEASE);
        if (used_in_sub) {
          system_page_flags_invalidated_in_submission_[block] |= page_bit;
        }
        if (!dirty_this_frame_.empty()) {
          dirty_this_frame_[block] |= page_bit;
        }
        if (!page_upload_hash_valid_.empty()) {
          __atomic_fetch_and(&page_upload_hash_valid_[block], ~page_bit,
                             __ATOMIC_RELAXED);
        }
        FireWatches(page, page, false);
      }
    }
    return;
  }
  while (valid) {
    uint32_t bit = 0;
    xe::bit_scan_forward(valid, &bit);
    valid &= valid - 1;
    const uint32_t page = (block << 6) + bit;
    if (!UnwatchedPageMatchesUpload(page)) {
      MemoryInvalidationCallbackImpl(page << page_size_log2_,
                                     uint32_t(1) << page_size_log2_, true,
                                     true);
    }
  }
}

void SharedMemory::OnFrameEndUnwatch() {
  if (dirty_score_.empty()) {
    return;
  }
  auto global_lock = global_critical_region_.Acquire();
  const bool unwatch_enabled =
      cvars::shared_memory_unwatch_dynamic_pages &&
      cvars::shm_coalesce_writewatch_pages;
  const int32_t decay_frames = std::max(cvars::shm_unwatch_decay_frames, 1);
  const uint8_t unwatch_score_max =
      uint8_t(std::clamp(cvars::shm_unwatch_score_max, 1, 255));
  const uint8_t unwatch_score = uint8_t(
      std::clamp(cvars::shm_unwatch_score, 1, int32_t(unwatch_score_max)));
  const bool decay_frame = ++unwatch_decay_counter_ >= uint32_t(decay_frames);
  if (decay_frame) {
    unwatch_decay_counter_ = 0;
  }
  for (uint32_t b = 0; b < num_system_page_flags_; ++b) {
    const uint64_t dirty = dirty_this_frame_[b];
    uint64_t live = dirty | dirty_score_nonzero_[b];
    uint64_t rewatch = unwatch_enabled ? 0 : unwatched_pages_[b];
    if (!live && !rewatch) {
      continue;
    }
    dirty_this_frame_[b] = 0;
    while (live) {
      uint32_t i = 0;
      xe::bit_scan_forward(live, &i);
      live &= live - 1;
      const uint64_t bit = uint64_t(1) << i;
      uint8_t& score = dirty_score_[(b << 6) + i];
      if (dirty & bit) {
        score = std::min(uint8_t(score + 1), unwatch_score_max);
      } else if (score && decay_frame) {
        --score;
      }
      if (unwatch_enabled && score >= unwatch_score) {
        unwatch_candidates_[b] |= bit;
      } else {
        unwatch_candidates_[b] &= ~bit;
      }
      if (score) {
        dirty_score_nonzero_[b] |= bit;
      } else {
        dirty_score_nonzero_[b] &= ~bit;
        rewatch |= unwatched_pages_[b] & bit;
      }
    }
    if (rewatch) {
      RewatchPages(b, rewatch);
    }
  }
}

void SharedMemory::UnlinkWatchRange(WatchRange* range) {
  uint32_t bucket =
      range->page_first << page_size_log2_ >> kWatchBucketSizeLog2;
  WatchNode* node = range->node_first;
  while (node != nullptr) {
    WatchNode* node_next = node->range_node_next;
    if (node->bucket_node_previous != nullptr) {
      node->bucket_node_previous->bucket_node_next = node->bucket_node_next;
    } else {
      watch_buckets_[bucket] = node->bucket_node_next;
    }
    if (node->bucket_node_next != nullptr) {
      node->bucket_node_next->bucket_node_previous = node->bucket_node_previous;
    }
    node->next_free = watch_node_first_free_;
    watch_node_first_free_ = node;
    node = node_next;
    ++bucket;
  }
  range->next_free = watch_range_first_free_;
  watch_range_first_free_ = range;
}

// todo: optimize, an enormous amount of cpu time (1.34%) is spent here.
bool SharedMemory::RequestRange(uint32_t start, uint32_t length) {
  if (!length) {
    // Some texture or buffer is empty, for example - safe to draw in this case.
    return true;
  }
  if (start > kBufferSize || (kBufferSize - start) < length) {
    return false;
  }

  SCOPE_profile_cpu_f("gpu");

  if (!EnsureHostGpuMemoryAllocated(start, length)) {
    return false;
  }

  unsigned int current_upload_range = 0;
  uint32_t page_first = start >> page_size_log2_;
  uint32_t page_last = (start + length - 1) >> page_size_log2_;

  uint32_t block_first = page_first >> 6;
  uint32_t block_last = page_last >> 6;

  if (unwatched_page_count_ && !unwatch_skip_verify_) {
    if (subpage_tracker_.active()) {
      VerifyUnwatchedSubpages(start, length);
    } else {
      VerifyUnwatchedPages(page_first, page_last);
    }
  }

  // Lockless fast-path: check if all pages are already valid.
  // This avoids lock acquisition for the common case where pages are resident.
  uint64_t* valid_flags = system_page_flags_valid_;
  if (valid_flags) {
    bool all_valid = true;
    for (uint32_t i = block_first; i <= block_last && all_valid; ++i) {
      uint64_t block_valid = __atomic_load_n(&valid_flags[i], __ATOMIC_ACQUIRE);
      if (i == block_first) {
        // Mask out pages before page_first
        uint64_t block_before = (uint64_t(1) << (page_first & 63)) - 1;
        block_valid |= block_before;
      }
      if (i == block_last && (page_last & 63) != 63) {
        // Mask out pages after page_last
        uint64_t block_after = ~((uint64_t(1) << ((page_last & 63) + 1)) - 1);
        block_valid |= block_after;
      }
      if (block_valid != ~uint64_t(0)) {
        all_valid = false;
      }
    }
    if (all_valid) {
      // All pages already valid, nothing to upload
      MarkRangeUsedInSubmission(start, length);
      return true;
    }
  }

  if (subpage_tracker_.active()) {
    {
      auto global_lock = global_critical_region_.Acquire();
      subpage_tracker_.CollectDirtySubpageRanges(start, length,
                                                 dirty_subpage_ranges_);
      if (dirty_subpage_ranges_.empty()) {
        MarkRangeUsedInSubmission(start, length);
        return true;
      }
      // Re-arm writewatch protection on covering non-unwatched host pages
      // BEFORE reading guest RAM for 4 KB hash filtering so any racing CPU
      // store faults cleanly and waits on global_critical_region_.
      if (memory_invalidation_callback_handle_) {
        const uint32_t shift = subpage_tracker_.subpages_per_host_page_log2();
        uint32_t protect_run_start = UINT32_MAX;
        uint32_t protect_run_end = 0;
        auto flush_protect = [&]() {
          if (protect_run_start != UINT32_MAX) {
            memory().EnablePhysicalMemoryAccessCallbacks(
                protect_run_start << page_size_log2_,
                (protect_run_end - protect_run_start) << page_size_log2_, true,
                false);
            protect_run_start = UINT32_MAX;
            protect_run_end = 0;
          }
        };
        for (const auto& dr : dirty_subpage_ranges_) {
          const uint32_t hp_first = dr.first >> shift;
          const uint32_t hp_last = (dr.first + dr.second - 1) >> shift;
          for (uint32_t hp = hp_first; hp <= hp_last; ++hp) {
            const bool is_unwatched =
                !unwatched_pages_.empty() &&
                ((unwatched_pages_[hp >> 6] >> (hp & 63)) & 1u) != 0;
            if (is_unwatched) {
              flush_protect();
              continue;
            }
            if (protect_run_start == UINT32_MAX) {
              protect_run_start = hp;
              protect_run_end = hp + 1;
            } else if (hp == protect_run_end) {
              ++protect_run_end;
            } else if (hp > protect_run_end) {
              flush_protect();
              protect_run_start = hp;
              protect_run_end = hp + 1;
            }
          }
        }
        flush_protect();
      }
      auto* phys_heap = memory().GetPhysicalHeap();
      subpage_tracker_.FilterUnchangedDirtySubpages(
          dirty_subpage_ranges_,
          [&](uint32_t sp, uint64_t& out_hash) -> bool {
            const uint32_t sp_addr = sp << SubpageTracker::kSubpageSizeLog2;
            uint32_t protect = 0;
            if (!phys_heap->QueryProtectUnlocked(sp_addr, &protect) ||
                !(protect & kMemoryProtectRead)) {
              return false;
            }
            out_hash = XXH3_64bits(
                memory().TranslatePhysical<const uint8_t*>(sp_addr),
                SubpageTracker::kSubpageSize);
            return true;
          },
          filtered_subpage_ranges_, newly_full_host_pages_);
      for (uint32_t hp : newly_full_host_pages_) {
        __atomic_fetch_or(&system_page_flags_valid_[hp >> 6],
                          uint64_t(1) << (hp & 63), __ATOMIC_RELEASE);
      }
    }
    if (filtered_subpage_ranges_.empty()) {
      MarkRangeUsedInSubmission(start, length);
      return true;
    }
    const bool uploaded = UploadSubpageRanges(
        filtered_subpage_ranges_.data(),
        uint32_t(filtered_subpage_ranges_.size()));
    MarkRangeUsedInSubmission(start, length);
    return uploaded;
  }

  upload_ranges_.clear();

  std::pair<uint32_t, uint32_t>* uploads =
      reinterpret_cast<std::pair<uint32_t, uint32_t>*>(upload_ranges_.data());

  // swcache::PrefetchL1(&system_page_flags_[block_first]);
  uint32_t range_start = UINT32_MAX;

  {
    auto global_lock = global_critical_region_.Acquire();
    TryFindUploadRange(block_first, block_last, page_first, page_last,
                       range_start, current_upload_range, uploads);
  }
  if (range_start != UINT32_MAX) {
    uploads[current_upload_range++] =
        (std::make_pair(range_start, page_last + 1 - range_start));
  }
  if (!current_upload_range) {
    MarkRangeUsedInSubmission(start, length);
    return true;
  }

  const bool uploaded = UploadRanges(uploads, current_upload_range);
  MarkRangeUsedInSubmission(start, length);
  return uploaded;
}

bool SharedMemory::IsRangeValid(uint32_t start, uint32_t length) const {
  if (!length) {
    return true;
  }
  if (start > kBufferSize || (kBufferSize - start) < length) {
    return false;
  }
  if (subpage_tracker_.active()) {
    return subpage_tracker_.IsRangeValid(start, length);
  }

  const uint32_t page_first = start >> page_size_log2_;
  const uint32_t page_last = (start + length - 1) >> page_size_log2_;
  const uint32_t block_first = page_first >> 6;
  const uint32_t block_last = page_last >> 6;

  for (uint32_t i = block_first; i <= block_last; ++i) {
    uint64_t valid_mask = UINT64_MAX;
    if (i == block_first) {
      valid_mask &= ~((uint64_t(1) << (page_first & 63)) - 1);
    }
    if (i == block_last && (page_last & 63) != 63) {
      valid_mask &= (uint64_t(1) << ((page_last & 63) + 1)) - 1;
    }
    if ((system_page_flags_valid_[i] & valid_mask) != valid_mask) {
      return false;
    }
  }
  return true;
}

bool SharedMemory::IsRangeGpuWritten(uint32_t start, uint32_t length) const {
  if (!length) {
    return false;
  }
  if (start > kBufferSize || (kBufferSize - start) < length) {
    return false;
  }
  if (subpage_tracker_.active()) {
    return subpage_tracker_.IsRangeGpuWritten(start, length);
  }

  const uint32_t page_first = start >> page_size_log2_;
  const uint32_t page_last = (start + length - 1) >> page_size_log2_;
  const uint32_t block_first = page_first >> 6;
  const uint32_t block_last = page_last >> 6;

  for (uint32_t i = block_first; i <= block_last; ++i) {
    uint64_t valid_mask = UINT64_MAX;
    if (i == block_first) {
      valid_mask &= ~((uint64_t(1) << (page_first & 63)) - 1);
    }
    if (i == block_last && (page_last & 63) != 63) {
      valid_mask &= (uint64_t(1) << ((page_last & 63) + 1)) - 1;
    }
    if ((system_page_flags_valid_and_gpu_written_[i] & valid_mask) != valid_mask) {
      return false;
    }
  }
  return true;
}

template <typename T>
XE_FORCEINLINE XE_NOALIAS static T mod_shift_left(T value, uint32_t by) {
#if XE_ARCH_AMD64 == 1
  // arch has modular shifts
  return value << by;
#else
  return value << (by % (sizeof(T) * CHAR_BIT));
#endif
}
void SharedMemory::TryFindUploadRange(const uint32_t& block_first,
                                      const uint32_t& block_last,
                                      const uint32_t& page_first,
                                      const uint32_t& page_last,
                                      uint32_t& range_start,
                                      unsigned int& current_upload_range,
                                      std::pair<uint32_t, uint32_t>* uploads) {
  for (uint32_t i = block_first; i <= block_last; ++i) {
    uint64_t block_valid = system_page_flags_valid_[i];
    if (i == block_first) {
      uint64_t block_before = mod_shift_left(uint64_t(1), page_first) - 1;
      block_valid |= block_before;
    }
    if (i == block_last && (page_last & 63) != 63) {
      uint64_t block_inside = mod_shift_left(uint64_t(1), page_last + 1) - 1;
      block_valid |= ~block_inside;
    }
    TryGetNextUploadRange(range_start, block_valid, i, current_upload_range,
                          uploads);
  }
}

static bool UploadRange_DoBestScanForward(uint64_t v, uint32_t* out) {
#if XE_ARCH_AMD64 == 1 && XE_PLATFORM_WIN32
  if (!v) {
    return false;
  }
  if (amd64::GetFeatureFlags() & amd64::kX64EmitBMI1) {
    *out = static_cast<uint32_t>(_tzcnt_u64(v));
  } else {
    unsigned char bsfres = _BitScanForward64((unsigned long*)out, v);

    XE_MSVC_ASSUME(bsfres == 1);
  }
  return true;
#else
  return xe::bit_scan_forward(v, out);
#endif
}

void SharedMemory::TryGetNextUploadRange(
    uint32_t& range_start, uint64_t& block_valid, const uint32_t& i,
    unsigned int& current_upload_range,
    std::pair<uint32_t, uint32_t>* uploads) {
  while (true) {
    uint32_t block_page = 0;
    if (range_start == UINT32_MAX) {
      // Check if need to open a new range.
      if (!UploadRange_DoBestScanForward(~block_valid, &block_page)) {
        break;
      }
      range_start = (i << 6) + block_page;
    } else {
      // Check if need to close the range.
      // Ignore the valid pages before the beginning of the range.
      uint64_t block_valid_from_start = block_valid;
      if (i == (range_start >> 6)) {
        block_valid_from_start &=
            ~(mod_shift_left(uint64_t(1), range_start) - 1);
      }
      if (!UploadRange_DoBestScanForward(block_valid_from_start, &block_page)) {
        break;
      }
      if (current_upload_range + 1 < MAX_UPLOAD_RANGES) {
        uploads[current_upload_range++] =
            std::make_pair(range_start, (i << 6) + block_page - range_start);
        // In the next iteration within this block, consider this range valid
        // since it has been queued for upload.
        block_valid |= (uint64_t(1) << block_page) - 1;
        range_start = UINT32_MAX;
      } else {
        xe::FatalError(
            "Hit max upload ranges in shared_memory.cc, tell a dev to "
            "raise the limit!");
      }
    }
  }
}

void SharedMemory::TakeInvalidationStats(uint64_t& pages,
                                         uint64_t& repeated_pages) {
  auto global_lock = global_critical_region_.Acquire();
  pages = invalidated_pages_;
  repeated_pages = repeated_invalidated_pages_;
  invalidated_pages_ = 0;
  repeated_invalidated_pages_ = 0;
}

std::pair<uint32_t, uint32_t> SharedMemory::MemoryInvalidationCallbackThunk(
    void* context_ptr, uint32_t physical_address_start, uint32_t length,
    bool exact_range) {
  return reinterpret_cast<SharedMemory*>(context_ptr)
      ->MemoryInvalidationCallback(physical_address_start, length, exact_range);
}

std::pair<uint32_t, uint32_t> SharedMemory::MemoryInvalidationCallback(
    uint32_t physical_address_start, uint32_t length, bool exact_range) {
  return MemoryInvalidationCallbackImpl(physical_address_start, length,
                                        exact_range, false);
}

std::pair<uint32_t, uint32_t> SharedMemory::MemoryInvalidationCallbackImpl(
    uint32_t physical_address_start, uint32_t length, bool exact_range,
    bool from_unwatch_verify) {
  if (length == 0 || physical_address_start >= kBufferSize) {
    return std::make_pair(uint32_t(0), UINT32_MAX);
  }
  length = std::min(length, kBufferSize - physical_address_start);
  uint32_t physical_address_last = physical_address_start + (length - 1);

  uint32_t page_first = physical_address_start >> page_size_log2_;
  uint32_t page_last = physical_address_last >> page_size_log2_;
  uint32_t block_first = page_first >> 6;
  uint32_t block_last = page_last >> 6;

  auto global_lock = global_critical_region_.Acquire();

  if (!exact_range) {
    // Coalesce fault invalidation to an aligned 64 KB block (16 pages on 4 KB
    // host kernels, 4 pages on 16 KB host kernels) when
    // shm_coalesce_writewatch_pages is enabled, or 64 pages (256 KB) otherwise,
    // stopping before any GPU-written page so GPU-authoritative data is kept.
    const uint32_t subblock_pages =
        cvars::shm_coalesce_writewatch_pages
            ? std::clamp<uint32_t>(uint32_t(64 * 1024) >> page_size_log2_, 1u,
                                   64u)
            : 64u;
    const uint32_t subblock_mask = subblock_pages - 1;
    const uint32_t offset_in_subblock = page_first & subblock_mask;
    if (offset_in_subblock != 0) {
      const uint32_t subblock_base_in_word =
          (page_first & 63) & ~subblock_mask;
      const uint64_t gpu_written_before =
          (system_page_flags_valid_and_gpu_written_[block_first] >>
           subblock_base_in_word) &
          ((uint64_t(1) << offset_in_subblock) - 1);
      page_first = (page_first & ~uint32_t(63)) + subblock_base_in_word +
                   (64 - xe::lzcnt(gpu_written_before));
    }
    const uint32_t offset_last = page_last & subblock_mask;
    if (offset_last != subblock_mask) {
      const uint32_t rem = subblock_mask - offset_last;
      uint64_t gpu_written_after =
          system_page_flags_valid_and_gpu_written_[block_last] >>
          ((page_last & 63) + 1);
      gpu_written_after |= uint64_t(1) << rem;
      page_last += uint32_t(xe::tzcnt(gpu_written_after));
    }
  }

  if (cvars::shared_memory_preserve_gpu_writes) {
    // The GPU-written pages among the invalidated ones (the wider range above
    // stops at them, so only the written ones) go to guest memory. A guest
    // write faulting in (not exact) lands on them afterwards, so all of them.
    // An exact range was written by the host already (a file read straight
    // into physical memory notifies after reading) or isn't kept (released
    // memory) - only the parts of its pages outside it, around it.
    const uint32_t written_end = physical_address_last + 1;
    for (uint32_t i = block_first; i <= block_last; ++i) {
      // Not the pages of exports read back into guest memory.
      uint64_t gpu_written_bits = system_page_flags_valid_and_gpu_written_[i] &
                                  ~system_page_flags_preserve_exempt_[i];
      if (i == block_first) {
        gpu_written_bits &= ~((uint64_t(1) << (page_first & 63)) - 1);
      }
      if (i == block_last && (page_last & 63) != 63) {
        gpu_written_bits &= (uint64_t(1) << ((page_last & 63) + 1)) - 1;
      }
      uint32_t run_first;
      while (xe::bit_scan_forward(gpu_written_bits, &run_first)) {
        uint32_t run_length = xe::tzcnt(~(gpu_written_bits >> run_first));
        uint32_t run_page_first = (i << 6) + run_first;
        const uint32_t run_start = run_page_first << page_size_log2_;
        const uint32_t run_end = (run_page_first + run_length)
                                 << page_size_log2_;
        uint32_t copied = 0;
        if (subpage_tracker_.active()) {
          // Copy only subpages that were genuinely GPU-written so CPU writes
          // in neighboring 4 KB subpages are never overwritten.
          const uint32_t shift = subpage_tracker_.subpages_per_host_page_log2();
          const uint32_t sp_per_hp = subpage_tracker_.subpages_per_host_page();
          for (uint32_t hp = run_page_first; hp < run_page_first + run_length;
               ++hp) {
            const uint8_t gw_mask = subpage_tracker_.gpu_written_mask(hp);
            if (!gw_mask) {
              continue;
            }
            const uint32_t base_sp = hp << shift;
            for (uint32_t local = 0; local < sp_per_hp; ++local) {
              if (!(gw_mask & uint8_t(uint32_t(1) << local))) {
                continue;
              }
              const uint32_t sp_start =
                  (base_sp + local) << SubpageTracker::kSubpageSizeLog2;
              const uint32_t sp_end = sp_start + SubpageTracker::kSubpageSize;
              if (!exact_range) {
                if (CopyToGuestMemory(sp_start, sp_end - sp_start)) {
                  copied += sp_end - sp_start;
                  NoteGuestCopy(sp_start, sp_end - sp_start,
                                GuestCopySource::kPreserveGuestWrite);
                }
              } else {
                if (sp_start < physical_address_start) {
                  const uint32_t head_end =
                      std::min(sp_end, physical_address_start);
                  if (CopyToGuestMemory(sp_start, head_end - sp_start)) {
                    copied += head_end - sp_start;
                    NoteGuestCopy(sp_start, head_end - sp_start,
                                  GuestCopySource::kPreserveAroundHostWrite);
                  }
                }
                if (sp_end > written_end) {
                  const uint32_t tail_start = std::max(sp_start, written_end);
                  if (CopyToGuestMemory(tail_start, sp_end - tail_start)) {
                    copied += sp_end - tail_start;
                    NoteGuestCopy(tail_start, sp_end - tail_start,
                                  GuestCopySource::kPreserveAroundHostWrite);
                  }
                }
              }
            }
          }
        } else if (!exact_range) {
          if (CopyToGuestMemory(run_start, run_end - run_start)) {
            copied = run_end - run_start;
            NoteGuestCopy(run_start, run_end - run_start,
                          GuestCopySource::kPreserveGuestWrite);
          }
        } else {
          if (run_start < physical_address_start) {
            const uint32_t head_end = std::min(run_end, physical_address_start);
            if (CopyToGuestMemory(run_start, head_end - run_start)) {
              copied += head_end - run_start;
              NoteGuestCopy(run_start, head_end - run_start,
                            GuestCopySource::kPreserveAroundHostWrite);
            }
          }
          if (run_end > written_end) {
            const uint32_t tail_start = std::max(run_start, written_end);
            if (CopyToGuestMemory(tail_start, run_end - tail_start)) {
              copied += run_end - tail_start;
              NoteGuestCopy(tail_start, run_end - tail_start,
                            GuestCopySource::kPreserveAroundHostWrite);
            }
          }
        }
        if (copied) {
          static std::atomic<uint32_t> logged{0};
          if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
            XELOGI(
                "Shared memory: {} write to {} GPU-written page(s) at {:08X} - "
                "{} bytes of them copied to guest memory before the reupload",
                exact_range ? "a host" : "a guest", run_length, run_start,
                copied);
          }
        }
        gpu_written_bits &=
            ~((run_length >= 64 ? UINT64_MAX
                                : (uint64_t(1) << run_length) - 1)
              << run_first);
      }
    }
  }

  uint32_t dirty_blocks_mask = 0;
  for (uint32_t i = block_first; i <= block_last; ++i) {
    uint64_t invalidate_bits = UINT64_MAX;
    if (i == block_first) {
      invalidate_bits &= ~((uint64_t(1) << (page_first & 63)) - 1);
    }
    if (i == block_last && (page_last & 63) != 63) {
      invalidate_bits &= (uint64_t(1) << ((page_last & 63) + 1)) - 1;
    }
    __atomic_fetch_and(&system_page_flags_valid_[i], ~invalidate_bits,
                       __ATOMIC_RELEASE);
    system_page_flags_valid_and_gpu_written_[i] &= ~invalidate_bits;
    if (from_unwatch_verify && !system_page_flags_used_in_submission_.empty()) {
      system_page_flags_invalidated_in_submission_[i] |=
          invalidate_bits & system_page_flags_used_in_submission_[i];
    } else {
      system_page_flags_invalidated_in_submission_[i] |= invalidate_bits;
    }
    if (!dirty_this_frame_.empty()) {
      dirty_this_frame_[i] |= invalidate_bits;
    }
    if (!page_upload_hash_valid_.empty()) {
      __atomic_fetch_and(&page_upload_hash_valid_[i], ~invalidate_bits,
                         __ATOMIC_RELAXED);
    }
    if (cvars::shm_unwatch_release_exclusion && !texture_watched_pages_.empty()) {
      // FireWatches below unlinks every one-shot range watch on these pages, so
      // they no longer need dynamic-unwatch exclusion unless re-watched later.
      texture_watched_pages_[i] &= ~invalidate_bits;
    }
    // Track which chunks changed for partial copying.
    dirty_blocks_mask |= (1u << (i >> 6));
  }

  if (subpage_tracker_.active()) {
    if (exact_range) {
      if (from_unwatch_verify && !system_page_flags_used_in_submission_.empty()) {
        for (uint32_t hp = page_first; hp <= page_last; ++hp) {
          const bool used_in_sub =
              ((system_page_flags_used_in_submission_[hp >> 6] >> (hp & 63)) &
               1u) != 0;
          subpage_tracker_.InvalidateExactByteRangeOnUnprotectedHostPages(
              physical_address_start, length, hp, hp, used_in_sub);
        }
      } else {
        subpage_tracker_.InvalidateExactByteRangeOnUnprotectedHostPages(
            physical_address_start, length, page_first, page_last, true);
      }
    } else {
      subpage_tracker_.InvalidateHostPages(page_first, page_last, true);
    }
  }

  // GPU-written flags changed due to CPU invalidation.
  gpu_written_data_dirty_ = true;
  dirty_blocks_ |= dirty_blocks_mask;
  invalidated_pages_ += page_last - page_first + 1;
  if (!page_history_.empty()) {
    for (uint32_t page = page_first; page <= page_last; ++page) {
      const uint32_t last_invalidation = page_history_[page].invalidation;
      if (last_invalidation && last_invalidation + 1 >= history_frame_) {
        ++repeated_invalidated_pages_;
      }
    }
  }
  NotePages<&PageHistory::invalidation>(
      page_first << page_size_log2_,
      (page_last - page_first + 1) << page_size_log2_);

  FireWatches(page_first, page_last, false);

  return std::make_pair(page_first << page_size_log2_,
                        (page_last - page_first + 1) << page_size_log2_);
}

void SharedMemory::PrepareForTraceDownload() {
  ReleaseTraceDownloadRanges();
  assert_true(trace_download_ranges_.empty());
  assert_zero(trace_download_page_count_);

  // Invalidate the entire memory CPU->GPU memory copy so all the history
  // doesn't have to be written into every frame trace, and collect the list of
  // ranges with data modified on the GPU.

  uint32_t fire_watches_range_start = UINT32_MAX;
  uint32_t gpu_written_range_start = UINT32_MAX;
  auto global_lock = global_critical_region_.Acquire();
  for (uint32_t i = 0; i < num_system_page_flags_; ++i) {
    // SystemPageFlagsBlock& page_flags_block = system_page_flags_[i];
    uint64_t previously_valid_block = system_page_flags_valid_[i];
    uint64_t gpu_written_block = system_page_flags_valid_and_gpu_written_[i];
    uint64_t fire_watches_block = 0;
    if (subpage_tracker_.active()) {
      uint64_t new_valid_block = 0;
      const uint32_t base_hp = i << 6;
      const uint32_t hp_limit =
          std::min(base_hp + 64u, subpage_tracker_.host_page_count());
      for (uint32_t hp = base_hp; hp < hp_limit; ++hp) {
        const uint64_t bit = uint64_t(1) << (hp & 63);
        if (subpage_tracker_.SetHostPageValidFromGpuWritten(hp) != 0) {
          fire_watches_block |= bit;
        }
        if (subpage_tracker_.IsHostPageFullyValid(hp)) {
          new_valid_block |= bit;
        }
      }
      __atomic_store_n(&system_page_flags_valid_[i], new_valid_block,
                       __ATOMIC_RELEASE);
    } else {
      __atomic_store_n(&system_page_flags_valid_[i], gpu_written_block,
                       __ATOMIC_RELEASE);
      fire_watches_block = previously_valid_block & ~gpu_written_block;
    }

    // Fire watches on the invalidated pages.
    system_page_flags_invalidated_in_submission_[i] |= fire_watches_block;
    uint64_t fire_watches_break_block = ~fire_watches_block;
    while (true) {
      uint32_t fire_watches_block_page;
      if (!xe::bit_scan_forward(fire_watches_range_start == UINT32_MAX
                                    ? fire_watches_block
                                    : fire_watches_break_block,
                                &fire_watches_block_page)) {
        break;
      }
      uint32_t fire_watches_page = (i << 6) + fire_watches_block_page;
      if (fire_watches_range_start == UINT32_MAX) {
        fire_watches_range_start = fire_watches_page;
      } else {
        FireWatches(fire_watches_range_start, fire_watches_page - 1, false);
        fire_watches_range_start = UINT32_MAX;
      }
      uint64_t fire_watches_block_mask =
          ~((uint64_t(1) << fire_watches_block_page) - 1);
      fire_watches_block &= fire_watches_block_mask;
      fire_watches_break_block &= fire_watches_block_mask;
    }

    // Add to the GPU-written ranges.
    uint64_t gpu_written_break_block = ~gpu_written_block;
    while (true) {
      uint32_t gpu_written_block_page;
      if (!xe::bit_scan_forward(gpu_written_range_start == UINT32_MAX
                                    ? gpu_written_block
                                    : gpu_written_break_block,
                                &gpu_written_block_page)) {
        break;
      }
      uint32_t gpu_written_page = (i << 6) + gpu_written_block_page;
      if (gpu_written_range_start == UINT32_MAX) {
        gpu_written_range_start = gpu_written_page;
      } else {
        uint32_t gpu_written_range_length =
            gpu_written_page - gpu_written_range_start;
        // Call EnsureHostGpuMemoryAllocated in case the page was marked as
        // GPU-written not as a result to an actual write to the shared memory
        // buffer, but, for instance, by resolving with resolution scaling (to a
        // separate buffer).
        if (EnsureHostGpuMemoryAllocated(
                gpu_written_range_start << page_size_log2_,
                gpu_written_range_length << page_size_log2_)) {
          trace_download_ranges_.push_back(
              std::make_pair(gpu_written_range_start << page_size_log2_,
                             gpu_written_range_length << page_size_log2_));
          trace_download_page_count_ += gpu_written_range_length;
        }
        gpu_written_range_start = UINT32_MAX;
      }
      uint64_t gpu_written_block_mask =
          ~((uint64_t(1) << gpu_written_block_page) - 1);
      gpu_written_block &= gpu_written_block_mask;
      gpu_written_break_block &= gpu_written_block_mask;
    }
  }
  uint32_t page_count = kBufferSize >> page_size_log2_;
  if (fire_watches_range_start != UINT32_MAX) {
    FireWatches(fire_watches_range_start, page_count - 1, false);
  }
  if (gpu_written_range_start != UINT32_MAX) {
    uint32_t gpu_written_range_length = page_count - gpu_written_range_start;
    if (EnsureHostGpuMemoryAllocated(
            gpu_written_range_start << page_size_log2_,
            gpu_written_range_length << page_size_log2_)) {
      trace_download_ranges_.push_back(
          std::make_pair(gpu_written_range_start << page_size_log2_,
                         gpu_written_range_length << page_size_log2_));
      trace_download_page_count_ += gpu_written_range_length;
    }
  }
}

void SharedMemory::ReleaseTraceDownloadRanges() {
  trace_download_ranges_.clear();
  trace_download_ranges_.shrink_to_fit();
  trace_download_page_count_ = 0;
}

bool SharedMemory::EnsureHostGpuMemoryAllocated(uint32_t start,
                                                uint32_t length) {
  if (host_gpu_memory_sparse_granularity_log2_ != UINT32_MAX && length) {
    if (start <= kBufferSize && (kBufferSize - start) >= length) {
      uint32_t page_first = start >> page_size_log2_;
      uint32_t page_last = (start + length - 1) >> page_size_log2_;
      uint32_t allocation_first = page_first << page_size_log2_ >>
                                  host_gpu_memory_sparse_granularity_log2_;
      uint32_t allocation_last = page_last << page_size_log2_ >>
                                 host_gpu_memory_sparse_granularity_log2_;
      while (true) {
        std::pair<size_t, size_t> allocation_range =
            xe::bit_range::NextUnsetRange(
                host_gpu_memory_sparse_allocated_.data(), allocation_first,
                allocation_last - allocation_first + 1);
        if (!allocation_range.second) {
          break;
        }
        if (!AllocateSparseHostGpuMemoryRange(
                uint32_t(allocation_range.first),
                uint32_t(allocation_range.second))) {
          return false;
        }
        xe::bit_range::SetRange(host_gpu_memory_sparse_allocated_.data(),
                                allocation_range.first,
                                allocation_range.second);
        ++host_gpu_memory_sparse_allocations_;
        COUNT_profile_set(
            "gpu/shared_memory/host_gpu_memory_sparse_allocations",
            host_gpu_memory_sparse_allocations_);
        host_gpu_memory_sparse_used_bytes_ +=
            uint32_t(allocation_range.second)
            << host_gpu_memory_sparse_granularity_log2_;
        COUNT_profile_set(
            "gpu/shared_memory/host_gpu_memory_sparse_used_mb",
            (host_gpu_memory_sparse_used_bytes_ + ((1 << 20) - 1)) >> 20);
        allocation_first =
            uint32_t(allocation_range.first + allocation_range.second);
      }
    } else {
      return false;
    }
  } else {
    return true;
  }
  return true;
}

}  // namespace gpu
}  // namespace xe
