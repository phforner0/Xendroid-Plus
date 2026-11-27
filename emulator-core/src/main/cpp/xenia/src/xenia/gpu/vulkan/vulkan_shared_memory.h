/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2020 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_VULKAN_VULKAN_SHARED_MEMORY_H_
#define XENIA_GPU_VULKAN_VULKAN_SHARED_MEMORY_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace xe {
namespace gpu {
namespace vulkan {

// Strictly parses Android debug.xendroid.* boolean properties ('0' or '1').
inline bool ParseBinaryDebugProperty(const char* value, bool& out_enabled) {
  if (!value || value[1] != '\0') {
    return false;
  }
  if (value[0] == '0') {
    out_enabled = false;
    return true;
  }
  if (value[0] == '1') {
    out_enabled = true;
    return true;
  }
  return false;
}

struct ByteRange {
  uint64_t offset = 0;
  uint64_t size = 0;
};

// Aligns written byte ranges outward to nonCoherentAtomSize (clamped to
// buffer_size) and coalesces overlapping or adjacent ranges so
// vkFlushMappedMemoryRanges flushes strictly the written bytes, never globally.
inline std::vector<ByteRange> CoalesceAlignedFlushRanges(
    const std::vector<ByteRange>& written_ranges, uint64_t atom_size,
    uint64_t buffer_size) {
  std::vector<ByteRange> aligned;
  if (written_ranges.empty() || buffer_size == 0) {
    return aligned;
  }
  const uint64_t atom = std::max<uint64_t>(atom_size, 1);
  aligned.reserve(written_ranges.size());
  for (const ByteRange& r : written_ranges) {
    if (r.size == 0 || r.offset >= buffer_size) {
      continue;
    }
    const uint64_t end =
        (r.size > buffer_size - r.offset) ? buffer_size : (r.offset + r.size);
    const uint64_t start_aligned = (r.offset / atom) * atom;
    uint64_t end_aligned = ((end + atom - 1) / atom) * atom;
    if (end_aligned < end || end_aligned > buffer_size) {
      end_aligned = buffer_size;
    }
    if (end_aligned > start_aligned) {
      aligned.push_back({start_aligned, end_aligned - start_aligned});
    }
  }
  if (aligned.empty()) {
    return aligned;
  }
  std::sort(aligned.begin(), aligned.end(),
            [](const ByteRange& a, const ByteRange& b) {
              return a.offset < b.offset;
            });
  std::vector<ByteRange> merged;
  merged.reserve(aligned.size());
  for (const ByteRange& r : aligned) {
    if (!merged.empty() &&
        r.offset <= merged.back().offset + merged.back().size) {
      const uint64_t merged_end = std::max(
          merged.back().offset + merged.back().size, r.offset + r.size);
      merged.back().size = merged_end - merged.back().offset;
    } else {
      merged.push_back(r);
    }
  }
  return merged;
}

// Tracks the highest GPU submission index that may read or write each 4 KB
// logical page of VulkanSharedMemory::buffer_. A direct CPU write into
// host_mapped_data_ is safe against WAR/WAW hazards ONLY when every 4 KB slot
// touched has last_gpu_submission <= completed_submission (which rejects both
// in-flight prior submissions AND earlier commands in the open submission).
class ZeroCopyPageSubmissionTracker {
 public:
  static constexpr uint32_t kSlotSizeLog2 = 12;
  static constexpr uint32_t kSlotSize = uint32_t(1) << kSlotSizeLog2;

  void Initialize(uint32_t buffer_size_log2) {
    buffer_size_ = uint32_t(1) << buffer_size_log2;
    slot_count_ =
        buffer_size_log2 >= kSlotSizeLog2
            ? (uint32_t(1) << (buffer_size_log2 - kSlotSizeLog2))
            : 0u;
    last_gpu_submission_.assign(slot_count_, 0);
  }

  void Reset() {
    buffer_size_ = 0;
    slot_count_ = 0;
    last_gpu_submission_.clear();
  }

  bool initialized() const { return !last_gpu_submission_.empty(); }

  void MarkByteRangeUsed(uint32_t start, uint32_t length, uint64_t submission) {
    if (!length || start >= buffer_size_ || last_gpu_submission_.empty()) {
      return;
    }
    length = std::min(length, buffer_size_ - start);
    const uint32_t slot_first = start >> kSlotSizeLog2;
    const uint32_t slot_last = (start + length - 1) >> kSlotSizeLog2;
    for (uint32_t slot = slot_first; slot <= slot_last; ++slot) {
      if (submission > last_gpu_submission_[slot]) {
        last_gpu_submission_[slot] = submission;
      }
    }
  }

  bool IsByteRangeSafeForDirectWrite(uint32_t start, uint32_t length,
                                     uint64_t completed_submission,
                                     bool coherent,
                                     uint64_t atom_size) const {
    if (!length || start >= buffer_size_ || last_gpu_submission_.empty()) {
      return false;
    }
    if (length > buffer_size_ - start) {
      return false;
    }
    if (!coherent) {
      const uint64_t atom = std::max<uint64_t>(atom_size, 1);
      if ((uint64_t(start) % atom) != 0 || (uint64_t(length) % atom) != 0) {
        return false;
      }
    }
    const uint32_t slot_first = start >> kSlotSizeLog2;
    const uint32_t slot_last = (start + length - 1) >> kSlotSizeLog2;
    for (uint32_t slot = slot_first; slot <= slot_last; ++slot) {
      if (last_gpu_submission_[slot] > completed_submission) {
        return false;
      }
    }
    return true;
  }

 private:
  uint32_t buffer_size_ = 0;
  uint32_t slot_count_ = 0;
  std::vector<uint64_t> last_gpu_submission_;
};

inline void PartitionZeroCopySpans(
    const ZeroCopyPageSubmissionTracker& tracker, uint32_t span_start,
    uint32_t span_len, uint32_t unit_size_log2, uint64_t completed_submission,
    bool coherent, uint64_t atom_size,
    std::vector<std::pair<uint32_t, uint32_t>>& out_zero_copy_spans,
    std::vector<std::pair<uint32_t, uint32_t>>& out_staging_spans) {
  const uint32_t unit_bytes = uint32_t(1) << unit_size_log2;
  uint32_t zc_run_start = UINT32_MAX;
  uint32_t st_run_start = UINT32_MAX;
  auto flush_zc = [&](uint32_t end_unit) {
    if (zc_run_start != UINT32_MAX) {
      out_zero_copy_spans.emplace_back(zc_run_start, end_unit - zc_run_start);
      zc_run_start = UINT32_MAX;
    }
  };
  auto flush_st = [&](uint32_t end_unit) {
    if (st_run_start != UINT32_MAX) {
      out_staging_spans.emplace_back(st_run_start, end_unit - st_run_start);
      st_run_start = UINT32_MAX;
    }
  };
  for (uint32_t u = span_start; u < span_start + span_len; ++u) {
    const uint32_t byte_offset = u << unit_size_log2;
    const bool safe = tracker.IsByteRangeSafeForDirectWrite(
        byte_offset, unit_bytes, completed_submission, coherent, atom_size);
    if (safe) {
      flush_st(u);
      if (zc_run_start == UINT32_MAX) {
        zc_run_start = u;
      }
    } else {
      flush_zc(u);
      if (st_run_start == UINT32_MAX) {
        st_run_start = u;
      }
    }
  }
  flush_zc(span_start + span_len);
  flush_st(span_start + span_len);
}

// Describes why sparse shared memory is bypassed in favor of a dense
// host-mapped 512 MiB buffer so startup logs reflect the true cause (zero-copy
// upload vs UMA readback vs resolve/memexport without host-memory import).
inline const char* DescribeDenseSharedMemoryReason(
    bool uma_readback, bool zero_copy_upload,
    bool readback_or_memexport_without_host_buffer) {
  const uint32_t mask =
      (uint32_t(uma_readback) << 0) | (uint32_t(zero_copy_upload) << 1) |
      (uint32_t(readback_or_memexport_without_host_buffer) << 2);
  switch (mask) {
    case 0b001u:
      return "UMA resolve readback (readback_resolve=uma)";
    case 0b010u:
      return "zero-copy CPU upload (vulkan_shared_memory_zero_copy_upload)";
    case 0b011u:
      return "UMA resolve readback + zero-copy CPU upload";
    case 0b100u:
      return "resolve/memexport readback without host-memory import";
    case 0b101u:
      return "UMA/resolve/memexport readback without host-memory import";
    case 0b110u:
      return "zero-copy CPU upload + resolve/memexport readback without "
             "host-memory import";
    case 0b111u:
      return "UMA readback + zero-copy CPU upload + resolve/memexport readback "
             "without host-memory import";
    default:
      return "host-mapped shared memory";
  }
}

}  // namespace vulkan
}  // namespace gpu
}  // namespace xe

#ifndef XENIA_GPU_VULKAN_SHARED_MEMORY_PURE_LOGIC_ONLY

#include <memory>

#include "xenia/base/cvar.h"
#include "xenia/gpu/draw_export_tracker.h"
#include "xenia/gpu/shared_memory.h"
#include "xenia/gpu/trace_writer.h"
#include "xenia/memory.h"
#include "xenia/ui/vulkan/vulkan_upload_buffer_pool.h"

namespace xe {
namespace gpu {
namespace vulkan {

class VulkanCommandProcessor;

class VulkanSharedMemory : public SharedMemory {
 public:
  VulkanSharedMemory(VulkanCommandProcessor& command_processor, Memory& memory,
                     TraceWriter& trace_writer,
                     VkPipelineStageFlags guest_shader_pipeline_stages);
  ~VulkanSharedMemory() override;

  bool Initialize();
  void Shutdown(bool from_destructor = false);
  void ClearCache() override;

  void CompletedSubmissionUpdated();
  void EndSubmission();

  enum class Usage {
    // Index buffer, vfetch, compute read, transfer source.
    kRead,
    // Index buffer, vfetch, memexport.
    kGuestDrawReadWrite,
    kComputeWrite,
    kTransferDestination,
  };
  // Inserts a pipeline barrier for the target usage, also ensuring consecutive
  // read-write accesses are ordered with each other.
  void Use(Usage usage, std::pair<uint32_t, uint32_t> written_range = {});

  // What UseForGuestDraw did, for VkMemexport.
  enum class DrawBarrier {
    // None was needed, as with Use.
    kNone,
    // Use would have inserted one here.
    kElided,
    // A draw reads what an earlier draw exported.
    kReadAfterExport,
    // A draw exports over what an earlier draw exported.
    kExportAfterExport,
    // A draw exports after draws that read other memory, or over what an
    // earlier exporting draw read: execution only.
    kExportAfterRead,
    // The first export after the buffer was used otherwise, as with Use. With
    // vulkan_memexport_barriers_by_range off, the barrier of any exporting draw.
    kFirstExport,
    kCount,
  };
  // Use for a guest draw that reads and exports these byte ranges. Consecutive
  // draws with memory export otherwise get a barrier between each two, ending
  // the render pass, whether or not they touch the same memory. Here, there's
  // one only where a draw reads or writes what an earlier one has exported (or
  // writes what an earlier one has read) - later uses of the buffer by anything
  // else go through Use, which orders them after all the exports.
  // export_batch: see DrawExportTracker.
  // Not for local read attachments, which Use is for.
  DrawBarrier UseForGuestDraw(
      const std::pair<uint32_t, uint32_t>* reads, size_t read_count,
      const std::pair<uint32_t, uint32_t>* exports, size_t export_count,
      uint64_t export_batch);

  // Whether UseForGuestDraw needs a draw's ranges even if it doesn't export:
  // exports not ordered with later draws yet may be in its way.
  bool NeedsDrawRanges() const {
    return last_usage_ == Usage::kGuestDrawReadWrite;
  }

  VkBuffer buffer() const { return buffer_; }
  VkDeviceAddress buffer_device_address() const {
    return buffer_device_address_;
  }

  // Notes a range written by an in-pass draw without inserting a barrier at
  // the write point - ordering happens at the next Use().
  void MarkInPassWrite(std::pair<uint32_t, uint32_t> written_range) {
    NoteByteRangeUsedByGpu(written_range.first, written_range.second);
    if (last_written_range_.second) {
      uint32_t start =
          std::min(last_written_range_.first, written_range.first);
      uint32_t end =
          std::max(last_written_range_.first + last_written_range_.second,
                   written_range.first + written_range.second);
      last_written_range_ = std::make_pair(start, end - start);
    } else {
      last_written_range_ = written_range;
    }
  }

  // Whether the shared-memory buffer is persistently host-mapped so guest
  // physical memory can be read directly by the CPU (readback_resolve=uma).
  // True only for a dense, host-visible allocation.
  bool IsHostMapped() const { return host_mapped_data_ != nullptr; }
  // Copies `length` bytes of guest physical memory at `guest_address` out of
  // the mapped shared-memory buffer into `dest`. Invalidates the CPU cache
  // first when the memory type is not host-coherent. The caller is responsible
  // for GPU/CPU ordering; this is the approximate (fast-class) readback path.
  // Returns false if the buffer is not host-mapped or the range is out of
  // bounds.
  bool ReadHostMapped(uint32_t guest_address, uint32_t length,
                      void* dest) const;

  // A host-imported (guest RAM) copy of the buffer, or VK_NULL_HANDLE if
  // unavailable. Bound instead of buffer() for memexport-touching draws so
  // their output is coherent with the CPU (no clobber) - it aliases guest RAM.
  VkBuffer host_buffer() const { return host_buffer_; }

  // True when the buffer aliases guest RAM directly
  // (VK_EXT_external_memory_host import). Uploads and memexport readback copies
  // are then unnecessary.
  bool is_zero_copy() const { return zero_copy_; }

  // Returns true if any downloads were submitted to the command processor.
  bool InitializeTraceSubmitDownloads();
  void InitializeTraceCompleteDownloads();

 protected:
  bool AllocateSparseHostGpuMemoryRange(uint32_t offset_allocations,
                                        uint32_t length_allocations) override;

  bool CopyToGuestMemory(uint32_t start, uint32_t length) override;

  bool UploadRanges(const std::pair<uint32_t, uint32_t>* upload_page_ranges,
                    uint32_t num_ranges) override;

  bool SupportsSubpageUpload() const override { return true; }
  bool UploadSubpageRanges(
      const std::pair<uint32_t, uint32_t>* upload_subpage_ranges,
      uint32_t num_upload_ranges) override;

  void NoteByteRangeUsedByGpu(uint32_t start, uint32_t length) override;

 private:
  void PollZeroCopyUploadProperty();
  bool UploadUnitRanges(const std::pair<uint32_t, uint32_t>* upload_ranges,
                        uint32_t num_ranges, uint32_t unit_size_log2);
  void GetUsageMasks(Usage usage, VkPipelineStageFlags& stage_mask,
                     VkAccessFlags& access_mask) const;

  VulkanCommandProcessor& command_processor_;
  TraceWriter& trace_writer_;
  VkPipelineStageFlags guest_shader_pipeline_stages_;

  // Creates a non-sparse buffer bound to imported guest RAM
  // (VK_EXT_external_memory_host). Returns true and fills out_buffer/out_memory
  // on success; false (nothing to clean up) otherwise.
  bool CreateImportedGuestRamBuffer(VkBuffer& out_buffer,
                                    VkDeviceMemory& out_memory);
  // Attempts to alias guest RAM as the whole buffer (full zero-copy). Returns
  // true on success (buffer_/buffer_memory_ populated, zero_copy_ set); false
  // to fall back to the normal device-local path.
  bool TryInitializeZeroCopy();
  // Attempts to create host_buffer_ for the hybrid two-buffer path. No-op on
  // failure (host_buffer_ stays null).
  void TryInitializeHostBuffer();

  VkBuffer buffer_ = VK_NULL_HANDLE;
  VkDeviceAddress buffer_device_address_ = 0;
  uint32_t buffer_memory_type_;
  // Single for non-sparse, every allocation so far for sparse.
  std::vector<VkDeviceMemory> buffer_memory_;
  // Buffer memory is imported guest RAM - no uploads or readback copies needed.
  bool zero_copy_ = false;

  // Second buffer aliasing guest RAM, bound for memexport-touching draws.
  VkBuffer host_buffer_ = VK_NULL_HANDLE;
  VkDeviceMemory host_buffer_memory_ = VK_NULL_HANDLE;

  // Persistent host mapping of the dense buffer for direct CPU readback
  // (readback_resolve=uma). Null unless the buffer is a single host-visible
  // allocation.
  uint8_t* host_mapped_data_ = nullptr;
  bool host_mapped_coherent_ = false;

  Usage last_usage_;
  std::pair<uint32_t, uint32_t> last_written_range_;
  DrawExportTracker draw_exports_;

  std::unique_ptr<ui::vulkan::VulkanUploadBufferPool> upload_buffer_pool_;
  std::vector<VkBufferCopy> upload_regions_;
  // Upload ranges left after the memexport ones are refreshed on the GPU.
  std::vector<std::pair<uint32_t, uint32_t>> cpu_upload_ranges_;
  // Zero-copy upload hazard tracking and scratch buffers.
  ZeroCopyPageSubmissionTracker zero_copy_submission_tracker_;
  std::vector<std::pair<uint32_t, uint32_t>> zero_copy_spans_;
  std::vector<std::pair<uint32_t, uint32_t>> staging_spans_;
  std::vector<ByteRange> zero_copy_flush_ranges_;
  std::vector<VkMappedMemoryRange> vk_flush_ranges_;
  uint32_t zero_copy_prop_poll_counter_ = 0;

  // Created temporarily, only for downloading.
  VkBuffer trace_download_buffer_ = VK_NULL_HANDLE;
  VkDeviceMemory trace_download_buffer_memory_ = VK_NULL_HANDLE;
  uint32_t trace_download_buffer_memory_type_ = UINT32_MAX;
  void ResetTraceDownload();
};

}  // namespace vulkan
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_VULKAN_SHARED_MEMORY_PURE_LOGIC_ONLY

#endif  // XENIA_GPU_VULKAN_VULKAN_SHARED_MEMORY_H_
