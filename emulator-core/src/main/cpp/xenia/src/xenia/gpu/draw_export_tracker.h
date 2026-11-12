/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_DRAW_EXPORT_TRACKER_H_
#define XENIA_GPU_DRAW_EXPORT_TRACKER_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace xe {
namespace gpu {

// XenDroid: what guest draws with memory export have exported and read since the
// last barrier that ordered them, to tell which draw needs a barrier before it.
//
// Exporting draws of the same export batch, which the caller ends at whatever a
// title could wait for exports with (another draw, like one of hundreds of empty
// vertices, or a fence), aren't ordered with each other where they update memory
// in place: a Xenos draw can't rely on what the ones just before it have
// exported. So titles update a buffer in place with such batches, each draw in
// its own elements, though the ranges known here are the buffer's for all of
// them. A draw reading what the batch has exported without exporting it itself
// is a chain (A to B, then B to C), and does get a barrier.
class DrawExportTracker {
 public:
  // Byte address and length.
  using Range = std::pair<uint32_t, uint32_t>;

  enum class Need {
    kNothing,
    // A draw reads or writes what an earlier one has exported.
    kCommit,
    // A write over reads, of the same memory or unknown: only has to wait.
    kWaitForReads,
  };

  // Past this many separate ranges a draw gets a barrier instead.
  static constexpr size_t kMaxRanges = 256;

  // A barrier has ordered everything so far.
  void Clear() {
    pending_exports_.clear();
    export_reads_.clear();
    batch_exports_.clear();
    batch_reads_.clear();
    batch_ = 0;
    other_reads_ = false;
  }

  // Reads of ranges not known here happened.
  void NoteOtherReads() { other_reads_ = true; }

  // What a draw needs before it, with draws that export in export_batch.
  Need Check(const Range* reads, size_t read_count, const Range* exports,
             size_t export_count, uint64_t export_batch) {
    if (!export_count || export_batch != batch_) {
      EndBatch();
    }
    for (size_t i = 0; i < read_count; ++i) {
      if (Overlaps(pending_exports_, reads[i]) ||
          (Overlaps(batch_exports_, reads[i]) &&
           !Contains(exports, export_count, reads[i]))) {
        return Need::kCommit;
      }
    }
    for (size_t i = 0; i < export_count; ++i) {
      if (Overlaps(pending_exports_, exports[i])) {
        return Need::kCommit;
      }
    }
    if (pending_exports_.size() + batch_exports_.size() + export_count >
        kMaxRanges) {
      return Need::kCommit;
    }
    if (export_count) {
      if (other_reads_) {
        return Need::kWaitForReads;
      }
      for (size_t i = 0; i < export_count; ++i) {
        if (Overlaps(export_reads_, exports[i])) {
          return Need::kWaitForReads;
        }
      }
    }
    return Need::kNothing;
  }

  // An execution barrier was made for the reads so far.
  void ClearReads() {
    export_reads_.clear();
    other_reads_ = false;
  }

  // Notes a draw that goes on without a barrier, or with the one it needed.
  void AddDraw(const Range* reads, size_t read_count, const Range* exports,
               size_t export_count, uint64_t export_batch) {
    if (!export_count) {
      // Draws with no export aren't tracked by range.
      if (read_count) {
        other_reads_ = true;
      }
      return;
    }
    for (size_t i = 0; i < export_count; ++i) {
      Add(batch_exports_, exports[i]);
    }
    for (size_t i = 0; i < read_count; ++i) {
      Add(batch_reads_, reads[i]);
    }
    batch_ = export_batch;
    if (batch_reads_.size() > kMaxRanges) {
      batch_reads_.clear();
      other_reads_ = true;
    }
  }

 private:
  using Ranges = std::vector<Range>;

  void EndBatch() {
    for (const Range& range : batch_exports_) {
      Add(pending_exports_, range);
    }
    for (const Range& range : batch_reads_) {
      Add(export_reads_, range);
    }
    batch_exports_.clear();
    batch_reads_.clear();
    batch_ = 0;
  }

  static bool Contains(const Range* ranges, size_t count, Range range) {
    for (size_t i = 0; i < count; ++i) {
      if (ranges[i].first <= range.first &&
          range.first + range.second <= ranges[i].first + ranges[i].second) {
        return true;
      }
    }
    return false;
  }

  // The ranges are sorted and disjoint, so their ends are sorted too.
  static bool Overlaps(const Ranges& ranges, Range range) {
    if (!range.second) {
      return false;
    }
    auto it = std::upper_bound(
        ranges.begin(), ranges.end(), range.first,
        [](uint32_t address, const Range& other) {
          return address < other.first + other.second;
        });
    return it != ranges.end() && it->first < range.first + range.second;
  }

  // Merges with all that it overlaps or touches.
  static void Add(Ranges& ranges, Range range) {
    if (!range.second) {
      return;
    }
    auto first = std::lower_bound(
        ranges.begin(), ranges.end(), range.first,
        [](const Range& other, uint32_t address) {
          return other.first + other.second < address;
        });
    uint32_t start = range.first, end = range.first + range.second;
    auto last = first;
    while (last != ranges.end() && last->first <= end) {
      start = std::min(start, last->first);
      end = std::max(end, last->first + last->second);
      ++last;
    }
    if (first != last) {
      *first = Range(start, end - start);
      ranges.erase(first + 1, last);
    } else {
      ranges.insert(first, Range(start, end - start));
    }
  }

  Ranges pending_exports_;
  Ranges export_reads_;
  // The batch of exporting draws going on, not in the above yet.
  Ranges batch_exports_;
  Ranges batch_reads_;
  uint64_t batch_ = 0;
  bool other_reads_ = false;
};

}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_DRAW_EXPORT_TRACKER_H_
