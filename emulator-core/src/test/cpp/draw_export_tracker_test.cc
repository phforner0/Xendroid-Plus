#include <cassert>
#include <cstdint>
#include <cstdio>
#include <vector>
#include "xenia/gpu/draw_export_tracker.h"

using xe::gpu::DrawExportTracker;
using Need = DrawExportTracker::Need;
using Range = DrawExportTracker::Range;

constexpr uint64_t kShaderA = 0xA, kShaderB = 0xB;

// As UseForGuestDraw does, with no barrier in between unless the test makes one.
static Need Draw(DrawExportTracker& tracker, std::vector<Range> reads,
                 std::vector<Range> exports, uint64_t batch) {
  const Need need = tracker.Check(reads.data(), reads.size(), exports.data(),
                                  exports.size(), batch);
  tracker.AddDraw(reads.data(), reads.size(), exports.data(), exports.size(),
                  batch);
  return need;
}

static void NothingIsNeededWithoutExports() {
  DrawExportTracker t;
  assert(Draw(t, {{0, 64}}, {}, 0) == Need::kNothing);
  assert(Draw(t, {{0, 64}}, {}, 0) == Need::kNothing);
}

static void TheDrawsOfABatchArentOrdered() {
  // Halo: Reach updates its particles in place, a draw per block of them, with
  // the whole buffer as the range both read and exported.
  DrawExportTracker t;
  assert(Draw(t, {{0x1000, 0x8000}}, {{0x1000, 0x8000}}, kShaderA) ==
         Need::kNothing);
  assert(Draw(t, {{0x1000, 0x8000}}, {{0x1000, 0x8000}}, kShaderA) ==
         Need::kNothing);
  assert(Draw(t, {{0x1000, 0x8000}}, {{0x1000, 0x8000}}, kShaderA) ==
         Need::kNothing);
}

static void AChainInABatchIsOrdered() {
  // The same shader from A to B, then from B to C.
  DrawExportTracker t;
  assert(Draw(t, {{0x1000, 0x100}}, {{0x2000, 0x100}}, kShaderA) ==
         Need::kNothing);
  assert(Draw(t, {{0x2000, 0x100}}, {{0x3000, 0x100}}, kShaderA) ==
         Need::kCommit);
}

static void AReadWithinItsOwnExportIsntOrdered() {
  DrawExportTracker t;
  assert(Draw(t, {{0x1000, 0x8000}}, {{0x1000, 0x8000}}, kShaderA) ==
         Need::kNothing);
  assert(Draw(t, {{0x2000, 0x100}}, {{0x1000, 0x8000}}, kShaderA) ==
         Need::kNothing);
}

static void AReadPastItsOwnExportIsOrdered() {
  DrawExportTracker t;
  assert(Draw(t, {{0x1000, 0x8000}}, {{0x1000, 0x8000}}, kShaderA) ==
         Need::kNothing);
  assert(Draw(t, {{0x1000, 0x8000}}, {{0x1000, 0x100}}, kShaderA) ==
         Need::kCommit);
}

static void AReadOfWhatWasExportedCommits() {
  DrawExportTracker t;
  Draw(t, {}, {{0x1000, 0x100}}, kShaderA);
  assert(Draw(t, {{0x2000, 0x100}}, {}, 0) == Need::kNothing);
  DrawExportTracker u;
  Draw(u, {}, {{0x1000, 0x100}}, kShaderA);
  assert(u.Check(std::vector<Range>{{0x10FF, 1}}.data(), 1, nullptr, 0, 0) ==
         Need::kCommit);
}

static void TheBoundsOfARangeAreExact() {
  DrawExportTracker t;
  Draw(t, {}, {{0x1000, 0x100}}, kShaderA);
  std::vector<Range> before = {{0xF00, 0x100}}, after = {{0x1100, 0x100}};
  assert(t.Check(before.data(), 1, nullptr, 0, 0) == Need::kNothing);
  assert(t.Check(after.data(), 1, nullptr, 0, 0) == Need::kNothing);
}

static void AnotherBatchOverTheExportsCommits() {
  DrawExportTracker t;
  Draw(t, {}, {{0x1000, 0x100}}, kShaderA);
  assert(Draw(t, {}, {{0x1000, 0x10}}, kShaderB) == Need::kCommit);
}

static void AnotherBatchElsewhereNeedsNothing() {
  DrawExportTracker t;
  Draw(t, {}, {{0x1000, 0x100}}, kShaderA);
  assert(Draw(t, {}, {{0x4000, 0x100}}, kShaderB) == Need::kNothing);
}

static void ADrawInBetweenEndsTheBatch() {
  // The empty draws titles wait for exports with: the next ones are ordered.
  DrawExportTracker t;
  Draw(t, {}, {{0x1000, 0x100}}, kShaderA);
  assert(Draw(t, {}, {}, 0) == Need::kNothing);
  assert(Draw(t, {}, {{0x1000, 0x100}}, kShaderA) == Need::kCommit);
}

static void AnExportWaitsForOtherReads() {
  DrawExportTracker t;
  Draw(t, {}, {{0x1000, 0x100}}, kShaderA);
  // A draw with no export reads elsewhere, by ranges not tracked.
  assert(Draw(t, {{0x8000, 0x100}}, {}, 0) == Need::kNothing);
  std::vector<Range> exports = {{0x4000, 0x100}};
  assert(t.Check(nullptr, 0, exports.data(), 1, kShaderB) == Need::kWaitForReads);
  t.ClearReads();
  assert(t.Check(nullptr, 0, exports.data(), 1, kShaderB) == Need::kNothing);
}

static void AnExportWaitsForAnotherBatchReadingThere() {
  DrawExportTracker t;
  Draw(t, {{0x4000, 0x100}}, {{0x1000, 0x100}}, kShaderA);
  assert(Draw(t, {}, {{0x4080, 0x10}}, kShaderB) == Need::kWaitForReads);
}

static void ClearForgetsEverything() {
  DrawExportTracker t;
  Draw(t, {{0x4000, 0x100}}, {{0x1000, 0x100}}, kShaderA);
  t.NoteOtherReads();
  t.Clear();
  assert(Draw(t, {{0x1000, 0x100}}, {}, 0) == Need::kNothing);
  t.Clear();
  assert(Draw(t, {}, {{0x4000, 0x100}}, kShaderB) == Need::kNothing);
}

static void EmptyRangesTouchNothing() {
  DrawExportTracker t;
  Draw(t, {}, {{0x1000, 0}}, kShaderA);
  assert(Draw(t, {{0x1000, 0x100}}, {}, 0) == Need::kNothing);
}

static void TooManyRangesCommit() {
  DrawExportTracker t;
  uint32_t address = 0;
  uint64_t batch = 1;
  Need need = Need::kNothing;
  for (size_t i = 0; i <= DrawExportTracker::kMaxRanges && need == Need::kNothing;
       ++i) {
    need = Draw(t, {}, {{address, 16}}, batch++);
    address += 64;
  }
  assert(need == Need::kCommit);
}

static void TouchingRangesMerge() {
  DrawExportTracker t;
  Draw(t, {}, {{0x1000, 0x100}}, kShaderA);
  Draw(t, {}, {{0x1200, 0x100}}, kShaderB);
  Draw(t, {}, {{0x1100, 0x100}}, kShaderA);
  // Ends the batch; all three as one range now.
  std::vector<Range> middle = {{0x1180, 8}};
  assert(t.Check(middle.data(), 1, nullptr, 0, 0) == Need::kCommit);
}

int main() {
  NothingIsNeededWithoutExports();
  TheDrawsOfABatchArentOrdered();
  AChainInABatchIsOrdered();
  AReadWithinItsOwnExportIsntOrdered();
  AReadPastItsOwnExportIsOrdered();
  AReadOfWhatWasExportedCommits();
  TheBoundsOfARangeAreExact();
  AnotherBatchOverTheExportsCommits();
  AnotherBatchElsewhereNeedsNothing();
  ADrawInBetweenEndsTheBatch();
  AnExportWaitsForOtherReads();
  AnExportWaitsForAnotherBatchReadingThere();
  ClearForgetsEverything();
  EmptyRangesTouchNothing();
  TooManyRangesCommit();
  TouchingRangesMerge();
  std::puts("draw export tracker: passed");
}
