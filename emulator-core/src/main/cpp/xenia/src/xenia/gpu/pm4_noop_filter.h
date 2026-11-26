/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_PM4_NOOP_FILTER_H_
#define XENIA_GPU_PM4_NOOP_FILTER_H_

#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace xe {
namespace gpu {
namespace pm4_noop {

// Pure-logic 4-wide (128-bit AArch64 NEON / 4-wide scalar fallback) no-op
// register write filter for Onda 3B.
//
// Safety contract (see onda3-minispec.md §3.1):
// 1. NEVER drops writes whose new value differs from the current value in
//    register_file_->values[] (100% bit-exact for PM4_REG_TO_MEM,
//    PM4_COND_WRITE, PM4_REG_RMW, PM4_WAIT_REG_MEM, and save-states).
// 2. Mandatory side-effect registers (SCRATCH_REG0..7, COHER_STATUS_HOST,
//    DC_LUT_RW_INDEX..DC_LUT_30_COLOR) ALWAYS invoke their write side-effect
//    handler even when new_value == old_value.
// 3. When 4 consecutive non-side-effect registers all match new == old:
//    - skips the store (avoids dirtying the RegisterFile cacheline), and
//    - skips dirtying downstream trackers (sysconst_dirty_tracker_ /
//      kConstantBufferSystem).
//
// Integration interface for VulkanCommandProcessor (post-3A merge):
// - In WriteRegistersFromMem (`regular` lambda, big-endian guest memory):
//     if (cvars::vulkan_pm4_noop_reg_filter_4wide) {
//       auto res = pm4_noop::FilterWriteRegisters4WideBigEndian(
//           register_file_->values, index, src, count,
//           [this](uint32_t reg) {
//             sysconst_dirty_tracker_.NoteRegisterWrite(reg);
//           });
//       if (res.system_constants_dirty) {
//         current_constant_buffers_up_to_date_ &=
//             ~(UINT32_C(1) << SpirvShaderTranslator::kConstantBufferSystem);
//       }
//     }
// - In WritePossiblySpecialRegistersFromMem (big-endian guest memory):
//     if (cvars::vulkan_pm4_noop_reg_filter_4wide) {
//       auto res = pm4_noop::FilterWritePossiblySpecialRegistersBigEndian(
//           register_file_->values, start_index, base, num_registers,
//           [this](uint32_t reg, uint32_t val) {
//             HandleSpecialRegisterWrite(reg, val);
//           },
//           [this](uint32_t reg) {
//             sysconst_dirty_tracker_.NoteRegisterWrite(reg);
//           });
//       if (res.system_constants_dirty) {
//         current_constant_buffers_up_to_date_ &=
//             ~(UINT32_C(1) << SpirvShaderTranslator::kConstantBufferSystem);
//       }
//     }
// - In IB cache host-endian replay (already wired via CommandProcessor):
//     pm4_noop::DispatchChangedRegisters4WideHostEndian(...) filters unchanged
//     dwords 4-wide without byte-swap before dispatching changed registers to
//     WriteRegister(), and post-3A VulkanCommandProcessor can also override
//     WriteRegistersHostEndian() using FilterWriteRegisters4WideHostEndian().

// Canonical Xenos register indices (static_asserted against XE_GPU_REG_* in
// command_processor.cc).
constexpr uint32_t kRegScratchReg0 = 0x0578u;
constexpr uint32_t kRegScratchReg7 = 0x057Fu;
constexpr uint32_t kRegCoherStatusHost = 0x0A31u;
constexpr uint32_t kRegDcLutRwIndex = 0x1922u;
constexpr uint32_t kRegDcLutSeqColor = 0x1923u;
constexpr uint32_t kRegDcLutPwlData = 0x1924u;
constexpr uint32_t kRegDcLut30Color = 0x1925u;

constexpr uint32_t kRegVgtMaxVtxIndx = 0x2100u;
constexpr uint32_t kRegVgtMinVtxIndx = 0x2101u;
constexpr uint32_t kRegVgtIndxOffset = 0x2102u;
constexpr uint32_t kRegVgtDmaSize = 0x21FBu;
constexpr uint32_t kRegPaClClipCntl = 0x2204u;
constexpr uint32_t kRegVgtHosMaxTessLevel = 0x2286u;
constexpr uint32_t kRegVgtHosMinTessLevel = 0x2287u;
constexpr uint32_t kRegPaClUcp0X = 0x2388u;
constexpr uint32_t kRegPaClUcp5W = 0x239Fu;

// Returns true iff writing `reg_index` has a mandatory hardware/emulator side
// effect that MUST execute even when `new_value == old_value`.
constexpr bool HasMandatoryWriteSideEffect(uint32_t reg_index) noexcept {
  return (reg_index - kRegScratchReg0 < 8u) ||
         (reg_index == kRegCoherStatusHost) ||
         ((reg_index - kRegDcLutRwIndex) <=
          (kRegDcLut30Color - kRegDcLutRwIndex));
}

// Returns true iff the half-open register range [start_index, start_index + count)
// overlaps any mandatory side-effect register.
constexpr bool RangeHasMandatoryWriteSideEffect(uint32_t start_index,
                                                uint32_t count) noexcept {
  if (!count) {
    return false;
  }
  const uint64_t lo = start_index;
  const uint64_t hi = uint64_t(start_index) + uint64_t(count);
  auto overlaps = [lo, hi](uint32_t a, uint32_t b_incl) {
    return lo <= b_incl && hi > a;
  };
  return overlaps(kRegScratchReg0, kRegScratchReg7) ||
         overlaps(kRegCoherStatusHost, kRegCoherStatusHost) ||
         overlaps(kRegDcLutRwIndex, kRegDcLut30Color);
}

// Returns true iff changing `reg_index` invalidates kConstantBufferSystem
// (mirrors VulkanCommandProcessor::WritePossiblySpecialRegistersFromMem).
constexpr bool AffectsSystemConstantBuffer(uint32_t reg_index) noexcept {
  return (reg_index >= kRegVgtMaxVtxIndx && reg_index <= kRegVgtIndxOffset) ||
         reg_index == kRegVgtDmaSize || reg_index == kRegPaClClipCntl ||
         reg_index == kRegVgtHosMaxTessLevel ||
         reg_index == kRegVgtHosMinTessLevel ||
         (reg_index >= kRegPaClUcp0X && reg_index <= kRegPaClUcp5W);
}

struct NoopFilterWriteResult {
  bool any_changed = false;
  bool system_constants_dirty = false;
  uint32_t first_changed_index = 0;
  uint32_t last_changed_index = 0;
  uint32_t changed_count = 0;
  uint32_t skipped_count = 0;
  uint32_t side_effect_count = 0;
};

namespace detail {

template <bool kSwapInput>
inline uint32_t LoadDword(const uint32_t* ptr) noexcept {
  uint32_t value = 0;
  std::memcpy(&value, ptr, sizeof(value));
  if constexpr (kSwapInput) {
#if defined(_MSC_VER)
    return _byteswap_ulong(value);
#else
    return __builtin_bswap32(value);
#endif
  } else {
    return value;
  }
}

inline void RecordChangedLane(NoopFilterWriteResult& result,
                              uint32_t reg_index) noexcept {
  if (!result.any_changed) {
    result.any_changed = true;
    result.first_changed_index = reg_index;
  }
  result.last_changed_index = reg_index;
  ++result.changed_count;
  if (AffectsSystemConstantBuffer(reg_index)) {
    result.system_constants_dirty = true;
  }
}

struct NoopDirtyCallback {
  constexpr void operator()(uint32_t) const noexcept {}
};

// Compares 4 consecutive dwords against `cur_ptr` and writes all 4 lanes to
// `dst_ptr` (if non-null) ONLY when at least one lane differs.
// Returns a 4-bit mask where bit `k` is 1 iff lane `k` differed (0 means all
// 4 lanes matched and no store was performed).
template <bool kSwapInput, bool kWriteDest>
inline uint32_t CompareAndMaybeStore4Wide(uint32_t* dst_ptr,
                                          const uint32_t* cur_ptr,
                                          const uint32_t* src_ptr,
                                          uint32_t out_vals[4]) noexcept {
#if defined(__aarch64__)
  const uint32x4_t cur = vld1q_u32(cur_ptr);
  uint32x4_t val;
  if constexpr (kSwapInput) {
    const uint8x16_t raw = vld1q_u8(reinterpret_cast<const uint8_t*>(src_ptr));
    val = vreinterpretq_u32_u8(vrev32q_u8(raw));
  } else {
    val = vreinterpretq_u32_u8(
        vld1q_u8(reinterpret_cast<const uint8_t*>(src_ptr)));
  }
  const uint32x4_t eq = vceqq_u32(val, cur);
  if (vminvq_u32(eq) == 0xFFFFFFFFu) {
    return 0u;
  }
  if constexpr (kWriteDest) {
    vst1q_u32(dst_ptr, val);
  } else {
    vst1q_u32(out_vals, val);
  }
  const uint32_t e0 = vgetq_lane_u32(eq, 0);
  const uint32_t e1 = vgetq_lane_u32(eq, 1);
  const uint32_t e2 = vgetq_lane_u32(eq, 2);
  const uint32_t e3 = vgetq_lane_u32(eq, 3);
  return ((e0 == 0u) ? 1u : 0u) | ((e1 == 0u) ? 2u : 0u) |
         ((e2 == 0u) ? 4u : 0u) | ((e3 == 0u) ? 8u : 0u);
#else
  const uint32_t v0 = LoadDword<kSwapInput>(src_ptr + 0);
  const uint32_t v1 = LoadDword<kSwapInput>(src_ptr + 1);
  const uint32_t v2 = LoadDword<kSwapInput>(src_ptr + 2);
  const uint32_t v3 = LoadDword<kSwapInput>(src_ptr + 3);
  const uint32_t d0 = cur_ptr[0] ^ v0;
  const uint32_t d1 = cur_ptr[1] ^ v1;
  const uint32_t d2 = cur_ptr[2] ^ v2;
  const uint32_t d3 = cur_ptr[3] ^ v3;
  if ((d0 | d1 | d2 | d3) == 0u) {
    return 0u;
  }
  if constexpr (kWriteDest) {
    dst_ptr[0] = v0;
    dst_ptr[1] = v1;
    dst_ptr[2] = v2;
    dst_ptr[3] = v3;
  } else {
    out_vals[0] = v0;
    out_vals[1] = v1;
    out_vals[2] = v2;
    out_vals[3] = v3;
  }
  return ((d0 != 0u) ? 1u : 0u) | ((d1 != 0u) ? 2u : 0u) |
         ((d2 != 0u) ? 4u : 0u) | ((d3 != 0u) ? 8u : 0u);
#endif
}

template <bool kSwapInput, typename DirtyRegCallback>
inline void FilterWriteNoSideEffectRangeInto(
    uint32_t* register_values, uint32_t start_index, const uint32_t* src,
    uint32_t count, NoopFilterWriteResult& result,
    DirtyRegCallback&& on_dirty_reg) {
  uint32_t idx = start_index;
  uint32_t remaining = count;
  uint32_t* dst = register_values + start_index;
  uint32_t unused_vals[4];

  while (remaining >= 4u) {
    const uint32_t diff_mask =
        CompareAndMaybeStore4Wide<kSwapInput, true>(dst, dst, src, unused_vals);
    if (diff_mask == 0u) {
      result.skipped_count += 4u;
    } else {
      for (uint32_t lane = 0; lane < 4u; ++lane) {
        if (diff_mask & (1u << lane)) {
          const uint32_t reg_index = idx + lane;
          RecordChangedLane(result, reg_index);
          on_dirty_reg(reg_index);
        } else {
          ++result.skipped_count;
        }
      }
    }
    dst += 4;
    src += 4;
    idx += 4;
    remaining -= 4;
  }

  while (remaining > 0u) {
    const uint32_t val = LoadDword<kSwapInput>(src);
    if (*dst != val) {
      *dst = val;
      RecordChangedLane(result, idx);
      on_dirty_reg(idx);
    } else {
      ++result.skipped_count;
    }
    ++dst;
    ++src;
    ++idx;
    --remaining;
  }
}

}  // namespace detail

// Store-in-place 4-wide filter for regular (non-side-effect) register ranges.
// Caller must ensure [start_index, start_index + count) is within bounds and
// does not contain mandatory side-effect registers (use
// FilterWritePossiblySpecialRegisters when the range may overlap them).
template <bool kSwapInput, typename DirtyRegCallback = detail::NoopDirtyCallback>
inline NoopFilterWriteResult FilterWriteRegisters4Wide(
    uint32_t* register_values, uint32_t start_index, const uint32_t* src,
    uint32_t count, DirtyRegCallback&& on_dirty_reg = {}) {
  NoopFilterWriteResult result;
  if (!count) {
    return result;
  }
  detail::FilterWriteNoSideEffectRangeInto<kSwapInput>(
      register_values, start_index, src, count, result,
      std::forward<DirtyRegCallback>(on_dirty_reg));
  return result;
}

template <typename DirtyRegCallback = detail::NoopDirtyCallback>
inline NoopFilterWriteResult FilterWriteRegisters4WideBigEndian(
    uint32_t* register_values, uint32_t start_index, const uint32_t* src,
    uint32_t count, DirtyRegCallback&& on_dirty_reg = {}) {
  return FilterWriteRegisters4Wide<true>(
      register_values, start_index, src, count,
      std::forward<DirtyRegCallback>(on_dirty_reg));
}

template <typename DirtyRegCallback = detail::NoopDirtyCallback>
inline NoopFilterWriteResult FilterWriteRegisters4WideHostEndian(
    uint32_t* register_values, uint32_t start_index, const uint32_t* src,
    uint32_t count, DirtyRegCallback&& on_dirty_reg = {}) {
  return FilterWriteRegisters4Wide<false>(
      register_values, start_index, src, count,
      std::forward<DirtyRegCallback>(on_dirty_reg));
}

// Store-in-place 4-wide filter for ranges that MAY overlap mandatory
// side-effect registers (SCRATCH_REG0..7, COHER_STATUS_HOST,
// DC_LUT_RW_INDEX..DC_LUT_30_COLOR).
// - Mandatory side-effect registers ALWAYS store into `register_values[index]`
//   and ALWAYS invoke `on_special_write(index, value)` even if new == old.
// - All non-side-effect subranges execute via the 4-wide NEON/scalar fast path
//   in strict ascending register order, invoking `on_dirty_reg(index)` and
//   setting `system_constants_dirty` ONLY when `new != old`.
template <bool kSwapInput, typename SpecialCallback,
          typename DirtyRegCallback = detail::NoopDirtyCallback>
inline NoopFilterWriteResult FilterWritePossiblySpecialRegisters(
    uint32_t* register_values, uint32_t start_index, const uint32_t* src,
    uint32_t count, SpecialCallback&& on_special_write,
    DirtyRegCallback&& on_dirty_reg = {}) {
  NoopFilterWriteResult result;
  if (!count) {
    return result;
  }
  if (!RangeHasMandatoryWriteSideEffect(start_index, count)) {
    detail::FilterWriteNoSideEffectRangeInto<kSwapInput>(
        register_values, start_index, src, count, result,
        std::forward<DirtyRegCallback>(on_dirty_reg));
    return result;
  }

  uint32_t current = start_index;
  const uint32_t end = start_index + count;
  const uint32_t* cur_src = src;

  auto run_regular_until = [&](uint32_t limit) {
    if (current < limit) {
      const uint32_t seg_end = limit < end ? limit : end;
      const uint32_t seg_count = seg_end - current;
      detail::FilterWriteNoSideEffectRangeInto<kSwapInput>(
          register_values, current, cur_src, seg_count, result, on_dirty_reg);
      current += seg_count;
      cur_src += seg_count;
    }
  };

  auto run_special_until = [&](uint32_t limit) {
    if (current < limit) {
      const uint32_t seg_end = limit < end ? limit : end;
      while (current < seg_end) {
        const uint32_t value = detail::LoadDword<kSwapInput>(cur_src);
        const uint32_t old_value = register_values[current];
        register_values[current] = value;
        if (value != old_value) {
          detail::RecordChangedLane(result, current);
          on_dirty_reg(current);
        }
        on_special_write(current, value);
        ++result.side_effect_count;
        ++current;
        ++cur_src;
      }
    }
  };

  // Partition in ascending order around the three mandatory side-effect islands:
  //   [0x0578..0x057F] SCRATCH_REG0..7
  //   [0x0A31..0x0A31] COHER_STATUS_HOST
  //   [0x1922..0x1925] DC_LUT_RW_INDEX..DC_LUT_30_COLOR
  run_regular_until(kRegScratchReg0);
  run_special_until(kRegScratchReg7 + 1u);
  run_regular_until(kRegCoherStatusHost);
  run_special_until(kRegCoherStatusHost + 1u);
  run_regular_until(kRegDcLutRwIndex);
  run_special_until(kRegDcLut30Color + 1u);
  run_regular_until(end);

  return result;
}

template <typename SpecialCallback,
          typename DirtyRegCallback = detail::NoopDirtyCallback>
inline NoopFilterWriteResult FilterWritePossiblySpecialRegistersBigEndian(
    uint32_t* register_values, uint32_t start_index, const uint32_t* src,
    uint32_t count, SpecialCallback&& on_special_write,
    DirtyRegCallback&& on_dirty_reg = {}) {
  return FilterWritePossiblySpecialRegisters<true>(
      register_values, start_index, src, count,
      std::forward<SpecialCallback>(on_special_write),
      std::forward<DirtyRegCallback>(on_dirty_reg));
}

template <typename SpecialCallback,
          typename DirtyRegCallback = detail::NoopDirtyCallback>
inline NoopFilterWriteResult FilterWritePossiblySpecialRegistersHostEndian(
    uint32_t* register_values, uint32_t start_index, const uint32_t* src,
    uint32_t count, SpecialCallback&& on_special_write,
    DirtyRegCallback&& on_dirty_reg = {}) {
  return FilterWritePossiblySpecialRegisters<false>(
      register_values, start_index, src, count,
      std::forward<SpecialCallback>(on_special_write),
      std::forward<DirtyRegCallback>(on_dirty_reg));
}

// Non-mutating 4-wide host-endian dispatcher for CommandProcessor's base
// WriteRegistersHostEndian() path (used by PM4 IB cache replay across all
// backends before a backend overrides WriteRegistersHostEndian()).
//
// IMPORTANT: Does NOT pre-write `current_values` before invoking `on_write`,
// so backend overrides of WriteRegister(index, value) (e.g. VulkanCommandProcessor
// comparing `register_file_->values[index] == value` for fetch/float constants)
// observe the true previous value and invalidate derived caches accurately.
template <typename WriteCallback>
inline NoopFilterWriteResult DispatchChangedRegisters4WideHostEndian(
    const uint32_t* current_values, uint32_t start_index,
    const uint32_t* host_values, uint32_t count, WriteCallback&& on_write) {
  NoopFilterWriteResult result;
  if (!count) {
    return result;
  }

  uint32_t current = start_index;
  const uint32_t end = start_index + count;
  const uint32_t* cur_src = host_values;

  auto run_regular_until = [&](uint32_t limit) {
    if (current >= limit) {
      return;
    }
    const uint32_t seg_end = limit < end ? limit : end;
    uint32_t remaining = seg_end - current;
    uint32_t vals[4];
    while (remaining >= 4u) {
      const uint32_t diff_mask =
          detail::CompareAndMaybeStore4Wide<false, false>(
              nullptr, current_values + current, cur_src, vals);
      if (diff_mask == 0u) {
        result.skipped_count += 4u;
      } else {
        for (uint32_t lane = 0; lane < 4u; ++lane) {
          if (diff_mask & (1u << lane)) {
            const uint32_t reg_index = current + lane;
            detail::RecordChangedLane(result, reg_index);
            on_write(reg_index, vals[lane]);
          } else {
            ++result.skipped_count;
          }
        }
      }
      current += 4u;
      cur_src += 4u;
      remaining -= 4u;
    }
    while (remaining > 0u) {
      const uint32_t val = detail::LoadDword<false>(cur_src);
      if (current_values[current] != val) {
        detail::RecordChangedLane(result, current);
        on_write(current, val);
      } else {
        ++result.skipped_count;
      }
      ++current;
      ++cur_src;
      --remaining;
    }
  };

  auto run_special_until = [&](uint32_t limit) {
    if (current >= limit) {
      return;
    }
    const uint32_t seg_end = limit < end ? limit : end;
    while (current < seg_end) {
      const uint32_t val = detail::LoadDword<false>(cur_src);
      if (current_values[current] != val) {
        detail::RecordChangedLane(result, current);
      }
      on_write(current, val);
      ++result.side_effect_count;
      ++current;
      ++cur_src;
    }
  };

  if (!RangeHasMandatoryWriteSideEffect(start_index, count)) {
    run_regular_until(end);
    return result;
  }

  run_regular_until(kRegScratchReg0);
  run_special_until(kRegScratchReg7 + 1u);
  run_regular_until(kRegCoherStatusHost);
  run_special_until(kRegCoherStatusHost + 1u);
  run_regular_until(kRegDcLutRwIndex);
  run_special_until(kRegDcLut30Color + 1u);
  run_regular_until(end);

  return result;
}

}  // namespace pm4_noop
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_PM4_NOOP_FILTER_H_
