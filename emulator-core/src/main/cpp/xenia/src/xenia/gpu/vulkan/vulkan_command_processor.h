/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_VULKAN_VULKAN_COMMAND_PROCESSOR_H_
#define XENIA_GPU_VULKAN_VULKAN_COMMAND_PROCESSOR_H_

#include <algorithm>
#include <array>
#include <atomic>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <utility>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__SSSE3__)
#include <tmmintrin.h>
#endif

namespace xe {
namespace gpu {
namespace draw_cuts {

struct FloatConstantWriteResult {
  bool any_changed = false;
  bool vertex_dirty = false;
  bool pixel_dirty = false;
};

inline uint32_t LoadSwap32Unaligned(const uint32_t* ptr) {
  uint32_t raw;
  std::memcpy(&raw, ptr, sizeof(raw));
#if defined(_MSC_VER)
  return _byteswap_ulong(raw);
#else
  return __builtin_bswap32(raw);
#endif
}

// Byte-swaps big-endian guest float constant dwords into `dst` only where
// values actually differ (bitwise XOR exactness), and reports whether any
// changed vec4 is referenced by the active vertex/pixel float constant maps.
inline FloatConstantWriteResult CopySwapAndTrackFloatConstants(
    uint32_t* dst, const uint32_t* src, uint32_t start_index,
    uint32_t float_base_index, uint32_t num_registers,
    const uint64_t vertex_map[4], const uint64_t pixel_map[4],
    bool check_vertex, bool check_pixel) {
  FloatConstantWriteResult result;
  if (!num_registers) {
    return result;
  }

  auto mark_vec4_dirty = [&](uint32_t vec4_index) {
    if (vec4_index < 256u) {
      if (check_vertex &&
          (vertex_map[vec4_index >> 6] & (UINT64_C(1) << (vec4_index & 63)))) {
        result.vertex_dirty = true;
        check_vertex = false;
      }
    } else {
      const uint32_t pixel_index = vec4_index - 256u;
      if (pixel_index < 256u && check_pixel &&
          (pixel_map[pixel_index >> 6] & (UINT64_C(1) << (pixel_index & 63)))) {
        result.pixel_dirty = true;
        check_pixel = false;
      }
    }
  };

  uint32_t index = start_index;
  uint32_t remaining = num_registers;

  // Unaligned head dwords until vec4 boundary.
  while (remaining > 0 && ((index - float_base_index) & 3u) != 0) {
    const uint32_t val = LoadSwap32Unaligned(src);
    if (*dst != val) {
      *dst = val;
      result.any_changed = true;
      mark_vec4_dirty((index - float_base_index) >> 2);
    }
    ++dst;
    ++src;
    ++index;
    --remaining;
  }

  // Main vec4 (16-byte) SIMD/XOR loop.
#if defined(__SSSE3__)
  const __m128i kBswapMask =
      _mm_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
  const __m128i kZero = _mm_setzero_si128();
#endif
  uint32_t vec4_index = (index - float_base_index) >> 2;
  while (remaining >= 4u) {
    bool vec4_changed = false;
#if defined(__ARM_NEON) || defined(__aarch64__)
    const uint8x16_t raw = vld1q_u8(reinterpret_cast<const uint8_t*>(src));
    const uint32x4_t swapped = vreinterpretq_u32_u8(vrev32q_u8(raw));
    const uint32x4_t cur = vld1q_u32(dst);
    const uint64x2_t diff64 = vreinterpretq_u64_u32(veorq_u32(swapped, cur));
    if ((vgetq_lane_u64(diff64, 0) | vgetq_lane_u64(diff64, 1)) != 0) {
      vst1q_u32(dst, swapped);
      vec4_changed = true;
    }
#elif defined(__SSSE3__)
    const __m128i raw =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
    const __m128i swapped = _mm_shuffle_epi8(raw, kBswapMask);
    const __m128i cur =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(dst));
    const __m128i diff = _mm_xor_si128(swapped, cur);
    if (_mm_movemask_epi8(_mm_cmpeq_epi32(diff, kZero)) != 0xFFFF) {
      _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), swapped);
      vec4_changed = true;
    }
#else
    const uint32_t v0 = LoadSwap32Unaligned(src + 0);
    const uint32_t v1 = LoadSwap32Unaligned(src + 1);
    const uint32_t v2 = LoadSwap32Unaligned(src + 2);
    const uint32_t v3 = LoadSwap32Unaligned(src + 3);
    if (((dst[0] ^ v0) | (dst[1] ^ v1) | (dst[2] ^ v2) | (dst[3] ^ v3)) != 0) {
      dst[0] = v0;
      dst[1] = v1;
      dst[2] = v2;
      dst[3] = v3;
      vec4_changed = true;
    }
#endif
    if (vec4_changed) {
      result.any_changed = true;
      mark_vec4_dirty(vec4_index);
    }
    dst += 4;
    src += 4;
    index += 4;
    ++vec4_index;
    remaining -= 4;
  }

  // Trailing 0..3 dwords.
  while (remaining > 0) {
    const uint32_t val = LoadSwap32Unaligned(src);
    if (*dst != val) {
      *dst = val;
      result.any_changed = true;
      mark_vec4_dirty((index - float_base_index) >> 2);
    }
    ++dst;
    ++src;
    ++index;
    --remaining;
  }

  return result;
}

// Dirty-group bitmask for UpdateSystemConstantValues memoization.
constexpr uint32_t kSysConstGroupFlags = UINT32_C(1) << 0;
constexpr uint32_t kSysConstGroupIndex = UINT32_C(1) << 1;
constexpr uint32_t kSysConstGroupUcp = UINT32_C(1) << 2;
constexpr uint32_t kSysConstGroupTess = UINT32_C(1) << 3;
constexpr uint32_t kSysConstGroupPoint = UINT32_C(1) << 4;
constexpr uint32_t kSysConstGroupTextures = UINT32_C(1) << 5;
constexpr uint32_t kSysConstGroupColorExpBias = UINT32_C(1) << 6;
constexpr uint32_t kSysConstGroupFsi = UINT32_C(1) << 7;
constexpr uint32_t kSysConstGroupAll = 0xFFu;

inline uint32_t ClassifySysConstRegisterDirtyGroups(uint32_t reg_index) {
  if (reg_index >= 0x4800u && reg_index <= 0x48BFu) {
    return kSysConstGroupTextures;
  }
  if (reg_index < 0x2000u || reg_index > 0x239Fu) {
    return 0u;
  }
  if (reg_index >= 0x2388u && reg_index <= 0x239Fu) {
    // PA_CL_UCP_0_X .. PA_CL_UCP_5_W
    return kSysConstGroupUcp;
  }
  if (reg_index >= 0x2380u && reg_index <= 0x2383u) {
    // PA_SU_POLY_OFFSET_FRONT/BACK_SCALE/OFFSET (FSI)
    return kSysConstGroupFsi;
  }
  switch (reg_index) {
    case 0x2000u:  // RB_SURFACE_INFO
    case 0x2002u:  // RB_DEPTH_INFO
      return kSysConstGroupFlags | kSysConstGroupFsi;
    case 0x2001u:  // RB_COLOR_INFO
    case 0x2003u:  // RB_COLOR1_INFO
    case 0x2004u:  // RB_COLOR2_INFO
    case 0x2005u:  // RB_COLOR3_INFO
      return kSysConstGroupFlags | kSysConstGroupColorExpBias |
             kSysConstGroupFsi;
    case 0x2100u:  // VGT_MAX_VTX_INDX
    case 0x2101u:  // VGT_MIN_VTX_INDX
      return kSysConstGroupTess;
    case 0x2102u:  // VGT_INDX_OFFSET
      return kSysConstGroupIndex | kSysConstGroupTess;
    case 0x2105u:  // RB_BLEND_RED
    case 0x2106u:  // RB_BLEND_GREEN
    case 0x2107u:  // RB_BLEND_BLUE
    case 0x2108u:  // RB_BLEND_ALPHA
    case 0x210Cu:  // RB_STENCILREFMASK_BF
    case 0x210Du:  // RB_STENCILREFMASK
      return kSysConstGroupFsi;
    case 0x210Eu:  // RB_ALPHA_REF
      return kSysConstGroupFlags;
    case 0x21FCu:  // VGT_DRAW_INITIATOR
      return kSysConstGroupFlags | kSysConstGroupIndex | kSysConstGroupPoint;
    case 0x2200u:  // RB_DEPTHCONTROL
    case 0x2201u:  // RB_BLENDCONTROL0
    case 0x2203u:  // RB_COLOR_MASK
    case 0x2205u:  // PA_SU_SC_MODE_CNTL
    case 0x2209u:  // RB_BLENDCONTROL1
    case 0x220Au:  // RB_BLENDCONTROL2
    case 0x220Bu:  // RB_BLENDCONTROL3
      return kSysConstGroupFsi;
    case 0x2202u:  // RB_COLORCONTROL
      return kSysConstGroupFlags | kSysConstGroupFsi;
    case 0x2204u:  // PA_CL_CLIP_CNTL
      return kSysConstGroupUcp;
    case 0x2206u:  // PA_CL_VTE_CNTL
    case 0x2284u:  // VGT_OUTPUT_PATH_CNTL
      return kSysConstGroupFlags;
    case 0x2280u:  // PA_SU_POINT_SIZE
    case 0x2281u:  // PA_SU_POINT_MINMAX
      return kSysConstGroupPoint;
    case 0x2286u:  // VGT_HOS_MAX_TESS_LEVEL
    case 0x2287u:  // VGT_HOS_MIN_TESS_LEVEL
      return kSysConstGroupTess;
    default:
      return 0u;
  }
}

inline uint32_t ClassifySysConstRangeDirtyGroups(uint32_t start_index,
                                                 uint32_t num_registers) {
  if (!num_registers) {
    return 0u;
  }
  const uint64_t lo = start_index;
  const uint64_t hi = uint64_t(start_index) + uint64_t(num_registers);
  uint32_t mask = 0u;
  auto overlaps = [lo, hi](uint32_t a, uint32_t b_incl) {
    return lo <= b_incl && hi > a;
  };
  if (overlaps(0x4800u, 0x48BFu)) {
    mask |= kSysConstGroupTextures;
  }
  if (!overlaps(0x2000u, 0x239Fu)) {
    return mask;
  }
  if (overlaps(0x2000u, 0x2005u)) {
    mask |= kSysConstGroupFlags | kSysConstGroupFsi;
    if (overlaps(0x2001u, 0x2001u) || overlaps(0x2003u, 0x2005u)) {
      mask |= kSysConstGroupColorExpBias;
    }
  }
  if (overlaps(0x2100u, 0x2102u)) {
    mask |= kSysConstGroupTess;
    if (overlaps(0x2102u, 0x2102u)) {
      mask |= kSysConstGroupIndex;
    }
  }
  if (overlaps(0x2105u, 0x210Du)) {
    mask |= kSysConstGroupFsi;
  }
  if (overlaps(0x210Eu, 0x210Eu)) {
    mask |= kSysConstGroupFlags;
  }
  if (overlaps(0x21FCu, 0x21FCu)) {
    mask |= kSysConstGroupFlags | kSysConstGroupIndex | kSysConstGroupPoint;
  }
  if (overlaps(0x2200u, 0x220Bu)) {
    if (overlaps(0x2200u, 0x2203u) || overlaps(0x2205u, 0x2205u) ||
        overlaps(0x2209u, 0x220Bu)) {
      mask |= kSysConstGroupFsi;
    }
    if (overlaps(0x2202u, 0x2202u) || overlaps(0x2206u, 0x2206u)) {
      mask |= kSysConstGroupFlags;
    }
    if (overlaps(0x2204u, 0x2204u)) {
      mask |= kSysConstGroupUcp;
    }
  }
  if (overlaps(0x2280u, 0x2287u)) {
    if (overlaps(0x2280u, 0x2281u)) {
      mask |= kSysConstGroupPoint;
    }
    if (overlaps(0x2284u, 0x2284u)) {
      mask |= kSysConstGroupFlags;
    }
    if (overlaps(0x2286u, 0x2287u)) {
      mask |= kSysConstGroupTess;
    }
  }
  if (overlaps(0x2380u, 0x2383u)) {
    mask |= kSysConstGroupFsi;
  }
  if (overlaps(0x2388u, 0x239Fu)) {
    mask |= kSysConstGroupUcp;
  }
  return mask;
}

class SystemConstantsDirtyTracker {
 public:
  void InvalidateAll() { dirty_mask_ = kSysConstGroupAll; }
  void ClearAll() { dirty_mask_ = 0u; }
  void MarkDirty(uint32_t groups) { dirty_mask_ |= groups; }
  void NoteRegisterWrite(uint32_t reg_index) {
    dirty_mask_ |= ClassifySysConstRegisterDirtyGroups(reg_index);
  }
  void NoteRegisterRangeWrite(uint32_t start_index, uint32_t num_registers) {
    dirty_mask_ |=
        ClassifySysConstRangeDirtyGroups(start_index, num_registers);
  }
  uint32_t dirty_mask() const { return dirty_mask_; }

 private:
  uint32_t dirty_mask_ = kSysConstGroupAll;
};

// Lock-free atomic epoch write-watch cache for PM4_IM_LOAD shader loads.
// Guest CPU threads run OnMemoryInvalidated() inside the fault handler with
// zero locks or heap allocations; the GPU command thread calls Lookup(),
// BeginArmAndSnapshot(), and CommitIfUnchanged() exclusively.
constexpr uint32_t kShaderWatchPhysicalMemoryMask = 0x1FFFFFFFu;
constexpr uint32_t kShaderWatchPhysicalMemorySize = 512u * 1024u * 1024u;
constexpr uint32_t kShaderWatchPageShift = 12u;
constexpr uint32_t kShaderWatchPageCount =
    kShaderWatchPhysicalMemorySize >> kShaderWatchPageShift;
constexpr uint32_t kShaderWatchMaxPagesPerShader = 8u;
constexpr size_t kShaderWatchMaxEntries = 4096u;

struct ShaderWatchSnapshot {
  uint32_t phys_address = 0;
  uint32_t byte_length = 0;
  uint32_t first_page = 0;
  uint32_t page_count = 0;
  uint64_t global_epoch = 0;
  std::array<uint32_t, kShaderWatchMaxPagesPerShader> page_epochs{};
};

template <typename ShaderT>
class ShaderLoadWatchCache {
 public:
  ShaderLoadWatchCache()
      : page_epochs_(std::make_unique<std::atomic<uint32_t>[]>(
            kShaderWatchPageCount)),
        armed_pages_(std::make_unique<std::atomic<uint64_t>[]>(
            kShaderWatchPageCount / 64u)) {
    for (uint32_t i = 0; i < kShaderWatchPageCount; ++i) {
      page_epochs_[i].store(0u, std::memory_order_relaxed);
    }
    for (uint32_t i = 0; i < kShaderWatchPageCount / 64u; ++i) {
      armed_pages_[i].store(0u, std::memory_order_relaxed);
    }
  }

  // Called from Guest CPU fault handler (or trace playback): wait-free,
  // lock-free, no allocations. Skips non-shader pages via the 16 KiB bitset.
  // Returns the safe unwatch range (start, length) for Memory's fault handler:
  // if another armed shader page exists inside Memory's 4 MiB max unwatch
  // excess window outside [physical_address, physical_address + length), clamps
  // the unwatch range to (physical_address, length) so adjacent cached shader
  // pages are never silently unprotected without bumping their epochs.
  std::pair<uint32_t, uint32_t> OnMemoryInvalidatedForUnwatch(
      uint32_t physical_address, uint32_t length, bool exact_range) {
    if (!length) {
      return {0u, UINT32_MAX};
    }
    const uint32_t phys = physical_address & kShaderWatchPhysicalMemoryMask;
    const uint32_t first_page = phys >> kShaderWatchPageShift;
    if (first_page >= kShaderWatchPageCount) {
      return {0u, UINT32_MAX};
    }
    const uint64_t end_excl = std::min<uint64_t>(
        uint64_t(phys) + uint64_t(length), kShaderWatchPhysicalMemorySize);
    if (end_excl <= phys) {
      return {0u, UINT32_MAX};
    }
    const uint32_t last_page =
        static_cast<uint32_t>((end_excl - 1u) >> kShaderWatchPageShift);
    for (uint32_t page = first_page; page <= last_page; ++page) {
      const uint64_t word =
          armed_pages_[page >> 6].load(std::memory_order_acquire);
      if (word & (UINT64_C(1) << (page & 63u))) {
        page_epochs_[page].fetch_add(1u, std::memory_order_acq_rel);
      }
    }
    if (exact_range) {
      return {physical_address, length};
    }
    // Check Memory's kMaxUnwatchExcess (4 MiB) window for any OTHER armed
    // shader page; if present, clamp unwatch to the invalidated range so that
    // sibling shader pages keep PROT_READ.
    constexpr uint32_t kMaxUnwatchExcess = 4u * 1024u * 1024u;
    const uint32_t win_first_addr = phys & ~(kMaxUnwatchExcess - 1u);
    const uint32_t win_last_addr = std::min<uint32_t>(
        static_cast<uint32_t>((end_excl - 1u) | (kMaxUnwatchExcess - 1u)),
        kShaderWatchPhysicalMemorySize - 1u);
    const uint32_t win_first_word = (win_first_addr >> kShaderWatchPageShift) >> 6;
    const uint32_t win_last_word = (win_last_addr >> kShaderWatchPageShift) >> 6;
    for (uint32_t w = win_first_word; w <= win_last_word; ++w) {
      uint64_t armed = armed_pages_[w].load(std::memory_order_acquire);
      if (!armed) {
        continue;
      }
      const uint32_t word_base_page = w << 6;
      const uint32_t word_end_page = word_base_page + 63u;
      if (first_page <= word_end_page && last_page >= word_base_page) {
        const uint32_t lo_bit =
            first_page > word_base_page ? first_page - word_base_page : 0u;
        const uint32_t hi_bit =
            last_page < word_end_page ? last_page - word_base_page : 63u;
        const uint32_t span = hi_bit - lo_bit + 1u;
        const uint64_t invalidated_mask =
            span >= 64u ? ~UINT64_C(0)
                        : (((UINT64_C(1) << span) - 1u) << lo_bit);
        armed &= ~invalidated_mask;
      }
      if (armed != 0u) {
        return {physical_address, length};
      }
    }
    return {0u, UINT32_MAX};
  }

  void OnMemoryInvalidated(uint32_t physical_address, uint32_t length) {
    OnMemoryInvalidatedForUnwatch(physical_address, length, true);
  }

  // GPU thread only: invalidates all cached entries immediately.
  void InvalidateAll() {
    global_epoch_.fetch_add(1u, std::memory_order_acq_rel);
    entries_.clear();
  }

  // GPU thread only: returns cached shader pointer if all watched pages still
  // match the snapshot epochs recorded before the shader was hashed.
  ShaderT* Lookup(uint32_t shader_type, uint32_t guest_address,
                  uint32_t dword_count) const {
    if (!dword_count) {
      return nullptr;
    }
    const uint32_t phys = guest_address & kShaderWatchPhysicalMemoryMask;
    auto it = entries_.find(MakeKey(shader_type, phys, dword_count));
    if (it == entries_.end()) {
      return nullptr;
    }
    const Entry& e = it->second;
    if (e.global_epoch != global_epoch_.load(std::memory_order_acquire)) {
      return nullptr;
    }
    for (uint32_t i = 0; i < e.page_count; ++i) {
      if (page_epochs_[e.first_page + i].load(std::memory_order_acquire) !=
          e.page_epochs[i]) {
        return nullptr;
      }
    }
    return e.shader;
  }

  // GPU thread only: prepares a watch snapshot BEFORE arming OS write-watch
  // and hashing microcode. Returns false if the shader range cannot be cached.
  bool BeginArmAndSnapshot(uint32_t guest_address, uint32_t dword_count,
                           ShaderWatchSnapshot& out) {
    if (!dword_count) {
      return false;
    }
    const uint32_t phys = guest_address & kShaderWatchPhysicalMemoryMask;
    const uint64_t bytes = uint64_t(dword_count) * sizeof(uint32_t);
    if (uint64_t(phys) + bytes > kShaderWatchPhysicalMemorySize) {
      return false;
    }
    const uint32_t first_page = phys >> kShaderWatchPageShift;
    const uint32_t last_page =
        static_cast<uint32_t>((uint64_t(phys) + bytes - 1u) >>
                              kShaderWatchPageShift);
    const uint32_t page_count = last_page - first_page + 1u;
    if (page_count > kShaderWatchMaxPagesPerShader) {
      return false;
    }
    out.phys_address = phys;
    out.byte_length = static_cast<uint32_t>(bytes);
    out.first_page = first_page;
    out.page_count = page_count;
    // Mark pages armed BEFORE reading epochs so any concurrent CPU fault
    // observes the armed bit and bumps the page epoch.
    for (uint32_t i = 0; i < page_count; ++i) {
      const uint32_t p = first_page + i;
      armed_pages_[p >> 6].fetch_or(UINT64_C(1) << (p & 63u),
                                    std::memory_order_acq_rel);
    }
    out.global_epoch = global_epoch_.load(std::memory_order_acquire);
    for (uint32_t i = 0; i < page_count; ++i) {
      out.page_epochs[i] =
          page_epochs_[first_page + i].load(std::memory_order_acquire);
    }
    return true;
  }

  // GPU thread only: commits `shader` into the watch cache ONLY if zero
  // invalidations happened across [BeginArmAndSnapshot .. now].
  bool CommitIfUnchanged(uint32_t shader_type, uint32_t guest_address,
                         uint32_t dword_count,
                         const ShaderWatchSnapshot& snap, ShaderT* shader) {
    if (!shader || !snap.page_count) {
      return false;
    }
    if (global_epoch_.load(std::memory_order_acquire) != snap.global_epoch) {
      return false;
    }
    for (uint32_t i = 0; i < snap.page_count; ++i) {
      if (page_epochs_[snap.first_page + i].load(std::memory_order_acquire) !=
          snap.page_epochs[i]) {
        return false;
      }
    }
    if (entries_.size() >= kShaderWatchMaxEntries) {
      entries_.clear();
    }
    const uint32_t phys = guest_address & kShaderWatchPhysicalMemoryMask;
    Entry& e = entries_[MakeKey(shader_type, phys, dword_count)];
    e.shader = shader;
    e.first_page = snap.first_page;
    e.page_count = snap.page_count;
    e.global_epoch = snap.global_epoch;
    e.page_epochs = snap.page_epochs;
    return true;
  }

 private:
  struct Entry {
    ShaderT* shader = nullptr;
    uint32_t first_page = 0;
    uint32_t page_count = 0;
    uint64_t global_epoch = 0;
    std::array<uint32_t, kShaderWatchMaxPagesPerShader> page_epochs{};
  };

  static uint64_t MakeKey(uint32_t shader_type, uint32_t phys_address,
                          uint32_t dword_count) {
    // phys_address fits in 29 bits (512 MiB), dword_count in 16 bits, type in 2 bits.
    return (uint64_t(shader_type & 0x3u) << 48) |
           (uint64_t(dword_count & 0xFFFFu) << 32) | uint64_t(phys_address);
  }

  std::unique_ptr<std::atomic<uint32_t>[]> page_epochs_;
  std::unique_ptr<std::atomic<uint64_t>[]> armed_pages_;
  std::atomic<uint64_t> global_epoch_{1};
  std::unordered_map<uint64_t, Entry> entries_;
};

// VK_KHR_push_descriptor driver blacklist + Eden maxPushDescriptors clamp.
constexpr uint32_t kMaxPushDescriptorsClamp = 32u;

struct PushDescriptorPolicyResult {
  bool enabled = false;
  uint32_t clamped_max_push_descriptors = 0;
};

inline PushDescriptorPolicyResult EvaluatePushDescriptorPolicy(
    bool cvar_enabled, bool extension_enabled, bool function_loaded,
    uint32_t vendor_id, uint32_t driver_id,
    uint32_t reported_max_push_descriptors, uint32_t required_bindings) {
  PushDescriptorPolicyResult result;
  if (!cvar_enabled || !extension_enabled || !function_loaded) {
    return result;
  }
  constexpr uint32_t kVendorIdArm = 0x13B5u;
  constexpr uint32_t kVendorIdQualcomm = 0x5143u;
  constexpr uint32_t kDriverIdQualcommProprietary = 8u;
  constexpr uint32_t kDriverIdArmProprietary = 9u;

  // Strict blacklist: Qualcomm proprietary Adreno blob and Arm Mali proprietary
  // blob (including pre-1.2 driverID == 0). Mesa Turnip (18) and PanVK /
  // malisx2 (20) are explicitly permitted.
  const bool is_qualcomm_proprietary =
      driver_id == kDriverIdQualcommProprietary ||
      (driver_id == 0u && vendor_id == kVendorIdQualcomm);
  const bool is_arm_proprietary =
      driver_id == kDriverIdArmProprietary ||
      (driver_id == 0u && vendor_id == kVendorIdArm);
  if (is_qualcomm_proprietary || is_arm_proprietary) {
    return result;
  }

  // Eden lesson: clamp reported maxPushDescriptors to a safe upper bound.
  const uint32_t clamped =
      std::min(reported_max_push_descriptors, kMaxPushDescriptorsClamp);
  if (clamped < required_bindings || required_bindings == 0u) {
    return result;
  }
  result.enabled = true;
  result.clamped_max_push_descriptors = clamped;
  return result;
}

// Asynchronous ring state machine for small UMA resolve readbacks (<= 4 KiB).
constexpr uint32_t kSmallReadbackMaxBytes = 4096u;
constexpr uint32_t kSmallReadbackRingSlotCount = 64u;

inline bool IsSmallReadbackRingEligible(uint32_t length_bytes) {
  return length_bytes > 0u && length_bytes <= kSmallReadbackMaxBytes;
}

struct SmallReadbackRingSlot {
  uint64_t submission = 0;
  uint64_t generation = 0;
  uint32_t guest_address = 0;
  uint32_t length = 0;
  uint32_t slot_index = 0;
};

class SmallReadbackRingState {
 public:
  bool empty() const { return count_ == 0u; }
  bool full() const { return count_ >= kSmallReadbackRingSlotCount; }
  uint32_t pending_count() const { return count_; }
  uint64_t current_generation() const { return generation_; }

  uint64_t OldestSubmission() const {
    return count_ > 0u ? slots_[head_].submission : 0u;
  }

  uint64_t NewestSubmission() const {
    if (count_ == 0u) {
      return 0u;
    }
    const uint32_t last_idx =
        (tail_ + kSmallReadbackRingSlotCount - 1u) %
        kSmallReadbackRingSlotCount;
    return slots_[last_idx].submission;
  }

  // Marks all currently queued slots as discarded (e.g. on InvalidateGpuMemory)
  // WITHOUT resetting head_/tail_/count_, so in-flight GPU copies still retire
  // before their ring slots can be reused (preventing WAW on the ring buffer).
  void InvalidatePending() { ++generation_; }

  bool TryEnqueue(uint64_t submission, uint32_t guest_address, uint32_t length,
                  uint32_t& slot_index_out) {
    if (full() || !IsSmallReadbackRingEligible(length)) {
      return false;
    }
    const uint32_t idx = tail_;
    SmallReadbackRingSlot& slot = slots_[idx];
    slot.submission = submission;
    slot.generation = generation_;
    slot.guest_address = guest_address;
    slot.length = length;
    slot.slot_index = idx;
    tail_ = (tail_ + 1u) % kSmallReadbackRingSlotCount;
    ++count_;
    slot_index_out = idx;
    return true;
  }

  template <typename Fn>
  uint32_t DrainCompleted(uint64_t completed_submission, Fn&& on_slot) {
    uint32_t drained = 0u;
    while (count_ > 0u && slots_[head_].submission <= completed_submission) {
      const SmallReadbackRingSlot slot = slots_[head_];
      head_ = (head_ + 1u) % kSmallReadbackRingSlotCount;
      --count_;
      if (slot.generation == generation_) {
        on_slot(slot);
      }
      ++drained;
    }
    return drained;
  }

  void Reset() {
    head_ = 0u;
    tail_ = 0u;
    count_ = 0u;
    generation_ = 1u;
  }

 private:
  std::array<SmallReadbackRingSlot, kSmallReadbackRingSlotCount> slots_{};
  uint32_t head_ = 0u;
  uint32_t tail_ = 0u;
  uint32_t count_ = 0u;
  uint64_t generation_ = 1u;
};

// Guards fixed-size per-submission Vulkan timestamp query/readback rings
// (frame/resolve/pass pools, 32 slots each) against wraparound and reuse
// hazards when bursts of submissions outpace GPU completion or CPU consumption.
template <uint32_t kSlotCount>
class TimestampSlotRingGuard {
 public:
  static constexpr uint32_t kInvalidSlot = UINT32_MAX;

  void Reset() {
    slot_submission_.fill(0u);
    dropped_count_ = 0u;
  }

  // Returns slot in [0, kSlotCount) if safe to reset/write for
  // `submission_index`, or kInvalidSlot if the slot may still be in flight on
  // the GPU or unread in vk_submit_times_.
  uint32_t TryAcquireSlot(uint64_t submission_index,
                          uint64_t completed_submission,
                          uint64_t oldest_unread_submission = 0u) {
    if (submission_index == 0u) {
      ++dropped_count_;
      return kInvalidSlot;
    }
    if (submission_index > completed_submission &&
        (submission_index - completed_submission) >= kSlotCount) {
      ++dropped_count_;
      return kInvalidSlot;
    }
    const uint32_t slot = uint32_t(submission_index % kSlotCount);
    const uint64_t prev_sub = slot_submission_[slot];
    if (prev_sub != 0u) {
      if (prev_sub > completed_submission) {
        ++dropped_count_;
        return kInvalidSlot;
      }
      if (oldest_unread_submission != 0u &&
          prev_sub >= oldest_unread_submission) {
        ++dropped_count_;
        return kInvalidSlot;
      }
    }
    return slot;
  }

  void CommitSlot(uint32_t slot, uint64_t submission_index) {
    if (slot < kSlotCount) {
      slot_submission_[slot] = submission_index;
    }
  }

  uint64_t slot_submission(uint32_t slot) const {
    return slot < kSlotCount ? slot_submission_[slot] : 0u;
  }

  uint64_t dropped_count() const { return dropped_count_; }

 private:
  std::array<uint64_t, kSlotCount> slot_submission_{};
  uint64_t dropped_count_ = 0u;
};

// Periodic nice reapplication controller for the GPU Submit Worker thread
// (reapplies every 1s to survive Android cgroup/ActivityManager resets without
// spamming logs).
class WorkerNiceReapplyTracker {
 public:
  static constexpr uint64_t kReapplyIntervalNs = 1000000000ull;

  struct CheckResult {
    bool should_apply = false;
    bool valid_range = true;
    bool should_log_invalid = false;
  };

  CheckResult Check(int32_t desired_nice, uint64_t now_ns) {
    CheckResult res{};
    if (desired_nice < -20 || desired_nice > 19) {
      res.valid_range = false;
      if (last_logged_nice_ != desired_nice) {
        last_logged_nice_ = desired_nice;
        res.should_log_invalid = true;
      }
      last_desired_nice_ = desired_nice;
      return res;
    }
    const bool desired_changed =
        !initialized_ || desired_nice != last_desired_nice_;
    const bool period_elapsed =
        initialized_ && desired_nice != 0 && now_ns >= last_apply_ns_ &&
        (now_ns - last_apply_ns_) >= kReapplyIntervalNs;
    if (!desired_changed && !period_elapsed) {
      return res;
    }
    initialized_ = true;
    last_desired_nice_ = desired_nice;
    last_apply_ns_ = now_ns;
    res.should_apply = true;
    return res;
  }

  bool already_logged(int32_t desired_nice) const {
    return last_logged_nice_ == desired_nice;
  }

  bool ShouldLogOutcome(int32_t desired_nice) {
    if (last_logged_nice_ != desired_nice) {
      last_logged_nice_ = desired_nice;
      return true;
    }
    return false;
  }

 private:
  bool initialized_ = false;
  int32_t last_desired_nice_ = 0;
  int32_t last_logged_nice_ = INT32_MIN;
  uint64_t last_apply_ns_ = 0;
};

// Decision policy for VulkanCommandProcessor::PrepareForWait() covering
// pending non-memexport host writes (small resolve readbacks, ZPD, VIZ).
struct PrepareForWaitFlushAction {
  bool close_open_submission = false;
  bool drain_submit_worker = false;
  bool poll_completion = false;
  uint64_t await_submission = 0;
};

inline PrepareForWaitFlushAction EvaluatePrepareForWaitFlush(
    bool device_lost, bool submission_open, bool async_submit_worker_active,
    uint64_t current_submission, uint64_t submitted_submission,
    uint64_t completed_submission, bool has_pending_small_readbacks,
    uint64_t newest_small_readback_submission, bool has_pending_zpd_batch,
    bool has_in_flight_zpd_resolves, uint64_t newest_zpd_resolve_submission,
    bool has_pending_zpd_retire, bool has_pending_viz_resolves) {
  PrepareForWaitFlushAction action{};
  if (device_lost) {
    return action;
  }
  const bool has_non_memexport_pending =
      has_pending_small_readbacks || has_pending_zpd_batch ||
      has_in_flight_zpd_resolves || has_pending_zpd_retire ||
      has_pending_viz_resolves;
  if (!has_non_memexport_pending) {
    return action;
  }
  const bool pending_in_open_submission =
      submission_open &&
      ((has_pending_small_readbacks &&
        newest_small_readback_submission >= current_submission) ||
       has_pending_zpd_batch ||
       (has_in_flight_zpd_resolves &&
        newest_zpd_resolve_submission >= current_submission));
  action.close_open_submission = pending_in_open_submission;

  const uint64_t max_pending_submission = std::max(
      has_pending_small_readbacks ? newest_small_readback_submission : 0u,
      has_in_flight_zpd_resolves ? newest_zpd_resolve_submission : 0u);
  const uint64_t effective_target =
      pending_in_open_submission
          ? std::max(max_pending_submission, current_submission)
          : max_pending_submission;
  const bool retire_or_viz_unsubmitted =
      (has_pending_zpd_retire || has_pending_viz_resolves) &&
      current_submission > 1 && (current_submission - 1) > submitted_submission;
  if (async_submit_worker_active &&
      (pending_in_open_submission || effective_target > submitted_submission ||
       retire_or_viz_unsubmitted)) {
    action.drain_submit_worker = true;
  }
  if (has_pending_small_readbacks &&
      newest_small_readback_submission > completed_submission) {
    action.await_submission = newest_small_readback_submission;
  }
  action.poll_completion =
      action.await_submission == 0 &&
      (has_pending_small_readbacks || has_in_flight_zpd_resolves ||
       has_pending_zpd_retire || has_pending_viz_resolves);
  return action;
}

}  // namespace draw_cuts
}  // namespace gpu
}  // namespace xe

#if !defined(XENIA_GPU_DRAW_CUTS_PURE_LOGIC_ONLY)

#include <deque>
#include <filesystem>
#include <functional>
#include <map>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include "xenia/base/assert.h"
#include "xenia/base/hash.h"
#include "../../../../../xe_wave5_perf_logic.h"
#include "xenia/gpu/command_processor.h"
#include "xenia/gpu/draw_util.h"
#include "xenia/gpu/gpu_flags.h"
#include "xenia/gpu/registers.h"
#include "xenia/gpu/spirv_shader_translator.h"
#include "xenia/gpu/vulkan/deferred_command_buffer.h"
#include "xenia/gpu/vulkan/thermal_vrs_controller.h"
#include "xenia/gpu/vulkan/vulkan_graphics_system.h"
#include "xenia/gpu/vulkan/vulkan_pipeline_cache.h"
#include "xenia/gpu/vulkan/vulkan_primitive_processor.h"
#include "xenia/gpu/vulkan/vulkan_render_target_cache.h"
#include "xenia/gpu/vulkan/vulkan_shader.h"
#include "xenia/gpu/vulkan/vulkan_shared_memory.h"
#include "xenia/gpu/vulkan/vulkan_submit_worker.h"
#include "xenia/gpu/vulkan/vulkan_texture_cache.h"
#include "xenia/gpu/vulkan/vulkan_zpd_query_pool.h"
#include "xenia/gpu/xenos.h"
#include "xenia/kernel/kernel_state.h"
#include "xenia/ui/vulkan/linked_type_descriptor_set_allocator.h"
#include "xenia/ui/vulkan/vulkan_descriptor_pool_chain.h"
#include "xenia/ui/vulkan/vulkan_gpu_completion_timeline.h"
#include "xenia/ui/vulkan/vulkan_presenter.h"
#include "xenia/ui/vulkan/vulkan_provider.h"
#include "xenia/ui/vulkan/vulkan_upload_buffer_pool.h"

DECLARE_bool(log_gpu_frame_time_breakdown);

namespace xe {
namespace gpu {
namespace vulkan {

class VulkanCommandProcessor final : public CommandProcessor {
 protected:
#define OVERRIDING_BASE_CMDPROCESSOR
#include "../pm4_command_processor_declare.h"
#undef OVERRIDING_BASE_CMDPROCESSOR
 public:
  // Single-descriptor layouts for use within a single frame.
  enum class SingleTransientDescriptorLayout {
    kStorageBufferCompute,
    kStorageBufferFragment,
    // Uniform buffer at binding 1 for shaders that keep binding 0 as push
    // constants for D3D-style root constants.
    kUniformBufferComputeB1,
    // Storage image at binding 0 for the resolve-to-texture fragment variant.
    kStorageImageFragment,
    // Storage image at binding 0 for texture loads straight into the image.
    kStorageImageCompute,
    kCount,
  };

  class ScratchBufferAcquisition {
   public:
    explicit ScratchBufferAcquisition() = default;
    explicit ScratchBufferAcquisition(VulkanCommandProcessor& command_processor,
                                      VkBuffer buffer,
                                      VkPipelineStageFlags stage_mask,
                                      VkAccessFlags access_mask)
        : command_processor_(&command_processor),
          buffer_(buffer),
          stage_mask_(stage_mask),
          access_mask_(access_mask) {}

    ScratchBufferAcquisition(const ScratchBufferAcquisition& acquisition) =
        delete;
    ScratchBufferAcquisition& operator=(
        const ScratchBufferAcquisition& acquisition) = delete;

    ScratchBufferAcquisition(ScratchBufferAcquisition&& acquisition) {
      command_processor_ = acquisition.command_processor_;
      buffer_ = acquisition.buffer_;
      stage_mask_ = acquisition.stage_mask_;
      access_mask_ = acquisition.access_mask_;
      acquisition.command_processor_ = nullptr;
      acquisition.buffer_ = VK_NULL_HANDLE;
      acquisition.stage_mask_ = 0;
      acquisition.access_mask_ = 0;
    }
    ScratchBufferAcquisition& operator=(
        ScratchBufferAcquisition&& acquisition) {
      if (this == &acquisition) {
        return *this;
      }
      command_processor_ = acquisition.command_processor_;
      buffer_ = acquisition.buffer_;
      stage_mask_ = acquisition.stage_mask_;
      access_mask_ = acquisition.access_mask_;
      acquisition.command_processor_ = nullptr;
      acquisition.buffer_ = VK_NULL_HANDLE;
      acquisition.stage_mask_ = 0;
      acquisition.access_mask_ = 0;
      return *this;
    }

    ~ScratchBufferAcquisition() {
      if (buffer_ != VK_NULL_HANDLE) {
        assert_true(command_processor_->scratch_buffer_used_);
        assert_true(command_processor_->scratch_buffer_ == buffer_);
        command_processor_->scratch_buffer_last_stage_mask_ = stage_mask_;
        command_processor_->scratch_buffer_last_access_mask_ = access_mask_;
        command_processor_->scratch_buffer_last_usage_submission_ =
            command_processor_->GetCurrentSubmission();
        command_processor_->scratch_buffer_used_ = false;
      }
    }

    // VK_NULL_HANDLE if failed to acquire or if moved.
    VkBuffer buffer() const { return buffer_; }

    VkPipelineStageFlags GetStageMask() const { return stage_mask_; }
    VkPipelineStageFlags SetStageMask(VkPipelineStageFlags new_stage_mask) {
      VkPipelineStageFlags old_stage_mask = stage_mask_;
      stage_mask_ = new_stage_mask;
      return old_stage_mask;
    }
    VkAccessFlags GetAccessMask() const { return access_mask_; }
    VkAccessFlags SetAccessMask(VkAccessFlags new_access_mask) {
      VkAccessFlags old_access_mask = access_mask_;
      access_mask_ = new_access_mask;
      return old_access_mask;
    }

   private:
    VulkanCommandProcessor* command_processor_ = nullptr;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkPipelineStageFlags stage_mask_ = 0;
    VkAccessFlags access_mask_ = 0;
  };

  VulkanCommandProcessor(VulkanGraphicsSystem* graphics_system,
                         kernel::KernelState* kernel_state);
  ~VulkanCommandProcessor();

  void ClearCaches() override;
  void InvalidateGpuMemory() override;
  void WaitUntilGpuIdleForBench() override;

  void TracePlaybackWroteMemory(uint32_t base_ptr, uint32_t length) override;

  void InitializeShaderStorage(
      const std::filesystem::path& cache_root, uint32_t title_id, bool blocking,
      std::function<void()> completion_callback = nullptr) override;

  void RestoreRegisters(uint32_t first_register,
                        const uint32_t* register_values,
                        uint32_t register_count,
                        bool execute_callbacks) override;
  void RestoreEdramSnapshot(const void* snapshot) override;

  void PrepareForWait() override;
  void PollCompletedSubmission() override;

  ui::vulkan::VulkanDevice* GetVulkanDevice() const {
    return static_cast<const ui::vulkan::VulkanProvider*>(
               graphics_system_->provider())
        ->vulkan_device();
  }

  Memory* memory() const { return memory_; }

  // Guest-address-aware shader load hook invoked by PM4_IM_LOAD so the
  // pipeline cache can hit its write-watch cache before hashing microcode.
  Shader* LoadShaderFromGuestAddress(xenos::ShaderType shader_type,
                                     uint32_t guest_physical_address,
                                     const uint32_t* host_address,
                                     uint32_t dword_count);

  // Returns the deferred drawing command list for the currently open
  // submission (points into the active SubmitChunk slot on the Recorder).
  DeferredCommandBuffer& deferred_command_buffer() {
    assert_true(submission_open_);
    return submit_chunks_[active_chunk_index_]->main_dcb;
  }

  // Deferred commands replayed at the head of the submission's command
  // buffer, before deferred_command_buffer() - holds shared-memory uploads
  // hoisted out of render passes.
  DeferredCommandBuffer& deferred_setup_command_buffer() {
    assert_true(submission_open_);
    return submit_chunks_[active_chunk_index_]->setup_dcb;
  }

  bool submission_open() const { return submission_open_; }
  bool in_render_pass() const { return in_render_pass_; }
  // GPU timing of work outside render passes and resolves
  // (log_gpu_frame_time_breakdown_passes), in the per-pass timestamp ring and
  // logged as VkMiscTime lines. A region must not begin a render pass;
  // OpenMiscTimestamp fails inside one.
  enum class MiscTimestampKind : uint32_t {
    // Texture load: guest format in bits 16:21, ceil(log2(texels)) in bits
    // 8:12, and the kMiscTimestampTexture* flags.
    kTextureLoad = 1,
    // The setup command buffer (hoisted shared memory uploads).
    kSetupCommands = 2,
    // The copy dispatches of a resolve, from the end of their barriers (the
    // render pass break, source layout transitions) to the last dispatch:
    // height / 8 in bits 0:7 and width / 8 in bits 8:15 (both saturated),
    // depth in bit 16, source xenos::MsaaSamples in bits 17:18, source format
    // in bits 19:22, and bit 23 for the direct host path (otherwise the copy
    // from the EDRAM buffer).
    kResolveCopyDispatch = 3,
  };
  static constexpr uint32_t kMiscTimestampKeyBit = UINT32_C(1) << 30;
  static constexpr uint32_t kMiscTimestampTextureGpuWritten = UINT32_C(1) << 7;
  static constexpr uint32_t kMiscTimestampTextureBase = UINT32_C(1) << 6;
  static constexpr uint32_t kMiscTimestampTextureMips = UINT32_C(1) << 5;
  // The copy of the untiled texels from the scratch buffer to the image (the
  // load's second half; without it, the untiling dispatches).
  static constexpr uint32_t kMiscTimestampTextureCopy = UINT32_C(1) << 4;
  static constexpr uint32_t MakeMiscTimestampKey(MiscTimestampKind kind,
                                                 uint32_t payload) {
    return kMiscTimestampKeyBit | (uint32_t(kind) << 24) |
           (payload & 0xFFFFFF);
  }
  bool misc_timestamps_enabled() const {
    return pass_timestamp_mapping_ != nullptr;
  }
  // Work counted per submission for VkSlowSubmission (a submission whose GPU
  // time passes vulkan_log_slow_submission_ms, with the frame timestamps).
  void NoteTextureLoad(uint64_t texels) {
    ++submission_work_.texture_loads;
    submission_work_.texture_load_texels += texels;
  }
  void NoteSharedMemoryUpload(uint64_t bytes) {
    submission_work_.upload_bytes += bytes;
  }
  bool OpenMiscTimestamp(uint32_t key);
  void CloseMiscTimestamp();
  // Called by the render target cache between the copy and the clear of a
  // resolve, splitting its VkResolveTime into the two.
  void MarkResolveCopyEnd();
  void FlushDeferredResolveClearsIfPending() {
    if (render_target_cache_ &&
        render_target_cache_->HasPendingDeferredResolveClears() &&
        BeginSubmission(true)) {
      render_target_cache_->FlushAllDeferredResolveClears();
    }
  }
  // debug.xendroid.pm4_bin_trace sequence note from the render target cache.
  void BinTraceNoteIfActive(const char* tag, uint32_t a, uint32_t b) {
    if (bin_trace_.frames_left) {
      BinTraceNote(tag, a, b);
    }
  }
  // Draws per <vertex shader hash, pixel shader hash, pass WxH> during
  // pm4_bin_trace frames, logged as PipeUse lines at the swap.
  struct PipelineUse {
    uint32_t draws = 0;
    uint64_t vertices = 0;
    // Pixel shader texture bindings summed over the draws, by the component
    // signs of each texture: all unsigned, any signed, any biased, any gamma
    // (a texture counts in the last of these that applies).
    uint32_t texture_signs[4] = {};
    // Draws with a pixel shader by alpha handling: specialized without alpha
    // operations (kNoAlphaTests), alpha test only, alpha to coverage, and
    // generic for other reasons (no color 0 write, depth modifiers).
    uint32_t alpha_modes[4] = {};
  };
  std::map<std::tuple<uint64_t, uint64_t, uint32_t>, PipelineUse>
      pipeline_use_;
  uint64_t GetCurrentSubmission() const {
    return completion_timeline_.GetRecordedSubmission();
  }
  uint64_t GetCompletedSubmission() const override {
    return completion_timeline_.GetCompletedSubmissionFromLastUpdate();
  }

  // Sparse binds are:
  // - In a single submission, all submitted in one vkQueueBindSparse.
  // - Sent to the queue without waiting for a semaphore.
  // Thus, multiple sparse binds between the completed and the current
  // submission, and within one submission, must not touch any overlapping
  // memory regions.
  void SparseBindBuffer(VkBuffer buffer, uint32_t bind_count,
                        const VkSparseMemoryBind* binds,
                        VkPipelineStageFlags wait_stage_mask);

  uint64_t GetCurrentFrame() const { return frame_current_; }
  uint64_t GetCompletedFrame() const { return frame_completed_; }

  // Submission must be open to insert barriers. If no pipeline stages access
  // the resource in a synchronization scope, the stage masks should be 0 (top /
  // bottom of pipe should be specified only if explicitly needed). Returning
  // true if the barrier has actually been inserted and not dropped.
  bool PushBufferMemoryBarrier(
      VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size,
      VkPipelineStageFlags src_stage_mask, VkPipelineStageFlags dst_stage_mask,
      VkAccessFlags src_access_mask, VkAccessFlags dst_access_mask,
      uint32_t src_queue_family_index = VK_QUEUE_FAMILY_IGNORED,
      uint32_t dst_queue_family_index = VK_QUEUE_FAMILY_IGNORED,
      bool skip_if_equal = true);
  bool PushImageMemoryBarrier(
      VkImage image, const VkImageSubresourceRange& subresource_range,
      VkPipelineStageFlags src_stage_mask, VkPipelineStageFlags dst_stage_mask,
      VkAccessFlags src_access_mask, VkAccessFlags dst_access_mask,
      VkImageLayout old_layout, VkImageLayout new_layout,
      uint32_t src_queue_family_index = VK_QUEUE_FAMILY_IGNORED,
      uint32_t dst_queue_family_index = VK_QUEUE_FAMILY_IGNORED,
      bool skip_if_equal = true);
  // Returns whether any barriers have been submitted - if true is returned, the
  // render pass will also be closed.
  bool SubmitBarriers(bool force_end_render_pass);

  // If [base_bytes, base_bytes + size_bytes) holds memexport output (which
  // under two-buffer routing lives in the host-imported buffer aliasing guest
  // RAM), copies it into the device-local buffer so a following device-buffer
  // read - a texture load - sees it. No-op for non-memexport ranges (already in
  // the device buffer) and when the host buffer is unavailable. Called by the
  // texture cache before loading a texture. Ends the render pass.
  // Returns whether it copied, so an upload of the same range can be skipped.
  bool EnsureMemexportRangeInDeviceBuffer(uint32_t base_bytes,
                                          uint32_t size_bytes);

  // If not started yet, begins a render pass from the render target cache.
  // Submission must be open.
  void SubmitBarriersAndEnterRenderTargetCacheRenderPass(
      VkRenderPass render_pass,
      const VulkanRenderTargetCache::Framebuffer* framebuffer);
  // Overload for transfer operations with dynamic rendering.
  // transfer_dest_view is the destination render target view for the transfer.
  void SubmitBarriersAndEnterRenderTargetCacheRenderPass(
      VkRenderPass render_pass,
      const VulkanRenderTargetCache::Framebuffer* framebuffer,
      VkImageView transfer_dest_view, bool transfer_dest_is_depth);
  // Must be called before doing anything outside the render pass scope,
  // including adding pipeline barriers that are not a part of the render pass
  // scope. Submission must be open.
  void EndRenderPass();

  // What ends render passes, counted with log_gpu_frame_time_breakdown
  // (VkPassEnd lines): the work in progress (a PassEndReasonScope), or the
  // kind of the pending barriers that SubmitBarriers had to end the pass for.
  enum class PassEndReason : uint32_t {
    kOther,
    // Entering another framebuffer, render target ownership transfers.
    kRenderTargets,
    kResolve,
    // Texture loads.
    kTextures,
    // Shared memory uploads.
    kSharedMemory,
    kPrimitiveProcessor,
    kQuery,
    kSubmission,
    kBufferBarriers,
    kImageBarriers,
    kBufferAndImageBarriers,
    // Descriptor and texture bindings of a draw (texture usage transitions).
    kBindings,
    kCount,
  };
  class PassEndReasonScope {
   public:
    PassEndReasonScope(VulkanCommandProcessor& command_processor,
                       PassEndReason reason)
        : command_processor_(command_processor),
          previous_reason_(command_processor.pass_end_reason_) {
      command_processor.pass_end_reason_ = reason;
    }
    ~PassEndReasonScope() {
      command_processor_.pass_end_reason_ = previous_reason_;
    }
    PassEndReasonScope(const PassEndReasonScope&) = delete;
    PassEndReasonScope& operator=(const PassEndReasonScope&) = delete;

   private:
    VulkanCommandProcessor& command_processor_;
    PassEndReason previous_reason_;
  };
  PassEndReason pass_end_reason_ = PassEndReason::kOther;
  // The PassEndReason scopes open when the pending barriers were pushed (a bit
  // per reason), for the origins of the barriers that end render passes.
  uint32_t pending_barrier_origins_ = 0;

  VkDescriptorSetLayout GetSingleTransientDescriptorLayout(
      SingleTransientDescriptorLayout transient_descriptor_layout) const {
    return descriptor_set_layouts_single_transient_[size_t(
        transient_descriptor_layout)];
  }
  // A frame must be open.
  VkDescriptorSet AllocateSingleTransientDescriptor(
      SingleTransientDescriptorLayout transient_descriptor_layout);

  // The returned reference is valid until a cache clear.
  VkDescriptorSetLayout GetTextureDescriptorSetLayout(bool is_vertex,
                                                      size_t texture_count,
                                                      size_t sampler_count);
  // The returned reference is valid until a cache clear.
  const VulkanPipelineCache::PipelineLayoutProvider* GetPipelineLayout(
      size_t texture_count_pixel, size_t sampler_count_pixel,
      size_t texture_count_vertex, size_t sampler_count_vertex);

  // Returns a single temporary GPU-side buffer within a submission for tasks
  // like texture untiling and resolving. May push a buffer memory barrier into
  // the initial usage. Submission must be open.
  ScratchBufferAcquisition AcquireScratchGpuBuffer(
      VkDeviceSize size, VkPipelineStageFlags initial_stage_mask,
      VkAccessFlags initial_access_mask);

  // Binds a graphics pipeline for host-specific purposes, invalidating the
  // affected state. keep_dynamic_* must be false (to invalidate the dynamic
  // state after binding the pipeline with the same state being static, or if
  // the caller changes the dynamic state bypassing the VulkanCommandProcessor)
  // unless the caller has these state variables as dynamic and uses the
  // tracking in VulkanCommandProcessor to modify them.
  void BindExternalGraphicsPipeline(VkPipeline pipeline,
                                    bool keep_dynamic_depth_bias = false,
                                    bool keep_dynamic_blend_constants = false,
                                    bool keep_dynamic_stencil_mask_ref = false);
  void BindExternalComputePipeline(VkPipeline pipeline);
  void SetViewport(const VkViewport& viewport);
  void SetScissor(const VkRect2D& scissor);

  std::string GetTitleStateSuffix() const override;

  // Debug marker methods - public so subsystems can annotate their operations.
  void PushDebugMarker(const char* format, ...);
  void PopDebugMarker();
  void InsertDebugMarker(const char* format, ...);
  bool debug_markers_enabled() const { return debug_markers_enabled_; }

 protected:
  bool SetupContext() override;
  void ShutdownContext() override;
  void OnPause() override;
  XE_FORCEINLINE
  void WriteRegister(uint32_t index, uint32_t value) override;
  XE_FORCEINLINE
  virtual void WriteRegistersFromMem(uint32_t start_index, uint32_t* base,
                                     uint32_t num_registers) override;
  void WriteRegisterRangeFromRing(xe::RingBuffer* ring, uint32_t base,
                                  uint32_t num_registers) override;

  // Range-segmented register write fast path (port of the D3D12 backend's
  // WriteRegisterRangeFromMem machinery, cvar vulkan_fast_register_ranges):
  // ranges with no side effects are bulk copy_and_swap'ed into the register
  // file, the constant families get range-level dirty handling instead of the
  // per-register virtual WriteRegister cascade.
  void WriteShaderConstantsFromMem(uint32_t start_index, uint32_t* base,
                                   uint32_t num_registers);
  void WriteBoolLoopFromMem(uint32_t start_index, uint32_t* base,
                            uint32_t num_registers);
  void WriteFetchFromMem(uint32_t start_index, uint32_t* base,
                         uint32_t num_registers);
  void WritePossiblySpecialRegistersFromMem(uint32_t start_index,
                                            uint32_t* base,
                                            uint32_t num_registers);
  XE_NOINLINE
  void WriteRegisterRangeFromRing_WraparoundCase(xe::RingBuffer* ring,
                                                 uint32_t base,
                                                 uint32_t num_registers);

  void OnGammaRamp256EntryTableValueWritten() override;
  void OnGammaRampPWLValueWritten() override;

  void IssueSwap(uint32_t frontbuffer_ptr, uint32_t frontbuffer_width,
                 uint32_t frontbuffer_height) override;
  // adrenotools_turbo_reassert_seconds, once per swap: requests the KGSL
  // power control again after a resume or once the period has passed.
  void ReassertGpuPowerControlIfDue();

  void OnPrimaryBufferEnd() override;

  Shader* LoadShader(xenos::ShaderType shader_type,
                     const uint32_t* host_address,
                     uint32_t dword_count) override;

  bool IssueDraw(xenos::PrimitiveType prim_type, uint32_t index_count,
                 IndexBufferInfo* index_buffer_info,
                 bool major_mode_explicit) override;
  // Set while IssueDraw issues a draw again after waiting for its pipeline
  // (it was already numbered and dumped the first time).
  bool issue_draw_pipeline_retry_ = false;
  bool IssueCopy() override;
  // vulkan_depth_4x_as_1x: the guest's 4x MSAA surface info rewritten for the
  // current draw (IssueDraw restores it) to the 1x surface of the samples, for
  // depth-only draws (and with vulkan_samples_as_pixels_simple_ps, draws with
  // simple pixel shaders) into surfaces never drawn with color otherwise (the
  // depth surfaces of multisampled scenes are left alone). pixel_shader is the
  // one the draw uses (null for depth-only). Returns whether the draw renders
  // the samples as pixels.
  bool RewriteMsaa4xSurfaceInfoForDraw(const Shader* pixel_shader,
                                       bool pixel_shader_uses_position);
  // The key of the current depth surface: depth base | pitch << 16.
  uint32_t GetMsaa4xDepthSurface() const;
  std::unordered_set<uint32_t> msaa_4x_scene_depth_surfaces_;
  // vulkan_samples_as_pixels_simple_ps and vulkan_samples_as_pixels_coarse when
  // msaa_4x_scene_depth_surfaces_ was filled.
  uint32_t msaa_4x_scene_depth_surfaces_options_ = 0;
  // vulkan_samples_as_pixels_coarse: this draw is shaded once per 2x2 pixels.
  bool draw_samples_as_pixels_coarse_ = false;

  void InitializeTrace() override;

 private:
  struct CommandBuffer {
    VkCommandPool pool;
    VkCommandBuffer buffer;
  };

  struct SparseBufferBind {
    VkBuffer buffer;
    size_t bind_offset;
    uint32_t bind_count;
  };

  union TextureDescriptorSetLayoutKey {
    uint32_t key;
    struct {
      // If texture and sampler counts are both 0, use
      // descriptor_set_layout_empty_ instead as these are owning references.
      uint32_t texture_count : 16;
      uint32_t sampler_count : 15;
      uint32_t is_vertex : 1;
    };

    TextureDescriptorSetLayoutKey() : key(0) {
      static_assert_size(*this, sizeof(key));
    }

    struct Hasher {
      size_t operator()(const TextureDescriptorSetLayoutKey& key) const {
        return std::hash<decltype(key.key)>{}(key.key);
      }
    };
    bool operator==(const TextureDescriptorSetLayoutKey& other_key) const {
      return key == other_key.key;
    }
    bool operator!=(const TextureDescriptorSetLayoutKey& other_key) const {
      return !(*this == other_key);
    }
  };

  union PipelineLayoutKey {
    uint64_t key;
    struct {
      // Pixel textures in the low bits since those are varied much more
      // commonly.
      uint16_t texture_count_pixel;
      uint16_t sampler_count_pixel;
      uint16_t texture_count_vertex;
      uint16_t sampler_count_vertex;
    };

    PipelineLayoutKey() : key(0) { static_assert_size(*this, sizeof(key)); }

    struct Hasher {
      size_t operator()(const PipelineLayoutKey& key) const {
        return std::hash<decltype(key.key)>{}(key.key);
      }
    };
    bool operator==(const PipelineLayoutKey& other_key) const {
      return key == other_key.key;
    }
    bool operator!=(const PipelineLayoutKey& other_key) const {
      return !(*this == other_key);
    }
  };

  class PipelineLayout : public VulkanPipelineCache::PipelineLayoutProvider {
   public:
    explicit PipelineLayout(
        VkPipelineLayout pipeline_layout,
        VkDescriptorSetLayout
            descriptor_set_layout_shared_memory_and_edram_ref,
        VkDescriptorSetLayout descriptor_set_layout_constants_ref,
        VkDescriptorSetLayout descriptor_set_layout_textures_vertex_ref,
        VkDescriptorSetLayout descriptor_set_layout_textures_pixel_ref,
        uint32_t texture_count_vertex, uint32_t sampler_count_vertex,
        uint32_t texture_count_pixel, uint32_t sampler_count_pixel)
        : pipeline_layout_(pipeline_layout),
          descriptor_set_layout_shared_memory_and_edram_ref_(
              descriptor_set_layout_shared_memory_and_edram_ref),
          descriptor_set_layout_constants_ref_(
              descriptor_set_layout_constants_ref),
          descriptor_set_layout_textures_vertex_ref_(
              descriptor_set_layout_textures_vertex_ref),
          descriptor_set_layout_textures_pixel_ref_(
              descriptor_set_layout_textures_pixel_ref),
          texture_count_vertex_(texture_count_vertex),
          sampler_count_vertex_(sampler_count_vertex),
          texture_count_pixel_(texture_count_pixel),
          sampler_count_pixel_(sampler_count_pixel) {}
    VkPipelineLayout GetPipelineLayout() const override {
      return pipeline_layout_;
    }
    uint32_t GetDescriptorSetLayouts(
        VkDescriptorSetLayout layouts_out[4]) const override {
      layouts_out[SpirvShaderTranslator::kDescriptorSetSharedMemoryAndEdram] =
          descriptor_set_layout_shared_memory_and_edram_ref_;
      layouts_out[SpirvShaderTranslator::kDescriptorSetConstants] =
          descriptor_set_layout_constants_ref_;
      layouts_out[SpirvShaderTranslator::kDescriptorSetTexturesVertex] =
          descriptor_set_layout_textures_vertex_ref_;
      layouts_out[SpirvShaderTranslator::kDescriptorSetTexturesPixel] =
          descriptor_set_layout_textures_pixel_ref_;
      return SpirvShaderTranslator::kDescriptorSetCount;
    }
    VkDescriptorSetLayout descriptor_set_layout_textures_vertex_ref() const {
      return descriptor_set_layout_textures_vertex_ref_;
    }
    VkDescriptorSetLayout descriptor_set_layout_textures_pixel_ref() const {
      return descriptor_set_layout_textures_pixel_ref_;
    }
    // The binding counts the texture set layouts were built for - lets the
    // draw path verify a set allocation against the layout it will use.
    uint32_t texture_count_vertex() const { return texture_count_vertex_; }
    uint32_t sampler_count_vertex() const { return sampler_count_vertex_; }
    uint32_t texture_count_pixel() const { return texture_count_pixel_; }
    uint32_t sampler_count_pixel() const { return sampler_count_pixel_; }

   private:
    VkPipelineLayout pipeline_layout_;
    VkDescriptorSetLayout descriptor_set_layout_shared_memory_and_edram_ref_;
    VkDescriptorSetLayout descriptor_set_layout_constants_ref_;
    VkDescriptorSetLayout descriptor_set_layout_textures_vertex_ref_;
    VkDescriptorSetLayout descriptor_set_layout_textures_pixel_ref_;
    uint32_t texture_count_vertex_;
    uint32_t sampler_count_vertex_;
    uint32_t texture_count_pixel_;
    uint32_t sampler_count_pixel_;
  };

  struct UsedSingleTransientDescriptor {
    uint64_t frame;
    SingleTransientDescriptorLayout layout;
    VkDescriptorSet set;
  };

  struct UsedTextureTransientDescriptorSet {
    uint64_t frame;
    TextureDescriptorSetLayoutKey layout;
    VkDescriptorSet set;
  };

  enum SwapApplyGammaDescriptorSet : uint32_t {
    kSwapApplyGammaDescriptorSetRamp,
    kSwapApplyGammaDescriptorSetSource,
    kSwapApplyGammaDescriptorSetDest,

    kSwapApplyGammaDescriptorSetCount,
  };

  // BeginSubmission and EndSubmission may be called at any time. If there's an
  // open non-frame submission, BeginSubmission(true) will promote it to a
  // frame. EndSubmission(true) will close the frame no matter whether the
  // submission has already been closed.
  // Unlike on Direct3D 12, submission boundaries do not imply any memory
  // barriers aside from an incoming host write (but not outgoing host read)
  // dependency.

  // Rechecks submission number and reclaims per-submission resources. Pass 0 as
  // the submission to await to simply check status, or pass
  // GetCurrentSubmission() to wait for all queue operations to be completed.
  void CheckSubmissionCompletionAndDeviceLoss(uint64_t await_submission);
  // If is_guest_command is true, a new full frame - with full cleanup of
  // resources and, if needed, starting capturing - is opened if pending (as
  // opposed to simply resuming after mid-frame synchronization). Returns
  // whether a submission is open currently and the device is not lost.
  bool BeginSubmission(bool is_guest_command);
  // If is_swap is true, a full frame is closed - with, if needed, cache
  // clearing and stopping capturing. Returns whether the submission was done
  // successfully, if it has failed, leaves it open.
  bool EndSubmission(bool is_swap);
  // Checks if ending a submission right now would not cause potentially more
  // delay than it would reduce - such as when there are unfinished graphics
  // pipeline creation requests.
  bool CanEndSubmissionImmediately() const;
  // Always inlined: VkAwaitSites attributes its waits to the callers.
  XE_FORCEINLINE bool AwaitAllQueueOperationsCompletion() {
    if (submission_open_) {
      EndSubmission(false);
    }
    DrainSubmitWorker();
    CheckSubmissionCompletionAndDeviceLoss(GetCurrentSubmission());
    const bool drained = !submission_open_ &&
                         GetCompletedSubmission() + 1u >= GetCurrentSubmission();
    if (drained) {
      drained_submission_ = GetCurrentSubmission();
      ReclaimCommandBuffersAndSemaphoresOnRecorder(GetCompletedSubmission());
    }
    return drained;
  }
  // command_processor_memexport.inc: the submission of the last export, not the
  // whole queue. Always inlined for VkAwaitSites too.
  XE_FORCEINLINE bool AwaitMemexportSubmission(uint64_t submission) {
    if (memexport_compute_batch_.size && !memexport_batch_flushing_) {
      FlushMemExportBatch();
    }
    CheckSubmissionCompletionAndDeviceLoss(submission);
    return !device_lost_ && GetCompletedSubmission() >= submission;
  }

  uint64_t TakeMemexportSignalSubmission() override;
  void EndSubmissionForHeldSignals(uint64_t submission) override;
  void AwaitSubmissionForHeldSignals(uint64_t submission) override;
  void AwaitVIZQueryResolve(uint64_t submission) override;

  // Open-pass upload splice guard (adapted from XDroid360 6e54a1f6f,
  // BSD-3-Clause): allows hoisting safe uploads ahead of the active pass while
  // rejecting ranges overlapping memexport or GPU-written shared memory.
  bool CanSpliceBeforeActivePass(uint32_t base_bytes,
                                 uint32_t size_bytes) const;
  bool BeginPrePassSplice();
  void EndPrePassSplice();
  void ClosePrePassSpliceSegment();

  void ClearTransientDescriptorPools();

  void SplitPendingBarrier();

  void DestroyScratchBuffer();

  // ZPD occlusion queries backend.
  // vkCmdBeginQuery is only valid inside a render pass, so segments split at
  // pass end and resume at the next pass begin. If BEGIN fires outside a pass,
  // segment_pending_begin waits for the next. Outside a render pass,
  // DiscardZPDQuery defers the slot release until the submission completes.
  // FSI queries clear a dedicated counter with vkCmdFillBuffer, so they may
  // need to open before a pass begins or split an active pass around the clear.

  void EnsureZPDQueryResources() override;
  void ShutdownZPDQueryResources() override {
    zpd_resolves_in_flight_.clear();
    zpd_deferred_releases_.clear();
    zpd_active_query_index_ = UINT32_MAX;
    zpd_active_query_generation_ = 0;
    zpd_active_query_is_fsi_ = false;
    zpd_fsi_counter_index_force_update_ = true;
    if (zpd_host_query_pool_) {
      zpd_host_query_pool_->Shutdown();
    }
  }

  bool IsZPDQueryPoolReady() const override;
  bool CanOpenZPDQuery() const override;

  QueryOpenResult OpenZPDQuery(ReportHandle report_handle,
                               bool can_close_submission) override;
  bool CloseZPDQuery(ReportHandle report_handle,
                     uint64_t& out_submission) override;
  bool DiscardZPDQuery() override;
  void PumpQueryResolves() override;
  bool AwaitQueryResolve(ReportHandle report_handle,
                         uint64_t wait_for_submission) override;

  void UpdateDynamicState(
      const draw_util::ViewportInfo& viewport_info, bool primitive_polygonal,
      reg::RB_DEPTHCONTROL normalized_depth_control,
      uint32_t draw_resolution_scale_x, uint32_t draw_resolution_scale_y,
      bool depth_bias_in_pixel_shader,
      const VulkanPipelineCache::DynamicState& pipeline_dynamic_state,
      bool color_writes_off = false);
  void UpdateSystemConstantValues(
      bool primitive_polygonal,
      const PrimitiveProcessor::ProcessingResult& primitive_processing_result,
      bool shader_32bit_index_dma, const draw_util::ViewportInfo& viewport_info,
      uint32_t used_texture_mask, reg::RB_DEPTHCONTROL normalized_depth_control,
      uint32_t normalized_color_mask,
      const draw_util::HostDepthPolygonOffset* host_depth_polygon_offset);
  // *_bindings_ready is IssueDraw's per-draw readiness snapshot, the same one
  // the sampler collection used - re-reading bindings_ready() here would race
  // translation finishing mid-draw.
  bool UpdateBindings(const VulkanShader* vertex_shader,
                      const VulkanShader* pixel_shader,
                      bool vertex_bindings_ready, bool pixel_bindings_ready,
                      bool interpreter_placeholder = false,
                      bool placeholder_pixel_shader = false);
  // Allocates a descriptor set and fills one or two VkWriteDescriptorSet
  // structure instances (for images and samplers).
  // The descriptor set layout must be the one for the given is_vertex,
  // texture_count, sampler_count (from GetTextureDescriptorSetLayout - may be
  // already available at the moment of the call, no need to locate it again).
  // Returns how many VkWriteDescriptorSet structure instances have been
  // written, or 0 if there was a failure to allocate the descriptor set or no
  // bindings were requested.
  uint32_t WriteTransientTextureBindings(
      bool is_vertex, uint32_t texture_count, uint32_t sampler_count,
      VkDescriptorSetLayout descriptor_set_layout,
      const VkDescriptorImageInfo* texture_image_info,
      const VkDescriptorImageInfo* sampler_image_info,
      VkWriteDescriptorSet* descriptor_set_writes_out);

  bool device_lost_ = false;

  // Rolling per-submission summary for diagnosing VK_ERROR_DEVICE_LOST.
  // Accumulated into submission_in_progress_ during a submission, snapshotted
  // into submission_history_ on successful submit, dumped on device-loss.
  struct SubmissionSummary {
    uint64_t submission_index = 0;
    uint64_t frame_index = 0;
    uint32_t draw_count = 0;
    uint32_t dispatch_count = 0;
    uint32_t resolve_count = 0;
    uint64_t last_vs_hash = 0;
    uint64_t last_ps_hash = 0;
    uint64_t last_render_pass_key = 0;
  };
  static constexpr size_t kSubmissionHistorySize = 16;
  std::array<SubmissionSummary, kSubmissionHistorySize> submission_history_{};
  size_t submission_history_next_ = 0;
  SubmissionSummary submission_in_progress_{};

  void LogRecentSubmissions(const char* context);

  bool cache_clear_requested_ = false;

  // Host shader types that guest shaders can be translated into - they can
  // access the shared memory (via vertex fetch, memory export, or manual index
  // buffer reading) and textures.
  VkPipelineStageFlags guest_shader_pipeline_stages_ = 0;
  VkShaderStageFlags guest_shader_vertex_stages_ = 0;

  std::vector<VkSemaphore> semaphores_free_;

  struct PendingQueryResolve {
    uint64_t submission = 0;
    uint32_t query_index = UINT32_MAX;
    uint32_t query_generation = 0;
    uint32_t scale_area = 1;
    bool uses_fsi_counter = false;
    ReportHandle report_handle = kInvalidReportHandle;
  };
  uint32_t zpd_active_query_index_ = UINT32_MAX;
  uint32_t zpd_active_query_generation_ = 0;
  bool zpd_active_query_is_fsi_ = false;
  bool zpd_fsi_counter_index_force_update_ = true;
  std::deque<PendingQueryResolve> zpd_resolves_in_flight_;
  // Fallback buffer for EDRAM descriptor binding 2.
  VkBuffer zpd_fsi_counter_sink_buffer_ = VK_NULL_HANDLE;
  VkDeviceMemory zpd_fsi_counter_sink_buffer_memory_ = VK_NULL_HANDLE;
  // Currently installed binding 2 buffer.
  VkBuffer zpd_fsi_counter_descriptor_buffer_ = VK_NULL_HANDLE;
  VkDeviceSize zpd_fsi_counter_descriptor_range_ = 0;

  ui::vulkan::VulkanGPUCompletionTimeline completion_timeline_;
  // Per-frame GPU synchronization diagnostics (worker thread only), reported
  // alongside the log_gpu_frame_time_breakdown breakdown.
  struct VkFrameSyncStats {
    uint64_t frames = 0;
    uint64_t awaits = 0;    // blocking waits for own submissions
    uint64_t await_ns = 0;  // time inside those waits
    uint64_t submissions = 0;
    uint64_t resolves = 0;  // EDRAM resolves issued (churn magnitude)
    uint64_t memexport_awaits = 0;
    // Draws with memory export, and what each draw's barrier was.
    uint64_t memexport_draws = 0;
    uint64_t memexport_draw_barriers[size_t(
        VulkanSharedMemory::DrawBarrier::kCount)] = {};
    uint64_t readback_awaits = 0;
    // Host-side latency from vkQueueSubmit to the fence being observed
    // completed - includes GPU queue wait, render time and detection delay.
    uint64_t sub_latency_ns = 0;
    uint64_t sub_latency_max_ns = 0;
    uint64_t sub_completions = 0;
    // Awaits that actually blocked, and the sum over them of
    // (current submission - awaited submission) to see how many submissions
    // of headroom the throttle really has.
    uint64_t blocking_awaits = 0;
    uint64_t await_delta = 0;
    // GPU-side wall time (top-of-pipe to bottom-of-pipe timestamps) of
    // completed submissions, and the idle gaps between consecutive ones.
    uint64_t gpu_exec_ns = 0;
    uint64_t gpu_exec_max_ns = 0;
    uint64_t gpu_gap_ns = 0;
    uint64_t gpu_samples = 0;
    // GPU time inside resolve emission regions (see the pairs in IssueCopy).
    uint64_t resolve_gpu_ns = 0;
    uint64_t resolve_gpu_max_ns = 0;
    uint64_t resolve_gpu_samples = 0;
    uint64_t resolve_ts_dropped = 0;
    // Draw/pass structure counters.
    uint64_t draws = 0;
    uint64_t render_pass_begins = 0;
    uint64_t primary_buffer_splits = 0;
    uint64_t rb_uma_direct = 0;
    uint64_t rb_uma_first_use = 0;
    // Resolves not copied: the CPU had not read their range
    // (readback_resolve_uma_read_watch).
    uint64_t rb_uma_unread = 0;
    // Bytes the direct readbacks copied into guest RAM.
    uint64_t rb_uma_bytes = 0;
    // vulkan_replay_stats: texture descriptor sets written, of them the ones
    // with the content of a set already written in the same frame, and
    // constant buffer descriptor sets written.
    uint64_t texture_set_writes = 0;
    uint64_t texture_set_rewrites = 0;
    uint64_t constants_set_writes = 0;
    // Texture descriptor sets reused from earlier in the frame
    // (vulkan_texture_descriptor_set_frame_reuse).
    uint64_t texture_set_reuses = 0;
    // Command processor thread time spent replaying the deferred command
    // buffer into Vulkan and submitting it (part of the draw/swap time).
    uint64_t replay_ns = 0;
    // Resolves that clear EDRAM, and how many of those were cleared inside
    // the guest render pass (vulkan_resolve_clear_in_guest_pass).
    uint64_t resolve_clears = 0;
    uint64_t resolve_clears_in_guest_pass = 0;
    // Render passes ended, by PassEndReason.
    uint64_t pass_ends[size_t(PassEndReason::kCount)] = {};
    // Guest passes reopened on the framebuffer the previous guest pass ended
    // on - breaks between draws into the same attachments - by the
    // PassEndReason of that end, and how many of them had a resolve between.
    uint64_t pass_reopens[size_t(PassEndReason::kCount)] = {};
    uint64_t pass_reopens_after_resolve = 0;
    // Render passes ended by barriers, by the PassEndReason scope that pushed
    // them (several per pass end if several did).
    uint64_t pass_ending_barrier_origins[size_t(PassEndReason::kCount)] = {};
    // In-pass render target transfers inside an open native occlusion query
    // segment, which was closed around them.
    uint64_t zpd_transfer_suspends = 0;
    uint64_t last_report_ns = 0;
  };
  VkFrameSyncStats vk_frame_sync_stats_;

 public:
  VkFrameSyncStats& vk_frame_sync_stats() { return vk_frame_sync_stats_; }

 private:
  // debug.xendroid.frame_dump: a new value dumps the next frame to
  // <storage root>/frame_dump/<value>/ - frame.txt with a line per draw (its
  // shaders, render target registers and texture fetch constants) and per
  // resolve (where it copied from and to), and r<index>.bin with the bytes each
  // resolve wrote to guest memory. Each resolve drains the GPU to read them.
  void PollFrameDump();
  void DumpFrameDraw(const VulkanShader& vertex_shader,
                     const VulkanShader* pixel_shader,
                     xenos::PrimitiveType prim_type, uint32_t index_count,
                     const IndexBufferInfo* index_buffer_info);
  void DumpFrameResolve(uint32_t written_address, uint32_t written_length);
  std::string frame_dump_value_;
  std::filesystem::path frame_dump_dir_;
  FILE* frame_dump_file_ = nullptr;
  uint32_t frame_dump_draws_ = 0;
  uint32_t frame_dump_resolves_ = 0;
  // Shaders whose microcode disassembly is already in the dump directory.
  std::unordered_set<uint64_t> frame_dump_shaders_;
  // The base levels of the textures drawn with in the dumped frame (guest
  // address and length), compared between guest memory and the GPU's copy
  // when the frame ends.
  struct FrameDumpTexture {
    uint32_t length;
    // Fetch constant words 0 and 1 (format, pitch, tiling, endianness) and 2
    // (size) of the first draw sampling it.
    uint32_t dword_0, dword_1, dword_2;
    // A 3D texture: its GPU copy is written even when it matches the guest's.
    bool is_3d;
  };
  std::map<uint32_t, FrameDumpTexture> frame_dump_textures_;
  // debug.xendroid.gpu_reload, the last value acted upon.
  std::string gpu_reload_value_;
  // The vertex and index buffers drawn with in the dumped frame (guest address
  // and length), compared like the textures.
  std::map<uint32_t, uint32_t> frame_dump_buffers_;
  // Per draw of the dumped frame, what its vertices are read from: scanned at
  // the end of the frame in the GPU's copy, for values a stretched triangle
  // would have (infinite, NaN, huge).
  struct FrameDumpVertexScan {
    struct Attribute {
      uint32_t base;  // Bytes.
      uint32_t size;  // Bytes, as the fetch constant gives it.
      uint32_t stride_words;
      int32_t offset_words;
      uint32_t fetch_constant;
      xenos::VertexFormat format;
      xenos::Endian endian;
    };
    uint32_t draw;
    uint32_t index_base;  // 0 for auto-indexed draws.
    uint32_t index_count;
    xenos::IndexFormat index_format;
    xenos::Endian index_endian;
    uint32_t index_offset;
    // VGT_MIN_VTX_INDX, VGT_MAX_VTX_INDX.
    uint32_t index_clamp_min, index_clamp_max;
    std::vector<Attribute> attributes;
    // Every vertex buffer binding, for dumping the ranges of a broken draw.
    struct Binding {
      uint32_t fetch_constant;
      uint32_t base;  // Bytes.
      uint32_t size;  // Bytes.
      uint32_t stride_words;
    };
    std::vector<Binding> bindings;
  };
  // Broken draws whose index buffer, vertex ranges and page histories were
  // written.
  uint32_t frame_dump_broken_draws_ = 0;
  // resolve_log_frames: the resolves of the first frames, one line each.
  FILE* resolve_log_file_ = nullptr;
  bool OpenResolveLog();
  void LogResolve(uint32_t written_address, uint32_t written_length);
  // And the memory-exporting draws: their export ranges and, with the GPU
  // drained, whether the GPU's copy of what they read matches guest memory.
  void LogMemexportDraw(const VulkanShader& vertex_shader,
                        const VulkanShader* pixel_shader, uint32_t index_count,
                        const IndexBufferInfo* index_buffer_info);
  // Exporting shaders whose disassembly is already in the storage root.
  std::unordered_set<uint64_t> memexport_logged_shaders_;
  std::vector<FrameDumpVertexScan> frame_dump_vertex_scans_;
  // Raw vertex files written for the attributes found broken.
  uint32_t frame_dump_vertex_files_ = 0;
  void WriteFrameDumpVertexScans();
  // Draws so far in the frame, counted where the frame dump numbers them, and
  // debug.xendroid.skip_draws=<first>[-<last>][!<first>[-<last>]]: the draws of
  // every frame not issued, except those after the "!", to find the ones
  // drawing something broken by elimination.
  uint32_t frame_draw_index_ = 0;
  std::string skip_draws_value_;
  uint32_t skip_draws_first_ = UINT32_MAX;
  uint32_t skip_draws_last_ = 0;
  uint32_t skip_draws_keep_first_ = UINT32_MAX;
  uint32_t skip_draws_keep_last_ = 0;
  // debug.xendroid.memexport_log=N: the next N deferred exports and awaits of
  // their output, with where in the frame and how long (needs
  // log_gpu_frame_time_breakdown for the awaits).
  std::string memexport_log_value_;
  uint32_t memexport_log_left_ = 0;
  // debug.xendroid.memexport_trace: logs the next exporting draws.
  std::string memexport_trace_value_;
  uint32_t memexport_trace_left_ = 0;
  // vulkan_memexport_barriers_by_range: this draw's ranges, kept to reuse their
  // storage.
  std::vector<std::pair<uint32_t, uint32_t>> memexport_barrier_reads_;
  std::vector<std::pair<uint32_t, uint32_t>> memexport_barrier_exports_;
  bool memexport_barriers_by_range_last_ = false;
  // The batch of exporting draws going on (DrawExportTracker), and what a draw
  // has to match to be a part of it.
  uint64_t memexport_batch_ = 0;
  uint64_t memexport_batch_vertex_shader_ = 0;
  uint64_t memexport_batch_pixel_shader_ = 0;
  uint64_t memexport_batch_next_draw_ = 0;
  uint64_t memexport_batch_sync_count_ = 0;
  // All guest draws so far, also those with no effect here.
  uint64_t guest_draw_count_ = 0;

  struct SubmissionWork {
    uint32_t draws = 0;
    uint32_t resolves = 0;
    uint32_t texture_loads = 0;
    uint64_t texture_load_texels = 0;
    uint64_t upload_bytes = 0;
  };
  SubmissionWork submission_work_;
  struct SubmitTimeRecord {
    uint64_t submission;
    uint64_t submit_ns;
    // Slot in frame_timestamp_pool_, UINT32_MAX if timestamps unavailable.
    uint32_t timestamp_slot;
    // Resolve timestamp pairs recorded in this submission.
    uint32_t resolve_slot_base;
    uint32_t resolve_pair_count;
    // Render-pass timestamp pairs recorded in this submission.
    uint32_t pass_slot_base;
    uint32_t pass_pair_count;
    SubmissionWork work;
  };
  std::deque<SubmitTimeRecord> vk_submit_times_;
  // GPU timestamps around each submission (2 per slot), copied in-buffer to
  // the readback buffer so completion never requires host query calls (see
  // the Turnip/kgsl note in the completion timeline about blocking fence
  // status polls - vkGetQueryPoolResults would have the same hazard).
  static constexpr uint32_t kFrameTimestampSlots = 32;
  draw_cuts::TimestampSlotRingGuard<kFrameTimestampSlots> timestamp_slot_guard_;
  uint32_t current_submission_timestamp_slot_ = UINT32_MAX;
  VkQueryPool frame_timestamp_pool_ = VK_NULL_HANDLE;
  // Draws recorded into the currently open submission, for
  // vulkan_mid_frame_submission_draws.
  uint32_t draws_since_submission_ = 0;
  // The first submission after the GPU was last awaited until it had nothing
  // left to do (vulkan_drained_submission_draws).
  uint64_t drained_submission_ = 0;
  VkBuffer frame_timestamp_buffer_ = VK_NULL_HANDLE;
  VkDeviceMemory frame_timestamp_buffer_memory_ = VK_NULL_HANDLE;
  VkDeviceSize frame_timestamp_buffer_size_ = 0;
  uint64_t* frame_timestamp_mapping_ = nullptr;
  uint64_t frame_timestamp_prev_end_ = 0;
  // GPU timestamps bracketing each resolve region, ring-buffered per
  // submission like the per-submission pair above (same no-host-query rule).
  // Three per resolve: begin, end of the copy (MarkResolveCopyEnd, or the end
  // if it wasn't reached) and end.
  static constexpr uint32_t kResolveTimestampPairsPerSubmission = 128;
  static constexpr uint32_t kResolveTimestampRingSubmissions = 32;
  static constexpr uint32_t kResolveTimestampsPerResolve = 3;
  VkQueryPool resolve_timestamp_pool_ = VK_NULL_HANDLE;
  VkBuffer resolve_timestamp_buffer_ = VK_NULL_HANDLE;
  VkDeviceMemory resolve_timestamp_buffer_memory_ = VK_NULL_HANDLE;
  VkDeviceSize resolve_timestamp_buffer_size_ = 0;
  uint64_t* resolve_timestamp_mapping_ = nullptr;
  uint64_t resolve_ts_submission_ = 0;
  uint32_t resolve_ts_count_ = 0;
  // The resolve being recorded, and whether its copy end is written.
  uint32_t resolve_ts_open_ = UINT32_MAX;
  bool resolve_ts_copy_end_written_ = false;
  // Kind and size of the resolve of each timestamp pair, and the GPU time per
  // key accumulated since the last report (VkResolveTime lines).
  std::array<uint32_t, size_t(kResolveTimestampPairsPerSubmission) *
                           kResolveTimestampRingSubmissions>
      resolve_ts_keys_{};
  struct ResolveBucketStat {
    uint64_t ns = 0;
    uint64_t max_ns = 0;
    uint64_t count = 0;
    // Split of ns at MarkResolveCopyEnd.
    uint64_t copy_ns = 0;
    uint64_t clear_ns = 0;
  };
  std::map<uint32_t, ResolveBucketStat> resolve_bucket_stats_;
  // GPU timestamps bracketing each render pass, bucketed CPU-side by
  // framebuffer extent (bit 31 = ownership-transfer pass). Same ring/readback
  // pattern; timestamps written OUTSIDE the pass (before begin / after end).
  // Also holds the VkMiscTime regions (keys with kMiscTimestampKeyBit). 192
  // dropped ~10 pairs per frame of Forza Horizon (a submission can hold most
  // of a frame's ~300 passes, resolves and loads).
  static constexpr uint32_t kPassTimestampPairsPerSubmission = 512;
  static constexpr uint32_t kPassTimestampRingSubmissions = 32;
  VkQueryPool pass_timestamp_pool_ = VK_NULL_HANDLE;
  VkBuffer pass_timestamp_buffer_ = VK_NULL_HANDLE;
  VkDeviceMemory pass_timestamp_buffer_memory_ = VK_NULL_HANDLE;
  VkDeviceSize pass_timestamp_buffer_size_ = 0;
  uint64_t* pass_timestamp_mapping_ = nullptr;
  uint64_t pass_ts_submission_ = 0;
  uint32_t pass_ts_count_ = 0;
  uint32_t pass_ts_open_pair_ = UINT32_MAX;
  uint64_t pass_ts_open_submission_ = 0;
  uint64_t pass_ts_dropped_ = 0;
  std::array<uint32_t, size_t(kPassTimestampPairsPerSubmission) *
                           kPassTimestampRingSubmissions>
      pass_ts_keys_{};
  std::array<uint32_t, size_t(kPassTimestampPairsPerSubmission) *
                           kPassTimestampRingSubmissions>
      pass_ts_draws_{};
  uint32_t pass_open_draws_ = 0;
  // Largest scissor corner seen among the draws of the currently open pass -
  // the region actually touched, as opposed to the full EDRAM-derived
  // framebuffer the render area is set to.
  uint32_t pass_open_scissor_w_ = 0;
  uint32_t pass_open_scissor_h_ = 0;
  // Same for the viewport, which bounds the draw even when the guest leaves
  // the scissor wide open.
  uint32_t pass_open_viewport_w_ = 0;
  uint32_t pass_open_viewport_h_ = 0;
  std::array<uint32_t, size_t(kPassTimestampPairsPerSubmission) *
                           kPassTimestampRingSubmissions>
      pass_ts_scissor_{};
  std::array<uint32_t, size_t(kPassTimestampPairsPerSubmission) *
                           kPassTimestampRingSubmissions>
      pass_ts_viewport_{};
  struct PassBucketStat {
    uint64_t ns = 0;
    uint64_t passes = 0;
    uint64_t draws = 0;
    uint32_t max_scissor_w = 0;
    uint32_t max_scissor_h = 0;
    uint32_t max_viewport_w = 0;
    uint32_t max_viewport_h = 0;
  };
  // Accumulated per bucket key since the last report.
  std::map<uint32_t, PassBucketStat> pass_bucket_stats_;
  void OpenPassTimestamp(uint32_t bucket_key);
  void ClosePassTimestamp();
  bool submission_open_ = false;
  // In case vkQueueSubmit fails after something like a successful
  // vkQueueBindSparse, to wait correctly on the next attempt.
  std::vector<VkSemaphore> current_submission_wait_semaphores_;
  std::vector<VkPipelineStageFlags> current_submission_wait_stage_masks_;
  std::deque<std::pair<uint64_t, VkSemaphore>>
      submissions_in_flight_semaphores_;

  static constexpr uint32_t kMaxFramesInFlight = 3;
  bool frame_open_ = false;
  // Guest frame index, since some transient resources can be reused across
  // submissions. Values updated in the beginning of a frame.
  uint64_t frame_current_ = 1;
  uint64_t frame_completed_ = 0;
  // Submission indices of frames that have already been submitted.
  uint64_t closed_frame_submissions_[kMaxFramesInFlight] = {};

  // <Submission where last used, resource>, sorted by the submission number.
  std::deque<std::pair<uint64_t, VkDeviceMemory>> destroy_memory_;
  std::deque<std::pair<uint64_t, VkBuffer>> destroy_buffers_;
  std::deque<std::pair<uint64_t, VkImage>> destroy_images_;
  std::deque<std::pair<uint64_t, VkImageView>> destroy_image_views_;
  std::deque<std::pair<uint64_t, VkFramebuffer>> destroy_framebuffers_;

  std::vector<CommandBuffer> command_buffers_writable_;
  std::deque<std::pair<uint64_t, CommandBuffer>> command_buffers_submitted_;

  // Bounded SPSC K = 4 SubmitChunk ring (Onda 3A). In sync mode, slot 0 is
  // used directly on the Recorder thread. In async mode, the Recorder writes
  // into submit_chunks_[active_chunk_index_] and pushes sealed chunks to the
  // Submit & Replay Worker thread.
  struct SubmitChunk {
    explicit SubmitChunk(const VulkanCommandProcessor& command_processor)
        : setup_dcb(command_processor, 64 * 1024),
          main_dcb(command_processor) {}

    DeferredCommandBuffer setup_dcb;
    DeferredCommandBuffer main_dcb;
    uint64_t submission_index = 0;
    bool is_swap = false;
    bool has_deferred_pipelines = false;
    uint64_t required_pipeline_ticket = 0;
    std::vector<SparseBufferBind> sparse_buffer_binds;
    std::vector<VkSparseMemoryBind> sparse_memory_binds;
    VkPipelineStageFlags sparse_bind_wait_stage_mask = 0;
    std::vector<VkSemaphore> external_wait_semaphores;
    std::vector<VkPipelineStageFlags> external_wait_stage_masks;
    submit_worker::SubmitTimestampSnapshot timestamps;
  };
  std::array<std::unique_ptr<SubmitChunk>,
             submit_worker::kSubmitChunkRingCapacity>
      submit_chunks_;
  submit_worker::SpscSlotRing<submit_worker::kSubmitChunkRingCapacity>
      submit_ring_;
  size_t active_chunk_index_ = 0;
  std::thread submit_worker_thread_;
  bool async_submit_worker_active_ = false;
  bool current_submission_has_deferred_pipelines_ = false;
  std::atomic<int32_t> desired_submit_worker_nice_{-10};
  std::atomic<uint64_t> async_worker_replay_ns_{0};
  std::vector<VkSemaphore> worker_wait_semaphores_;
  std::vector<VkPipelineStageFlags> worker_wait_stage_masks_;

  void PollAsyncSubmitWorkerProperties();
  void SyncAsyncSubmitWorkerMode();
  void StartSubmitWorker();
  void DrainSubmitWorker();
  void StopSubmitWorker();
  void SubmitWorkerThreadMain();
  void ExecuteAndSubmitChunkOnWorker(
      SubmitChunk& chunk, const std::function<void()>& release_slot);
  void ReclaimCommandBuffersAndSemaphoresOnWorker(
      uint64_t completed_submission);
  void ReclaimCommandBuffersAndSemaphoresOnRecorder(
      uint64_t completed_submission);
  bool EnsureSubmissionResourcesLockedOrOwned(bool needs_semaphore);
  void CaptureTimestampSnapshotOnRecorder(
      uint64_t submission_index, bool setup_has_commands,
      submit_worker::SubmitTimestampSnapshot& snapshot_out);

  std::vector<VkSparseMemoryBind> sparse_memory_binds_;
  std::vector<SparseBufferBind> sparse_buffer_binds_;
  // SparseBufferBind converted to VkSparseBufferMemoryBindInfo to this buffer
  // on submission (because pBinds should point to a place in std::vector, but
  // it may be reallocated).
  std::vector<VkSparseBufferMemoryBindInfo> sparse_buffer_bind_infos_temp_;
  VkPipelineStageFlags sparse_bind_wait_stage_mask_ = 0;

  // Temporary storage with reusable memory for creating descriptor set layouts.
  std::vector<VkDescriptorSetLayoutBinding> descriptor_set_layout_bindings_;
  // Temporary storage with reusable memory for writing image and sampler
  // descriptors.
  std::vector<VkDescriptorImageInfo> descriptor_write_image_info_;

  std::unique_ptr<ui::vulkan::VulkanUploadBufferPool> uniform_buffer_pool_;

  // Descriptor set layouts used by different shaders.
  VkDescriptorSetLayout descriptor_set_layout_empty_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout descriptor_set_layout_constants_ = VK_NULL_HANDLE;
  std::array<VkDescriptorSetLayout,
             size_t(SingleTransientDescriptorLayout::kCount)>
      descriptor_set_layouts_single_transient_{};
  VkDescriptorSetLayout descriptor_set_layout_shared_memory_and_edram_ =
      VK_NULL_HANDLE;

  // Guards the layout maps below - GetPipelineLayout is called both from the
  // draw thread and from pipeline creation threads (for deferred translation).
  std::mutex pipeline_layouts_mutex_;
  // Descriptor set layouts are referenced by pipeline_layouts_.
  std::unordered_map<TextureDescriptorSetLayoutKey, VkDescriptorSetLayout,
                     TextureDescriptorSetLayoutKey::Hasher>
      descriptor_set_layouts_textures_;
  // Pipeline layouts are referenced by VulkanPipelineCache.
  std::unordered_map<PipelineLayoutKey, PipelineLayout,
                     PipelineLayoutKey::Hasher>
      pipeline_layouts_;

  // No specific reason for 32768, just the "too much" descriptor count from
  // Direct3D 12 PIX warnings.
  static constexpr uint32_t kLinkedTypeDescriptorPoolSetCount = 32768;
  static const VkDescriptorPoolSize kDescriptorPoolSizeUniformBuffer;
  static const VkDescriptorPoolSize kDescriptorPoolSizeStorageBuffer;
  static const VkDescriptorPoolSize kDescriptorPoolSizeStorageImage;
  static const VkDescriptorPoolSize kDescriptorPoolSizeTextures[2];
  ui::vulkan::LinkedTypeDescriptorSetAllocator
      transient_descriptor_allocator_uniform_buffer_;
  ui::vulkan::LinkedTypeDescriptorSetAllocator
      transient_descriptor_allocator_storage_buffer_;
  ui::vulkan::LinkedTypeDescriptorSetAllocator
      transient_descriptor_allocator_storage_image_;
  std::deque<UsedSingleTransientDescriptor> single_transient_descriptors_used_;
  std::array<std::vector<VkDescriptorSet>,
             size_t(SingleTransientDescriptorLayout::kCount)>
      single_transient_descriptors_free_;
  // <Usage frame, set>.
  std::deque<std::pair<uint64_t, VkDescriptorSet>>
      constants_transient_descriptors_used_;
  std::vector<VkDescriptorSet> constants_transient_descriptors_free_;

  ui::vulkan::LinkedTypeDescriptorSetAllocator
      transient_descriptor_allocator_textures_;
  std::deque<UsedTextureTransientDescriptorSet>
      texture_transient_descriptor_sets_used_;
  std::unordered_map<TextureDescriptorSetLayoutKey,
                     std::vector<VkDescriptorSet>,
                     TextureDescriptorSetLayoutKey::Hasher>
      texture_transient_descriptor_sets_free_;

  std::unique_ptr<VulkanSharedMemory> shared_memory_;

  std::unique_ptr<VulkanPrimitiveProcessor> primitive_processor_;

  std::unique_ptr<VulkanRenderTargetCache> render_target_cache_;

  std::unique_ptr<VulkanZPDQueryPool> zpd_host_query_pool_;

  // Deferred query slot releases for discards that happen outside a render
  // pass, where vkCmdEndQuery cannot be issued.  The slot is held until the
  // submission containing the stale BeginQuery completes on the GPU.
  struct DeferredQueryRelease {
    uint64_t submission;
    uint32_t query_index;
    uint32_t query_generation;
  };
  std::deque<DeferredQueryRelease> zpd_deferred_releases_;

  std::unique_ptr<VulkanPipelineCache> pipeline_cache_;

  std::unique_ptr<VulkanTextureCache> texture_cache_;

  VkDescriptorPool shared_memory_and_edram_descriptor_pool_ = VK_NULL_HANDLE;
  VkDescriptorSet shared_memory_and_edram_descriptor_set_;
  // Second set identical to the one above except binding 0 points at the
  // host-imported shared-memory buffer (aliasing guest RAM). Bound instead of
  // the device set for memexport-touching draws so their GPU-written geometry
  // stays coherent with the CPU. VK_NULL_HANDLE when the host buffer is
  // unavailable - routing is then disabled and every draw uses the device set.
  VkDescriptorSet shared_memory_host_and_edram_descriptor_set_ = VK_NULL_HANDLE;
  // Two-buffer memexport page tracking (data + methods), shared with the D3D12
  // backend as a class-body fragment so each backend gets its own non-virtual,
  // inlinable copy. Requires <array> and SharedMemory, both included above.
#include "../command_processor_memexport.inc"

  // Bytes 0x0...0x3FF - 256-entry gamma ramp table with B10G10R10X2 data (read
  // as R10G10B10X2 with swizzle).
  // Bytes 0x400...0x9FF - 128-entry PWL R16G16 gamma ramp (R - base, G - delta,
  // low 6 bits of each are zero, 3 elements per entry).
  // kMaxFramesInFlight pairs of gamma ramps if in host-visible memory and
  // uploaded directly, one otherwise.
  VkDeviceMemory gamma_ramp_buffer_memory_ = VK_NULL_HANDLE;
  VkBuffer gamma_ramp_buffer_ = VK_NULL_HANDLE;
  // kMaxFramesInFlight pairs, only when the gamma ramp buffer is not
  // host-visible.
  VkDeviceMemory gamma_ramp_upload_buffer_memory_ = VK_NULL_HANDLE;
  VkBuffer gamma_ramp_upload_buffer_ = VK_NULL_HANDLE;
  VkDeviceSize gamma_ramp_upload_memory_size_;
  uint32_t gamma_ramp_upload_memory_type_;
  // Mapping of either gamma_ramp_buffer_memory_ (if it's host-visible) or
  // gamma_ramp_upload_buffer_memory_ (otherwise).
  void* gamma_ramp_upload_mapping_;
  std::array<VkBufferView, 2 * kMaxFramesInFlight> gamma_ramp_buffer_views_{};
  // UINT32_MAX if outdated.
  uint32_t gamma_ramp_256_entry_table_current_frame_ = UINT32_MAX;
  uint32_t gamma_ramp_pwl_current_frame_ = UINT32_MAX;

  VkDescriptorSetLayout swap_descriptor_set_layout_sampled_image_ =
      VK_NULL_HANDLE;
  VkDescriptorSetLayout swap_descriptor_set_layout_uniform_texel_buffer_ =
      VK_NULL_HANDLE;
  // Storage image descriptor set layout for compute shader destination.
  VkDescriptorSetLayout swap_descriptor_set_layout_storage_image_ =
      VK_NULL_HANDLE;

  // Descriptor pool for allocating descriptors needed for presentation, such as
  // the destination images and the gamma ramps.
  VkDescriptorPool swap_descriptor_pool_ = VK_NULL_HANDLE;
  // Interleaved 256-entry table and PWL texel buffer descriptors.
  // kMaxFramesInFlight pairs of gamma ramps if in host-visible memory and
  // uploaded directly, one otherwise.
  std::array<VkDescriptorSet, 2 * kMaxFramesInFlight>
      swap_descriptors_gamma_ramp_;
  // Sampled images for source.
  std::array<VkDescriptorSet, kMaxFramesInFlight> swap_descriptors_source_;
  // Storage images for destination.
  std::array<VkDescriptorSet, kMaxFramesInFlight> swap_descriptors_dest_;

  // Gamma ramp application compute pipeline constants.
  struct ApplyGammaConstants {
    uint32_t size[2];
  };
  VkPipelineLayout swap_apply_gamma_pipeline_layout_ = VK_NULL_HANDLE;
  VkPipeline swap_apply_gamma_256_entry_table_pipeline_ = VK_NULL_HANDLE;
  VkPipeline swap_apply_gamma_pwl_pipeline_ = VK_NULL_HANDLE;
  // Gamma pipelines that also compute FXAA luma in the alpha channel.
  VkPipeline swap_apply_gamma_256_entry_table_fxaa_luma_pipeline_ =
      VK_NULL_HANDLE;
  VkPipeline swap_apply_gamma_pwl_fxaa_luma_pipeline_ = VK_NULL_HANDLE;

  // FXAA post-processing.
  struct FxaaConstants {
    uint32_t size[2];
    float size_inv[2];
  };
  // Descriptor set layout for FXAA source (combined image sampler).
  VkDescriptorSetLayout fxaa_source_descriptor_set_layout_ = VK_NULL_HANDLE;
  VkPipelineLayout fxaa_pipeline_layout_ = VK_NULL_HANDLE;
  VkPipeline fxaa_pipeline_ = VK_NULL_HANDLE;
  VkPipeline fxaa_extreme_pipeline_ = VK_NULL_HANDLE;
  VkSampler fxaa_sampler_ = VK_NULL_HANDLE;

  // Resolve downscale compute shader for scaled resolution readback.
  // Downscales scaled resolve buffer data back to 1x resolution on the GPU,
  // avoiding expensive CPU-side downscaling and reducing transfer bandwidth.
  VkDescriptorSetLayout resolve_downscale_descriptor_set_layout_ =
      VK_NULL_HANDLE;
  VkPipelineLayout resolve_downscale_pipeline_layout_ = VK_NULL_HANDLE;
  VkPipeline resolve_downscale_pipeline_ = VK_NULL_HANDLE;
  // Descriptor pool chain for resolve downscale shader (2 storage buffers per
  // set). Uses pool chain to avoid mid-frame GPU stalls on pool exhaustion.
  std::unique_ptr<ui::vulkan::VulkanDescriptorPoolChain>
      resolve_downscale_descriptor_pool_chain_;
  // Intermediate buffer for downscaled output (device local, for compute).
  VkBuffer resolve_downscale_buffer_ = VK_NULL_HANDLE;
  VkDeviceMemory resolve_downscale_buffer_memory_ = VK_NULL_HANDLE;
  uint32_t resolve_downscale_buffer_size_ = 0;

  // FXAA source texture - R16G16B16A16_SFLOAT for luma in alpha.
  static constexpr VkFormat kFxaaSourceFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
  VkDeviceMemory fxaa_source_memory_ = VK_NULL_HANDLE;
  VkImage fxaa_source_image_ = VK_NULL_HANDLE;
  VkImageView fxaa_source_image_view_ = VK_NULL_HANDLE;
  uint32_t fxaa_source_width_ = 0;
  uint32_t fxaa_source_height_ = 0;
  uint64_t fxaa_source_last_submission_ = 0;
  // Descriptor sets for FXAA - source as combined image sampler (for FXAA
  // read).
  std::array<VkDescriptorSet, kMaxFramesInFlight> fxaa_source_descriptors_;
  // Descriptor sets for FXAA - source as storage image (for gamma+luma write).
  std::array<VkDescriptorSet, kMaxFramesInFlight>
      fxaa_source_storage_descriptors_;

  // Pending pipeline barriers.
  std::vector<VkBufferMemoryBarrier> pending_barriers_buffer_memory_barriers_;
  std::vector<VkImageMemoryBarrier> pending_barriers_image_memory_barriers_;
  struct PendingBarrier {
    VkPipelineStageFlags src_stage_mask = 0;
    VkPipelineStageFlags dst_stage_mask = 0;
    size_t buffer_memory_barriers_offset = 0;
    size_t image_memory_barriers_offset = 0;
  };
  std::vector<PendingBarrier> pending_barriers_;
  PendingBarrier current_pending_barrier_;

  // GPU-local scratch buffer.
  static constexpr VkDeviceSize kScratchBufferSizeIncrement = 16 * 1024 * 1024;
  VkDeviceMemory scratch_buffer_memory_ = VK_NULL_HANDLE;
  VkBuffer scratch_buffer_ = VK_NULL_HANDLE;
  VkDeviceSize scratch_buffer_size_ = 0;
  VkPipelineStageFlags scratch_buffer_last_stage_mask_ = 0;
  VkAccessFlags scratch_buffer_last_access_mask_ = 0;
  uint64_t scratch_buffer_last_usage_submission_ = 0;
  bool scratch_buffer_used_ = false;

  // The current dynamic state of the graphics pipeline bind point. Note that
  // binding any pipeline to the bind point with static state (even if it's
  // unused, like depth bias being disabled, but the values themselves still not
  // declared as dynamic in the pipeline) invalidates such dynamic state.
  VkViewport dynamic_viewport_;
  VkRect2D dynamic_scissor_;
  // Dynamic fixed-function depth bias, blend constants, stencil state are
  // applicable only to the render target implementations where they are
  // actually involved.
  float dynamic_depth_bias_constant_factor_;
  float dynamic_depth_bias_slope_factor_;
  float dynamic_blend_constants_[4];
  // The stencil values are pre-initialized (to D3D11_DEFAULT_STENCIL_*, and the
  // initial values for front and back are the same for portability subset
  // safety) because they're updated conditionally to avoid changing the back
  // face values when stencil is disabled and the primitive type is changed
  // between polygonal and non-polygonal.
  uint32_t dynamic_stencil_compare_mask_front_ = UINT8_MAX;
  uint32_t dynamic_stencil_compare_mask_back_ = UINT8_MAX;
  uint32_t dynamic_stencil_write_mask_front_ = UINT8_MAX;
  uint32_t dynamic_stencil_write_mask_back_ = UINT8_MAX;
  uint32_t dynamic_stencil_reference_front_ = 0;
  uint32_t dynamic_stencil_reference_back_ = 0;
  bool dynamic_viewport_update_needed_;
  bool dynamic_scissor_update_needed_;
  bool dynamic_depth_bias_update_needed_;
  bool dynamic_blend_constants_update_needed_;
  bool dynamic_stencil_compare_mask_front_update_needed_;
  bool dynamic_stencil_compare_mask_back_update_needed_;
  bool dynamic_stencil_write_mask_front_update_needed_;
  bool dynamic_stencil_write_mask_back_update_needed_;
  bool dynamic_stencil_reference_front_update_needed_;
  bool dynamic_stencil_reference_back_update_needed_;

  // Extended dynamic state (VK_EXT_extended_dynamic_state / state2 / state3).
  // Cached last-emitted values + dirty flags, mirroring the existing dynamic
  // state idiom. Initialized to invalid sentinels so the first draw of each
  // submission always emits. Only emitted when the matching capability bit in
  // the pipeline cache is set.
  VkPrimitiveTopology dynamic_primitive_topology_ = VK_PRIMITIVE_TOPOLOGY_MAX_ENUM;
  VkBool32 dynamic_primitive_restart_enable_ = VK_FALSE;
  VkCullModeFlags dynamic_cull_mode_ = VK_CULL_MODE_FLAG_BITS_MAX_ENUM;
  VkFrontFace dynamic_front_face_ = VK_FRONT_FACE_MAX_ENUM;
  VkBool32 dynamic_depth_test_enable_ = VK_FALSE;
  VkBool32 dynamic_depth_write_enable_ = VK_FALSE;
  VkCompareOp dynamic_depth_compare_op_ = VK_COMPARE_OP_MAX_ENUM;
  VkBool32 dynamic_stencil_test_enable_ = VK_FALSE;
  VkStencilOpState dynamic_stencil_op_front_ = {};
  VkStencilOpState dynamic_stencil_op_back_ = {};
  VkBool32 dynamic_depth_clamp_enable_ = VK_FALSE;
  VkPolygonMode dynamic_polygon_mode_ = VK_POLYGON_MODE_MAX_ENUM;
  VkBool32 dynamic_color_blend_enable_[xenos::kMaxColorRenderTargets] = {};
  VkColorBlendEquationEXT
      dynamic_color_blend_equation_[xenos::kMaxColorRenderTargets] = {};
  VkColorComponentFlags
      dynamic_color_write_mask_[xenos::kMaxColorRenderTargets] = {};
  bool dynamic_primitive_topology_update_needed_;
  bool dynamic_primitive_restart_enable_update_needed_;
  bool dynamic_cull_mode_update_needed_;
  bool dynamic_front_face_update_needed_;
  bool dynamic_depth_test_enable_update_needed_;
  bool dynamic_depth_write_enable_update_needed_;
  bool dynamic_depth_compare_op_update_needed_;
  bool dynamic_stencil_test_enable_update_needed_;
  bool dynamic_stencil_op_front_update_needed_;
  bool dynamic_stencil_op_back_update_needed_;
  bool dynamic_depth_clamp_enable_update_needed_;
  bool dynamic_polygon_mode_update_needed_;
  bool dynamic_color_blend_enable_update_needed_;
  bool dynamic_color_blend_equation_update_needed_;
  bool dynamic_color_write_mask_update_needed_;
  // vulkan_shading_rate: the rate the current draw wants and the one set in
  // the command buffer - bit 0: 2 pixels wide, bit 1: 2 pixels tall.
  uint32_t draw_shading_rate_ = 0;
  uint32_t dynamic_shading_rate_ = 0;
  bool dynamic_shading_rate_update_needed_ = true;
  ThermalVrsController thermal_vrs_controller_;
  uint64_t last_thermal_vrs_poll_ms_ = 0;

  // Currently used samplers.
  std::vector<std::pair<VulkanTextureCache::SamplerParameters, VkSampler>>
      current_samplers_vertex_;
  std::vector<std::pair<VulkanTextureCache::SamplerParameters, VkSampler>>
      current_samplers_pixel_;
  // Cross-draw sampler cache state (cvar vulkan_cache_sampler_parameters).
  // A stage's current_samplers_* contents are reusable for the next draw if:
  // the same shader provides the binding list (pointers are stable - shaders
  // are only destroyed at pipeline cache shutdown), no sampler was destroyed
  // since (destroy generation), and per fetch slot, the fetch constant wasn't
  // written since the parameters were derived (per-stage dirty masks, bit set
  // = up to date, maintained in WriteRegister and committed after a draw's
  // sampler pass). The per-stage submission stamp tracks when UseSampler was
  // last called for every entry - it must run at least once per submission so
  // the texture cache's LRU eviction (which only frees samplers whose last
  // usage submission has fully completed) can never destroy a cached handle.
  uint32_t current_samplers_fetch_up_to_date_vertex_ = 0;
  uint32_t current_samplers_fetch_up_to_date_pixel_ = 0;
  const VulkanShader* current_samplers_shader_vertex_ = nullptr;
  const VulkanShader* current_samplers_shader_pixel_ = nullptr;
  uint64_t current_samplers_submission_vertex_ = 0;
  uint64_t current_samplers_submission_pixel_ = 0;
  uint64_t current_samplers_destroy_generation_ = ~uint64_t(0);

  xendroid::PixelSpecCacheState pixel_spec_cache_{};
  std::array<uint8_t, 32> current_pixel_spec_sign_slots_{};
  uint32_t current_pixel_spec_sign_classes_ = 0;
  bool current_pixel_spec_exp_adjust_zero_ = false;
  uint32_t current_pixel_spec_iscale_used_ = UINT32_MAX;

  // Cache render pass currently started in the command buffer with the
  // framebuffer. For dynamic rendering, current_render_pass_ is VK_NULL_HANDLE
  // but in_render_pass_ is true.
  VkRenderPass current_render_pass_;
  const VulkanRenderTargetCache::Framebuffer* current_framebuffer_;
  VkImageView current_transfer_dest_view_ = VK_NULL_HANDLE;
  // For the VkPassEnd reopen counts: the framebuffer of the last guest pass
  // ended, why it ended, and whether a resolve was issued since.
  const VulkanRenderTargetCache::Framebuffer* last_ended_framebuffer_ = nullptr;
  PassEndReason last_pass_end_reason_ = PassEndReason::kOther;
  bool resolve_since_pass_end_ = false;
  // True when inside a render pass or dynamic rendering block.
  bool in_render_pass_ = false;
  // Draws since the last vulkan_debug_extra_pass_breaks break.
  uint32_t debug_extra_pass_break_draws_ = 0;
  // adrenotools_turbo_reassert_seconds: the last resume the KGSL power
  // control was requested for, and when it was last requested.
  uint32_t gpu_power_resume_count_seen_ = 0;
  uint64_t gpu_power_last_request_ms_ = 0;

  // Currently bound graphics pipeline, either from the pipeline cache (with
  // potentially deferred creation - current_external_graphics_pipeline_ is
  // VK_NULL_HANDLE in this case) or a non-Xenos one
  // (current_guest_graphics_pipeline_ is nullptr in this case).
  // current_guest_graphics_pipeline_ is a STABLE pointer to the pipeline
  // cache's atomic VkPipeline slot, not a handle value - with asynchronous
  // creation the handle may still be VK_NULL_HANDLE at record time, so the
  // bind de-dup compares slot pointers and the deferred command buffer
  // resolves the handle at replay.
  const std::atomic<VkPipeline>* current_guest_graphics_pipeline_;
  // The handle the last recorded deferred bind captured; the first draw after
  // the slot fills or swaps re-records with the new handle.
  VkPipeline current_guest_graphics_pipeline_handle_ = VK_NULL_HANDLE;
  VkPipeline current_external_graphics_pipeline_;
  VkPipeline current_external_compute_pipeline_;

  // The index buffer bound in the current submission's command buffer. Guest
  // DMA index buffers are bound at the start of the shared memory buffer and
  // selected with firstIndex, the others rebound only when they change.
  VkBuffer current_index_buffer_ = VK_NULL_HANDLE;
  VkDeviceSize current_index_buffer_offset_ = 0;
  VkIndexType current_index_type_ = VK_INDEX_TYPE_MAX_ENUM;

  // Pipeline layout of the current guest graphics pipeline.
  const PipelineLayout* current_guest_graphics_pipeline_layout_;
  VkDescriptorBufferInfo current_constant_buffer_infos_
      [SpirvShaderTranslator::kConstantBufferCount];
  // Whether up-to-date data has been written to constant (uniform) buffers, and
  // the buffer infos in current_constant_buffer_infos_ point to them.
  uint32_t current_constant_buffers_up_to_date_;
  // merge_tiling_bands: the band sequence the state below is for, the rows of
  // a band, the render targets to replicate before the draw (1 - color, 2 -
  // depth), and the row offset and the scissor expansion of the draw.
  uint32_t tiling_band_sequence_seen_ = UINT32_MAX;
  uint32_t tiling_band_rows_ = 0;
  uint32_t tiling_band_replicate_ = 0;
  int32_t tiling_band_draw_y_offset_ = 0;
  bool tiling_band_draw_expand_scissor_ = false;
  // spirv_texture_fetch_constants_decoded: the values decoded from each fetch
  // constant (appended to the fetch constant buffer), and the fetch
  // constants written since they were decoded.
  std::array<SpirvShaderTranslator::DecodedTextureFetchConstant, 32>
      fetch_constants_decoded_;
  uint32_t fetch_constants_decode_needed_ = UINT32_MAX;
  // Dynamic constant buffers: when true (gated at SetupContext on the device reporting
  // maxDescriptorSetUniformBuffersDynamic >= kConstantBufferCount and the
  // vulkan_dynamic_constant_buffers cvar), the kDescriptorSetConstants set uses
  // VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC: the descriptor base offset is 0
  // and the per-draw slice offset is supplied as a dynamic offset at bind time,
  // so the transient set is re-allocated and re-written ONLY when a binding's
  // backing VkBuffer changes (uniform_buffer_pool_ page rollover). Held constant
  // for the device lifetime; all of the layout binding type, the
  // VkWriteDescriptorSet type and the bind's pDynamicOffsets switch on it
  // together (a partial switch is a VUID violation).
  bool use_dynamic_constants_ = false;
  // VK_KHR_push_descriptor for set 1 (constants): mutually exclusive with
  // use_dynamic_constants_ (VUID-VkDescriptorSetLayoutCreateInfo-flags-00280).
  bool use_push_descriptors_ = false;
  uint32_t max_push_descriptors_ = 0;
  // Value-cache for the constants descriptor set (one shared set, 5 bindings).
  // Mirrors the sentinel-free validity discipline of the texture hash caches
  // above: constants_descriptor_set_buffers_[i] records the VkBuffer handle last
  // WRITTEN into binding i of the cached set in
  // current_graphics_descriptor_sets_[kDescriptorSetConstants]. When all five
  // still match the current_constant_buffer_infos_[i].buffer handles AND the
  // recorded ranges still match (no page rollover and no constant-count change),
  // the per-draw vkUpdateDescriptorSets + set allocation are skipped and only
  // fresh pDynamicOffsets are bound. constants_descriptor_set_valid_ is reset to
  // false at every point where the cached transient set can become invalid (new
  // frame / transient-pool reset / used->free reclaim) so a stale handle can
  // never let the GPU read recycled or freed descriptor memory. Only meaningful
  // when use_dynamic_constants_ is true.
  //
  // The range is tracked alongside the buffer because the dynamic path only
  // re-binds pDynamicOffsets per draw; the descriptor's range is baked at write
  // time. The float-constant buffers' sizes vary with the shader's float_count
  // (see UpdateBindings), and the upload pool returns the SAME page (only a new
  // offset) when the new size still fits, so a buffer-only rollover check would
  // keep a stale, too-small range and the shader's higher-index float constants
  // would read as clamped/zero -> wrong transforms -> flicker, with no VUID
  // (dynamicOffset + stale_range still fits the buffer). Comparing the range
  // forces the cheap re-write on a constant-count change (a shader switch, far
  // rarer than per-draw constant value changes).
  VkBuffer constants_descriptor_set_buffers_
      [SpirvShaderTranslator::kConstantBufferCount] = {};
  VkDeviceSize constants_descriptor_set_ranges_
      [SpirvShaderTranslator::kConstantBufferCount] = {};
  bool constants_descriptor_set_valid_ = false;
  VkDescriptorSet current_graphics_descriptor_sets_
      [SpirvShaderTranslator::kDescriptorSetCount];
  // Whether descriptor sets in current_graphics_descriptor_sets_ point to
  // up-to-date data.
  uint32_t current_graphics_descriptor_set_values_up_to_date_;
  // Value-cache for the texture/sampler descriptor sets (vertex and pixel,
  // tracked independently). 64-bit FNV-1a content hash over the packed layout
  // key + the resolved VkImageView and VkSampler handles of the last
  // successfully-written transient set for the stage. The cached set lives in
  // current_graphics_descriptor_sets_[kDescriptorSetTexturesVertex/Pixel]; the
  // hash gates whether we may SKIP clearing that stage's
  // current_graphics_descriptor_set_values_up_to_date_ bit (i.e. reuse the
  // already-written, still-valid transient set without a vkUpdateDescriptorSets
  // + re-bind). The *_valid_ flags are sentinel-free validity: they are reset
  // to false at every point where the cached transient set can become invalid
  // (new frame / transient-pool reset / pipeline-layout change) so a stale hash
  // can never let the GPU sample recycled or freed descriptor memory.
  uint64_t current_texture_descriptor_set_hash_vertex_ = 0;
  uint64_t current_texture_descriptor_set_hash_pixel_ = 0;
  bool current_texture_descriptor_set_hash_valid_vertex_ = false;
  bool current_texture_descriptor_set_hash_valid_pixel_ = false;
  uint64_t current_texture_set_usage_gen_vertex_ = UINT64_MAX;
  uint64_t current_texture_set_usage_gen_pixel_ = UINT64_MAX;
  uint64_t current_texture_set_sampler_gen_vertex_ = UINT64_MAX;
  uint64_t current_texture_set_sampler_gen_pixel_ = UINT64_MAX;
  bool current_texture_set_3d_to_2d_vertex_ = true;
  bool current_texture_set_3d_to_2d_pixel_ = true;
  // Scratch buffer for compute_texture_set_hash, reused across draws (cleared
  // each call, capacity persists) to gather the reuse-input words for a single
  // XXH3 call without a per-draw allocation. GPU-command-thread-only.
  std::vector<uint64_t> texture_set_hash_scratch_;
  // Whether the descriptor sets currently bound to the command buffer - only
  // low bits for the descriptor set layouts that remained the same are kept
  // when changing the pipeline layout. May be out of sync with
  // current_graphics_descriptor_set_values_up_to_date_, but should be ensured
  // to be a subset of it at some point when it becomes important; bits for
  // non-existent descriptor set layouts may also be set, but need to be ignored
  // when they start to matter.
  uint32_t current_graphics_descriptor_sets_bound_up_to_date_;
  static_assert(
      SpirvShaderTranslator::kDescriptorSetCount <=
          sizeof(current_graphics_descriptor_set_values_up_to_date_) * CHAR_BIT,
      "Bit fields storing descriptor set validity must be large enough");
  static_assert(
      SpirvShaderTranslator::kDescriptorSetCount <=
          sizeof(current_graphics_descriptor_sets_bound_up_to_date_) * CHAR_BIT,
      "Bit fields storing descriptor set validity must be large enough");
  // For reusing the texture/sampler descriptor sets across draws while their
  // contents stay valid - the shaders and the samplers the currently valid
  // texture descriptor sets were written for (texture image view changes are
  // tracked separately via VulkanTextureCache::texture_bindings_changed).
  const VulkanShader* current_textures_vertex_shader_ = nullptr;
  const VulkanShader* current_textures_pixel_shader_ = nullptr;
  std::vector<std::pair<VulkanTextureCache::SamplerParameters, VkSampler>>
      current_written_samplers_vertex_;
  std::vector<std::pair<VulkanTextureCache::SamplerParameters, VkSampler>>
      current_written_samplers_pixel_;

  // Float constant usage masks of the last draw call.
  uint64_t current_float_constant_map_vertex_[4];
  uint64_t current_float_constant_map_pixel_[4];
  // True if the vertex float constant buffer holds the full unpacked 256 float
  // constants for the ucode interpreter rather than a shader's packed subset.
  // Used to invalidate the binding when switching interpreter <-> normal.
  bool float_constants_vertex_are_full_ = false;

  // System shader constants, including user clip planes and tessellation
  // constants.
  SpirvShaderTranslator::SystemConstants system_constants_;

  // Dirty-mask memoization for UpdateSystemConstantValues (gated by
  // vulkan_memoize_system_constants).
  draw_cuts::SystemConstantsDirtyTracker sysconst_dirty_tracker_;
  struct SysConstMemoState {
    bool valid = false;
    bool edram_fsi = false;
    bool gamma_as_unorm16 = false;
    bool alpha_to_coverage_as_test = false;
    bool texture_point_snap = false;
    bool primitive_polygonal = false;
    bool shader_32bit_index_dma = false;
    bool host_builtin_index_dma = false;
    bool has_host_depth_polygon_offset = false;
    uint32_t draw_scale_x = 0;
    uint32_t draw_scale_y = 0;
    uint32_t rb_surface_info = 0;
    uint32_t vgt_draw_initiator = 0;
    uint32_t used_texture_mask = 0;
    uint32_t normalized_depth_control = 0;
    uint32_t normalized_color_mask = 0;
    uint32_t zpd_fsi_counter_index = UINT32_MAX;
    uint64_t texture_usage_generation = UINT64_MAX;
    uint32_t cached_base_flags = 0;
    uint32_t cached_fsi_flags = 0;
    xenos::Endian host_shader_index_endian = xenos::Endian::kNone;
    float ndc_scale[3] = {};
    float ndc_offset[3] = {};
    uint32_t viewport_xy_extent[2] = {};
    draw_util::HostDepthPolygonOffset host_depth_polygon_offset = {};
  };
  SysConstMemoState sysconst_memo_;

  // Host viewport of the previous draw, reused while the inputs it was derived
  // from stay the same.
  draw_util::GetViewportInfoArgs previous_viewport_info_args_{};
  draw_util::ViewportInfo previous_viewport_info_{};

  // Temporary storage for memexport stream constants used in the draw.
  std::vector<draw_util::MemExportRange> memexport_ranges_;

  // Vertex shaders with memory export translated to compute shaders
  // (kMemExportCompute) for GPUs without vertexPipelineStoresAndAtomics, plus
  // vulkan_memexport_batch multi-draw compute batching (adapted from
  // DEGROUCHO/xendroid-mali, BSD-3-Clause).
  struct MemExportComputePipelineKey {
    const void* translation;
    VkPipelineLayout pipeline_layout;
    bool operator==(const MemExportComputePipelineKey& other) const {
      return translation == other.translation &&
             pipeline_layout == other.pipeline_layout;
    }
    struct Hasher {
      size_t operator()(const MemExportComputePipelineKey& key) const {
        return std::hash<const void*>{}(key.translation) ^
               (std::hash<const void*>{}(
                    reinterpret_cast<const void*>(key.pipeline_layout))
                << 1);
      }
    };
  };
  std::unordered_map<MemExportComputePipelineKey, VkPipeline,
                     MemExportComputePipelineKey::Hasher>
      memexport_compute_pipelines_;
  VkPipeline GetMemExportComputePipeline(
      VulkanShader& vertex_shader,
      SpirvShaderTranslator::Modification vertex_shader_modification,
      VkPipelineLayout pipeline_layout, bool batched);

  struct MemExportBatch {
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorSet sets[SpirvShaderTranslator::kDescriptorSetCount] = {};
    uint32_t set_count = 0;
    uint32_t size = 0;
    uint32_t capacity = 0;
    uint32_t float_stride = 0;
    uint32_t fetch_stride = 0;
    bool fetch_decoded = false;
    uint32_t max_vertex_count = 0;
    std::vector<uint8_t> system, float_vertex, bool_loop, fetch;
    std::vector<std::pair<uint32_t, uint32_t>> exported_ranges;
  };
  MemExportBatch memexport_compute_batch_;
  bool memexport_batch_flushing_ = false;
  bool memexport_batch_supported_ = false;
  uint64_t memexport_batched_draws_ = 0;
  uint64_t memexport_batch_dispatches_ = 0;
  void PushMemexportBeforeBarrier();
  void PushMemexportAfterBarrier();
  void AddToMemExportBatch(VkPipeline pipeline, VkPipelineLayout layout,
                           uint32_t set_count, uint32_t float_constant_count,
                           uint32_t vertex_count,
                           const uint32_t* vfetch_addresses,
                           const uint32_t* vfetch_sizes, uint32_t vfetch_count,
                           uint32_t guest_index_base, uint32_t index_bytes);
  void FlushMemExportBatch();

  size_t pass_splice_placeholder_ = SIZE_MAX;
  bool pass_splice_open_ = false;
  uint32_t pass_splice_depth_ = 0;
  bool dynamic_depth_bias_exact_unorm_ = false;
  bool current_shader_objects_bound_ = false;

  // memexport_enable without guest RAM import (the readback fallback in
  // IssueDraw): the GPU copies the export ranges of the shared memory buffer
  // here right before the exporting draw, so only the bytes the draw changed
  // go back to guest RAM. A range is the stream's declared capacity, and the
  // buffer's copy of the bytes the draw doesn't write can be older than guest
  // RAM - copying it all back rolled Need for Speed Most Wanted's job objects,
  // which share pages with an export, back to freed states.
  bool EnsureMemexportSnapshotBuffer(uint32_t size);
  VkBuffer memexport_snapshot_buffer_ = VK_NULL_HANDLE;
  VkDeviceMemory memexport_snapshot_memory_ = VK_NULL_HANDLE;
  const uint8_t* memexport_snapshot_mapping_ = nullptr;
  bool memexport_snapshot_coherent_ = false;
  uint32_t memexport_snapshot_size_ = 0;
  std::vector<VkBufferCopy> memexport_snapshot_regions_;
  std::vector<uint8_t> memexport_readback_after_;
  // memexport_readback_deferred: the exports not read back yet, in draw order,
  // with their submission and where their snapshot is in the snapshot buffer
  // (filled from 0 by the exports since it was last empty).
  struct PendingMemexportReadback {
    uint64_t submission;
    uint32_t address;
    uint32_t size;
    uint32_t snapshot_offset;
  };
  std::deque<PendingMemexportReadback> pending_memexport_readbacks_;
  uint32_t pending_memexport_snapshot_used_ = 0;
  // The open submission has such exports: it ends with a barrier making the
  // shader writes visible to the host.
  bool memexport_readbacks_in_submission_ = false;
  // Stores the changes of the pending exports in guest memory once the
  // submissions of all of them have completed - as submissions complete, and
  // so by the time the guest can see the GPU as done (while any is pending,
  // its fences, interrupts, coherency requests and waits on registers or
  // memory, and the command processor running out of commands, await the GPU
  // - command_processor_memexport.inc).
  void ApplyCompletedMemexportReadbacks(uint64_t completed_submission);
  // Byte-wise: the bytes of a range that differ between the snapshot taken
  // before an export and the GPU's copy after it, to guest memory. False if
  // the GPU's copy could not be read.
  bool StoreMemexportChanges(uint32_t address, uint32_t size,
                             const uint8_t* before);

  // Backend-agnostic resolve-to-guest-RAM copy decisions and read-watch
  // consumption tracking, shared with the D3D12 backend.
#include "../command_processor_resolve_readwatch.inc"
  // readback_resolve=uma: last submission that resolved each destination key,
  // so the direct host read only happens once that submission has retired.
  std::unordered_map<uint64_t, uint64_t> uma_readback_last_write_;
  // readback_resolve_uma_unread_traffic's scratch destination.
  std::vector<uint8_t> uma_unread_traffic_buffer_;
  // vulkan_texture_descriptor_set_frame_reuse: the texture descriptor sets
  // written in the frame by their content hash, and the frame.
  std::unordered_map<uint64_t, VkDescriptorSet> texture_set_frame_reuse_;
  uint64_t texture_set_frame_reuse_frame_ = UINT64_MAX;
  // vulkan_replay_stats: the contents of the texture descriptor sets written
  // in the frame (CountTextureSetWrite).
  std::unordered_set<uint64_t> replay_stats_texture_set_hashes_;
  uint64_t replay_stats_texture_set_hashes_frame_ = UINT64_MAX;
  void CountTextureSetWrite(uint64_t content_hash);
  // Guest frame of each destination's last uma readback, for the bounded
  // fence collection staleness cap.
  std::unordered_map<uint64_t, uint64_t> uma_readback_last_read_frame_;
  // Asynchronous staging ring for small UMA resolve readbacks (<= 4 KiB),
  // gated by vulkan_small_readback_async_ring. Avoids first_use and
  // kMaxFramesInFlight stalls while preserving per-frame exposure readbacks.
  bool EnsureSmallReadbackRing();
  void DestroySmallReadbackRing();
  bool TryEnqueueSmallReadback(uint32_t written_address,
                               uint32_t written_length);
  void DrainCompletedSmallReadbacks(uint64_t completed_submission);
  draw_cuts::SmallReadbackRingState small_readback_ring_;
  VkBuffer small_readback_ring_buffer_ = VK_NULL_HANDLE;
  VkDeviceMemory small_readback_ring_memory_ = VK_NULL_HANDLE;
  const uint8_t* small_readback_ring_mapping_ = nullptr;
  bool small_readback_ring_coherent_ = false;
  // Per-backend trampoline from the memory read callback into the shared
  // MarkResolvePagesRead.
  static void ResolveReadCallbackThunk(void* context, uint32_t physical_address,
                                       uint32_t length);

  // Debug marker support for RenderDoc/debug tools.
  bool debug_markers_enabled_ = false;
  void UpdateDebugMarkersEnabled();
};

}  // namespace vulkan
}  // namespace gpu
}  // namespace xe

#endif  // !defined(XENIA_GPU_DRAW_CUTS_PURE_LOGIC_ONLY)

#endif  // XENIA_GPU_VULKAN_VULKAN_COMMAND_PROCESSOR_H_
