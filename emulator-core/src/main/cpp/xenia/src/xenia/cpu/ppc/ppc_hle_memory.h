/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_CPU_PPC_PPC_HLE_MEMORY_H_
#define XENIA_CPU_PPC_PPC_HLE_MEMORY_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>

#include "xenia/base/xxhash.h"

namespace xe {
namespace cpu {
namespace ppc {

struct PPCContext_s;
using PPCContext = PPCContext_s;

enum class HleMemoryOp : uint8_t {
  kNone = 0,
  kMemcpy,
  kMemset,
  kMemmove,
};

struct HleMemoryRangePlan {
  bool valid = false;
  bool dst_is_physical = false;
  bool src_is_physical = false;
  uint32_t dst_phys = 0;
  uint32_t src_phys = 0;
  bool needs_dst_writewatch = false;
  bool needs_src_readwatch = false;
};

constexpr const char* HleMemoryOpName(HleMemoryOp op) {
  switch (op) {
    case HleMemoryOp::kMemcpy:
      return "memcpy";
    case HleMemoryOp::kMemset:
      return "memset";
    case HleMemoryOp::kMemmove:
      return "memmove";
    case HleMemoryOp::kNone:
    default:
      return "none";
  }
}

constexpr uint32_t MaskPpcRelocationBits(uint32_t big_endian_word) {
  // Mask LI [25:2] only on relative call `bl` (opcode 18, AA=0, LK=1).
  if ((big_endian_word & 0xFC000003u) == 0x48000001u) {
    return big_endian_word & 0xFC000003u;
  }
  return big_endian_word;
}

inline uint64_t ComputePpcFunctionSignature(const uint32_t* big_endian_words,
                                            size_t word_count) {
  if (!big_endian_words || word_count < 4) {
    return 0;
  }
  constexpr size_t kInlineWords = 256;
  if (word_count <= kInlineWords) {
    alignas(16) std::array<uint8_t, kInlineWords * sizeof(uint32_t)> buf{};
    for (size_t i = 0; i < word_count; ++i) {
      const uint32_t word = big_endian_words[i];
      if (word == 0u) {
        return 0;
      }
      const uint32_t masked = MaskPpcRelocationBits(word);
      std::memcpy(buf.data() + i * sizeof(uint32_t), &masked, sizeof(masked));
    }
    const uint64_t hash =
        XXH3_64bits(buf.data(), word_count * sizeof(uint32_t));
    return hash == 0u ? 1u : hash;
  }

  XXH3_state_t state;
  XXH3_64bits_reset(&state);
  for (size_t i = 0; i < word_count; ++i) {
    const uint32_t word = big_endian_words[i];
    if (word == 0u) {
      return 0;
    }
    const uint32_t masked = MaskPpcRelocationBits(word);
    XXH3_64bits_update(&state, &masked, sizeof(masked));
  }
  const uint64_t hash = XXH3_64bits_digest(&state);
  return hash == 0u ? 1u : hash;
}

constexpr bool CheckGuestRange32NoWrap(uint32_t addr, uint32_t len) {
  if (len == 0u) {
    return false;
  }
  return static_cast<uint64_t>(addr) + static_cast<uint64_t>(len) - 1ull <=
         0xFFFFFFFFull;
}

constexpr bool CrossesSkew16KbBoundary(uint32_t addr, uint32_t len,
                                       uint32_t alloc_granularity) {
  if (alloc_granularity <= 0x1000u || len == 0u) {
    return false;
  }
  if (addr >= 0xE0000000u) {
    return false;
  }
  const uint64_t end = static_cast<uint64_t>(addr) + static_cast<uint64_t>(len);
  return end > 0xE0000000ull;
}

constexpr bool PhysicalRangesOverlap(uint32_t dst_phys, uint32_t src_phys,
                                     uint32_t len) {
  if (len == 0u) {
    return false;
  }
  const uint64_t d0 = dst_phys;
  const uint64_t d1 = d0 + static_cast<uint64_t>(len);
  const uint64_t s0 = src_phys;
  const uint64_t s1 = s0 + static_cast<uint64_t>(len);
  return d0 < s1 && s0 < d1;
}

namespace detail {

constexpr int HexDigitValue(char ch) {
  if (ch >= '0' && ch <= '9') {
    return ch - '0';
  }
  if (ch >= 'a' && ch <= 'f') {
    return 10 + (ch - 'a');
  }
  if (ch >= 'A' && ch <= 'F') {
    return 10 + (ch - 'A');
  }
  return -1;
}

constexpr bool ParseHexUInt64Strict(std::string_view text, uint64_t* out,
                                    size_t max_digits) {
  if (text.size() >= 2 && text[0] == '0' &&
      (text[1] == 'x' || text[1] == 'X')) {
    text.remove_prefix(2);
  }
  if (text.empty() || text.size() > max_digits) {
    return false;
  }
  uint64_t val = 0;
  for (char ch : text) {
    const int d = HexDigitValue(ch);
    if (d < 0) {
      return false;
    }
    val = (val << 4) | static_cast<uint64_t>(d);
  }
  *out = val;
  return true;
}

constexpr bool EqualsIgnoreCaseAscii(std::string_view a, std::string_view b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (size_t i = 0; i < a.size(); ++i) {
    char ca = a[i];
    char cb = b[i];
    if (ca >= 'A' && ca <= 'Z') {
      ca = static_cast<char>(ca - 'A' + 'a');
    }
    if (cb >= 'A' && cb <= 'Z') {
      cb = static_cast<char>(cb - 'A' + 'a');
    }
    if (ca != cb) {
      return false;
    }
  }
  return true;
}

constexpr HleMemoryOp ParseHleOpName(std::string_view name) {
  if (EqualsIgnoreCaseAscii(name, "memcpy")) {
    return HleMemoryOp::kMemcpy;
  }
  if (EqualsIgnoreCaseAscii(name, "memset")) {
    return HleMemoryOp::kMemset;
  }
  if (EqualsIgnoreCaseAscii(name, "memmove")) {
    return HleMemoryOp::kMemmove;
  }
  return HleMemoryOp::kNone;
}

constexpr bool HasValidBlrTermination(const uint32_t* big_endian_words,
                                      size_t word_count) {
  if (!big_endian_words || word_count < 4) {
    return false;
  }
  return big_endian_words[word_count - 1] == 0x4E800020u;
}

struct CanonicalHleSignature {
  HleMemoryOp op;
  uint32_t exact_word_count;
  uint64_t xxh3_hash;
};

// Canonical Xbox 360 / PowerPC CRT memory routine signatures (matched only when
// exact_word_count and trailing `blr` (0x4E800020) both hold).
inline constexpr std::array<CanonicalHleSignature, 5> kCanonicalHleSignatures = {{
    // Minimal byte memset: cmplwi r5,0; beqlr; mtctr r5; addi r6,r3,-1;
    // stbu r4,1(r6); bdnz -4; blr
    {HleMemoryOp::kMemset, 7u, 0x54242DB4826F145BULL},
    // Byte memset with clrlwi r4,r4,24 masking
    {HleMemoryOp::kMemset, 8u, 0x69CF984463B81CBAULL},
    // Minimal byte memcpy: cmplwi r5,0; beqlr; mtctr r5; addi r6,r3,-1;
    // addi r4,r4,-1; lbzu r0,1(r4); stbu r0,1(r6); bdnz -8; blr
    {HleMemoryOp::kMemcpy, 9u, 0x012460B3136CA583ULL},
    // Indexed byte memcpy: cmplwi r5,0; beqlr; mtctr r5; li r6,0;
    // lbzx r0,r4,r6; stbx r0,r3,r6; addi r6,r6,1; bdnz -12; blr
    {HleMemoryOp::kMemcpy, 9u, 0xAAEF509FA1FA0939ULL},
    // Canonical directional byte memmove (forward/backward copy + blr)
    {HleMemoryOp::kMemmove, 19u, 0x99597BF7D25DF86EULL},
}};

}  // namespace detail

constexpr HleMemoryOp ParseHleOverrideEntry(std::string_view list,
                                            uint32_t guest_address,
                                            uint64_t actual_hash) {
  if (list.empty() || actual_hash == 0u) {
    return HleMemoryOp::kNone;
  }

  size_t pos = 0;
  while (pos < list.size()) {
    const size_t sep = list.find_first_of(",; \t\r\n", pos);
    const std::string_view token =
        sep == std::string_view::npos ? list.substr(pos)
                                      : list.substr(pos, sep - pos);
    pos = sep == std::string_view::npos ? list.size() : sep + 1;
    if (token.empty()) {
      continue;
    }

    const size_t c1 = token.find(':');
    if (c1 == std::string_view::npos) {
      continue;
    }
    const size_t c2 = token.find(':', c1 + 1);
    if (c2 == std::string_view::npos) {
      continue;
    }
    if (token.find(':', c2 + 1) != std::string_view::npos) {
      continue;
    }

    const std::string_view addr_sv = token.substr(0, c1);
    const std::string_view op_sv = token.substr(c1 + 1, c2 - c1 - 1);
    const std::string_view hash_sv = token.substr(c2 + 1);

    uint64_t parsed_addr = 0;
    if (!detail::ParseHexUInt64Strict(addr_sv, &parsed_addr, 8u) ||
        static_cast<uint32_t>(parsed_addr) != guest_address) {
      continue;
    }

    const HleMemoryOp parsed_op = detail::ParseHleOpName(op_sv);
    if (parsed_op == HleMemoryOp::kNone) {
      continue;
    }

    uint64_t parsed_hash = 0;
    if (!detail::ParseHexUInt64Strict(hash_sv, &parsed_hash, 16u) ||
        parsed_hash != actual_hash) {
      continue;
    }

    return parsed_op;
  }

  return HleMemoryOp::kNone;
}

inline HleMemoryOp MatchHleMemorySignature(uint32_t guest_address,
                                           const uint32_t* be_words,
                                           size_t word_count,
                                           std::string_view override_list,
                                           uint64_t* out_hash = nullptr) {
  const uint64_t hash = ComputePpcFunctionSignature(be_words, word_count);
  if (out_hash) {
    *out_hash = hash;
  }
  if (hash == 0u) {
    return HleMemoryOp::kNone;
  }

  const HleMemoryOp override_op =
      ParseHleOverrideEntry(override_list, guest_address, hash);
  if (override_op != HleMemoryOp::kNone) {
    return override_op;
  }

  if (!detail::HasValidBlrTermination(be_words, word_count)) {
    return HleMemoryOp::kNone;
  }
  for (const auto& entry : detail::kCanonicalHleSignatures) {
    if (entry.exact_word_count == word_count && entry.xxh3_hash == hash) {
      return entry.op;
    }
  }
  return HleMemoryOp::kNone;
}

void HleMemcpy(PPCContext* ppc_context, void* arg0, void* arg1);
void HleMemset(PPCContext* ppc_context, void* arg0, void* arg1);
void HleMemmove(PPCContext* ppc_context, void* arg0, void* arg1);

}  // namespace ppc
}  // namespace cpu
}  // namespace xe

#endif  // XENIA_CPU_PPC_PPC_HLE_MEMORY_H_
