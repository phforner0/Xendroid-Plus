/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_PM4_IB_CACHE_H_
#define XENIA_GPU_PM4_IB_CACHE_H_

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace xe {
namespace gpu {
namespace pm4_ib {

// Pure-logic PM4_INDIRECT_BUFFER cache validated by lock-free atomic-epoch
// physical memory write-watches (Onda 3B, onda3-minispec.md §3.2).
//
// Threading model (identical to Onda 2B ShaderLoadWatchCache):
// - Guest CPU fault handler threads invoke OnMemoryInvalidatedForUnwatch() /
//   OnMemoryInvalidated() wait-free, lock-free, with zero heap allocations.
// - GPU command processor thread exclusively calls Lookup(), ObserveMiss(),
//   BeginArmAndSnapshot(), CommitIfUnchanged(), AdvanceFrame(),
//   InvalidateAll(), and MarkCandidateNonCacheable().

constexpr uint32_t kIbCachePhysicalMemoryMask = 0x1FFFFFFFu;
constexpr uint32_t kIbCachePhysicalMemorySize = 512u * 1024u * 1024u;
constexpr uint32_t kIbCachePageShift = 12u;
constexpr uint32_t kIbCachePageCount =
    kIbCachePhysicalMemorySize >> kIbCachePageShift;
constexpr uint32_t kIbCacheMaxDwords = 4096u;  // 16 KiB
constexpr uint32_t kIbCacheMaxPagesPerIb = 8u;
constexpr size_t kIbCacheMaxEntries = 1024u;
constexpr size_t kIbCacheCandidateTableSize = 256u;
constexpr size_t kIbCachePageCooldownTableSize = 256u;
constexpr uint32_t kIbCacheHysteresisHits = 2u;
constexpr uint64_t kIbCacheInvalidationCooldownFrames = 64u;
constexpr uint32_t kIbCacheMaxUnwatchExcess = 4u * 1024u * 1024u;

// Canonical PM4 Type-3 opcode constants (verified against xenos.h).
namespace opcode {
constexpr uint32_t kNop = 0x10u;
constexpr uint32_t kRegRmw = 0x21u;
constexpr uint32_t kDrawIndx = 0x22u;
constexpr uint32_t kVizQuery = 0x23u;
constexpr uint32_t kWaitForIdle = 0x26u;
constexpr uint32_t kImLoad = 0x27u;
constexpr uint32_t kImLoadImmediate = 0x2Bu;
constexpr uint32_t kSetConstant = 0x2Du;
constexpr uint32_t kLoadAluConstant = 0x2Fu;
constexpr uint32_t kDrawIndx2 = 0x36u;
constexpr uint32_t kIndirectBufferPfd = 0x37u;
constexpr uint32_t kInvalidateState = 0x3Bu;
constexpr uint32_t kWaitRegMem = 0x3Cu;
constexpr uint32_t kMemWrite = 0x3Du;
constexpr uint32_t kRegToMem = 0x3Eu;
constexpr uint32_t kIndirectBuffer = 0x3Fu;
constexpr uint32_t kCondWrite = 0x45u;
constexpr uint32_t kEventWrite = 0x46u;
constexpr uint32_t kMeInit = 0x48u;
constexpr uint32_t kSetBinMask = 0x50u;
constexpr uint32_t kSetBinSelect = 0x51u;
constexpr uint32_t kInterrupt = 0x54u;
constexpr uint32_t kSetConstant2 = 0x55u;
constexpr uint32_t kSetShaderConstants = 0x56u;
constexpr uint32_t kEventWriteShd = 0x58u;
constexpr uint32_t kEventWriteCfl = 0x59u;
constexpr uint32_t kEventWriteExt = 0x5Au;
constexpr uint32_t kEventWriteZpd = 0x5Bu;
constexpr uint32_t kContextUpdate = 0x5Eu;
constexpr uint32_t kSetBinMaskLo = 0x60u;
constexpr uint32_t kSetBinMaskHi = 0x61u;
constexpr uint32_t kSetBinSelectLo = 0x62u;
constexpr uint32_t kSetBinSelectHi = 0x63u;
constexpr uint32_t kXeSwap = 0x64u;
}  // namespace opcode

enum class IbOpKind : uint8_t {
  kType0WriteRange = 0,  // base_reg, count, one_reg_wr, data_offset
  kType1WritePair = 1,   // base_reg (reg1), aux_reg (reg2), data_offset (2 dwords)
  kSetConstant = 2,      // arg0=type_and_offset, count, data_offset, predicated
  kSetConstant2 = 3,     // base_reg, count, data_offset, predicated (also SET_SHADER_CONSTANTS)
  kLoadAluConstant = 4,  // arg0=addr(&0x3FFFFFFF), arg1=offset_type, arg2=size_dwords(&0xFFF), predicated
  kImLoad = 5,           // arg0=shader_type, arg1=addr, arg2=size_dwords, predicated
  kDrawIndx = 6,         // arg0=viz_cond, arg1=initiator, arg2=dma_base, arg3=dma_size, post_packet_byte_offset, predicated
  kDrawIndx2 = 7,        // arg0=0, arg1=initiator, arg2=dma_base, arg3=dma_size, post_packet_byte_offset, predicated
  kBinMaskLo = 8,        // arg0=value, predicated
  kBinMaskHi = 9,        // arg0=value, predicated
  kBinSelectLo = 10,     // arg0=value, predicated (calls OnBinSelectWritten)
  kBinSelectHi = 11,     // arg0=value, predicated
  kBinMask = 12,         // arg0=val_hi, arg1=val_lo, predicated
  kBinSelect = 13,       // arg0=val_hi, arg1=val_lo, predicated
  kNestedIb = 14,        // arg0=list_ptr_gpu, arg1=list_length, post_packet_byte_offset, predicated (NEVER inlined!)
};

struct DecodedIbOp {
  IbOpKind kind = IbOpKind::kType0WriteRange;
  bool predicated = false;
  bool one_reg_wr = false;
  uint8_t flags = 0;  // bit 0: is_indexed for DrawIndx/DrawIndx2
  uint16_t base_reg = 0;
  uint16_t aux_reg = 0;
  uint32_t count = 0;
  uint32_t data_offset = 0;
  // Exact byte offset inside the IB right past this packet, so replay can set
  // reader_.set_read_offset(post_packet_byte_offset) before
  // PrepareTilingBandDraw() and recursive ExecuteIndirectBuffer() (preserving
  // GuestReadPtrOffset() / ib_call_depth_ / ib_call_sites_ bit-for-bit).
  uint32_t post_packet_byte_offset = 0;
  uint32_t arg0 = 0;
  uint32_t arg1 = 0;
  uint32_t arg2 = 0;
  uint32_t arg3 = 0;
};

struct CompiledIndirectBuffer {
  uint32_t phys_address = 0;
  uint32_t dword_count = 0;
  bool has_nested_ib = false;
  std::vector<DecodedIbOp> ops;
  // Contiguous host-endian (pre-byte-swapped) register/constant payloads.
  std::vector<uint32_t> swapped_dwords;
};

enum class DecodeStatus : uint8_t {
  kOk = 0,
  kBailout = 1,    // Non-cacheable opcode (WAIT_REG_MEM, MEM_WRITE, etc.)
  kMalformed = 2,  // Truncated / overflowing / uninitialized packet
};

inline uint32_t LoadSwap32(const uint32_t* ptr) noexcept {
  uint32_t v = 0;
  std::memcpy(&v, ptr, sizeof(v));
#if defined(_MSC_VER)
  return _byteswap_ulong(v);
#else
  return __builtin_bswap32(v);
#endif
}

// Fast 64-bit content hash over raw guest IB dwords (used for 2-hit hysteresis).
inline uint64_t ComputeIbContentHash64(const uint32_t* host_ib_ptr,
                                       uint32_t dword_count) noexcept {
  uint64_t h = UINT64_C(0xCBF29CE484222325) ^ (uint64_t(dword_count) << 32);
  uint32_t i = 0;
  while (i + 2u <= dword_count) {
    uint64_t q = 0;
    std::memcpy(&q, host_ib_ptr + i, sizeof(q));
    h ^= q;
    h *= UINT64_C(0x100000001B3);
    h ^= (h >> 33);
    i += 2u;
  }
  if (i < dword_count) {
    uint32_t tail = 0;
    std::memcpy(&tail, host_ib_ptr + i, sizeof(tail));
    h ^= uint64_t(tail);
    h *= UINT64_C(0x100000001B3);
    h ^= (h >> 33);
  }
  h *= UINT64_C(0xFF51AFD7ED558CCD);
  h ^= (h >> 33);
  return h;
}

// Returns true iff a Type-3 opcode is explicitly disallowed from IB caching
// (sync, readback, memory write, interrupt, swap, query, embedded shader, etc.).
constexpr bool IsMandatoryBailoutType3Opcode(uint32_t op) noexcept {
  switch (op) {
    case opcode::kWaitRegMem:
    case opcode::kCondWrite:
    case opcode::kRegRmw:
    case opcode::kRegToMem:
    case opcode::kMemWrite:
    case opcode::kInterrupt:
    case opcode::kXeSwap:
    case opcode::kMeInit:
    case opcode::kEventWrite:
    case opcode::kEventWriteShd:
    case opcode::kEventWriteCfl:
    case opcode::kEventWriteExt:
    case opcode::kEventWriteZpd:
    case opcode::kVizQuery:
    case opcode::kImLoadImmediate:
    case opcode::kWaitForIdle:
    case opcode::kIndirectBufferPfd:
      return true;
    default:
      return false;
  }
}

// Fast header-only scan before arming an OS write-watch: rejects IBs that
// contain mandatory bailout opcodes or malformed lengths without touching
// OS page protections (zero wasted mprotect syscalls on non-cacheable IBs).
inline DecodeStatus PrecheckCacheableIndirectBuffer(const uint32_t* host_ib_ptr,
                                                    uint32_t dword_count) {
  if (!host_ib_ptr || dword_count == 0u || dword_count > kIbCacheMaxDwords) {
    return DecodeStatus::kMalformed;
  }
  uint32_t pos = 0;
  while (pos < dword_count) {
    const uint32_t packet = LoadSwap32(host_ib_ptr + pos++);
    if (packet == 0u || packet == 0x0BADF00Du) {
      continue;
    }
    if (packet == 0xCDCDCDCDu) {
      return DecodeStatus::kMalformed;
    }
    const uint32_t packet_type = packet >> 30;
    if (packet_type == 0u) {
      const uint32_t count = ((packet >> 16) & 0x3FFFu) + 1u;
      if (count > dword_count - pos) {
        return DecodeStatus::kMalformed;
      }
      pos += count;
    } else if (packet_type == 1u) {
      if (2u > dword_count - pos) {
        return DecodeStatus::kMalformed;
      }
      pos += 2u;
    } else if (packet_type == 2u) {
      continue;
    } else {
      const uint32_t op = (packet >> 8) & 0x7Fu;
      const uint32_t count = ((packet >> 16) & 0x3FFFu) + 1u;
      if (count > dword_count - pos) {
        return DecodeStatus::kMalformed;
      }
      if (IsMandatoryBailoutType3Opcode(op)) {
        return DecodeStatus::kBailout;
      }
      switch (op) {
        case opcode::kNop:
          break;
        case opcode::kSetConstant: {
          if (count < 1u) {
            return DecodeStatus::kMalformed;
          }
          const uint32_t offset_type = LoadSwap32(host_ib_ptr + pos);
          const uint32_t type = (offset_type >> 16) & 0xFFu;
          if (type > 4u) {
            return DecodeStatus::kBailout;
          }
          break;
        }
        case opcode::kSetConstant2:
        case opcode::kSetShaderConstants:
          if (count < 1u) {
            return DecodeStatus::kMalformed;
          }
          break;
        case opcode::kLoadAluConstant: {
          if (count != 3u) {
            return DecodeStatus::kBailout;
          }
          const uint32_t offset_type = LoadSwap32(host_ib_ptr + pos + 1u);
          const uint32_t type = (offset_type >> 16) & 0xFFu;
          if (type > 4u) {
            return DecodeStatus::kBailout;
          }
          break;
        }
        case opcode::kImLoad: {
          if (count != 2u) {
            return DecodeStatus::kBailout;
          }
          const uint32_t addr_type = LoadSwap32(host_ib_ptr + pos);
          const uint32_t start_size = LoadSwap32(host_ib_ptr + pos + 1u);
          if ((addr_type & 0x3u) > 1u || (start_size >> 16) != 0u) {
            return DecodeStatus::kBailout;
          }
          break;
        }
        case opcode::kDrawIndx:
        case opcode::kDrawIndx2: {
          const uint32_t init_offset = (op == opcode::kDrawIndx) ? 1u : 0u;
          if (count <= init_offset) {
            return DecodeStatus::kMalformed;
          }
          const uint32_t initiator = LoadSwap32(host_ib_ptr + pos + init_offset);
          const uint32_t src_sel = (initiator >> 6) & 0x3u;
          // SourceSelect::kDMA == 0, SourceSelect::kAutoIndex == 2.
          // SourceSelect::kImmediate == 1 or reserved 3 -> bailout.
          if (src_sel == 0u) {
            if (count < init_offset + 3u) {
              return DecodeStatus::kMalformed;
            }
          } else if (src_sel != 2u) {
            return DecodeStatus::kBailout;
          }
          break;
        }
        case opcode::kSetBinMaskLo:
        case opcode::kSetBinMaskHi:
        case opcode::kSetBinSelectLo:
        case opcode::kSetBinSelectHi:
          if (count != 1u) {
            return DecodeStatus::kMalformed;
          }
          break;
        case opcode::kSetBinMask:
        case opcode::kSetBinSelect:
          if (count != 2u) {
            return DecodeStatus::kMalformed;
          }
          break;
        case opcode::kIndirectBuffer: {
          if (count != 2u) {
            return DecodeStatus::kMalformed;
          }
          const uint32_t list_length = LoadSwap32(host_ib_ptr + pos + 1u);
          if ((list_length & ~0xFFFFFu) != 0u) {
            return DecodeStatus::kMalformed;
          }
          break;
        }
        case opcode::kInvalidateState:
          if (count != 1u) {
            return DecodeStatus::kBailout;
          }
          break;
        case opcode::kContextUpdate:
          if (count != 1u || LoadSwap32(host_ib_ptr + pos) != 0u) {
            return DecodeStatus::kBailout;
          }
          break;
        default:
          return DecodeStatus::kBailout;
      }
      pos += count;
    }
  }
  return DecodeStatus::kOk;
}

// Decodes a big-endian guest PM4 indirect buffer into a compact host-endian
// op stream (`out`). Never inlines nested IBs (`kNestedIb` preserves recursive
// ExecuteIndirectBuffer semantics, ib_call_depth_, and ib_call_sites_).
inline DecodeStatus DecodeIndirectBuffer(uint32_t phys_address,
                                         const uint32_t* host_ib_ptr,
                                         uint32_t dword_count,
                                         CompiledIndirectBuffer& out) {
  out.phys_address = phys_address & kIbCachePhysicalMemoryMask;
  out.dword_count = dword_count;
  out.has_nested_ib = false;
  out.ops.clear();
  out.swapped_dwords.clear();

  if (!host_ib_ptr || dword_count == 0u || dword_count > kIbCacheMaxDwords) {
    return DecodeStatus::kMalformed;
  }

  // Reserve approximate space up-front to avoid repeated reallocations.
  out.ops.reserve(std::min<uint32_t>(dword_count, 128u));
  out.swapped_dwords.reserve(dword_count);

  uint32_t pos = 0;
  while (pos < dword_count) {
    const uint32_t packet = LoadSwap32(host_ib_ptr + pos++);
    if (packet == 0u || packet == 0x0BADF00Du) {
      continue;
    }
    if (packet == 0xCDCDCDCDu) {
      return DecodeStatus::kMalformed;
    }
    const uint32_t packet_type = packet >> 30;
    if (packet_type == 0u) {
      const uint32_t count = ((packet >> 16) & 0x3FFFu) + 1u;
      if (count > dword_count - pos) {
        return DecodeStatus::kMalformed;
      }
      DecodedIbOp op{};
      op.kind = IbOpKind::kType0WriteRange;
      op.base_reg = static_cast<uint16_t>(packet & 0x7FFFu);
      op.one_reg_wr = ((packet >> 15) & 0x1u) != 0u;
      op.count = count;
      op.data_offset = static_cast<uint32_t>(out.swapped_dwords.size());
      for (uint32_t i = 0; i < count; ++i) {
        out.swapped_dwords.push_back(LoadSwap32(host_ib_ptr + pos + i));
      }
      pos += count;
      op.post_packet_byte_offset = pos * sizeof(uint32_t);
      out.ops.push_back(op);
    } else if (packet_type == 1u) {
      if (2u > dword_count - pos) {
        return DecodeStatus::kMalformed;
      }
      DecodedIbOp op{};
      op.kind = IbOpKind::kType1WritePair;
      op.base_reg = static_cast<uint16_t>(packet & 0x7FFu);
      op.aux_reg = static_cast<uint16_t>((packet >> 11) & 0x7FFu);
      op.count = 2u;
      op.data_offset = static_cast<uint32_t>(out.swapped_dwords.size());
      out.swapped_dwords.push_back(LoadSwap32(host_ib_ptr + pos));
      out.swapped_dwords.push_back(LoadSwap32(host_ib_ptr + pos + 1u));
      pos += 2u;
      op.post_packet_byte_offset = pos * sizeof(uint32_t);
      out.ops.push_back(op);
    } else if (packet_type == 2u) {
      // Type-2 no-op.
      continue;
    } else {
      const uint32_t pkt_op = (packet >> 8) & 0x7Fu;
      const uint32_t count = ((packet >> 16) & 0x3FFFu) + 1u;
      const bool predicated = (packet & 1u) != 0u;
      if (count > dword_count - pos) {
        return DecodeStatus::kMalformed;
      }
      if (IsMandatoryBailoutType3Opcode(pkt_op)) {
        return DecodeStatus::kBailout;
      }
      const uint32_t payload_pos = pos;
      pos += count;
      const uint32_t post_byte_offset = pos * sizeof(uint32_t);

      switch (pkt_op) {
        case opcode::kNop:
          break;
        case opcode::kSetConstant: {
          if (count < 1u) {
            return DecodeStatus::kMalformed;
          }
          const uint32_t offset_type = LoadSwap32(host_ib_ptr + payload_pos);
          const uint32_t type = (offset_type >> 16) & 0xFFu;
          if (type > 4u) {
            return DecodeStatus::kBailout;
          }
          const uint32_t countm1 = count - 1u;
          DecodedIbOp op{};
          op.kind = IbOpKind::kSetConstant;
          op.predicated = predicated;
          op.arg0 = offset_type;
          op.count = countm1;
          op.data_offset = static_cast<uint32_t>(out.swapped_dwords.size());
          op.post_packet_byte_offset = post_byte_offset;
          for (uint32_t i = 0; i < countm1; ++i) {
            out.swapped_dwords.push_back(
                LoadSwap32(host_ib_ptr + payload_pos + 1u + i));
          }
          out.ops.push_back(op);
          break;
        }
        case opcode::kSetConstant2:
        case opcode::kSetShaderConstants: {
          if (count < 1u) {
            return DecodeStatus::kMalformed;
          }
          const uint32_t offset_type = LoadSwap32(host_ib_ptr + payload_pos);
          const uint32_t countm1 = count - 1u;
          DecodedIbOp op{};
          op.kind = IbOpKind::kSetConstant2;
          op.predicated = predicated;
          op.base_reg = static_cast<uint16_t>(offset_type & 0xFFFFu);
          op.count = countm1;
          op.data_offset = static_cast<uint32_t>(out.swapped_dwords.size());
          op.post_packet_byte_offset = post_byte_offset;
          for (uint32_t i = 0; i < countm1; ++i) {
            out.swapped_dwords.push_back(
                LoadSwap32(host_ib_ptr + payload_pos + 1u + i));
          }
          out.ops.push_back(op);
          break;
        }
        case opcode::kLoadAluConstant: {
          if (count != 3u) {
            return DecodeStatus::kBailout;
          }
          const uint32_t address =
              LoadSwap32(host_ib_ptr + payload_pos) & 0x3FFFFFFFu;
          const uint32_t offset_type =
              LoadSwap32(host_ib_ptr + payload_pos + 1u);
          const uint32_t size_dwords =
              LoadSwap32(host_ib_ptr + payload_pos + 2u) & 0xFFFu;
          const uint32_t type = (offset_type >> 16) & 0xFFu;
          if (type > 4u) {
            return DecodeStatus::kBailout;
          }
          DecodedIbOp op{};
          op.kind = IbOpKind::kLoadAluConstant;
          op.predicated = predicated;
          op.arg0 = address;
          op.arg1 = offset_type;
          op.arg2 = size_dwords;
          op.post_packet_byte_offset = post_byte_offset;
          out.ops.push_back(op);
          break;
        }
        case opcode::kImLoad: {
          if (count != 2u) {
            return DecodeStatus::kBailout;
          }
          const uint32_t addr_type = LoadSwap32(host_ib_ptr + payload_pos);
          const uint32_t start_size =
              LoadSwap32(host_ib_ptr + payload_pos + 1u);
          const uint32_t shader_type = addr_type & 0x3u;
          const uint32_t start = start_size >> 16;
          if (shader_type > 1u || start != 0u) {
            return DecodeStatus::kBailout;
          }
          DecodedIbOp op{};
          op.kind = IbOpKind::kImLoad;
          op.predicated = predicated;
          op.arg0 = shader_type;
          op.arg1 = addr_type & ~0x3u;
          op.arg2 = start_size & 0xFFFFu;
          op.post_packet_byte_offset = post_byte_offset;
          out.ops.push_back(op);
          break;
        }
        case opcode::kDrawIndx:
        case opcode::kDrawIndx2: {
          const bool is_draw_indx = (pkt_op == opcode::kDrawIndx);
          const uint32_t init_offset = is_draw_indx ? 1u : 0u;
          if (count <= init_offset) {
            return DecodeStatus::kMalformed;
          }
          const uint32_t viz_cond =
              is_draw_indx ? LoadSwap32(host_ib_ptr + payload_pos) : 0u;
          const uint32_t initiator =
              LoadSwap32(host_ib_ptr + payload_pos + init_offset);
          const uint32_t src_sel = (initiator >> 6) & 0x3u;
          uint32_t dma_base = 0;
          uint32_t dma_size = 0;
          uint8_t flags = 0;
          if (src_sel == 0u) {  // SourceSelect::kDMA
            if (count < init_offset + 3u) {
              return DecodeStatus::kMalformed;
            }
            dma_base = LoadSwap32(host_ib_ptr + payload_pos + init_offset + 1u);
            dma_size = LoadSwap32(host_ib_ptr + payload_pos + init_offset + 2u);
            flags = 1u;
          } else if (src_sel != 2u) {  // SourceSelect::kAutoIndex
            return DecodeStatus::kBailout;
          }
          DecodedIbOp op{};
          op.kind = is_draw_indx ? IbOpKind::kDrawIndx : IbOpKind::kDrawIndx2;
          op.predicated = predicated;
          op.flags = flags;
          op.arg0 = viz_cond;
          op.arg1 = initiator;
          op.arg2 = dma_base;
          op.arg3 = dma_size;
          op.post_packet_byte_offset = post_byte_offset;
          out.ops.push_back(op);
          break;
        }
        case opcode::kSetBinMaskLo:
        case opcode::kSetBinMaskHi:
        case opcode::kSetBinSelectLo:
        case opcode::kSetBinSelectHi: {
          if (count != 1u) {
            return DecodeStatus::kMalformed;
          }
          DecodedIbOp op{};
          if (pkt_op == opcode::kSetBinMaskLo) {
            op.kind = IbOpKind::kBinMaskLo;
          } else if (pkt_op == opcode::kSetBinMaskHi) {
            op.kind = IbOpKind::kBinMaskHi;
          } else if (pkt_op == opcode::kSetBinSelectLo) {
            op.kind = IbOpKind::kBinSelectLo;
          } else {
            op.kind = IbOpKind::kBinSelectHi;
          }
          op.predicated = predicated;
          op.arg0 = LoadSwap32(host_ib_ptr + payload_pos);
          op.post_packet_byte_offset = post_byte_offset;
          out.ops.push_back(op);
          break;
        }
        case opcode::kSetBinMask:
        case opcode::kSetBinSelect: {
          if (count != 2u) {
            return DecodeStatus::kMalformed;
          }
          DecodedIbOp op{};
          op.kind = (pkt_op == opcode::kSetBinMask) ? IbOpKind::kBinMask
                                                    : IbOpKind::kBinSelect;
          op.predicated = predicated;
          op.arg0 = LoadSwap32(host_ib_ptr + payload_pos);       // val_hi
          op.arg1 = LoadSwap32(host_ib_ptr + payload_pos + 1u);  // val_lo
          op.post_packet_byte_offset = post_byte_offset;
          out.ops.push_back(op);
          break;
        }
        case opcode::kIndirectBuffer: {
          if (count != 2u) {
            return DecodeStatus::kMalformed;
          }
          const uint32_t list_ptr_raw = LoadSwap32(host_ib_ptr + payload_pos);
          const uint32_t list_length =
              LoadSwap32(host_ib_ptr + payload_pos + 1u);
          if ((list_length & ~0xFFFFFu) != 0u) {
            return DecodeStatus::kMalformed;
          }
          DecodedIbOp op{};
          op.kind = IbOpKind::kNestedIb;
          op.predicated = predicated;
          // Store raw swapped guest ptr (same value read before CpuToGpu /
          // GpuToCpu round-trip in ExecutePacketType3_INDIRECT_BUFFER).
          op.arg0 = list_ptr_raw;
          op.arg1 = list_length & 0xFFFFFu;
          op.post_packet_byte_offset = post_byte_offset;
          out.has_nested_ib = true;
          out.ops.push_back(op);
          break;
        }
        case opcode::kInvalidateState:
          if (count != 1u) {
            return DecodeStatus::kBailout;
          }
          break;
        case opcode::kContextUpdate:
          if (count != 1u || LoadSwap32(host_ib_ptr + payload_pos) != 0u) {
            return DecodeStatus::kBailout;
          }
          break;
        default:
          return DecodeStatus::kBailout;
      }
    }
  }
  return DecodeStatus::kOk;
}

struct IbWatchSnapshot {
  uint32_t phys_address = 0;
  uint32_t byte_length = 0;
  uint32_t first_page = 0;
  uint32_t page_count = 0;
  uint64_t global_epoch = 0;
  std::array<uint32_t, kIbCacheMaxPagesPerIb> page_epochs{};
};

enum class CandidateDecision : uint8_t {
  kSkipInvalidRange = 0,
  kInCooldown = 1,
  kNonCacheable = 2,
  kFirstSeen = 3,      // Recorded hash64; do NOT arm write-watch yet
  kDynamicReset = 4,   // Content changed between visits; reset hysteresis
  kReadyToCompile = 5, // 2 consecutive identical visits and not in cooldown
};

struct IbCacheStats {
  uint64_t hits = 0;
  uint64_t misses = 0;
  uint64_t compilations = 0;
  uint64_t watch_invalidations = 0;
  uint64_t cooldown_skips = 0;
  uint64_t non_cacheable_skips = 0;
  uint64_t evictions = 0;
};

class Pm4IndirectBufferCache {
 public:
  Pm4IndirectBufferCache()
      : page_epochs_(
            std::make_unique<std::atomic<uint32_t>[]>(kIbCachePageCount)),
        armed_pages_(std::make_unique<std::atomic<uint64_t>[]>(
            kIbCachePageCount / 64u)) {
    for (uint32_t i = 0; i < kIbCachePageCount; ++i) {
      page_epochs_[i].store(0u, std::memory_order_relaxed);
    }
    for (uint32_t i = 0; i < kIbCachePageCount / 64u; ++i) {
      armed_pages_[i].store(0u, std::memory_order_relaxed);
    }
  }

  // Wait-free, lock-free, allocation-free physical memory invalidation hook
  // (runs on Guest CPU fault handler thread or trace playback).
  // Clamps Memory's unwatch range to (physical_address, length) if any OTHER
  // armed IB page lives within the 4 MiB max unwatch excess window.
  std::pair<uint32_t, uint32_t> OnMemoryInvalidatedForUnwatch(
      uint32_t physical_address, uint32_t length, bool exact_range) {
    if (!length) {
      return {0u, UINT32_MAX};
    }
    const uint32_t phys = physical_address & kIbCachePhysicalMemoryMask;
    const uint32_t first_page = phys >> kIbCachePageShift;
    if (first_page >= kIbCachePageCount) {
      return {0u, UINT32_MAX};
    }
    const uint64_t end_excl = std::min<uint64_t>(
        uint64_t(phys) + uint64_t(length), kIbCachePhysicalMemorySize);
    if (end_excl <= phys) {
      return {0u, UINT32_MAX};
    }
    const uint32_t last_page =
        static_cast<uint32_t>((end_excl - 1u) >> kIbCachePageShift);
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
    const uint32_t win_first_addr = phys & ~(kIbCacheMaxUnwatchExcess - 1u);
    const uint32_t win_last_addr = std::min<uint32_t>(
        static_cast<uint32_t>((end_excl - 1u) |
                              (kIbCacheMaxUnwatchExcess - 1u)),
        kIbCachePhysicalMemorySize - 1u);
    const uint32_t win_first_word =
        (win_first_addr >> kIbCachePageShift) >> 6;
    const uint32_t win_last_word = (win_last_addr >> kIbCachePageShift) >> 6;
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

  // GPU thread only: advances the frame counter used for 64-frame cooldown.
  void AdvanceFrame() { ++current_frame_; }
  uint64_t current_frame() const { return current_frame_; }

  // GPU thread only: invalidates all cached entries, clears armed page bits
  // (so Memory stops clamping unwatches while disabled/cleared), and resets
  // hysteresis/cooldown tables.
  void InvalidateAll() {
    global_epoch_.fetch_add(1u, std::memory_order_acq_rel);
    entries_.clear();
    fifo_keys_.clear();
    for (uint32_t i = 0; i < kIbCachePageCount / 64u; ++i) {
      armed_pages_[i].store(0u, std::memory_order_release);
    }
    candidates_.fill(CandidateSlot{});
    page_cooldowns_.fill(PageCooldownSlot{});
  }

  // GPU thread only: returns a shared_ptr to the compiled host-endian IB if
  // present and every watched page epoch still matches its snapshot.
  // If a write-watch invalidation is observed on a cached entry, erases it and
  // places the modified page(s) into 64-frame anti-thrash cooldown.
  std::shared_ptr<const CompiledIndirectBuffer> Lookup(uint32_t guest_address,
                                                       uint32_t dword_count) {
    if (!dword_count || dword_count > kIbCacheMaxDwords) {
      return nullptr;
    }
    const uint32_t phys = guest_address & kIbCachePhysicalMemoryMask;
    const uint64_t key = MakeKey(phys, dword_count);
    auto it = entries_.find(key);
    if (it == entries_.end()) {
      ++stats_.misses;
      return nullptr;
    }
    const Entry& e = it->second;
    if (e.global_epoch != global_epoch_.load(std::memory_order_acquire)) {
      entries_.erase(it);
      ++stats_.misses;
      return nullptr;
    }
    bool invalidated_by_watch = false;
    for (uint32_t i = 0; i < e.page_count; ++i) {
      const uint32_t page = e.first_page + i;
      if (page_epochs_[page].load(std::memory_order_acquire) !=
          e.page_epochs[i]) {
        invalidated_by_watch = true;
        ArmPageCooldown(page);
      }
    }
    if (invalidated_by_watch) {
      ++stats_.watch_invalidations;
      ++stats_.misses;
      ResetCandidateFor(phys, dword_count);
      entries_.erase(it);
      return nullptr;
    }
    ++stats_.hits;
    return e.compiled;
  }

  // GPU thread only: applies 64-frame cooldown and 2-hit content hysteresis
  // on cache misses BEFORE arming any OS write-watch.
  CandidateDecision ObserveMiss(uint32_t guest_address,
                                const uint32_t* host_ib_ptr,
                                uint32_t dword_count) {
    uint32_t phys = 0, first_page = 0, page_count = 0;
    if (!host_ib_ptr ||
        !ComputePageRange(guest_address, dword_count, phys, first_page,
                          page_count)) {
      return CandidateDecision::kSkipInvalidRange;
    }
    if (IsPageRangeInCooldown(first_page, page_count)) {
      ++stats_.cooldown_skips;
      return CandidateDecision::kInCooldown;
    }
    const uint64_t hash64 = ComputeIbContentHash64(host_ib_ptr, dword_count);
    CandidateSlot& slot = candidates_[CandidateSlotIndex(phys, dword_count)];
    if (slot.valid && slot.phys_address == phys &&
        slot.dword_count == dword_count) {
      if (slot.content_hash == hash64) {
        if (slot.non_cacheable) {
          ++stats_.non_cacheable_skips;
          return CandidateDecision::kNonCacheable;
        }
        if (slot.consecutive_hits < kIbCacheHysteresisHits) {
          ++slot.consecutive_hits;
        }
        if (slot.consecutive_hits >= kIbCacheHysteresisHits) {
          return CandidateDecision::kReadyToCompile;
        }
        return CandidateDecision::kFirstSeen;
      }
      // Same address/length but bytes changed: dynamic IB, reset hysteresis.
      slot.content_hash = hash64;
      slot.consecutive_hits = 1u;
      slot.non_cacheable = false;
      return CandidateDecision::kDynamicReset;
    }
    slot.phys_address = phys;
    slot.dword_count = dword_count;
    slot.content_hash = hash64;
    slot.consecutive_hits = 1u;
    slot.non_cacheable = false;
    slot.valid = true;
    return CandidateDecision::kFirstSeen;
  }

  // Marks the current candidate matching (guest_address, dword_count) as
  // non-cacheable (bailout opcode encountered) so future calls skip arming.
  void MarkCandidateNonCacheable(uint32_t guest_address, uint32_t dword_count) {
    const uint32_t phys = guest_address & kIbCachePhysicalMemoryMask;
    CandidateSlot& slot = candidates_[CandidateSlotIndex(phys, dword_count)];
    if (slot.valid && slot.phys_address == phys &&
        slot.dword_count == dword_count) {
      slot.non_cacheable = true;
    }
  }

  bool IsRangeInCooldown(uint32_t guest_address, uint32_t dword_count) const {
    uint32_t phys = 0, first_page = 0, page_count = 0;
    if (!ComputePageRange(guest_address, dword_count, phys, first_page,
                          page_count)) {
      return false;
    }
    return IsPageRangeInCooldown(first_page, page_count);
  }

  // GPU thread only: marks pages armed and snapshots epochs BEFORE enabling OS
  // write-watch and decoding the IB payload.
  bool BeginArmAndSnapshot(uint32_t guest_address, uint32_t dword_count,
                           IbWatchSnapshot& out) {
    uint32_t phys = 0, first_page = 0, page_count = 0;
    if (!ComputePageRange(guest_address, dword_count, phys, first_page,
                          page_count)) {
      return false;
    }
    if (IsPageRangeInCooldown(first_page, page_count)) {
      return false;
    }
    out.phys_address = phys;
    out.byte_length = dword_count * uint32_t(sizeof(uint32_t));
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

  // GPU thread only: commits `compiled` ONLY if zero invalidations happened
  // across [BeginArmAndSnapshot .. now]. On page-epoch mismatch (TOCTOU race),
  // enters 64-frame cooldown on the modified page(s) and rejects the commit.
  bool CommitIfUnchanged(
      uint32_t guest_address, uint32_t dword_count, const IbWatchSnapshot& snap,
      std::shared_ptr<const CompiledIndirectBuffer> compiled) {
    if (!compiled || !snap.page_count) {
      return false;
    }
    if (global_epoch_.load(std::memory_order_acquire) != snap.global_epoch) {
      return false;
    }
    bool page_changed = false;
    for (uint32_t i = 0; i < snap.page_count; ++i) {
      const uint32_t page = snap.first_page + i;
      if (page_epochs_[page].load(std::memory_order_acquire) !=
          snap.page_epochs[i]) {
        page_changed = true;
        ArmPageCooldown(page);
      }
    }
    if (page_changed) {
      ++stats_.watch_invalidations;
      ResetCandidateFor(guest_address & kIbCachePhysicalMemoryMask,
                        dword_count);
      return false;
    }

    const uint32_t phys = guest_address & kIbCachePhysicalMemoryMask;
    const uint64_t key = MakeKey(phys, dword_count);
    auto existing = entries_.find(key);
    if (existing == entries_.end()) {
      EvictFifoIfFull();
    }
    const uint64_t seq = ++next_insert_seq_;
    Entry& e = entries_[key];
    e.compiled = std::move(compiled);
    e.first_page = snap.first_page;
    e.page_count = snap.page_count;
    e.global_epoch = snap.global_epoch;
    e.page_epochs = snap.page_epochs;
    e.insert_seq = seq;
    fifo_keys_.emplace_back(key, seq);
    if (fifo_keys_.size() > kIbCacheMaxEntries * 4u) {
      CompactFifoQueue();
    }
    ++stats_.compilations;
    return true;
  }

  size_t entry_count() const { return entries_.size(); }
  uint64_t global_epoch() const {
    return global_epoch_.load(std::memory_order_acquire);
  }
  const IbCacheStats& stats() const { return stats_; }

  static constexpr uint64_t MakeKey(uint32_t phys_address,
                                    uint32_t dword_count) noexcept {
    // phys_address fits in 29 bits (512 MiB), dword_count fits in 13 bits
    // (<= 4096). Zero key collisions.
    return (uint64_t(dword_count & 0xFFFFu) << 32) |
           uint64_t(phys_address & kIbCachePhysicalMemoryMask);
  }

 private:
  struct Entry {
    std::shared_ptr<const CompiledIndirectBuffer> compiled;
    uint32_t first_page = 0;
    uint32_t page_count = 0;
    uint64_t global_epoch = 0;
    uint64_t insert_seq = 0;
    std::array<uint32_t, kIbCacheMaxPagesPerIb> page_epochs{};
  };

  struct CandidateSlot {
    uint32_t phys_address = 0;
    uint32_t dword_count = 0;
    uint64_t content_hash = 0;
    uint32_t consecutive_hits = 0;
    bool non_cacheable = false;
    bool valid = false;
  };

  struct PageCooldownSlot {
    uint32_t page = UINT32_MAX;
    uint64_t cooldown_until_frame = 0;
  };

  static bool ComputePageRange(uint32_t guest_address, uint32_t dword_count,
                               uint32_t& out_phys, uint32_t& out_first_page,
                               uint32_t& out_page_count) noexcept {
    if (!dword_count || dword_count > kIbCacheMaxDwords) {
      return false;
    }
    const uint32_t phys = guest_address & kIbCachePhysicalMemoryMask;
    const uint64_t bytes = uint64_t(dword_count) * sizeof(uint32_t);
    if (uint64_t(phys) + bytes > kIbCachePhysicalMemorySize) {
      return false;
    }
    const uint32_t first_page = phys >> kIbCachePageShift;
    const uint32_t last_page = static_cast<uint32_t>(
        (uint64_t(phys) + bytes - 1u) >> kIbCachePageShift);
    const uint32_t page_count = last_page - first_page + 1u;
    if (page_count > kIbCacheMaxPagesPerIb) {
      return false;
    }
    out_phys = phys;
    out_first_page = first_page;
    out_page_count = page_count;
    return true;
  }

  static size_t CandidateSlotIndex(uint32_t phys_address,
                                   uint32_t dword_count) noexcept {
    const uint64_t h = (MakeKey(phys_address, dword_count) *
                        UINT64_C(0x9E3779B97F4A7C15)) >>
                       56;
    return static_cast<size_t>(h) & (kIbCacheCandidateTableSize - 1u);
  }

  static size_t PageCooldownIndex(uint32_t page) noexcept {
    const uint32_t h = (page * 2654435761u) >> 24;
    return static_cast<size_t>(h) & (kIbCachePageCooldownTableSize - 1u);
  }

  void ArmPageCooldown(uint32_t page) {
    PageCooldownSlot& slot = page_cooldowns_[PageCooldownIndex(page)];
    slot.page = page;
    slot.cooldown_until_frame =
        current_frame_ + kIbCacheInvalidationCooldownFrames;
  }

  bool IsPageRangeInCooldown(uint32_t first_page, uint32_t page_count) const {
    for (uint32_t i = 0; i < page_count; ++i) {
      const uint32_t page = first_page + i;
      const PageCooldownSlot& slot = page_cooldowns_[PageCooldownIndex(page)];
      if (slot.page == page && current_frame_ < slot.cooldown_until_frame) {
        return true;
      }
    }
    return false;
  }

  void ResetCandidateFor(uint32_t phys_address, uint32_t dword_count) {
    CandidateSlot& slot = candidates_[CandidateSlotIndex(phys_address, dword_count)];
    if (slot.valid && slot.phys_address == phys_address &&
        slot.dword_count == dword_count) {
      slot.consecutive_hits = 0;
      slot.non_cacheable = false;
      slot.valid = false;
    }
  }

  void CompactFifoQueue() {
    std::deque<std::pair<uint64_t, uint64_t>> compacted;
    for (const auto& item : fifo_keys_) {
      auto it = entries_.find(item.first);
      if (it != entries_.end() && it->second.insert_seq == item.second) {
        compacted.push_back(item);
      }
    }
    fifo_keys_.swap(compacted);
  }

  void EvictFifoIfFull() {
    while (entries_.size() >= kIbCacheMaxEntries) {
      if (fifo_keys_.empty()) {
        entries_.clear();
        ++stats_.evictions;
        return;
      }
      const auto oldest = fifo_keys_.front();
      fifo_keys_.pop_front();
      auto it = entries_.find(oldest.first);
      if (it != entries_.end() && it->second.insert_seq == oldest.second) {
        entries_.erase(it);
        ++stats_.evictions;
      }
    }
  }

  std::unique_ptr<std::atomic<uint32_t>[]> page_epochs_;
  std::unique_ptr<std::atomic<uint64_t>[]> armed_pages_;
  std::atomic<uint64_t> global_epoch_{1};
  std::unordered_map<uint64_t, Entry> entries_;
  std::deque<std::pair<uint64_t, uint64_t>> fifo_keys_;
  uint64_t next_insert_seq_ = 0;
  std::array<CandidateSlot, kIbCacheCandidateTableSize> candidates_{};
  std::array<PageCooldownSlot, kIbCachePageCooldownTableSize> page_cooldowns_{};
  uint64_t current_frame_ = 0;
  IbCacheStats stats_{};
};

}  // namespace pm4_ib
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_PM4_IB_CACHE_H_
