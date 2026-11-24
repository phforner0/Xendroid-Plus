/**
 ******************************************************************************
 * XenDroid: on-disk per-title .jprof JIT function warmup profile cache.      *
 ******************************************************************************
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_CPU_JIT_PROFILE_FILE_H_
#define XENIA_CPU_JIT_PROFILE_FILE_H_

// Header-only and pure C++20 (zero Emulator/Processor/Android dependencies) so
// host and NDK logic tests can exercise it directly.
//
// File layout (version 1, little-endian):
//   char     magic[8]      "XDJPROF\0"
//   uint32_t header_size   48
//   uint32_t version       1
//   uint32_t title_id      Title ID (uint32_t)
//   uint32_t entry_count   payload_size / sizeof(uint32_t)
//   uint64_t module_hash   UserModule::hash().value_or(0)
//   uint64_t payload_size  entry_count * sizeof(uint32_t)
//   uint64_t payload_hash  XXH3_64bits(payload, payload_size)
//   payload                Sorted, deduplicated 4-byte-aligned guest addresses.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <system_error>
#include <vector>

#include "xenia/base/xxhash.h"

namespace xe {
namespace cpu {
namespace jit_profile_file {

inline constexpr char kMagic[8] = {'X', 'D', 'J', 'P', 'R', 'O', 'F', 0};
inline constexpr uint32_t kFormatVersion = 1;
inline constexpr uint32_t kHeaderSize = 48;
inline constexpr uint64_t kDefaultMaxBytes = uint64_t(4) << 20;  // 4 MiB
inline constexpr uint32_t kDefaultMaxEntries = 524288;            // 2 MiB payload

struct Header {
  char magic[8];
  uint32_t header_size;
  uint32_t version;
  uint32_t title_id;
  uint32_t entry_count;
  uint64_t module_hash;
  uint64_t payload_size;
  uint64_t payload_hash;
};
static_assert(sizeof(Header) == kHeaderSize);

enum class LoadStatus {
  kMissing,
  kLoaded,
  kMismatch,
  kCorrupt,
  kTooLarge,
};

inline const char* LoadStatusName(LoadStatus status) {
  switch (status) {
    case LoadStatus::kMissing:
      return "missing";
    case LoadStatus::kLoaded:
      return "loaded";
    case LoadStatus::kMismatch:
      return "title/module mismatch";
    case LoadStatus::kCorrupt:
      return "corrupt";
    case LoadStatus::kTooLarge:
      return "too large";
  }
  return "unknown";
}

struct LoadResult {
  LoadStatus status = LoadStatus::kMissing;
  uint32_t title_id = 0;
  uint64_t module_hash = 0;
  std::vector<uint32_t> addresses;
};

inline std::vector<uint32_t> BuildPayload(
    const std::vector<uint32_t>& addresses,
    uint32_t max_entries = kDefaultMaxEntries) {
  std::vector<uint32_t> clean;
  clean.reserve(addresses.size());
  for (uint32_t addr : addresses) {
    if (addr != 0u && (addr & 3u) == 0u) {
      clean.push_back(addr);
    }
  }
  std::sort(clean.begin(), clean.end());
  clean.erase(std::unique(clean.begin(), clean.end()), clean.end());
  if (clean.size() > max_entries) {
    clean.resize(max_entries);
  }
  return clean;
}

inline LoadResult Load(const std::filesystem::path& path,
                       uint32_t expected_title_id,
                       uint64_t expected_module_hash,
                       uint64_t max_bytes = kDefaultMaxBytes) {
  LoadResult result;
  std::error_code ec;
  if (!std::filesystem::is_regular_file(path, ec)) {
    return result;
  }
  const uintmax_t file_size = std::filesystem::file_size(path, ec);
  if (ec) {
    result.status = LoadStatus::kCorrupt;
    return result;
  }
  if (file_size > max_bytes) {
    result.status = LoadStatus::kTooLarge;
    return result;
  }
  if (file_size < kHeaderSize) {
    result.status = LoadStatus::kCorrupt;
    return result;
  }

  std::vector<uint8_t> bytes(static_cast<size_t>(file_size));
  FILE* file = std::fopen(path.string().c_str(), "rb");
  const bool read_ok =
      file && (bytes.empty() ||
               std::fread(bytes.data(), 1, bytes.size(), file) == bytes.size());
  if (file) {
    std::fclose(file);
  }
  if (!read_ok) {
    result.status = LoadStatus::kCorrupt;
    return result;
  }

  Header hdr = {};
  std::memcpy(&hdr, bytes.data(), kHeaderSize);
  const uint64_t actual_payload_size = bytes.size() - kHeaderSize;
  if (std::memcmp(hdr.magic, kMagic, sizeof(kMagic)) != 0 ||
      hdr.header_size != kHeaderSize || hdr.version != kFormatVersion ||
      hdr.payload_size != actual_payload_size ||
      (actual_payload_size % sizeof(uint32_t)) != 0 ||
      hdr.entry_count != actual_payload_size / sizeof(uint32_t) ||
      hdr.entry_count > kDefaultMaxEntries) {
    result.status = LoadStatus::kCorrupt;
    return result;
  }

  const uint64_t actual_hash =
      XXH3_64bits(bytes.data() + kHeaderSize, static_cast<size_t>(actual_payload_size));
  if (hdr.payload_hash != actual_hash) {
    result.status = LoadStatus::kCorrupt;
    return result;
  }

  result.title_id = hdr.title_id;
  result.module_hash = hdr.module_hash;
  if (hdr.title_id != expected_title_id ||
      (expected_module_hash != 0 && hdr.module_hash != expected_module_hash)) {
    result.status = LoadStatus::kMismatch;
    return result;
  }

  if (hdr.entry_count > 0) {
    std::vector<uint32_t> raw_addrs(hdr.entry_count);
    std::memcpy(raw_addrs.data(), bytes.data() + kHeaderSize,
                static_cast<size_t>(actual_payload_size));
    result.addresses = BuildPayload(raw_addrs, kDefaultMaxEntries);
  }
  result.status = LoadStatus::kLoaded;
  return result;
}

inline bool WriteAtomically(const std::filesystem::path& path,
                            uint32_t title_id, uint64_t module_hash,
                            const std::vector<uint32_t>& addresses,
                            uint32_t max_entries = kDefaultMaxEntries) {
  const std::vector<uint32_t> clean = BuildPayload(addresses, max_entries);
  const uint64_t payload_size =
      static_cast<uint64_t>(clean.size()) * sizeof(uint32_t);
  const uint64_t payload_hash =
      XXH3_64bits(clean.data(), static_cast<size_t>(payload_size));

  Header hdr = {};
  std::memcpy(hdr.magic, kMagic, sizeof(kMagic));
  hdr.header_size = kHeaderSize;
  hdr.version = kFormatVersion;
  hdr.title_id = title_id;
  hdr.entry_count = static_cast<uint32_t>(clean.size());
  hdr.module_hash = module_hash;
  hdr.payload_size = payload_size;
  hdr.payload_hash = payload_hash;

  std::error_code ec;
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path(), ec);
  }

  std::filesystem::path temp = path;
  temp += ".tmp";
  FILE* file = std::fopen(temp.string().c_str(), "wb");
  if (!file) {
    return false;
  }
  bool ok = std::fwrite(&hdr, 1, sizeof(hdr), file) == sizeof(hdr) &&
            (payload_size == 0 ||
             std::fwrite(clean.data(), 1, static_cast<size_t>(payload_size),
                         file) == payload_size);
  ok = (std::fflush(file) == 0) && ok;
  ok = (std::fclose(file) == 0) && ok;
  if (ok) {
    std::filesystem::rename(temp, path, ec);
    ok = !ec;
  }
  if (!ok) {
    std::filesystem::remove(temp, ec);
  }
  return ok;
}

}  // namespace jit_profile_file
}  // namespace cpu
}  // namespace xe

#endif  // XENIA_CPU_JIT_PROFILE_FILE_H_
