// SPDX-License-Identifier: WTFPL
#ifndef XENIA_VFS_FATX_PACKAGE_NAME_H_
#define XENIA_VFS_FATX_PACKAGE_NAME_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace xe {
namespace vfs {

// The name a package installed from a file named |file_name| takes, one the
// content calls accept (FATX's rules, ContentManager::ResolvePackagePath).
// Each byte FATX refuses and each character past ASCII becomes '_'. Spaces
// and dots go from both ends: the lookup trims spaces, Windows drops a
// trailing dot, and ".xd-" names are the installer's own. The name fits the
// 42 bytes of XCONTENT_DATA::file_name_raw. A name that had to change ends
// in '~' and 4 hex digits from the file's name, so files that only differ
// in what changed keep apart; a name that is valid already stays.
inline std::string FatxPackageName(std::string_view file_name) {
  constexpr std::string_view kRefused = "\"*+,/:;<=>?|\\";
  constexpr size_t kMaxLength = 42;
  constexpr size_t kSuffixLength = 5;  // ~XXXX
  std::string name;
  for (size_t i = 0; i < file_name.size();) {
    const uint8_t c = uint8_t(file_name[i++]);
    if (c >= 0x80) {
      // The continuation bytes of the character go with its lead byte.
      while (i < file_name.size() && (uint8_t(file_name[i]) & 0xC0) == 0x80) {
        ++i;
      }
      name += '_';
    } else if (c < 0x20 || kRefused.find(char(c)) != std::string_view::npos) {
      name += '_';
    } else {
      name += char(c);
    }
  }
  auto trim = [&name]() {
    const size_t first = name.find_first_not_of(" .");
    if (first == std::string::npos) {
      name.clear();
      return;
    }
    name = name.substr(first, name.find_last_not_of(" .") - first + 1);
  };
  trim();
  if (!name.empty() && name == file_name && name.size() <= kMaxLength) {
    return name;
  }
  if (name.size() > kMaxLength - kSuffixLength) {
    name.resize(kMaxLength - kSuffixLength);
    trim();
  }
  if (name.empty()) {
    name = "Package";
  }
  // FNV-1a of the file's name.
  uint32_t hash = 2166136261u;
  for (char c : file_name) {
    hash = (hash ^ uint8_t(c)) * 16777619u;
  }
  constexpr char kHex[] = "0123456789ABCDEF";
  name += '~';
  for (int shift = 12; shift >= 0; shift -= 4) {
    name += kHex[((hash ^ (hash >> 16)) >> shift) & 0xF];
  }
  return name;
}

}  // namespace vfs
}  // namespace xe

#endif  // XENIA_VFS_FATX_PACKAGE_NAME_H_
