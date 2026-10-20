/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_KERNEL_XSOCKET_OPTIONS_H_
#define XENIA_KERNEL_XSOCKET_OPTIONS_H_

#include <cstddef>
#include <cstdint>

namespace xe {
namespace kernel {

// The value of a BOOL or DWORD socket option: its first byte when |optlen| is
// under four, otherwise a big-endian DWORD.
inline uint32_t SocketOptionValue(const void* optval, uint32_t optlen) {
  const auto* bytes = static_cast<const uint8_t*>(optval);
  if (optlen < 4) {
    return bytes[0];
  }
  return uint32_t(bytes[0]) << 24 | uint32_t(bytes[1]) << 16 |
         uint32_t(bytes[2]) << 8 | bytes[3];
}

// Stores a BOOL or DWORD socket option value the same way and sets |optlen| to
// the bytes written.
inline void StoreSocketOptionValue(void* optval, uint32_t& optlen,
                                   uint32_t value) {
  auto* bytes = static_cast<uint8_t*>(optval);
  if (optlen < 4) {
    bytes[0] = uint8_t(value);
    optlen = 1;
    return;
  }
  for (size_t i = 0; i < 4; ++i) {
    bytes[i] = uint8_t(value >> (24 - 8 * i));
  }
  optlen = 4;
}

}  // namespace kernel
}  // namespace xe

#endif  // XENIA_KERNEL_XSOCKET_OPTIONS_H_
