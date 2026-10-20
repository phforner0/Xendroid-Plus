// The socket option values (xenia/kernel/xsocket_options.h): a BOOL or DWORD
// option is its first byte when the guest's length is under four, otherwise a
// big-endian DWORD, and a value is stored back the same way.

#include <array>
#include <cstdint>
#include <cstdio>

#include "xenia/kernel/xsocket_options.h"

namespace {

int failures = 0;

void expect(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAILED: %s\n", what);
    ++failures;
  }
}

}  // namespace

int main() {
  using xe::kernel::SocketOptionValue;
  using xe::kernel::StoreSocketOptionValue;

  const uint8_t dword[] = {0x00, 0x00, 0x03, 0xE8};
  expect(SocketOptionValue(dword, 4) == 1000, "a DWORD is big-endian");
  // A one-byte BOOL and the stack bytes after it.
  const uint8_t bool_byte[] = {0x01, 0xCD, 0xCD, 0xCD};
  expect(SocketOptionValue(bool_byte, 1) == 1, "a one-byte TRUE");
  expect(SocketOptionValue(bool_byte, 3) == 1, "a value under four bytes is its first");
  const uint8_t false_byte[] = {0x00, 0xCD, 0xCD, 0xCD};
  expect(SocketOptionValue(false_byte, 1) == 0, "a one-byte FALSE");

  std::array<uint8_t, 4> stored;
  stored.fill(0xCD);
  uint32_t optlen = 4;
  StoreSocketOptionValue(stored.data(), optlen, 1000);
  expect(optlen == 4, "a DWORD is stored in four bytes");
  expect(stored == std::array<uint8_t, 4>{0x00, 0x00, 0x03, 0xE8},
         "a DWORD is stored big-endian");
  stored.fill(0xCD);
  optlen = 1;
  StoreSocketOptionValue(stored.data(), optlen, 1);
  expect(optlen == 1, "a short value is stored in one byte");
  expect(stored == std::array<uint8_t, 4>{0x01, 0xCD, 0xCD, 0xCD},
         "a short value leaves the bytes after it");

  if (failures) {
    std::fprintf(stderr, "socket_option_test: %d failed\n", failures);
    return 1;
  }
  std::printf("socket_option_test: ok\n");
  return 0;
}
