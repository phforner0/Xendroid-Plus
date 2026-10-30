// The name a package installed from a file takes (xenia/vfs/fatx_package_name.h):
// one the content calls accept, by the rules ContentManager::ResolvePackagePath
// checks after trimming spaces (no byte FATX refuses, nothing past ASCII, not
// "." or ".."), at most the 42 bytes of XCONTENT_DATA::file_name_raw. A name
// that had to change ends in '~' and 4 hex digits from the file's name.

#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>

#include "xenia/vfs/fatx_package_name.h"

namespace {

int failures = 0;

void expect(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAILED: %s\n", what);
    ++failures;
  }
}

// What ResolvePackagePath accepts, after its trim.
bool ContentCallsTake(const std::string& name) {
  constexpr std::string_view kRefused = "\"*+,/:;<=>?|\\";
  if (name.empty() || name.size() > 42 || name == "." || name == ".." ||
      name.front() == ' ' || name.back() == ' ') {
    return false;
  }
  for (char c : name) {
    if (uint8_t(c) < 0x20 || uint8_t(c) >= 0x80 ||
        kRefused.find(c) != std::string_view::npos) {
      return false;
    }
  }
  return true;
}

// '~' and the 4 hex digits a changed name ends in: FNV-1a of the file's name,
// its halves xored.
std::string Suffix(std::string_view file_name) {
  uint32_t hash = 2166136261u;
  for (char c : file_name) {
    hash = (hash ^ uint8_t(c)) * 16777619u;
  }
  char suffix[6];
  std::snprintf(suffix, sizeof(suffix), "~%04X", (hash ^ (hash >> 16)) & 0xFFFF);
  return suffix;
}

void expect_name(std::string_view file_name, const std::string& expected,
                 const char* what) {
  const std::string name = xe::vfs::FatxPackageName(file_name);
  if (name != expected) {
    std::fprintf(stderr, "  \"%s\" -> \"%s\", expected \"%s\"\n",
                 std::string(file_name).c_str(), name.c_str(),
                 expected.c_str());
  }
  expect(name == expected, what);
  expect(ContentCallsTake(name), "the content calls take the name");
  expect(xe::vfs::FatxPackageName(name) == name, "the name stays the same");
}

}  // namespace

int main() {
  expect_name("TU_16L61V6_0000014000000.00000000000O4",
              "TU_16L61V6_0000014000000.00000000000O4",
              "a console's name stays");
  expect_name("Halo 3 TU5 (v2).bin", "Halo 3 TU5 (v2).bin",
              "a valid name stays");
  expect_name("Jogo TU5, v2.bin",
              "Jogo TU5_ v2.bin" + Suffix("Jogo TU5, v2.bin"),
              "a comma becomes '_'");
  expect_name("a+b;c=d\"e*f<g>h?i|j:k",
              "a_b_c_d_e_f_g_h_i_j_k" + Suffix("a+b;c=d\"e*f<g>h?i|j:k"),
              "every byte FATX refuses becomes '_'");
  expect_name("a\\b/c", "a_b_c" + Suffix("a\\b/c"), "separators become '_'");
  expect_name("\x01x\x7F", "_x\x7F" + Suffix("\x01x\x7F"),
              "control bytes become '_'");
  expect_name("Atualiza\xC3\xA7\xC3\xA3o.bin",
              "Atualiza__o.bin" + Suffix("Atualiza\xC3\xA7\xC3\xA3o.bin"),
              "one '_' for each character past ASCII");
  expect_name("\xF0\x9F\x98\x80x", "_x" + Suffix("\xF0\x9F\x98\x80x"),
              "a 4-byte character is one '_'");
  expect_name("\x80\x80y", "_y" + Suffix("\x80\x80y"),
              "stray continuation bytes are one '_'");
  expect_name("  name.  ", "name" + Suffix("  name.  "),
              "spaces and dots go from the ends");
  expect_name(".xd-install-foo", "xd-install-foo" + Suffix(".xd-install-foo"),
              "no name the installer keeps for itself");
  expect_name("...", "Package" + Suffix("..."), "a name of dots only");
  expect_name("", "Package" + Suffix(""), "an empty name");
  expect_name(std::string(42, 'a'), std::string(42, 'a'),
              "42 bytes are kept whole");
  expect_name(std::string(50, 'a'),
              std::string(37, 'a') + Suffix(std::string(50, 'a')),
              "a longer name is cut for the suffix");
  expect_name(std::string(36, 'a') + " bcdefg",
              std::string(36, 'a') + Suffix(std::string(36, 'a') + " bcdefg"),
              "a cut ending in a space is trimmed");
  const std::string beatles = "Rock Band 3 - Song Pack - The Beatles Vol ";
  expect(xe::vfs::FatxPackageName(beatles + "1.bin") !=
             xe::vfs::FatxPackageName(beatles + "2.bin"),
         "names cut to the same start keep apart");
  expect(xe::vfs::FatxPackageName("Jogo, TU.bin") !=
             xe::vfs::FatxPackageName("Jogo+ TU.bin"),
         "names that only differ in what changed keep apart");

  if (failures) {
    std::fprintf(stderr, "fatx_package_name_test: %d failed\n", failures);
    return 1;
  }
  std::printf("fatx_package_name_test: ok\n");
  return 0;
}
