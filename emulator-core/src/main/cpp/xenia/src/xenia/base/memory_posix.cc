/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/base/memory.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>

#if XE_PLATFORM_LINUX || XE_PLATFORM_xendroid || XE_PLATFORM_ANDROID
#include <sys/syscall.h>
#if defined(__has_include)
#if __has_include(<linux/memfd.h>)
#include <linux/memfd.h>
#endif
#endif
#ifndef MFD_CLOEXEC
#define MFD_CLOEXEC 0x0001u
#endif
#endif

#if XE_PLATFORM_MAC
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <mach/vm_region.h>
#endif  // XE_PLATFORM_MAC

#include "xenia/base/cvar.h"
#include "xenia/base/logging.h"
#include "xenia/base/math.h"
#include "xenia/base/platform.h"
#include "xenia/base/string.h"

#if XE_PLATFORM_xendroid
#include <android/api-level.h>
#include <android/sharedmem.h>
#endif
// Both ANDROID and xendroid are defined on the fork; the allocation path below is
// exclusive (#if xendroid #elif ANDROID), but upstream's standalone dlopen setup
// block is bare #if ANDROID, so its includes must be available even on xendroid.
#if XE_PLATFORM_ANDROID
#include <dlfcn.h>
#include <linux/ashmem.h>
#include <string.h>
#include <sys/ioctl.h>

#include "xenia/base/main_android.h"
#endif

DEFINE_bool(
    memory_huge_pages, true,
    "Advise the OS to back large guest RAM and JIT code cache mappings with "
    "2 MiB transparent huge pages (MADV_HUGEPAGE on Linux/Android).",
    "Memory");

namespace xe {
namespace memory {

#if XE_PLATFORM_ANDROID
// May be null if no dynamically loaded functions are required.
static void* libandroid_;
// API 26+.
static int (*android_ASharedMemory_create_)(const char* name, size_t size);

void AndroidInitialize() {
  if (xe::GetAndroidApiLevel() >= 26) {
    libandroid_ = dlopen("libandroid.so", RTLD_NOW);
    assert_not_null(libandroid_);
    if (libandroid_) {
      android_ASharedMemory_create_ =
          reinterpret_cast<decltype(android_ASharedMemory_create_)>(
              dlsym(libandroid_, "ASharedMemory_create"));
      assert_not_null(android_ASharedMemory_create_);
    }
  }
}

void AndroidShutdown() {
  android_ASharedMemory_create_ = nullptr;
  if (libandroid_) {
    dlclose(libandroid_);
    libandroid_ = nullptr;
  }
}
#endif

size_t page_size() {
  static const size_t value = static_cast<size_t>(getpagesize());
  return value;
}
size_t allocation_granularity() { return page_size(); }

uint32_t ToPosixProtectFlags(PageAccess access) {
  switch (access) {
    case PageAccess::kNoAccess:
      return PROT_NONE;
    case PageAccess::kReadOnly:
      return PROT_READ;
    case PageAccess::kReadWrite:
      return PROT_READ | PROT_WRITE;
    case PageAccess::kExecuteReadOnly:
      return PROT_READ | PROT_EXEC;
    case PageAccess::kExecuteReadWrite:
      return PROT_READ | PROT_WRITE | PROT_EXEC;
    default:
      assert_unhandled_case(access);
      return PROT_NONE;
  }
}

PageAccess ToXeniaProtectFlags(const char* protection) {
  if (protection[0] == 'r' && protection[1] == 'w' && protection[2] == 'x') {
    return PageAccess::kExecuteReadWrite;
  }
  if (protection[0] == 'r' && protection[1] == '-' && protection[2] == 'x') {
    return PageAccess::kExecuteReadOnly;
  }
  if (protection[0] == 'r' && protection[1] == 'w' && protection[2] == '-') {
    return PageAccess::kReadWrite;
  }
  if (protection[0] == 'r' && protection[1] == '-' && protection[2] == '-') {
    return PageAccess::kReadOnly;
  }
  return PageAccess::kNoAccess;
}

#if XE_PLATFORM_MAC
bool IsWritableExecutableMemorySupported() {
  // macOS allows RWX only on anonymous MAP_JIT regions. Callers that see
  // true must allocate via AllocFixed (which sets MAP_JIT) and toggle
  // pthread_jit_write_protect_np around writes. MAP_JIT requires the
  // com.apple.security.cs.allow-jit entitlement; without it the probe
  // fails and JIT is disabled.
  static const bool supported = []() {
    const size_t test_size = page_size();
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_JIT
    flags |= MAP_JIT;
#endif
    void* test_mapping = mmap(nullptr, test_size,
                              PROT_READ | PROT_WRITE | PROT_EXEC, flags, -1, 0);
    if (test_mapping == MAP_FAILED) {
      XELOGE("MAP_JIT probe failed: {} ({}); JIT will not work",
             strerror(errno), errno);
      return false;
    }
    munmap(test_mapping, test_size);
    return true;
  }();
  return supported;
}

DualMappingWxBackend ProbeDualMappingWxSupport() {
  return DualMappingWxBackend::kUnsupported;
}
#else
namespace {
int CreateMemfd(const char* name, size_t length) {
#if defined(__NR_memfd_create)
  const char* safe_name = (name && name[0] != '\0') ? name : "xenia_jit";
  int fd = static_cast<int>(syscall(__NR_memfd_create, safe_name, MFD_CLOEXEC));
  if (fd < 0) {
    return -1;
  }
  const size_t aligned_length = xe::round_up(length, page_size());
  if (ftruncate(fd, static_cast<off_t>(aligned_length)) < 0) {
    close(fd);
    return -1;
  }
  return fd;
#else
  (void)name;
  (void)length;
  return -1;
#endif
}

bool TestDualMappingFd(int fd, size_t probe_size) {
  if (fd < 0 || probe_size < sizeof(uint32_t)) {
    return false;
  }
  void* rx_view =
      mmap(nullptr, probe_size, PROT_READ | PROT_EXEC, MAP_SHARED, fd, 0);
  if (rx_view == MAP_FAILED) {
    return false;
  }
  void* rw_view =
      mmap(nullptr, probe_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (rw_view == MAP_FAILED) {
    munmap(rx_view, probe_size);
    return false;
  }
  *reinterpret_cast<volatile uint32_t*>(rw_view) = kDualMappingWxProbeSentinel;
  std::atomic_thread_fence(std::memory_order_seq_cst);
  const uint32_t readback =
      *reinterpret_cast<const volatile uint32_t*>(rx_view);
  munmap(rw_view, probe_size);
  munmap(rx_view, probe_size);
  return readback == kDualMappingWxProbeSentinel;
}

#if XE_PLATFORM_xendroid || XE_PLATFORM_ANDROID
int32_t GetDeviceAndroidApiLevel() {
#if XE_PLATFORM_xendroid
  return android_get_device_api_level();
#else
  return xe::GetAndroidApiLevel();
#endif
}
#endif
}  // namespace

bool IsWritableExecutableMemorySupported() {
  static const bool supported = []() {
    const size_t test_size = page_size();
    void* test_mapping = mmap(nullptr, test_size,
                              PROT_READ | PROT_WRITE | PROT_EXEC,
                              MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (test_mapping == MAP_FAILED) {
      XELOGW("Anonymous RWX probe failed: {} ({}); W^X dual-mapping required",
             strerror(errno), errno);
      return false;
    }
    munmap(test_mapping, test_size);
    return true;
  }();
  return supported;
}

DualMappingWxBackend ProbeDualMappingWxSupport() {
  static std::once_flag probe_once;
  static DualMappingWxBackend cached_backend =
      DualMappingWxBackend::kUnsupported;
  std::call_once(probe_once, []() {
    const size_t probe_size = page_size();
#if XE_PLATFORM_xendroid || XE_PLATFORM_ANDROID
    // Prefer memfd_create on Android API >= 30 if SELinux permits r-xs + rw-s.
    if (GetDeviceAndroidApiLevel() >= 30) {
      int fd = CreateMemfd("xenia_wx_probe", probe_size);
      if (fd >= 0) {
        const bool ok = TestDualMappingFd(fd, probe_size);
        close(fd);
        if (ok) {
          cached_backend = DualMappingWxBackend::kMemfd;
          XELOGI("W^X dual-mapping probe: memfd_create (r-xs + rw-s) OK");
          return;
        }
      }
    }
    // Fallback to ASharedMemory_create (ashmem / Android shared memory).
    int ashmem_fd = -1;
#if XE_PLATFORM_xendroid
    ashmem_fd = ASharedMemory_create("xenia_wx_probe", probe_size);
#elif XE_PLATFORM_ANDROID
    if (android_ASharedMemory_create_) {
      ashmem_fd = android_ASharedMemory_create_("xenia_wx_probe", probe_size);
    }
#endif
    if (ashmem_fd >= 0) {
      const bool ok = TestDualMappingFd(ashmem_fd, probe_size);
      close(ashmem_fd);
      if (ok) {
        cached_backend = DualMappingWxBackend::kAshmem;
        XELOGI("W^X dual-mapping probe: ASharedMemory (r-xs + rw-s) OK");
        return;
      }
    }
    XELOGW(
        "W^X dual-mapping probe failed (SELinux/kernel denied r-xs + rw-s on "
        "memfd and ASharedMemory); falling back cleanly to RWX");
#else
    int fd = CreateMemfd("xenia_wx_probe", probe_size);
    if (fd >= 0) {
      const bool ok = TestDualMappingFd(fd, probe_size);
      close(fd);
      if (ok) {
        cached_backend = DualMappingWxBackend::kMemfd;
        return;
      }
    }
    char shm_name[64];
    std::snprintf(shm_name, sizeof(shm_name), "/xenia_wx_probe_%d",
                  static_cast<int>(getpid()));
    int shm_fd = shm_open(shm_name, O_RDWR | O_CREAT | O_EXCL, 0600);
    if (shm_fd >= 0) {
      shm_unlink(shm_name);
      bool ok = false;
      if (ftruncate(shm_fd, static_cast<off_t>(probe_size)) == 0) {
        ok = TestDualMappingFd(shm_fd, probe_size);
      }
      close(shm_fd);
      if (ok) {
        cached_backend = DualMappingWxBackend::kPosixShm;
        return;
      }
    }
    XELOGW("W^X dual-mapping probe failed on POSIX host; falling back to RWX");
#endif
  });
  return cached_backend;
}
#endif  // XE_PLATFORM_MAC

struct MappedFileRange {
  uintptr_t region_begin;
  uintptr_t region_end;
};

std::vector<MappedFileRange> mapped_file_ranges;
std::mutex g_mapped_file_ranges_mutex;

// Track shm file names for cleanup on exit
std::vector<std::string> g_shm_file_names;
std::mutex g_shm_file_names_mutex;
static bool g_cleanup_handlers_installed = false;

#if !XE_PLATFORM_ANDROID&& !XE_PLATFORM_xendroid
static void CleanupAtExit() {
  for (const auto& name : g_shm_file_names) {
    shm_unlink(name.c_str());
  }
}

static void InstallCleanupHandlers() {
  if (g_cleanup_handlers_installed) {
    return;
  }
  g_cleanup_handlers_installed = true;

  std::atexit(CleanupAtExit);
  std::at_quick_exit(CleanupAtExit);
}
#endif  // !XE_PLATFORM_ANDROID

// Lets a Win32-style length-0 release find the reservation's extent.
static std::mutex g_reservations_mutex;
static std::unordered_map<void*, size_t> g_reservations;

// Address space held by a PROT_NONE reservation that file views replace, and
// that comes back when they are unmapped. Windows holds the same with
// placeholders.
static std::mutex g_view_reservations_mutex;
static std::vector<MappedFileRange> g_view_reservations;

// Index of the view reservation the range sits in, or -1. The caller holds
// g_view_reservations_mutex.
static ptrdiff_t FindViewReservation(const void* base_address, size_t length) {
  const auto begin = reinterpret_cast<uintptr_t>(base_address);
  for (size_t i = 0; i < g_view_reservations.size(); ++i) {
    // Length is compared against the distance to the end so it cannot wrap.
    if (begin >= g_view_reservations[i].region_begin &&
        begin <= g_view_reservations[i].region_end &&
        length <= g_view_reservations[i].region_end - begin) {
      return ptrdiff_t(i);
    }
  }
  return -1;
}

static bool IsViewReservation(const void* base_address, size_t length) {
  std::lock_guard guard(g_view_reservations_mutex);
  return FindViewReservation(base_address, length) >= 0;
}

// Puts a reservation over the range, dropping whatever is mapped in it.
static bool ReserveRange(void* base_address, size_t length) {
  void* result = mmap(base_address, length, PROT_NONE,
                      MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
  if (result == base_address) {
    return true;
  }
  // Callers drop the result, so the log is the only sign that a view stayed
  // mapped where the reservation should have come back.
  XELOGE("mmap(PROT_NONE, {}, 0x{:X}) failed: {} ({})", base_address, length,
         strerror(errno), errno);
  return false;
}

static void RememberReservation(void* base_address, size_t length) {
  std::lock_guard guard(g_reservations_mutex);
  g_reservations[base_address] = length;
}

// Erase before munmap: a concurrent AllocFixed may reuse the address.
static size_t TakeReservationLength(void* base_address) {
  std::lock_guard guard(g_reservations_mutex);
  auto it = g_reservations.find(base_address);
  if (it == g_reservations.end()) {
    return 0;
  }
  const size_t length = it->second;
  g_reservations.erase(it);
  return length;
}

void* AllocFixed(void* base_address, size_t length,
                 AllocationType allocation_type, PageAccess access) {
  // mmap does not support reserve / commit, so ignore allocation_type.
  uint32_t prot = ToPosixProtectFlags(access);
  int flags = MAP_PRIVATE | MAP_ANONYMOUS;

#if XE_PLATFORM_MAC
  if (access == PageAccess::kExecuteReadWrite ||
      access == PageAccess::kExecuteReadOnly) {
    flags |= MAP_JIT;
  }
#endif  // XE_PLATFORM_MAC

  if (base_address != nullptr) {
    if (allocation_type == AllocationType::kCommit) {
      if (Protect(base_address, length, access)) {
        return base_address;
      }
      return nullptr;
    }

    flags |= MAP_FIXED;
  }

  const size_t host_page = page_size();
  void* mmap_base = base_address;
  size_t mmap_length = length;
  if (host_page > 0x1000) {
    if (mmap_base != nullptr) {
      const uintptr_t raw_begin = reinterpret_cast<uintptr_t>(mmap_base);
      const uintptr_t aligned_begin = raw_begin & ~(host_page - 1);
      const uintptr_t aligned_end =
          (raw_begin + mmap_length + host_page - 1) & ~(host_page - 1);
      mmap_base = reinterpret_cast<void*>(aligned_begin);
      mmap_length = aligned_end - aligned_begin;
    } else if (mmap_length != 0) {
      mmap_length = (mmap_length + host_page - 1) & ~(host_page - 1);
    }
  }

  void* result = mmap(mmap_base, mmap_length, prot, flags, -1, 0);

  if (result == MAP_FAILED) {
    return nullptr;
  }

  if (mmap_base != nullptr && result != mmap_base) {
    munmap(result, mmap_length);
    return nullptr;
  }

  RememberReservation(result, mmap_length);
  return base_address != nullptr ? base_address : result;
}

bool DeallocFixed(void* base_address, size_t length,
                  DeallocationType deallocation_type) {
  const auto region_begin = reinterpret_cast<uintptr_t>(base_address);
  const uintptr_t region_end =
      reinterpret_cast<uintptr_t>(base_address) + length;

  std::lock_guard guard(g_mapped_file_ranges_mutex);
  for (const auto& mapped_range : mapped_file_ranges) {
    if (region_begin >= mapped_range.region_begin &&
        region_begin < mapped_range.region_end &&
        region_end <= mapped_range.region_end) {
      switch (deallocation_type) {
        case DeallocationType::kDecommit:
          return Protect(base_address, length, PageAccess::kNoAccess);
        case DeallocationType::kRelease:
          return false;
        default:
          assert_unhandled_case(deallocation_type);
      }
    }
  }

  switch (deallocation_type) {
    case DeallocationType::kDecommit:
      return Protect(base_address, length, PageAccess::kNoAccess);
    case DeallocationType::kRelease: {
      // memory_win.cc passes length 0 for MEM_RELEASE; munmap rejects 0.
      const size_t recorded = length ? 0 : TakeReservationLength(base_address);
      const size_t release_length = length ? length : recorded;
      if (!release_length) {
        XELOGE(
            "DeallocFixed: release of {} with length 0, but that address is "
            "not a known reservation; refusing to guess",
            base_address);
        return false;
      }
      if (munmap(base_address, release_length) != 0) {
        if (recorded) {
          RememberReservation(base_address, recorded);
        }
        return false;
      }
      return true;
    }
    default:
      assert_unhandled_case(deallocation_type);
  }
}

bool Protect(void* base_address, size_t length, PageAccess access,
             PageAccess* out_old_access) {
  if (length == 0) {
    return true;
  }
  const size_t host_page = page_size();
  const uintptr_t raw_begin = reinterpret_cast<uintptr_t>(base_address);
  const uintptr_t aligned_begin = raw_begin & ~(host_page - 1);
  const uintptr_t aligned_end =
      (raw_begin + length + host_page - 1) & ~(host_page - 1);
  void* aligned_address = reinterpret_cast<void*>(aligned_begin);
  const size_t aligned_length = aligned_end - aligned_begin;

  if (out_old_access) {
    size_t length_copy = aligned_length;
    QueryProtect(aligned_address, length_copy, *out_old_access);
  }

  uint32_t prot = ToPosixProtectFlags(access);
  int ret = mprotect(aligned_address, aligned_length, prot);
  if (ret != 0) {
    XELOGE("mprotect({}, 0x{:X}, {}) failed: {} ({})", aligned_address,
           aligned_length, prot, strerror(errno), errno);
  }
  return ret == 0;
}

bool QueryProtect(void* base_address, size_t& length, PageAccess& access_out) {
#if XE_PLATFORM_MAC
  mach_vm_address_t address = reinterpret_cast<mach_vm_address_t>(base_address);
  mach_vm_size_t region_size = 0;
  vm_region_basic_info_data_64_t info;
  mach_msg_type_number_t info_count = VM_REGION_BASIC_INFO_COUNT_64;
  mach_port_t object_name;

  kern_return_t kr = mach_vm_region(
      mach_task_self(), &address, &region_size, VM_REGION_BASIC_INFO_64,
      reinterpret_cast<vm_region_info_t>(&info), &info_count, &object_name);

  if (kr != KERN_SUCCESS) {
    return false;
  }

  if (address > reinterpret_cast<mach_vm_address_t>(base_address)) {
    return false;
  }

  length =
      static_cast<size_t>((address + region_size) -
                          reinterpret_cast<mach_vm_address_t>(base_address));

  if ((info.protection & (VM_PROT_READ | VM_PROT_WRITE | VM_PROT_EXECUTE)) ==
      (VM_PROT_READ | VM_PROT_WRITE | VM_PROT_EXECUTE)) {
    access_out = PageAccess::kExecuteReadWrite;
  } else if ((info.protection & (VM_PROT_READ | VM_PROT_EXECUTE)) ==
             (VM_PROT_READ | VM_PROT_EXECUTE)) {
    access_out = PageAccess::kExecuteReadOnly;
  } else if ((info.protection & (VM_PROT_READ | VM_PROT_WRITE)) ==
             (VM_PROT_READ | VM_PROT_WRITE)) {
    access_out = PageAccess::kReadWrite;
  } else if (info.protection & VM_PROT_READ) {
    access_out = PageAccess::kReadOnly;
  } else {
    access_out = PageAccess::kNoAccess;
  }

  return true;
#else
  // No generic POSIX solution exists. The Linux solution should work on all
  // Linux kernel based OS, including Android.
  std::ifstream memory_maps;
  memory_maps.open("/proc/self/maps", std::ios_base::in);
  std::string maps_entry_string;

  while (std::getline(memory_maps, maps_entry_string)) {
    std::stringstream entry_stream(maps_entry_string);
    uintptr_t map_region_begin, map_region_end;
    char separator;
    char protection[5];  // 4 chars (e.g., "r-xp") + null terminator

    entry_stream >> std::hex >> map_region_begin >> separator >>
        map_region_end >> protection;

    if (map_region_begin <= reinterpret_cast<uintptr_t>(base_address) &&
        map_region_end > reinterpret_cast<uintptr_t>(base_address)) {
      length = map_region_end - reinterpret_cast<uintptr_t>(base_address);

      access_out = ToXeniaProtectFlags(protection);

      // Look at the next consecutive mappings
      while (std::getline(memory_maps, maps_entry_string)) {
        std::stringstream next_entry_stream(maps_entry_string);
        uintptr_t next_map_region_begin, next_map_region_end;
        char next_protection[5];  // 4 chars (e.g., "r-xp") + null terminator

        next_entry_stream >> std::hex >> next_map_region_begin >> separator >>
            next_map_region_end >> next_protection;
        if (map_region_end == next_map_region_begin &&
            access_out == ToXeniaProtectFlags(next_protection)) {
          length =
              next_map_region_end - reinterpret_cast<uintptr_t>(base_address);
          continue;
        }
        break;
      }

      memory_maps.close();
      return true;
    }
  }

  memory_maps.close();
  return false;
#endif  // XE_PLATFORM_MAC
}

void AdviseHugePages(void* base_address, size_t length) {
  if (!cvars::memory_huge_pages || !base_address ||
      length < (2u * 1024u * 1024u)) {
    return;
  }
#ifdef MADV_HUGEPAGE
  const size_t host_page = page_size();
  const uintptr_t raw_begin = reinterpret_cast<uintptr_t>(base_address);
  const uintptr_t aligned_begin = raw_begin & ~(host_page - 1);
  const uintptr_t aligned_end =
      (raw_begin + length + host_page - 1) & ~(host_page - 1);
  if (aligned_end > aligned_begin) {
    (void)madvise(reinterpret_cast<void*>(aligned_begin),
                  aligned_end - aligned_begin, MADV_HUGEPAGE);
  }
#endif
}

FileMappingHandle CreateFileMappingHandle(const std::filesystem::path& path,
                                          size_t length, PageAccess access,
                                          bool commit) {
#if XE_PLATFORM_xendroid
  const bool executable_mapping = (access == PageAccess::kExecuteReadWrite ||
                                   access == PageAccess::kExecuteReadOnly);
  if (executable_mapping && !IsWritableExecutableMemoryPreferred() &&
      ProbeDualMappingWxSupport() == DualMappingWxBackend::kMemfd) {
    std::string memfd_name = path.filename().string();
    if (memfd_name.empty()) {
      memfd_name = "xenia_jit";
    }
    int memfd = CreateMemfd(memfd_name.c_str(), length);
    if (memfd >= 0) {
      return memfd;
    }
    XELOGW("memfd_create({}, 0x{:X}) failed: {} ({}); falling back to "
           "ASharedMemory_create",
           memfd_name, length, strerror(errno), errno);
  }
  int sharedmem_fd = ASharedMemory_create(path.c_str(), length);
  return sharedmem_fd >= 0 ? sharedmem_fd : kFileMappingHandleInvalid;
#elif XE_PLATFORM_ANDROID
  const bool executable_mapping = (access == PageAccess::kExecuteReadWrite ||
                                   access == PageAccess::kExecuteReadOnly);
  if (executable_mapping && !IsWritableExecutableMemoryPreferred() &&
      ProbeDualMappingWxSupport() == DualMappingWxBackend::kMemfd) {
    std::string memfd_name = path.filename().string();
    if (memfd_name.empty()) {
      memfd_name = "xenia_jit";
    }
    int memfd = CreateMemfd(memfd_name.c_str(), length);
    if (memfd >= 0) {
      return memfd;
    }
  }
  if (android_ASharedMemory_create_) {
    int sharedmem_fd = android_ASharedMemory_create_(path.c_str(), length);
    return sharedmem_fd >= 0 ? sharedmem_fd : kFileMappingHandleInvalid;
  }

  // Use /dev/ashmem on API versions below 26, which added ASharedMemory.
  // /dev/ashmem was disabled on API 29 for apps targeting it.
  // https://chromium.googlesource.com/chromium/src/+/master/third_party/ashmem/ashmem-dev.c
  int ashmem_fd = open("/" ASHMEM_NAME_DEF, O_RDWR);
  if (ashmem_fd < 0) {
    return kFileMappingHandleInvalid;
  }
  char ashmem_name[ASHMEM_NAME_LEN];
  strlcpy(ashmem_name, path.c_str(), xe::countof(ashmem_name));
  if (ioctl(ashmem_fd, ASHMEM_SET_NAME, ashmem_name) < 0 ||
      ioctl(ashmem_fd, ASHMEM_SET_SIZE, length) < 0) {
    close(ashmem_fd);
    return kFileMappingHandleInvalid;
  }
  return ashmem_fd;
#else
  int oflag;
  switch (access) {
    case PageAccess::kNoAccess:
      oflag = 0;
      break;
    case PageAccess::kReadOnly:
    case PageAccess::kExecuteReadOnly:
      oflag = O_RDONLY;
      break;
    case PageAccess::kReadWrite:
    case PageAccess::kExecuteReadWrite:
      oflag = O_RDWR;
      break;
    default:
      assert_always();
      return kFileMappingHandleInvalid;
  }
  oflag |= O_CREAT;

#if XE_PLATFORM_MAC
  std::string shm_name = "/" + path.filename().string();
  if (shm_name.size() > 30) {
    std::size_t h = std::hash<std::string>{}(shm_name);
    char hash_buf[24];
    std::snprintf(hash_buf, sizeof(hash_buf), "/%016zx", h);
    shm_name = hash_buf;
  }
  int ret = shm_open(shm_name.c_str(), oflag, 0777);
  if (ret < 0) {
    XELOGE("shm_open({}) failed: {} ({})", shm_name, strerror(errno), errno);
    return kFileMappingHandleInvalid;
  }
  if (ftruncate(ret, length) < 0) {
    XELOGE("ftruncate({}, 0x{:X}) failed: {} ({})", shm_name, length,
           strerror(errno), errno);
    close(ret);
    shm_unlink(shm_name.c_str());
    return kFileMappingHandleInvalid;
  }
  // Track for cleanup on abnormal exit and install cleanup handlers
  {
    std::lock_guard guard(g_shm_file_names_mutex);
    g_shm_file_names.push_back(shm_name);
  }
  InstallCleanupHandlers();
  return ret;
#else
  auto full_path = "/" / path;
  int ret = shm_open(full_path.c_str(), oflag, 0777);
  if (ret < 0) {
    XELOGE("shm_open({}) failed: {} ({})", full_path.string(), strerror(errno),
           errno);
    return kFileMappingHandleInvalid;
  }
  if (ftruncate(ret, length) < 0) {
    XELOGE("ftruncate({}, 0x{:X}) failed: {} ({})", full_path.string(), length,
           strerror(errno), errno);
    close(ret);
    shm_unlink(full_path.c_str());
    return kFileMappingHandleInvalid;
  }
  // Track for cleanup on abnormal exit and install cleanup handlers
  {
    std::lock_guard guard(g_shm_file_names_mutex);
    g_shm_file_names.push_back(full_path.string());
  }
  InstallCleanupHandlers();
  return ret;
#endif  // XE_PLATFORM_MAC
#endif
}

void CloseFileMappingHandle(FileMappingHandle handle,
                            const std::filesystem::path& path) {
  close(handle);
#if !XE_PLATFORM_ANDROID && !XE_PLATFORM_xendroid
#if XE_PLATFORM_MAC
  std::string shm_name = "/" + path.filename().string();
  if (shm_name.size() > 30) {
    std::size_t h = std::hash<std::string>{}(shm_name);
    char hash_buf[24];
    std::snprintf(hash_buf, sizeof(hash_buf), "/%016zx", h);
    shm_name = hash_buf;
  }
  shm_unlink(shm_name.c_str());
  // Remove from tracking
  {
    std::lock_guard guard(g_shm_file_names_mutex);
    auto it =
        std::find(g_shm_file_names.begin(), g_shm_file_names.end(), shm_name);
    if (it != g_shm_file_names.end()) {
      g_shm_file_names.erase(it);
    }
  }
#else
  auto full_path = "/" / path;
  shm_unlink(full_path.c_str());
  // Remove from tracking
  {
    std::lock_guard guard(g_shm_file_names_mutex);
    auto it = std::find(g_shm_file_names.begin(), g_shm_file_names.end(),
                        full_path.string());
    if (it != g_shm_file_names.end()) {
      g_shm_file_names.erase(it);
    }
  }
#endif  // XE_PLATFORM_MAC
#endif
}

void* MapFileView(FileMappingHandle handle, void* base_address, size_t length,
                  PageAccess access, size_t file_offset) {
  uint32_t prot = ToPosixProtectFlags(access);

  // A view over a reservation needs no range of its own.
  const bool over_reservation =
      base_address != nullptr && IsViewReservation(base_address, length);

  int flags = MAP_SHARED;
  // Views inside a ReserveFileViewPages reservation intentionally replace the
  // PROT_NONE placeholder covering that range. Outside a managed reservation,
  // never clobber an existing unrelated mapping with MAP_FIXED.
  if (over_reservation) {
    flags |= MAP_FIXED;
  } else if (base_address != nullptr) {
#ifdef MAP_FIXED_NOREPLACE
    flags |= MAP_FIXED_NOREPLACE;
#endif
  }

  void* result = mmap(base_address, length, prot, flags, handle, file_offset);

#ifdef MAP_FIXED_NOREPLACE
  // Older kernels (< Linux 4.17) may reject MAP_FIXED_NOREPLACE with
  // EOPNOTSUPP/EINVAL; fall back to passing base_address as a non-clobbering
  // hint and enforce exact address equality below.
  if (result == MAP_FAILED && !over_reservation && base_address != nullptr &&
      (errno == EOPNOTSUPP || errno == EINVAL)) {
    flags &= ~MAP_FIXED_NOREPLACE;
    result = mmap(base_address, length, prot, flags, handle, file_offset);
  }
#endif

  if (result == MAP_FAILED) {
    return nullptr;
  }

  // Without MAP_FIXED_NOREPLACE (e.g. macOS or older kernels), a non-null
  // base_address is just a hint. Enforce the caller's contract by failing on
  // address mismatch so callers can retry at a different base, matching
  // AllocFixed's behavior.
  if (base_address != nullptr && result != base_address) {
    munmap(result, length);
    return nullptr;
  }

  if (over_reservation) {
    return result;
  }

  std::lock_guard guard(g_mapped_file_ranges_mutex);
  mapped_file_ranges.push_back({reinterpret_cast<uintptr_t>(result),
                                reinterpret_cast<uintptr_t>(result) + length});
  return result;
}

bool UnmapFileView(FileMappingHandle handle, void* base_address,
                   size_t length) {
  if (IsViewReservation(base_address, length)) {
    // The reservation comes back, and the view was never tracked.
    return ReserveRange(base_address, length);
  }

  std::lock_guard guard(g_mapped_file_ranges_mutex);

#if XE_PLATFORM_MAC
  uintptr_t unmap_begin = reinterpret_cast<uintptr_t>(base_address);
  uintptr_t unmap_end = unmap_begin + length;

  for (auto mapped_range = mapped_file_ranges.begin();
       mapped_range != mapped_file_ranges.end(); ++mapped_range) {
    if (unmap_begin >= mapped_range->region_begin &&
        unmap_end <= mapped_range->region_end) {
      uintptr_t orig_begin = mapped_range->region_begin;
      uintptr_t orig_end = mapped_range->region_end;
      mapped_file_ranges.erase(mapped_range);

      if (orig_begin < unmap_begin) {
        mapped_file_ranges.push_back({orig_begin, unmap_begin});
      }
      if (unmap_end < orig_end) {
        mapped_file_ranges.push_back({unmap_end, orig_end});
      }

      return munmap(base_address, length) == 0;
    }
  }

  return munmap(base_address, length) == 0;
#else
  for (auto mapped_range = mapped_file_ranges.begin();
       mapped_range != mapped_file_ranges.end();) {
    if (mapped_range->region_begin ==
            reinterpret_cast<uintptr_t>(base_address) &&
        mapped_range->region_end ==
            reinterpret_cast<uintptr_t>(base_address) + length) {
      mapped_file_ranges.erase(mapped_range);
      return munmap(base_address, length) == 0;
    }
    ++mapped_range;
  }
  // TODO: Implement partial file unmapping.
  assert_always("Error: Partial unmapping of files not yet supported.");
  return munmap(base_address, length) == 0;
#endif  // XE_PLATFORM_MAC
}

bool ReserveFileViewPages(void* base_address, size_t length) {
  std::lock_guard guard(g_view_reservations_mutex);
  if (FindViewReservation(base_address, length) >= 0) {
    // Part of a reservation already, with no view mapped in it.
    return true;
  }
  int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_FIXED_NOREPLACE
  flags |= MAP_FIXED_NOREPLACE;
#endif
  void* reservation = mmap(base_address, length, PROT_NONE, flags, -1, 0);
  if (reservation == MAP_FAILED) {
    return false;
  }
  if (reservation != base_address) {
    // Without MAP_FIXED_NOREPLACE the base is only a hint, and a reservation
    // elsewhere holds nothing the caller asked for.
    munmap(reservation, length);
    return false;
  }
  g_view_reservations.push_back(
      {reinterpret_cast<uintptr_t>(base_address),
       reinterpret_cast<uintptr_t>(base_address) + length});
  return true;
}

void* MapFileViewPages(FileMappingHandle handle, void* base_address,
                       size_t length, PageAccess access, size_t file_offset) {
  return MapFileView(handle, base_address, length, access, file_offset);
}

bool ReleaseFileViewPages(FileMappingHandle handle, void* base_address,
                          size_t length) {
  std::lock_guard guard(g_view_reservations_mutex);
  const ptrdiff_t index = FindViewReservation(base_address, length);
  if (index < 0) {
    return munmap(base_address, length) == 0;
  }
  const auto begin = reinterpret_cast<uintptr_t>(base_address);
  if (g_view_reservations[index].region_begin == begin &&
      g_view_reservations[index].region_end == begin + length) {
    // The whole reservation, so the address space goes back to the host.
    g_view_reservations.erase(g_view_reservations.begin() + index);
    return munmap(base_address, length) == 0;
  }
  // Part of a reservation, so it comes back over the views released here.
  return ReserveRange(base_address, length);
}

}  // namespace memory
}  // namespace xe
