// SPDX-License-Identifier: BSD-2-Clause
// Copyright © 2021 Billy Laws

#include <array>
#include <cstdio>
#include <cstdlib>
#include <errno.h>
#include <dlfcn.h>
#include <unistd.h>
#include <fcntl.h>
#include <android/dlext.h>
#include <android/log.h>
#include <android/api-level.h>
#include <sys/auxv.h>
#include <sys/mman.h>
#include "elf_soname_patcher.h"
#include "android_linker_ns.h"

using loader_android_create_namespace_t = android_namespace_t *(*)(const char *, const char *, const char *, uint64_t, const char *, android_namespace_t *, const void *);
static loader_android_create_namespace_t loader_android_create_namespace;

static bool lib_loaded;

static void resolve_linker_symbols();

static android_namespace_t *lazy_android_get_exported_namespace(const char *name);
static bool lazy_android_link_namespaces_all_libs(android_namespace_t *from, android_namespace_t *to);
static bool lazy_android_link_namespaces(android_namespace_t *from, android_namespace_t *to, const char *shared_libs_sonames);

/* Public API */
bool linkernsbypass_load_status() {
    // Resolve lazily on first use instead of from a library constructor so
    // loading a shared library that links linkernsbypass never touches linker
    // internals at startup (avoids Android 17 ARM64 BTI SIGILL).
    static const bool resolved{[]() {
        resolve_linker_symbols();
        return true;
    }()};
    (void)resolved;
    return lib_loaded;
}

struct android_namespace_t *android_create_namespace(const char *name,
                                                     const char *ld_library_path,
                                                     const char *default_library_path,
                                                     uint64_t type,
                                                     const char *permitted_when_isolated_path,
                                                     android_namespace_t *parent_namespace) {
    if (!linkernsbypass_load_status() || !loader_android_create_namespace)
        return nullptr;
    auto caller{__builtin_return_address(0)};
    return loader_android_create_namespace(name, ld_library_path, default_library_path, type,
                                           permitted_when_isolated_path, parent_namespace, caller);
}

struct android_namespace_t *android_create_namespace_escape(const char *name,
                                                            const char *ld_library_path,
                                                            const char *default_library_path,
                                                            uint64_t type,
                                                            const char *permitted_when_isolated_path,
                                                            android_namespace_t *parent_namespace) {
    if (!linkernsbypass_load_status() || !loader_android_create_namespace)
        return nullptr;
    auto caller{reinterpret_cast<void *>(&dlopen)};
    return loader_android_create_namespace(name, ld_library_path, default_library_path, type,
                                           permitted_when_isolated_path, parent_namespace, caller);
}

android_get_exported_namespace_t android_get_exported_namespace = &lazy_android_get_exported_namespace;

android_link_namespaces_all_libs_t android_link_namespaces_all_libs = &lazy_android_link_namespaces_all_libs;

android_link_namespaces_t android_link_namespaces = &lazy_android_link_namespaces;

static android_namespace_t *lazy_android_get_exported_namespace(const char *name) {
    if (!linkernsbypass_load_status() || !android_get_exported_namespace ||
        android_get_exported_namespace == &lazy_android_get_exported_namespace)
        return nullptr;
    return android_get_exported_namespace(name);
}

static bool lazy_android_link_namespaces_all_libs(android_namespace_t *from, android_namespace_t *to) {
    if (!linkernsbypass_load_status() || !android_link_namespaces_all_libs ||
        android_link_namespaces_all_libs == &lazy_android_link_namespaces_all_libs)
        return false;
    return android_link_namespaces_all_libs(from, to);
}

static bool lazy_android_link_namespaces(android_namespace_t *from, android_namespace_t *to, const char *shared_libs_sonames) {
    if (!linkernsbypass_load_status() || !android_link_namespaces ||
        android_link_namespaces == &lazy_android_link_namespaces)
        return false;
    return android_link_namespaces(from, to, shared_libs_sonames);
}

bool linkernsbypass_link_namespace_to_default_all_libs(android_namespace_t *to) {
    if (!linkernsbypass_load_status())
        return false;
    // Creating a shared namespace with the default parent will give a copy of the default namespace that we can actually access
    // This is needed since there is no way to access a direct handle to the default namespace as it's not exported
    static auto defaultNs{android_create_namespace_escape("default_copy", nullptr, nullptr, ANDROID_NAMESPACE_TYPE_SHARED, nullptr, nullptr)};
    if (!defaultNs)
        return false;
    return android_link_namespaces_all_libs(to, defaultNs);
}

void *linkernsbypass_namespace_dlopen(const char *filename, int flags, android_namespace_t *ns) {
    android_dlextinfo extInfo{
        .flags = ANDROID_DLEXT_USE_NAMESPACE,
        .library_namespace = ns
    };

    return android_dlopen_ext(filename, flags, &extInfo);
}

#ifndef __NR_memfd_create
    #if defined(__aarch64__)
        #define __NR_memfd_create 279
    #else
        #error Unsupported target architecture!
    #endif
#endif

void *linkernsbypass_namespace_dlopen_unique(const char *libPath, const char *libTargetDir, int flags, android_namespace_t *ns) {
    static std::array<char, PATH_MAX> PathBuf{};

    // Used as a unique ID for overwriting soname and creating target lib files
    static uint16_t TargetId{};

    int libTargetFd{[&] () {
        if (libTargetDir) {
            snprintf(PathBuf.data(), PathBuf.size(), "%s/%d_patched.so", libTargetDir, TargetId);
            return open(PathBuf.data(), O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
        } else {
            // If memfd isn't supported errno will contain ENOSYS after calling
            errno = 0;
            int fd{static_cast<int>(syscall(__NR_memfd_create, libPath, 0))};
            if (errno == ENOSYS || fd < 0)
                return -1;
            else
                return fd;
        }
    }()};
    if (libTargetFd == -1)
        return nullptr;

    // Partially overwrite soname with 3 digits (replacing lib...) with to make sure a cached so isn't loaded
    std::array<char, 3> sonameOverwrite{};
    snprintf(sonameOverwrite.data(), sonameOverwrite.size(), "%03u", TargetId++);

    if (!elf_soname_patch(libPath, libTargetFd, sonameOverwrite.data()))
        return nullptr;

    // Load our patched library into the hook namespace
    android_dlextinfo hookExtInfo{
        .flags = ANDROID_DLEXT_USE_NAMESPACE | ANDROID_DLEXT_USE_LIBRARY_FD,
        .library_fd = libTargetFd,
        .library_namespace = ns
    };

    // Make a path that looks about right
    snprintf(PathBuf.data(), PathBuf.size(), "/proc/self/fd/%d", libTargetFd);

    return android_dlopen_ext(PathBuf.data(), flags, &hookExtInfo);
}

static size_t get_page_size() {
    const long page_size{sysconf(_SC_PAGESIZE)};
    return page_size > 0 ? static_cast<size_t>(page_size) : 4096u;
}

static void *align_ptr(void *ptr, size_t page_size) {
    return reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(ptr) & ~(static_cast<uintptr_t>(page_size) - 1));
}

#ifndef HWCAP2_BTI
    #define HWCAP2_BTI (1UL << 17)
#endif

// An indirect call (BLR) into a BTI-guarded AArch64 page may only land on
// BTI c, BTI jc, PACIASP, or PACIBSP.
static bool is_call_landing_pad(uint32_t instr) {
    return instr == 0xd503245fu || instr == 0xd50324dfu ||
           instr == 0xd503233fu || instr == 0xd503237fu;
}

/* Private */
static void resolve_linker_symbols() {
    using loader_dlopen_t = void *(*)(const char *, int, const void *);

    if (android_get_device_api_level() < 28)
        return;

    const size_t page_size{get_page_size()};

    // ARM64 specific function walking to locate the internal dlopen handler
    auto loader_dlopen{[page_size]() -> loader_dlopen_t {
        struct BranchLinked {
            uint32_t raw;

            bool Verify() const {
                return (raw >> 26) == 0x25u;
            }

            int32_t Offset() const {
                return static_cast<int32_t>(raw << 6) >> 6;
            }
        };
        static_assert(sizeof(BranchLinked) == 4, "BranchLinked is wrong size");

        constexpr int MaxWrapperInstrs{32};
        const uintptr_t start_addr{reinterpret_cast<uintptr_t>(&dlopen)};
        const uintptr_t aligned_start{start_addr & ~(static_cast<uintptr_t>(page_size) - 1)};
        const uintptr_t end_addr{start_addr + static_cast<uintptr_t>(MaxWrapperInstrs) * sizeof(BranchLinked)};
        const size_t span{static_cast<size_t>(((end_addr - aligned_start) + page_size - 1) & ~(static_cast<uintptr_t>(page_size) - 1))};

        // Some legacy devices ship with --X (XOM) mappings for executables; request
        // PROT_READ | PROT_EXEC only (never RWX).
        (void)mprotect(reinterpret_cast<void *>(aligned_start), span, PROT_READ | PROT_EXEC);

        // dlopen is just a wrapper for __loader_dlopen that passes the return address as the third arg hence we can just walk it to find __loader_dlopen.
        // Give up safely if the wrapper no longer contains a BL within MaxWrapperInstrs.
        auto blInstr{reinterpret_cast<const BranchLinked *>(&dlopen)};
        for (int i{}; i < MaxWrapperInstrs; ++i, ++blInstr) {
            if (blInstr->Verify())
                return reinterpret_cast<loader_dlopen_t>(blInstr + blInstr->Offset());
        }
        return nullptr;
    }()};
    if (!loader_dlopen)
        return;

    // Strip the BTI attribute from __loader_dlopen's actual code page (not the
    // stack local variable's address) using PROT_READ | PROT_EXEC (no RWX).
    void *loader_dlopen_page{align_ptr(reinterpret_cast<void *>(loader_dlopen), page_size)};
    const bool bti_removed{mprotect(loader_dlopen_page, page_size, PROT_READ | PROT_EXEC) == 0};

    // If mprotect failed, only allow the indirect call when the CPU has no BTI
    // or the target instruction is a valid BTI indirect-call landing pad.
    if (!bti_removed && (getauxval(AT_HWCAP2) & HWCAP2_BTI) != 0 &&
        !is_call_landing_pad(*reinterpret_cast<const uint32_t *>(loader_dlopen))) {
        __android_log_print(
            ANDROID_LOG_WARN, "linkernsbypass",
            "Refusing indirect call into BTI-guarded __loader_dlopen (mprotect errno=%d)",
            errno);
        return;
    }

    // Passing dlopen as a caller address tricks the linker into using the internal unrestricted namespace letting us access libraries that are normally forbidden in the classloader namespace imposed on apps
    auto ldHandle{loader_dlopen("ld-android.so", RTLD_LAZY, reinterpret_cast<void *>(&dlopen))};
    if (!ldHandle)
        return;

    auto link_all_sym{reinterpret_cast<android_link_namespaces_all_libs_t>(dlsym(ldHandle, "__loader_android_link_namespaces_all_libs"))};
    if (!link_all_sym)
        return;

    auto link_sym{reinterpret_cast<android_link_namespaces_t>(dlsym(ldHandle, "__loader_android_link_namespaces"))};
    if (!link_sym)
        return;

    auto libdlAndroidHandle{loader_dlopen("libdl_android.so", RTLD_LAZY, reinterpret_cast<void *>(&dlopen))};
    if (!libdlAndroidHandle)
        return;

    auto create_ns_sym{reinterpret_cast<loader_android_create_namespace_t>(dlsym(libdlAndroidHandle, "__loader_android_create_namespace"))};
    if (!create_ns_sym)
        return;

    auto exported_ns_sym{reinterpret_cast<android_get_exported_namespace_t>(dlsym(libdlAndroidHandle, "__loader_android_get_exported_namespace"))};
    if (!exported_ns_sym)
        return;

    android_link_namespaces_all_libs = link_all_sym;
    android_link_namespaces = link_sym;
    loader_android_create_namespace = create_ns_sym;
    android_get_exported_namespace = exported_ns_sym;

    // Lib is now safe to use
    lib_loaded = true;
}
