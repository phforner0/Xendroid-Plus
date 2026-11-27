/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2020 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/gpu/vulkan/vulkan_shared_memory.h"

#include <algorithm>
#include <cstring>

#if defined(__ANDROID__)
#include <sys/system_properties.h>
#endif

#include "xenia/base/assert.h"
#include "xenia/base/cvar.h"
#include "xenia/base/logging.h"
#include "xenia/base/math.h"
#include "xenia/base/memory.h"
#include "xenia/gpu/vulkan/deferred_command_buffer.h"
#include "xenia/gpu/vulkan/vulkan_command_processor.h"
#include "xenia/ui/vulkan/vulkan_util.h"

DECLARE_bool(gpu_allow_invalid_upload_range);
DECLARE_bool(memexport_enable);
DECLARE_bool(shared_memory_zero_copy);
DECLARE_string(readback_resolve);

DEFINE_bool(vulkan_sparse_shared_memory, true,
            "Enable sparse binding for shared memory emulation. Disabling it "
            "increases video memory usage - a 512 MB buffer is created - but "
            "allows graphics debuggers that don't support sparse binding to "
            "work.",
            "Vulkan");

DEFINE_bool(vulkan_hoist_shmem_uploads, true,
            "Record shared-memory uploads whose pages were not invalidated "
            "since the current submission opened at the start of the "
            "submission's command buffer instead of breaking the current "
            "render pass (expensive on tile-based GPUs).",
            "Vulkan");

DEFINE_bool(vulkan_shared_memory_host_visible, true,
            "On unified-memory GPUs (Adreno, integrated), allocate the "
            "shared-memory buffer from a host-visible cached memory type and "
            "map it so resolve readback (readback_resolve=uma) can read guest "
            "memory directly without a GPU staging copy. Only applies to the "
            "dense (non-sparse) buffer. Measured on Adreno 830 / Turnip in "
            "TDU2: +11.6% fps (22.6 -> 25.2, p<0.001), from dropping ~153 "
            "staging copies per frame. Pass breaks shift (resolve readback "
            "breaks out, texture-upload breaks in) but net out, and per-"
            "submission GPU time barely moves - the win is bandwidth, not "
            "pass structure.",
            "Vulkan");

DEFINE_bool(
    vulkan_shared_memory_zero_copy_upload, true,
    "When the shared-memory buffer is densely host-mapped "
    "(DEVICE_LOCAL | HOST_VISIBLE | HOST_CACHED), write CPU uploads directly "
    "into host_mapped_data_ for pages whose last GPU submission has already "
    "completed, avoiding staging copies and pass breaks. Falls back to "
    "upload_buffer_pool_ whenever a page may still be in flight or was used "
    "earlier in the current submission. debug.xendroid.zero_copy_upload "
    "('0'|'1') overrides on Android.",
    "Vulkan");

namespace xe {
namespace gpu {
namespace vulkan {

VulkanSharedMemory::VulkanSharedMemory(
    VulkanCommandProcessor& command_processor, Memory& memory,
    TraceWriter& trace_writer,
    VkPipelineStageFlags guest_shader_pipeline_stages)
    : SharedMemory(memory),
      command_processor_(command_processor),
      trace_writer_(trace_writer),
      guest_shader_pipeline_stages_(guest_shader_pipeline_stages) {}

VulkanSharedMemory::~VulkanSharedMemory() { Shutdown(true); }

bool VulkanSharedMemory::Initialize() {
  if (!InitializeCommon()) {
    return false;
  }

  PollZeroCopyUploadProperty();
  zero_copy_submission_tracker_.Initialize(kBufferSizeLog2);
  zero_copy_prop_poll_counter_ = 0;

  const ui::vulkan::VulkanDevice* const vulkan_device =
      command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();

  // Zero-copy: alias guest RAM as the buffer. On success the buffer is a single
  // fully-resident non-sparse allocation backed by guest RAM, so the normal
  // sparse/device-local paths below are skipped.
  if (cvars::shared_memory_zero_copy && TryInitializeZeroCopy()) {
    last_usage_ = Usage::kTransferDestination;
    last_written_range_ = std::make_pair<uint32_t, uint32_t>(0, 0);
    upload_buffer_pool_ = std::make_unique<ui::vulkan::VulkanUploadBufferPool>(
        vulkan_device, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        xe::align(ui::vulkan::VulkanUploadBufferPool::kDefaultPageSize,
                  size_t(1) << page_size_log2()));
    return true;
  }

  const VkBufferCreateFlags sparse_flags =
      VK_BUFFER_CREATE_SPARSE_BINDING_BIT |
      VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT;

  // When a dedicated transfer queue is used for resolve readback, the buffer is
  // read from both the graphics/compute and transfer families, so it must be
  // shared concurrently between them (no compression penalty for a buffer).
  const uint32_t transfer_family = vulkan_device->queue_family_transfer();
  const uint32_t concurrent_queue_families[2] = {
      vulkan_device->queue_family_graphics_compute(), transfer_family};

  const VkBufferUsageFlags base_buffer_usage =
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
  const bool device_has_bda = vulkan_device->properties().bufferDeviceAddress;

  // Try to create a sparse buffer.
  VkBufferCreateInfo buffer_create_info;
  buffer_create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_create_info.pNext = nullptr;
  buffer_create_info.flags = sparse_flags;
  buffer_create_info.size = kBufferSize;
  buffer_create_info.usage = base_buffer_usage;
  if (transfer_family != UINT32_MAX) {
    buffer_create_info.sharingMode = VK_SHARING_MODE_CONCURRENT;
    buffer_create_info.queueFamilyIndexCount = 2;
    buffer_create_info.pQueueFamilyIndices = concurrent_queue_families;
  } else {
    buffer_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    buffer_create_info.queueFamilyIndexCount = 0;
    buffer_create_info.pQueueFamilyIndices = nullptr;
  }
  // Second (host-imported) buffer for memexport and kFast/kAll resolve
  // readbacks when VK_EXT_external_memory_host is available. Probed before the
  // sparse/dense choice below so unified-memory GPUs without host-memory import
  // (Adreno) fall back to a dense host-mapped buffer whenever readback or
  // memexport needs CPU visibility.
  TryInitializeHostBuffer();

  const ReadbackResolveMode readback_mode = GetReadbackResolveMode();
  const bool uma_readback = readback_mode == ReadbackResolveMode::kUma;
  const bool zero_copy_upload_enabled =
      cvars::vulkan_shared_memory_zero_copy_upload;
  const bool readback_or_memexport_no_host_buf =
      host_buffer_ == VK_NULL_HANDLE &&
      (readback_mode != ReadbackResolveMode::kDisabled ||
       cvars::memexport_enable);
  const bool needs_host_mapping = uma_readback || zero_copy_upload_enabled ||
                                  readback_or_memexport_no_host_buf;
  if (needs_host_mapping && cvars::vulkan_sparse_shared_memory) {
    XELOGI(
        "Shared memory: sparse binding bypassed for dense host-mapped 512 MB "
        "buffer (reason: {})",
        DescribeDenseSharedMemoryReason(uma_readback, zero_copy_upload_enabled,
                                        readback_or_memexport_no_host_buf));
  }
  buffer_device_address_ = 0;
  bool buffer_has_bda = false;
  auto try_create_sparse_buffer = [&]() {
    if (!vulkan_device->properties().sparseResidencyBuffer) {
      return;
    }
    buffer_create_info.flags |= sparse_flags;
    for (int bda_attempt = 0;
         bda_attempt < (device_has_bda ? 2 : 1) && buffer_ == VK_NULL_HANDLE;
         ++bda_attempt) {
      const bool try_bda = device_has_bda && (bda_attempt == 0);
      buffer_create_info.usage =
          base_buffer_usage |
          (try_bda ? VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT : 0);
      if (dfn.vkCreateBuffer(device, &buffer_create_info, nullptr, &buffer_) ==
          VK_SUCCESS) {
        VkMemoryRequirements buffer_memory_requirements;
        dfn.vkGetBufferMemoryRequirements(device, buffer_,
                                          &buffer_memory_requirements);
        if (xe::bit_scan_forward(buffer_memory_requirements.memoryTypeBits &
                                     vulkan_device->memory_types().device_local,
                                 &buffer_memory_type_)) {
          uint32_t allocation_size_log2;
          xe::bit_scan_forward(
              std::max(uint64_t(buffer_memory_requirements.alignment),
                       uint64_t(1)),
              &allocation_size_log2);
          if (allocation_size_log2 < kBufferSizeLog2) {
            // Maximum of 1024 allocations in the worst case for all of the
            // buffer because of the overall 4096 allocation count limit on
            // Windows drivers.
            InitializeSparseHostGpuMemory(
                std::max(allocation_size_log2,
                         std::max(kHostGpuMemoryOptimalSparseAllocationLog2,
                                  kBufferSizeLog2 - uint32_t(10))));
            buffer_has_bda = try_bda;
          } else {
            // Shouldn't happen on any real platform, but no point allocating
            // the buffer sparsely.
            dfn.vkDestroyBuffer(device, buffer_, nullptr);
            buffer_ = VK_NULL_HANDLE;
            break;
          }
        } else {
          if (try_bda) {
            XELOGW(
                "Shared memory: No device-local memory type for sparse buffer "
                "with BDA; retrying without SHADER_DEVICE_ADDRESS_BIT");
          } else {
            XELOGE(
                "Shared memory: Failed to get a device-local Vulkan memory "
                "type for the sparse buffer");
          }
          dfn.vkDestroyBuffer(device, buffer_, nullptr);
          buffer_ = VK_NULL_HANDLE;
        }
      } else if (try_bda) {
        XELOGW(
            "Shared memory: Sparse buffer creation with BDA failed; retrying "
            "without SHADER_DEVICE_ADDRESS_BIT");
      } else {
        XELOGE("Shared memory: Failed to create the {} MB Vulkan sparse buffer",
               kBufferSize >> 20);
      }
    }
  };
  if (cvars::vulkan_sparse_shared_memory && !needs_host_mapping) {
    try_create_sparse_buffer();
  }

  // Create a non-sparse buffer if sparse was bypassed or failed.
  if (buffer_ == VK_NULL_HANDLE) {
    XELOGGPU(
        "Vulkan sparse binding is not used for shared memory emulation - video "
        "memory usage may increase significantly because a full {} MB buffer "
        "will be created",
        kBufferSize >> 20);
    buffer_create_info.flags &= ~sparse_flags;
    bool buffer_host_visible = false;
    bool buffer_host_coherent = false;
    VkDeviceMemory buffer_memory = VK_NULL_HANDLE;
    bool dense_alloc_failed = false;
    for (int bda_attempt = 0; bda_attempt < (device_has_bda ? 2 : 1);
         ++bda_attempt) {
      const bool try_bda = device_has_bda && (bda_attempt == 0);
      buffer_create_info.usage =
          base_buffer_usage |
          (try_bda ? VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT : 0);
      if (dfn.vkCreateBuffer(device, &buffer_create_info, nullptr, &buffer_) !=
          VK_SUCCESS) {
        buffer_ = VK_NULL_HANDLE;
        if (try_bda) {
          XELOGW(
              "Shared memory: Dense buffer creation with BDA failed; retrying "
              "without SHADER_DEVICE_ADDRESS_BIT");
          continue;
        }
        XELOGE("Shared memory: Failed to create the {} MB Vulkan buffer",
               kBufferSize >> 20);
        dense_alloc_failed = true;
        break;
      }
      VkMemoryRequirements buffer_memory_requirements;
      dfn.vkGetBufferMemoryRequirements(device, buffer_,
                                        &buffer_memory_requirements);
      const ui::vulkan::VulkanDevice::MemoryTypes& memory_types =
          vulkan_device->memory_types();
      const uint32_t buffer_memory_type_bits =
          buffer_memory_requirements.memoryTypeBits;
      // Map for direct CPU readback when a host-visible type is available, so
      // resolve readback needs no staging copy. Prefer cached-coherent, accept
      // cached non-coherent; never map uncached (CPU reads are slower than
      // staging). Test this buffer's own candidate types - a whole-device test
      // fails on Adreno, whose LAZILY_ALLOCATED type is not host-visible.
      const bool is_uma =
          memory_types.device_local &&
          (memory_types.device_local & memory_types.host_visible) ==
              memory_types.device_local;
      buffer_host_visible = false;
      buffer_host_coherent = false;
      if (cvars::vulkan_shared_memory_host_visible) {
        const uint32_t cached_coherent = buffer_memory_type_bits &
                                         memory_types.device_local &
                                         memory_types.host_visible &
                                         memory_types.host_cached &
                                         memory_types.host_coherent;
        const uint32_t cached = buffer_memory_type_bits &
                                memory_types.device_local &
                                memory_types.host_visible &
                                memory_types.host_cached;
        if (xe::bit_scan_forward(cached_coherent, &buffer_memory_type_)) {
          buffer_host_visible = true;
          buffer_host_coherent = true;
        } else if (xe::bit_scan_forward(cached, &buffer_memory_type_)) {
          buffer_host_visible = true;
        }
      }
      XELOGI(
          "Shared memory: host-map decision: cvar={} is_uma={} bda={} "
          "type_bits={:#x} device_local={:#x} host_visible={:#x} "
          "host_cached={:#x} host_coherent={:#x} -> host_visible={} "
          "coherent={}",
          cvars::vulkan_shared_memory_host_visible, is_uma, try_bda,
          buffer_memory_type_bits, memory_types.device_local,
          memory_types.host_visible, memory_types.host_cached,
          memory_types.host_coherent, buffer_host_visible,
          buffer_host_coherent);

      if (!buffer_host_visible &&
          !xe::bit_scan_forward(
              buffer_memory_type_bits & memory_types.device_local,
              &buffer_memory_type_)) {
        dfn.vkDestroyBuffer(device, buffer_, nullptr);
        buffer_ = VK_NULL_HANDLE;
        if (try_bda) {
          XELOGW(
              "Shared memory: No device-local memory type for dense buffer "
              "with BDA; retrying without SHADER_DEVICE_ADDRESS_BIT");
          continue;
        }
        XELOGE(
            "Shared memory: Failed to get a device-local Vulkan memory type "
            "for the buffer");
        dense_alloc_failed = true;
        break;
      }
      VkMemoryAllocateInfo buffer_memory_allocate_info;
      buffer_memory_allocate_info.sType =
          VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
      buffer_memory_allocate_info.pNext = nullptr;
      buffer_memory_allocate_info.allocationSize =
          buffer_memory_requirements.size;
      buffer_memory_allocate_info.memoryTypeIndex = buffer_memory_type_;
      VkMemoryDedicatedAllocateInfo buffer_memory_dedicated_allocate_info;
      if (vulkan_device->extensions().ext_1_1_KHR_dedicated_allocation) {
        buffer_memory_dedicated_allocate_info.sType =
            VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
        buffer_memory_dedicated_allocate_info.pNext =
            buffer_memory_allocate_info.pNext;
        buffer_memory_dedicated_allocate_info.image = VK_NULL_HANDLE;
        buffer_memory_dedicated_allocate_info.buffer = buffer_;
        buffer_memory_allocate_info.pNext =
            &buffer_memory_dedicated_allocate_info;
      }
      VkMemoryAllocateFlagsInfo buffer_memory_allocate_flags_info = {};
      if (try_bda) {
        buffer_memory_allocate_flags_info.sType =
            VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        buffer_memory_allocate_flags_info.pNext =
            buffer_memory_allocate_info.pNext;
        buffer_memory_allocate_flags_info.flags =
            VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        buffer_memory_allocate_flags_info.deviceMask = 0;
        buffer_memory_allocate_info.pNext = &buffer_memory_allocate_flags_info;
      }
      if (dfn.vkAllocateMemory(device, &buffer_memory_allocate_info, nullptr,
                               &buffer_memory) != VK_SUCCESS) {
        dfn.vkDestroyBuffer(device, buffer_, nullptr);
        buffer_ = VK_NULL_HANDLE;
        buffer_memory = VK_NULL_HANDLE;
        if (try_bda) {
          XELOGW(
              "Shared memory: vkAllocateMemory with DEVICE_ADDRESS_BIT failed; "
              "retrying without SHADER_DEVICE_ADDRESS_BIT");
          continue;
        }
        XELOGE(
            "Shared memory: Failed to allocate {} MB of memory for the Vulkan "
            "buffer",
            kBufferSize >> 20);
        dense_alloc_failed = true;
        break;
      }
      if (dfn.vkBindBufferMemory(device, buffer_, buffer_memory, 0) !=
          VK_SUCCESS) {
        dfn.vkFreeMemory(device, buffer_memory, nullptr);
        dfn.vkDestroyBuffer(device, buffer_, nullptr);
        buffer_ = VK_NULL_HANDLE;
        buffer_memory = VK_NULL_HANDLE;
        if (try_bda) {
          XELOGW(
              "Shared memory: vkBindBufferMemory with BDA failed; retrying "
              "without SHADER_DEVICE_ADDRESS_BIT");
          continue;
        }
        XELOGE("Shared memory: Failed to bind memory to the Vulkan buffer");
        dense_alloc_failed = true;
        break;
      }
      buffer_memory_.push_back(buffer_memory);
      buffer_has_bda = try_bda;
      break;
    }

    if (dense_alloc_failed) {
      // Low-RAM fallback: if dense 512 MB allocation failed when sparse was
      // bypassed solely for host mapping (zero-copy upload / UMA), degrade
      // cleanly to a sparse device-local buffer (staging uploads).
      if (needs_host_mapping && cvars::vulkan_sparse_shared_memory &&
          vulkan_device->properties().sparseResidencyBuffer) {
        XELOGW(
            "Shared memory: dense {} MB allocation failed; falling back to "
            "sparse device-local buffer (zero-copy upload / direct host "
            "mapping disabled for this session)",
            kBufferSize >> 20);
        try_create_sparse_buffer();
      }
      if (buffer_ == VK_NULL_HANDLE) {
        Shutdown();
        return false;
      }
    } else if (buffer_host_visible && buffer_memory != VK_NULL_HANDLE) {
      // Persistently map the buffer for direct CPU readback (readback_resolve=uma)
      // and zero-copy uploads when it landed on a host-visible type.
      void* mapped_data;
      if (dfn.vkMapMemory(device, buffer_memory, 0, VK_WHOLE_SIZE, 0,
                          &mapped_data) == VK_SUCCESS) {
        host_mapped_data_ = static_cast<uint8_t*>(mapped_data);
        host_mapped_coherent_ = buffer_host_coherent;
        XELOGGPU(
            "Shared memory: buffer is host-mapped for direct readback "
            "(readback_resolve=uma available), coherent={}",
            buffer_host_coherent ? "yes" : "no");
      } else {
        XELOGW(
            "Shared memory: failed to map the host-visible buffer; direct "
            "readback (uma) disabled");
      }
    }
  }

  // Cache the static VkDeviceAddress once at initialization whenever the buffer
  // was created with SHADER_DEVICE_ADDRESS_BIT so IssueDraw never calls
  // vkGetBufferDeviceAddress per draw (even if BDA is enabled post-init).
  if (buffer_ != VK_NULL_HANDLE && buffer_has_bda && !zero_copy_) {
    VkBufferDeviceAddressInfo bda_info = {};
    bda_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    bda_info.buffer = buffer_;
    if (dfn.vkGetBufferDeviceAddress) {
      buffer_device_address_ = dfn.vkGetBufferDeviceAddress(device, &bda_info);
    } else if (dfn.vkGetBufferDeviceAddressKHR) {
      buffer_device_address_ =
          dfn.vkGetBufferDeviceAddressKHR(device, &bda_info);
    }
  }

  // The first usage will likely be uploading.
  last_usage_ = Usage::kTransferDestination;
  last_written_range_ = std::make_pair<uint32_t, uint32_t>(0, 0);

  upload_buffer_pool_ = std::make_unique<ui::vulkan::VulkanUploadBufferPool>(
      vulkan_device, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      xe::align(ui::vulkan::VulkanUploadBufferPool::kDefaultPageSize,
                size_t(1) << page_size_log2()));

  if (cvars::memexport_enable && host_buffer_ == VK_NULL_HANDLE) {
    if (IsHostMapped()) {
      XELOGI("Shared memory: guest RAM import unavailable - memexport uses "
             "host-mapped readback");
    } else {
      XELOGW("Shared memory: memexport needs guest RAM import or a host-mapped "
             "buffer");
    }
  }

  return true;
}

bool VulkanSharedMemory::CreateImportedGuestRamBuffer(
    VkBuffer& out_buffer, VkDeviceMemory& out_memory) {
  const ui::vulkan::VulkanDevice* const vulkan_device =
      command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();

  PFN_vkGetMemoryHostPointerPropertiesEXT get_host_pointer_properties =
      vulkan_device->vkGetMemoryHostPointerPropertiesEXT();
  if (!vulkan_device->extensions().ext_EXT_external_memory_host ||
      !get_host_pointer_properties) {
    XELOGI(
        "Shared memory host import: VK_EXT_external_memory_host not available");
    return false;
  }

  void* const guest_ram = memory().TranslatePhysical(0);
  if (!guest_ram) {
    XELOGE("Shared memory host import: guest RAM is null");
    return false;
  }

  // The host pointer and the imported size must both be aligned to the import
  // granularity.
  VkDeviceSize import_alignment =
      vulkan_device->properties().minImportedHostPointerAlignment;
  if (import_alignment == 0) {
    import_alignment = 1;
  }
  if ((reinterpret_cast<uintptr_t>(guest_ram) % import_alignment) != 0 ||
      (VkDeviceSize(kBufferSize) % import_alignment) != 0) {
    XELOGI(
        "Shared memory host import: guest RAM 0x{:X} not aligned to import "
        "granularity {}",
        reinterpret_cast<uintptr_t>(guest_ram), import_alignment);
    return false;
  }

  // Which memory types can back this exact host pointer.
  VkMemoryHostPointerPropertiesEXT host_pointer_properties = {
      VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
  if (get_host_pointer_properties(
          device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
          guest_ram, &host_pointer_properties) != VK_SUCCESS) {
    XELOGI(
        "Shared memory host import: vkGetMemoryHostPointerPropertiesEXT "
        "failed");
    return false;
  }

  // A plain non-sparse buffer to bind the imported memory to. It must declare
  // the external handle type it will be bound to, or vkBindBufferMemory is
  // invalid (VUID-vkBindBufferMemory-memory-02985).
  const uint32_t transfer_family = vulkan_device->queue_family_transfer();
  const uint32_t concurrent_queue_families[2] = {
      vulkan_device->queue_family_graphics_compute(), transfer_family};
  VkExternalMemoryBufferCreateInfo external_memory_buffer_create_info;
  external_memory_buffer_create_info.sType =
      VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
  external_memory_buffer_create_info.pNext = nullptr;
  external_memory_buffer_create_info.handleTypes =
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
  const VkBufferUsageFlags base_buffer_usage =
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
  const bool device_has_bda = vulkan_device->properties().bufferDeviceAddress;

  for (int bda_attempt = 0; bda_attempt < (device_has_bda ? 2 : 1);
       ++bda_attempt) {
    const bool try_bda = device_has_bda && (bda_attempt == 0);
    VkBufferCreateInfo buffer_create_info;
    buffer_create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_create_info.pNext = &external_memory_buffer_create_info;
    buffer_create_info.flags = 0;
    buffer_create_info.size = kBufferSize;
    buffer_create_info.usage =
        base_buffer_usage |
        (try_bda ? VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT : 0);
    if (transfer_family != UINT32_MAX) {
      buffer_create_info.sharingMode = VK_SHARING_MODE_CONCURRENT;
      buffer_create_info.queueFamilyIndexCount = 2;
      buffer_create_info.pQueueFamilyIndices = concurrent_queue_families;
    } else {
      buffer_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      buffer_create_info.queueFamilyIndexCount = 0;
      buffer_create_info.pQueueFamilyIndices = nullptr;
    }
    VkBuffer buffer = VK_NULL_HANDLE;
    if (dfn.vkCreateBuffer(device, &buffer_create_info, nullptr, &buffer) !=
        VK_SUCCESS) {
      if (try_bda) {
        XELOGI(
            "Shared memory host import: buffer creation with BDA rejected; "
            "retrying without SHADER_DEVICE_ADDRESS_BIT");
        continue;
      }
      XELOGE("Shared memory host import: failed to create the buffer");
      return false;
    }

    VkMemoryRequirements buffer_memory_requirements;
    dfn.vkGetBufferMemoryRequirements(device, buffer,
                                      &buffer_memory_requirements);

    // The memory type must satisfy the buffer, accept the host pointer, and be
    // host-visible + host-coherent so guest and GPU observe the same bytes.
    const uint32_t memory_type_bits =
        buffer_memory_requirements.memoryTypeBits &
        host_pointer_properties.memoryTypeBits &
        vulkan_device->memory_types().host_visible &
        vulkan_device->memory_types().host_coherent;
    uint32_t memory_type;
    if (!xe::bit_scan_forward(memory_type_bits, &memory_type)) {
      dfn.vkDestroyBuffer(device, buffer, nullptr);
      if (try_bda) {
        XELOGI(
            "Shared memory host import: no coherent host-pointer memory type "
            "with BDA; retrying without SHADER_DEVICE_ADDRESS_BIT");
        continue;
      }
      XELOGI(
          "Shared memory host import: no host-visible coherent memory type "
          "accepts the guest RAM pointer");
      return false;
    }

    VkMemoryAllocateFlagsInfo memory_allocate_flags_info = {};
    memory_allocate_flags_info.sType =
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    memory_allocate_flags_info.pNext = nullptr;
    memory_allocate_flags_info.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    memory_allocate_flags_info.deviceMask = 0;

    VkImportMemoryHostPointerInfoEXT import_info;
    import_info.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT;
    import_info.pNext = try_bda ? &memory_allocate_flags_info : nullptr;
    import_info.handleType =
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    import_info.pHostPointer = guest_ram;

    VkMemoryAllocateInfo memory_allocate_info;
    memory_allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memory_allocate_info.pNext = &import_info;
    memory_allocate_info.allocationSize = kBufferSize;
    memory_allocate_info.memoryTypeIndex = memory_type;

    VkDeviceMemory imported_memory = VK_NULL_HANDLE;
    if (dfn.vkAllocateMemory(device, &memory_allocate_info, nullptr,
                             &imported_memory) != VK_SUCCESS) {
      dfn.vkDestroyBuffer(device, buffer, nullptr);
      if (try_bda) {
        XELOGI(
            "Shared memory host import: vkAllocateMemory with "
            "DEVICE_ADDRESS_BIT rejected; retrying without BDA");
        continue;
      }
      XELOGE("Shared memory host import: failed to import {} MB of guest RAM",
             kBufferSize >> 20);
      return false;
    }

    if (dfn.vkBindBufferMemory(device, buffer, imported_memory, 0) !=
        VK_SUCCESS) {
      dfn.vkFreeMemory(device, imported_memory, nullptr);
      dfn.vkDestroyBuffer(device, buffer, nullptr);
      if (try_bda) {
        XELOGI(
            "Shared memory host import: vkBindBufferMemory with BDA rejected; "
            "retrying without BDA");
        continue;
      }
      XELOGE("Shared memory host import: failed to bind imported memory");
      return false;
    }

    out_buffer = buffer;
    out_memory = imported_memory;
    // The import pins guest RAM - a later mprotect on the alias would fail the
    // next submit with EFAULT.
    memory().SetPhysicalAliasSkipHostProtect(true);
    return true;
  }
  return false;
}

bool VulkanSharedMemory::TryInitializeZeroCopy() {
  VkDeviceMemory buffer_memory;
  if (!CreateImportedGuestRamBuffer(buffer_, buffer_memory)) {
    buffer_ = VK_NULL_HANDLE;
    return false;
  }
  buffer_memory_.push_back(buffer_memory);
  zero_copy_ = true;
  SetUnwatchSkipVerify(true);
  XELOGI("Shared memory: using zero-copy guest RAM aliasing");
  return true;
}

void VulkanSharedMemory::TryInitializeHostBuffer() {
  const ReadbackResolveMode readback_mode = GetReadbackResolveMode();
  if (!cvars::memexport_enable &&
      readback_mode != ReadbackResolveMode::kAll &&
      readback_mode != ReadbackResolveMode::kFast) {
    return;
  }
  // A second, host-imported (guest RAM) buffer used for memexport-touching
  // draws and non-UMA resolve readbacks while the main buffer stays fast
  // device-local. Non-sparse, so it can accept host memory where the sparse
  // buffer can't.
  if (!CreateImportedGuestRamBuffer(host_buffer_, host_buffer_memory_)) {
    host_buffer_ = VK_NULL_HANDLE;
    host_buffer_memory_ = VK_NULL_HANDLE;
    return;
  }
  XELOGI("Shared memory: host buffer for guest RAM import ready");
}

void VulkanSharedMemory::Shutdown(bool from_destructor) {
  ResetTraceDownload();

  upload_buffer_pool_.reset();

  const ui::vulkan::VulkanDevice* const vulkan_device =
      command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();

  if (host_mapped_data_ != nullptr && !buffer_memory_.empty()) {
    dfn.vkUnmapMemory(device, buffer_memory_.front());
    host_mapped_data_ = nullptr;
    host_mapped_coherent_ = false;
  }

  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyBuffer, device, buffer_);
  buffer_device_address_ = 0;
  for (VkDeviceMemory memory : buffer_memory_) {
    dfn.vkFreeMemory(device, memory, nullptr);
  }
  buffer_memory_.clear();
  zero_copy_ = false;
  SetUnwatchSkipVerify(false);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyBuffer, device,
                                         host_buffer_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device,
                                         host_buffer_memory_);
  // No import left to pin the alias.
  memory().SetPhysicalAliasSkipHostProtect(false);

  zero_copy_submission_tracker_.Reset();
  zero_copy_spans_.clear();
  staging_spans_.clear();
  zero_copy_flush_ranges_.clear();
  vk_flush_ranges_.clear();
  zero_copy_prop_poll_counter_ = 0;

  // If calling from the destructor, the SharedMemory destructor will call
  // ShutdownCommon.
  if (!from_destructor) {
    ShutdownCommon();
  }
}

void VulkanSharedMemory::ClearCache() {
  SharedMemory::ClearCache();

  upload_buffer_pool_->ClearCache();
}

void VulkanSharedMemory::CompletedSubmissionUpdated() {
  upload_buffer_pool_->Reclaim(command_processor_.GetCompletedSubmission());
}

void VulkanSharedMemory::EndSubmission() {
  upload_buffer_pool_->FlushWrites();
  if (++zero_copy_prop_poll_counter_ >= 30) {
    zero_copy_prop_poll_counter_ = 0;
    PollZeroCopyUploadProperty();
  }
}

void VulkanSharedMemory::PollZeroCopyUploadProperty() {
#if defined(__ANDROID__)
  char value[PROP_VALUE_MAX] = {};
  if (__system_property_get("debug.xendroid.zero_copy_upload", value) > 0) {
    bool enabled = false;
    if (ParseBinaryDebugProperty(value, enabled) &&
        cvars::vulkan_shared_memory_zero_copy_upload != enabled) {
      cvars::vulkan_shared_memory_zero_copy_upload = enabled;
      XELOGI(
          "debug.xendroid.zero_copy_upload: "
          "vulkan_shared_memory_zero_copy_upload = {}",
          enabled);
    }
  }
#endif
}

void VulkanSharedMemory::NoteByteRangeUsedByGpu(uint32_t start,
                                                uint32_t length) {
  if (!length || start >= kBufferSize) {
    return;
  }
  length = std::min(length, kBufferSize - start);
  if (!subpage_tracking_active()) {
    const uint32_t page_first = start >> page_size_log2();
    const uint32_t page_last = (start + length - 1) >> page_size_log2();
    start = page_first << page_size_log2();
    length = (page_last - page_first + 1) << page_size_log2();
  }
  zero_copy_submission_tracker_.MarkByteRangeUsed(
      start, length, command_processor_.GetCurrentSubmission());
}

void VulkanSharedMemory::Use(Usage usage,
                             std::pair<uint32_t, uint32_t> written_range) {
  written_range.first = std::min(written_range.first, kBufferSize);
  written_range.second =
      std::min(written_range.second, kBufferSize - written_range.first);
  assert_true(usage != Usage::kRead || !written_range.second);
  if (written_range.second) {
    NoteByteRangeUsedByGpu(written_range.first, written_range.second);
  }
  if (last_usage_ != usage || last_written_range_.second) {
    VkPipelineStageFlags src_stage_mask, dst_stage_mask;
    VkAccessFlags src_access_mask, dst_access_mask;
    GetUsageMasks(last_usage_, src_stage_mask, src_access_mask);
    GetUsageMasks(usage, dst_stage_mask, dst_access_mask);
    VkDeviceSize offset, size;
    if (last_usage_ == usage) {
      // Committing the previous write, while not changing the access mask
      // (passing false as whether to skip the barrier if no masks are changed
      // for this reason).
      offset = VkDeviceSize(last_written_range_.first);
      size = VkDeviceSize(last_written_range_.second);
    } else {
      // Changing the stage and access mask - all preceding writes must be
      // available not only to the source stage, but to the destination as well.
      offset = 0;
      size = VK_WHOLE_SIZE;
      last_usage_ = usage;
    }
    command_processor_.PushBufferMemoryBarrier(
        buffer_, offset, size, src_stage_mask, dst_stage_mask, src_access_mask,
        dst_access_mask, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        false);
    // All the draws' accesses so far are ordered by it.
    draw_exports_.Clear();
    // Memexport-touching draws are routed to the host-imported buffer, so their
    // writes and the reads that consume them go through host_buffer_, not
    // buffer_. Mirror the same ordering barrier onto it so those GPU
    // write->read dependencies are synchronized too.
    if (host_buffer_ != VK_NULL_HANDLE) {
      command_processor_.PushBufferMemoryBarrier(
          host_buffer_, offset, size, src_stage_mask, dst_stage_mask,
          src_access_mask, dst_access_mask, VK_QUEUE_FAMILY_IGNORED,
          VK_QUEUE_FAMILY_IGNORED, false);
    }
  }
  // Reads with no barrier after them, for an export to wait for.
  if (usage == Usage::kRead) {
    draw_exports_.NoteOtherReads();
  }
  last_written_range_ = written_range;
}

VulkanSharedMemory::DrawBarrier VulkanSharedMemory::UseForGuestDraw(
    const std::pair<uint32_t, uint32_t>* reads, size_t read_count,
    const std::pair<uint32_t, uint32_t>* exports, size_t export_count,
    uint64_t export_batch) {
  for (size_t i = 0; i < read_count; ++i) {
    NoteByteRangeUsedByGpu(reads[i].first, reads[i].second);
  }
  for (size_t i = 0; i < export_count; ++i) {
    NoteByteRangeUsedByGpu(exports[i].first, exports[i].second);
  }
  // The extent of the exports: what the next barrier has to make available.
  uint32_t export_start = UINT32_MAX, export_end = 0;
  for (size_t i = 0; i < export_count; ++i) {
    export_start = std::min(export_start, exports[i].first);
    export_end = std::max(export_end, exports[i].first + exports[i].second);
  }
  const bool exporting = export_start < export_end;
  const auto add_draw = [&]() {
    draw_exports_.AddDraw(reads, read_count, exports, export_count,
                          export_batch);
    if (exporting) {
      const bool written = last_written_range_.second != 0;
      const uint32_t start =
          written ? std::min(last_written_range_.first, export_start)
                  : export_start;
      const uint32_t end =
          written ? std::max(last_written_range_.first +
                                 last_written_range_.second,
                             export_end)
                  : export_end;
      last_written_range_ = std::make_pair(start, end - start);
    }
  };
  // As ever from other uses of the buffer, and the first export after them,
  // which aren't tracked, has to wait for them.
  if ((last_usage_ != Usage::kRead &&
       last_usage_ != Usage::kGuestDrawReadWrite) ||
      (last_usage_ == Usage::kRead && exporting)) {
    const Usage usage = exporting ? Usage::kGuestDrawReadWrite : Usage::kRead;
    Use(usage);
    add_draw();
    return exporting ? DrawBarrier::kFirstExport : DrawBarrier::kNone;
  }
  DrawBarrier result =
      last_written_range_.second ? DrawBarrier::kElided : DrawBarrier::kNone;
  VkPipelineStageFlags stage_mask;
  VkAccessFlags access_mask;
  GetUsageMasks(Usage::kGuestDrawReadWrite, stage_mask, access_mask);
  switch (draw_exports_.Check(reads, read_count, exports, export_count,
                              export_batch)) {
    case DrawExportTracker::Need::kCommit:
      if (!exporting) {
        Use(Usage::kRead);
        draw_exports_.AddDraw(reads, read_count, exports, export_count,
                              export_batch);
        return DrawBarrier::kReadAfterExport;
      }
      // Between two draws of the same usage, the masks on both sides are that
      // usage's. Only the ranges written have to be made visible.
      command_processor_.PushBufferMemoryBarrier(
          buffer_, VkDeviceSize(last_written_range_.first),
          VkDeviceSize(last_written_range_.second), stage_mask, stage_mask,
          access_mask, access_mask, VK_QUEUE_FAMILY_IGNORED,
          VK_QUEUE_FAMILY_IGNORED, false);
      draw_exports_.Clear();
      last_written_range_ = std::make_pair(uint32_t(0), uint32_t(0));
      result = DrawBarrier::kExportAfterExport;
      break;
    case DrawExportTracker::Need::kWaitForReads:
      command_processor_.PushBufferMemoryBarrier(
          buffer_, 0, VK_WHOLE_SIZE, stage_mask, stage_mask,
          VK_ACCESS_INDEX_READ_BIT | VK_ACCESS_SHADER_READ_BIT, access_mask,
          VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, false);
      draw_exports_.ClearReads();
      result = DrawBarrier::kExportAfterRead;
      break;
    default:
      break;
  }
  add_draw();
  return result;
}

bool VulkanSharedMemory::CopyToGuestMemory(uint32_t start, uint32_t length) {
  // Zero-copy: the GPU wrote guest memory itself.
  if (zero_copy_ || host_mapped_data_ == nullptr) {
    return false;
  }
  return ReadHostMapped(start, length, memory().TranslatePhysical(start));
}

bool VulkanSharedMemory::ReadHostMapped(uint32_t guest_address, uint32_t length,
                                        void* dest) const {
  if (host_mapped_data_ == nullptr || !length) {
    return false;
  }
  if (uint64_t(guest_address) + length > uint64_t(kBufferSize)) {
    return false;
  }
  if (!host_mapped_coherent_) {
    // Cached but not coherent: pull the GPU-written bytes out of DRAM into the
    // CPU cache before reading them.
    const ui::vulkan::VulkanDevice* const vulkan_device =
        command_processor_.GetVulkanDevice();
    const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
    const VkDeviceSize atom_size = vulkan_device->properties().nonCoherentAtomSize;
    VkMappedMemoryRange range;
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.pNext = nullptr;
    range.memory = buffer_memory_.front();
    range.offset = (VkDeviceSize(guest_address) / atom_size) * atom_size;
    range.size = std::min(xe::round_up(uint64_t(guest_address) + length,
                                       uint64_t(atom_size)),
                          uint64_t(kBufferSize)) -
                 range.offset;
    dfn.vkInvalidateMappedMemoryRanges(vulkan_device->device(), 1, &range);
  }
  memory::vastcpy(static_cast<uint8_t*>(dest), host_mapped_data_ + guest_address,
                  length);
  return true;
}

bool VulkanSharedMemory::InitializeTraceSubmitDownloads() {
  ResetTraceDownload();
  PrepareForTraceDownload();
  uint32_t download_page_count = trace_download_page_count();
  if (!download_page_count) {
    return false;
  }

  if (!ui::vulkan::util::CreateDedicatedAllocationBuffer(
          command_processor_.GetVulkanDevice(),
          download_page_count << page_size_log2(),
          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          ui::vulkan::util::MemoryPurpose::kReadback, trace_download_buffer_,
          trace_download_buffer_memory_,
          &trace_download_buffer_memory_type_)) {
    XELOGE(
        "Shared memory: Failed to create a {} KB GPU-written memory download "
        "buffer for frame tracing",
        download_page_count << page_size_log2() >> 10);
    ResetTraceDownload();
    return false;
  }

  Use(Usage::kRead);
  command_processor_.SubmitBarriers(true);
  DeferredCommandBuffer& command_buffer =
      command_processor_.deferred_command_buffer();

  command_processor_.InsertDebugMarker(
      "Trace Download: %u KB, %zu ranges",
      download_page_count << page_size_log2() >> 10,
      trace_download_ranges().size());

  size_t download_range_count = trace_download_ranges().size();
  VkBufferCopy* download_regions = command_buffer.CmdCopyBufferEmplace(
      buffer_, trace_download_buffer_, uint32_t(download_range_count));
  VkDeviceSize download_buffer_offset = 0;
  for (size_t i = 0; i < download_range_count; ++i) {
    VkBufferCopy& download_region = download_regions[i];
    const std::pair<uint32_t, uint32_t>& download_range =
        trace_download_ranges()[i];
    NoteByteRangeUsedByGpu(download_range.first, download_range.second);
    download_region.srcOffset = download_range.first;
    download_region.dstOffset = download_buffer_offset;
    download_region.size = download_range.second;
    download_buffer_offset += download_range.second;
  }

  command_processor_.PushBufferMemoryBarrier(
      trace_download_buffer_, 0, VK_WHOLE_SIZE, VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
      VK_ACCESS_HOST_READ_BIT);

  return true;
}

void VulkanSharedMemory::InitializeTraceCompleteDownloads() {
  if (!trace_download_buffer_memory_) {
    return;
  }
  const ui::vulkan::VulkanDevice* const vulkan_device =
      command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  void* download_mapping;
  if (dfn.vkMapMemory(device, trace_download_buffer_memory_, 0, VK_WHOLE_SIZE,
                      0, &download_mapping) == VK_SUCCESS) {
    // Readback memory may be cached without being coherent on mobile GPUs.
    ui::vulkan::util::InvalidateMappedMemoryRange(
        vulkan_device, trace_download_buffer_memory_,
        trace_download_buffer_memory_type_);
    uint32_t download_buffer_offset = 0;
    for (const auto& download_range : trace_download_ranges()) {
      trace_writer_.WriteMemoryRead(
          download_range.first, download_range.second,
          reinterpret_cast<const uint8_t*>(download_mapping) +
              download_buffer_offset);
      download_buffer_offset += download_range.second;
    }
    dfn.vkUnmapMemory(device, trace_download_buffer_memory_);
  } else {
    XELOGE(
        "Shared memory: Failed to map the GPU-written memory download buffer "
        "for frame tracing");
  }
  ResetTraceDownload();
}

bool VulkanSharedMemory::AllocateSparseHostGpuMemoryRange(
    uint32_t offset_allocations, uint32_t length_allocations) {
  if (!length_allocations) {
    return true;
  }

  const ui::vulkan::VulkanDevice* const vulkan_device =
      command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();

  VkMemoryAllocateFlagsInfo memory_allocate_flags_info = {};
  memory_allocate_flags_info.sType =
      VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
  memory_allocate_flags_info.pNext = nullptr;
  memory_allocate_flags_info.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
  memory_allocate_flags_info.deviceMask = 0;

  VkMemoryAllocateInfo memory_allocate_info;
  memory_allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  memory_allocate_info.pNext = vulkan_device->properties().bufferDeviceAddress
                                   ? &memory_allocate_flags_info
                                   : nullptr;
  memory_allocate_info.allocationSize =
      length_allocations << host_gpu_memory_sparse_granularity_log2();
  memory_allocate_info.memoryTypeIndex = buffer_memory_type_;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkResult alloc_result =
      dfn.vkAllocateMemory(device, &memory_allocate_info, nullptr, &memory);
  if (alloc_result != VK_SUCCESS && memory_allocate_info.pNext != nullptr) {
    memory_allocate_info.pNext = nullptr;
    alloc_result =
        dfn.vkAllocateMemory(device, &memory_allocate_info, nullptr, &memory);
  }
  if (alloc_result != VK_SUCCESS) {
    XELOGE("Shared memory: Failed to allocate sparse buffer memory");
    return false;
  }
  buffer_memory_.push_back(memory);

  VkSparseMemoryBind bind;
  bind.resourceOffset = offset_allocations
                        << host_gpu_memory_sparse_granularity_log2();
  bind.size = memory_allocate_info.allocationSize;
  bind.memory = memory;
  bind.memoryOffset = 0;
  bind.flags = 0;
  VkPipelineStageFlags bind_wait_stage_mask =
      VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
  if (vulkan_device->properties().tessellationShader) {
    bind_wait_stage_mask |=
        VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT;
  }
  command_processor_.SparseBindBuffer(buffer_, 1, &bind, bind_wait_stage_mask);

  return true;
}

bool VulkanSharedMemory::UploadRanges(
    const std::pair<uint32_t, uint32_t>* upload_page_ranges,
    uint32_t num_upload_ranges) {
  return UploadUnitRanges(upload_page_ranges, num_upload_ranges,
                          page_size_log2());
}

bool VulkanSharedMemory::UploadSubpageRanges(
    const std::pair<uint32_t, uint32_t>* upload_subpage_ranges,
    uint32_t num_upload_ranges) {
  return UploadUnitRanges(upload_subpage_ranges, num_upload_ranges,
                          SubpageTracker::kSubpageSizeLog2);
}

bool VulkanSharedMemory::UploadUnitRanges(
    const std::pair<uint32_t, uint32_t>* upload_ranges, uint32_t num_ranges,
    uint32_t unit_size_log2) {
  if (!num_ranges) {
    return true;
  }

  if (zero_copy_) {
    // The buffer aliases guest RAM - nothing to copy, just mark the pages valid
    // so RequestRange stops asking to upload them.
    for (uint32_t i = 0; i < num_ranges; ++i) {
      trace_writer_.WriteMemoryRead(upload_ranges[i].first << unit_size_log2,
                                    upload_ranges[i].second << unit_size_log2);
      MakeRangeValid(upload_ranges[i].first << unit_size_log2,
                     upload_ranges[i].second << unit_size_log2, false);
    }
    return true;
  }

  const uint64_t submission_current = command_processor_.GetCurrentSubmission();

  // Ranges holding memexport output live in host_buffer_ (guest RAM) and may
  // still be being written by the GPU, so reading them with the CPU below races
  // those writes. Refresh those on the GPU and upload only what is left.
  cpu_upload_ranges_.clear();
  for (uint32_t i = 0; i < num_ranges; ++i) {
    uint32_t range_base = upload_ranges[i].first << unit_size_log2;
    uint32_t range_size = upload_ranges[i].second << unit_size_log2;
    if (command_processor_.EnsureMemexportRangeInDeviceBuffer(range_base,
                                                              range_size)) {
      MakeRangeValid(range_base, range_size, false);
      NoteByteRangeUsedByGpu(range_base, range_size);
      continue;
    }
    cpu_upload_ranges_.push_back(upload_ranges[i]);
  }
  if (cpu_upload_ranges_.empty()) {
    return true;
  }
  upload_ranges = cpu_upload_ranges_.data();
  num_ranges = uint32_t(cpu_upload_ranges_.size());

  const ui::vulkan::VulkanDevice* const vulkan_device =
      command_processor_.GetVulkanDevice();
  const bool can_zero_copy = cvars::vulkan_shared_memory_zero_copy_upload &&
                             host_mapped_data_ != nullptr &&
                             !buffer_memory_.empty();
  const uint64_t completed_submission =
      command_processor_.GetCompletedSubmission();
  const uint64_t atom_size =
      host_mapped_coherent_
          ? uint64_t(1)
          : std::max<uint64_t>(
                vulkan_device->properties().nonCoherentAtomSize, uint64_t(1));

  zero_copy_spans_.clear();
  staging_spans_.clear();

  // Marks decommitted units valid without copying them: the buffer keeps its
  // last-uploaded contents, which is what the console's GPU would have read
  // out of physical RAM.
  auto skip_span = [&](uint32_t unit_first, uint32_t unit_end) {
    MakeRangeValid(unit_first << unit_size_log2,
                   (unit_end - unit_first) << unit_size_log2, false);
    XELOGW(
        "Vulkan shared memory: skipped upload of {} decommitted unit(s) at "
        "{:08X}, keeping last-uploaded contents",
        unit_end - unit_first, unit_first << unit_size_log2);
  };

  auto route_readable_span = [&](uint32_t span_start, uint32_t span_len) {
    if (!span_len) {
      return;
    }
    if (can_zero_copy) {
      PartitionZeroCopySpans(zero_copy_submission_tracker_, span_start,
                             span_len, unit_size_log2, completed_submission,
                             host_mapped_coherent_, atom_size, zero_copy_spans_,
                             staging_spans_);
    } else {
      staging_spans_.emplace_back(span_start, span_len);
    }
  };

  // Pages the guest decommitted are PROT_NONE host-side, so they can be
  // neither copied nor refused: a refusal is retried by every draw that still
  // references the memory, and games legally free memory that live textures
  // point at (the console GPU reads physical RAM, ignoring CPU page tables).
  for (uint32_t i = 0; i < num_ranges; ++i) {
    const uint32_t upload_range_start = upload_ranges[i].first;
    const uint32_t upload_range_length = upload_ranges[i].second;
    trace_writer_.WriteMemoryRead(upload_range_start << unit_size_log2,
                                  upload_range_length << unit_size_log2);
    if (!upload_range_length) {
      continue;
    }

    auto* heap = memory().GetPhysicalHeap();
    if (cvars::gpu_allow_invalid_upload_range ||
        heap->QueryRangeAccess(
            upload_range_start << unit_size_log2,
            (upload_range_start + upload_range_length - 1) << unit_size_log2) !=
            xe::memory::PageAccess::kNoAccess) {
      route_readable_span(upload_range_start, upload_range_length);
      continue;
    }

    uint32_t span_start = UINT32_MAX;
    uint32_t skipped_start = UINT32_MAX;
    for (uint32_t unit = upload_range_start;
         unit <= upload_range_start + upload_range_length; ++unit) {
      const uint32_t unit_addr = unit << unit_size_log2;
      const bool readable =
          unit < upload_range_start + upload_range_length &&
          heap->QueryRangeAccess(unit_addr, unit_addr) !=
              xe::memory::PageAccess::kNoAccess;
      if (readable) {
        if (skipped_start != UINT32_MAX) {
          skip_span(skipped_start, unit);
          skipped_start = UINT32_MAX;
        }
        if (span_start == UINT32_MAX) {
          span_start = unit;
        }
      } else {
        if (span_start != UINT32_MAX) {
          route_readable_span(span_start, unit - span_start);
          span_start = UINT32_MAX;
        }
        if (unit < upload_range_start + upload_range_length &&
            skipped_start == UINT32_MAX) {
          skipped_start = unit;
        }
      }
    }
    if (skipped_start != UINT32_MAX) {
      skip_span(skipped_start, upload_range_start + upload_range_length);
    }
  }

  // Zero-copy direct CPU write path into persistently mapped
  // DEVICE_LOCAL | HOST_VISIBLE | HOST_CACHED buffer for spans proven hazard-
  // free against all in-flight and current-submission GPU accesses.
  if (!zero_copy_spans_.empty()) {
    zero_copy_flush_ranges_.clear();
    for (const auto& span : zero_copy_spans_) {
      const uint32_t byte_offset = span.first << unit_size_log2;
      const uint32_t byte_size = span.second << unit_size_log2;
      MakeRangeValid(byte_offset, byte_size, false);
      command_processor_.NoteSharedMemoryUpload(byte_size);

      uint8_t* guest_src = memory().TranslatePhysical(byte_offset);
      if (unit_size_log2 == page_size_log2()) {
        SetPageUploadHashes(span.first, span.second, guest_src);
      }
      if (subpage_tracking_active()) {
        SetSubpageUploadHashes(byte_offset, byte_size, guest_src);
      }
      std::atomic_thread_fence(std::memory_order_acquire);

      if (byte_size > 8192) {
        memory::vastcpy(host_mapped_data_ + byte_offset, guest_src, byte_size);
        swcache::WriteFence();
      } else {
        std::memcpy(host_mapped_data_ + byte_offset, guest_src, byte_size);
      }
      NoteByteRangeUsedByGpu(byte_offset, byte_size);
      if (!host_mapped_coherent_) {
        zero_copy_flush_ranges_.push_back(
            {uint64_t(byte_offset), uint64_t(byte_size)});
      }
    }
    if (!host_mapped_coherent_ && !zero_copy_flush_ranges_.empty()) {
      const std::vector<ByteRange> coalesced = CoalesceAlignedFlushRanges(
          zero_copy_flush_ranges_, atom_size, kBufferSize);
      if (!coalesced.empty()) {
        vk_flush_ranges_.clear();
        vk_flush_ranges_.reserve(coalesced.size());
        const VkDeviceMemory mapped_memory = buffer_memory_.front();
        for (const ByteRange& r : coalesced) {
          VkMappedMemoryRange& range = vk_flush_ranges_.emplace_back();
          range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
          range.pNext = nullptr;
          range.memory = mapped_memory;
          range.offset = VkDeviceSize(r.offset);
          range.size = VkDeviceSize(r.size);
        }
        vulkan_device->functions().vkFlushMappedMemoryRanges(
            vulkan_device->device(), uint32_t(vk_flush_ranges_.size()),
            vk_flush_ranges_.data());
      }
    }
  }

  if (staging_spans_.empty()) {
    return true;
  }

  auto& range_front = staging_spans_.front();
  auto& range_back = staging_spans_.back();

  // Copies of pages never invalidated while this submission was recording
  // can't have been read by an already-recorded command, so they can execute
  // at the head of the submission's command buffer without breaking the
  // current render pass.
  bool hoist = cvars::vulkan_hoist_shmem_uploads &&
               !AnyRangeInvalidatedSinceSubmissionOpen(
                   staging_spans_.data(), uint32_t(staging_spans_.size()),
                   unit_size_log2);

  if (!hoist) {
    // staging_spans_ are sorted - the bounds give the barrier range.
    Use(Usage::kTransferDestination,
        std::make_pair(
            range_front.first << unit_size_log2,
            (range_back.first + range_back.second - range_front.first)
                << unit_size_log2));
    // Submit barriers (may end render pass) before pushing debug marker so
    // EndRenderPass is not inside the SharedMem Upload marker.
    command_processor_.SubmitBarriers(true);

    uint32_t total_upload_bytes =
        (range_back.first + range_back.second - range_front.first)
        << unit_size_log2;
    command_processor_.PushDebugMarker(
        "UploadRanges (SharedMem): 0x%08X-0x%08X (%u KB, %zu ranges)",
        range_front.first << unit_size_log2,
        (range_back.first + range_back.second) << unit_size_log2,
        total_upload_bytes / 1024, staging_spans_.size());
  }
  DeferredCommandBuffer& command_buffer =
      hoist ? command_processor_.deferred_setup_command_buffer()
            : command_processor_.deferred_command_buffer();
  if (hoist && command_buffer.empty()) {
    // Once per submission: order the head copies after shared-memory writes
    // from previous submissions.
    VkMemoryBarrier barrier;
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.pNext = nullptr;
    barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    command_buffer.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1,
                                        &barrier, 0, nullptr, 0, nullptr);
  }

  bool successful = true;
  upload_regions_.clear();
  VkBuffer upload_buffer_previous = VK_NULL_HANDLE;

  auto upload_span = [&](uint32_t upload_range_start,
                         uint32_t upload_range_length) -> bool {
    while (upload_range_length) {
      VkBuffer upload_buffer;
      VkDeviceSize upload_buffer_offset, upload_buffer_size;
      uint8_t* upload_buffer_mapping = upload_buffer_pool_->RequestPartial(
          submission_current, size_t(upload_range_length) << unit_size_log2,
          size_t(1) << unit_size_log2, upload_buffer, upload_buffer_offset,
          upload_buffer_size);
      if (upload_buffer_mapping == nullptr) {
        XELOGE("Shared memory: Failed to get a Vulkan upload buffer");
        return false;
      }
      const uint32_t byte_offset = upload_range_start << unit_size_log2;
      const uint32_t byte_size = uint32_t(upload_buffer_size);
      MakeRangeValid(byte_offset, byte_size, false);
      command_processor_.NoteSharedMemoryUpload(upload_buffer_size);

      uint32_t upload_buffer_units =
          uint32_t(upload_buffer_size >> unit_size_log2);
      uint8_t* guest_src = memory().TranslatePhysical(byte_offset);
      if (unit_size_log2 == page_size_log2()) {
        SetPageUploadHashes(upload_range_start, upload_buffer_units, guest_src);
      }
      if (subpage_tracking_active()) {
        SetSubpageUploadHashes(byte_offset, byte_size, guest_src);
      }
      std::atomic_thread_fence(std::memory_order_acquire);

      if (upload_buffer_size < (1ULL << 32) && upload_buffer_size > 8192) {
        memory::vastcpy(upload_buffer_mapping, guest_src,
                        static_cast<uint32_t>(upload_buffer_size));
        swcache::WriteFence();
      } else {
        std::memcpy(upload_buffer_mapping, guest_src, upload_buffer_size);
      }
      if (upload_buffer_previous != upload_buffer && !upload_regions_.empty()) {
        assert_true(upload_buffer_previous != VK_NULL_HANDLE);
        command_buffer.CmdVkCopyBuffer(upload_buffer_previous, buffer_,
                                       uint32_t(upload_regions_.size()),
                                       upload_regions_.data());
        upload_regions_.clear();
      }
      upload_buffer_previous = upload_buffer;
      VkBufferCopy& upload_region = upload_regions_.emplace_back();
      upload_region.srcOffset = upload_buffer_offset;
      upload_region.dstOffset = VkDeviceSize(byte_offset);
      upload_region.size = upload_buffer_size;
      NoteByteRangeUsedByGpu(byte_offset, byte_size);
      upload_range_start += upload_buffer_units;
      upload_range_length -= upload_buffer_units;
    }
    return true;
  };

  for (const auto& span : staging_spans_) {
    if (!upload_span(span.first, span.second)) {
      successful = false;
      break;
    }
  }
  if (!upload_regions_.empty()) {
    assert_true(upload_buffer_previous != VK_NULL_HANDLE);
    command_buffer.CmdVkCopyBuffer(upload_buffer_previous, buffer_,
                                   uint32_t(upload_regions_.size()),
                                   upload_regions_.data());
    upload_regions_.clear();
  }
  if (!hoist) {
    command_processor_.PopDebugMarker();
  }
  return successful;
}

void VulkanSharedMemory::GetUsageMasks(Usage usage,
                                       VkPipelineStageFlags& stage_mask,
                                       VkAccessFlags& access_mask) const {
  switch (usage) {
    case Usage::kComputeWrite:
      stage_mask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      access_mask = VK_ACCESS_SHADER_WRITE_BIT;
      return;
    case Usage::kTransferDestination:
      stage_mask = VK_PIPELINE_STAGE_TRANSFER_BIT;
      access_mask = VK_ACCESS_TRANSFER_WRITE_BIT;
      return;
    default:
      break;
  }
  stage_mask =
      VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | guest_shader_pipeline_stages_;
  access_mask = VK_ACCESS_INDEX_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
  switch (usage) {
    case Usage::kRead:
      stage_mask |=
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
      access_mask |= VK_ACCESS_TRANSFER_READ_BIT;
      break;
    case Usage::kGuestDrawReadWrite:
      access_mask |= VK_ACCESS_SHADER_WRITE_BIT;
      break;
    default:
      assert_unhandled_case(usage);
  }
}

void VulkanSharedMemory::ResetTraceDownload() {
  const ui::vulkan::VulkanDevice* const vulkan_device =
      command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyBuffer, device,
                                         trace_download_buffer_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device,
                                         trace_download_buffer_memory_);
  ReleaseTraceDownloadRanges();
}

}  // namespace vulkan
}  // namespace gpu
}  // namespace xe
