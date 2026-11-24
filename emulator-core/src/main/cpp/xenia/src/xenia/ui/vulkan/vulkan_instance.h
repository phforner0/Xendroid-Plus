/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_UI_VULKAN_VULKAN_INSTANCE_H_
#define XENIA_UI_VULKAN_VULKAN_INSTANCE_H_

#include <memory>
#include <string>
#include <vector>

#include "xenia/base/cvar.h"
#include "xenia/base/platform.h"
#include "xenia/ui/renderdoc_api.h"
#include "xenia/ui/vulkan/vulkan_api.h"

#if XE_PLATFORM_WIN32
#include "xenia/base/platform_win.h"
#endif

DECLARE_bool(vulkan_adreno_bcenabler);
DECLARE_bool(gpu_headless);

namespace xe {
namespace ui {
namespace vulkan {

// Requests the KGSL GPU power control state (Custom Driver Force Max Clocks)
// again, once the custom driver path made the first request at instance
// creation; logs with the reason if one is given, failures always. False
// where it is not in use or the kernel refuses.
bool RequestGpuPowerControl(const char* reason);

// Releases forced KGSL GPU power control back to the kernel governor (e.g. on
// surface disconnect / app pause, device teardown, or stock-driver startup).
bool ReleaseGpuPowerControl(const char* reason);

class VulkanInstance {
 public:
  // validation_level: 0=off, 1=standard layer, 2=+synchronization validation.
  static std::unique_ptr<VulkanInstance> Create(bool with_surface,
                                                int validation_level);

  VulkanInstance(const VulkanInstance&) = delete;
  VulkanInstance& operator=(const VulkanInstance&) = delete;
  VulkanInstance(VulkanInstance&&) = delete;
  VulkanInstance& operator=(VulkanInstance&&) = delete;

  ~VulkanInstance();

  // nullptr if RenderDoc is not connected.
  RenderDocAPI* renderdoc_api() const { return renderdoc_api_.get(); }

  struct Functions {
    // From the loader module.
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;
    PFN_vkDestroyInstance vkDestroyInstance = nullptr;

    // From vkGetInstanceProcAddr for nullptr.
    PFN_vkCreateInstance vkCreateInstance = nullptr;
    PFN_vkEnumerateInstanceExtensionProperties
        vkEnumerateInstanceExtensionProperties = nullptr;
    PFN_vkEnumerateInstanceLayerProperties vkEnumerateInstanceLayerProperties =
        nullptr;
    // Vulkan 1.1.
    PFN_vkEnumerateInstanceVersion vkEnumerateInstanceVersion = nullptr;

    // From vkGetInstanceProcAddr for the instance.
#define XE_UI_VULKAN_FUNCTION(name) PFN_##name name = nullptr;
#define XE_UI_VULKAN_FUNCTION_PROMOTED(extension_name, core_name) \
  PFN_##core_name core_name = nullptr;
#include "xenia/ui/vulkan/functions/instance_1_0.inc"
    // VK_KHR_surface (#1)
#include "xenia/ui/vulkan/functions/instance_khr_surface.inc"
    // VK_KHR_xcb_surface (#6)
#ifdef VK_USE_PLATFORM_XCB_KHR
#include "xenia/ui/vulkan/functions/instance_khr_xcb_surface.inc"
#endif
    // VK_KHR_wayland_surface (#7)
#ifdef VK_USE_PLATFORM_WAYLAND_KHR
#include "xenia/ui/vulkan/functions/instance_khr_wayland_surface.inc"
#endif
    // VK_KHR_android_surface (#9)
#ifdef VK_USE_PLATFORM_ANDROID_KHR
#include "xenia/ui/vulkan/functions/instance_khr_android_surface.inc"
#endif
    // VK_EXT_metal_surface (#218)
#ifdef VK_USE_PLATFORM_METAL_EXT
#include "xenia/ui/vulkan/functions/instance_ext_metal_surface.inc"
#endif
    // VK_KHR_win32_surface (#10)
#ifdef VK_USE_PLATFORM_WIN32_KHR
#include "xenia/ui/vulkan/functions/instance_khr_win32_surface.inc"
#endif
    // VK_KHR_get_physical_device_properties2 (#60, promoted to 1.1)
#include "xenia/ui/vulkan/functions/instance_1_1_khr_get_physical_device_properties2.inc"
    // VK_EXT_debug_utils (#129)
#include "xenia/ui/vulkan/functions/instance_ext_debug_utils.inc"
    // VK_EXT_headless_surface (#293)
#ifdef VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME
    PFN_vkCreateHeadlessSurfaceEXT vkCreateHeadlessSurfaceEXT = nullptr;
#endif
#undef XE_UI_VULKAN_FUNCTION_PROMOTED
#undef XE_UI_VULKAN_FUNCTION
  };

  const Functions& functions() const { return functions_; }

  uint32_t api_version() const { return api_version_; }

  // Also set to true if the version of the Vulkan API they were promoted to it
  // supported (with the `ext_major_minor_` prefix rather than `ext_`).
  struct Extensions {
    bool ext_KHR_surface = false;  // #1
#ifdef VK_USE_PLATFORM_XCB_KHR
    bool ext_KHR_xcb_surface = false;  // #6
#endif
#ifdef VK_USE_PLATFORM_WAYLAND_KHR
    bool ext_KHR_wayland_surface = false;  // #7
#endif
#ifdef VK_USE_PLATFORM_ANDROID_KHR
    bool ext_KHR_android_surface = false;  // #9
#endif
#ifdef VK_USE_PLATFORM_METAL_EXT
    bool ext_EXT_metal_surface = false;  // #218
#endif
#ifdef VK_USE_PLATFORM_WIN32_KHR
    bool ext_KHR_win32_surface = false;  // #10
#endif
    bool ext_1_1_KHR_get_physical_device_properties2 = false;  // #60
    bool ext_KHR_get_surface_capabilities2 = false;            // #120
    bool ext_EXT_debug_utils = false;                          // #129
    bool ext_KHR_portability_enumeration = false;              // #395
    bool ext_EXT_headless_surface = false;                     // #293
  };

  const Extensions& extensions() const { return extensions_; }

  VkInstance instance() const { return instance_; }

  // The custom driver library actually loaded through adrenotools
  // (vulkan_lib_path), or empty when the system loader is in use.
  const std::string& custom_driver_path() const { return custom_driver_path_; }

  void EnumeratePhysicalDevices(
      std::vector<VkPhysicalDevice>& physical_devices_out) const;

  // Creates a headless VkSurfaceKHR when VK_EXT_headless_surface is active.
  bool CreateHeadlessSurface(VkSurfaceKHR* out_surface) const;

 private:
  explicit VulkanInstance() = default;

  std::unique_ptr<RenderDocAPI> renderdoc_api_;

#if XE_PLATFORM_LINUX
  void* loader_ = nullptr;
#elif XE_PLATFORM_WIN32
  HMODULE loader_ = nullptr;
#endif
  std::string custom_driver_path_;

  Functions functions_;

  uint32_t api_version_ = VK_MAKE_API_VERSION(0, 1, 0, 0);

  Extensions extensions_;

  VkInstance instance_ = nullptr;

  static VkBool32 DebugUtilsMessengerCallback(
      VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
      VkDebugUtilsMessageTypeFlagsEXT message_types,
      const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
      void* user_data);

  VkDebugUtilsMessengerEXT debug_utils_messenger_ = VK_NULL_HANDLE;
};

}  // namespace vulkan
}  // namespace ui
}  // namespace xe

#endif  // XENIA_UI_VULKAN_VULKAN_INSTANCE_H_
