// Pure-logic and source-contract tests for Onda 4A:
// 1. EDRAM row-on-demand height policy (rt_height_policy.h)
// 2. Mali proprietary quirks matrix (r34p0 / r40p0 / r47p0 / r54p1) vs
//    malisx2 (PanVK-on-kbase) vs armsx2-turnip vs Qualcomm proprietary
//    (vulkan_driver_quirks.h)
// 3. armsx2-turnip opt-in env-var helpers and strict '0'|'1' prop parsing
// 4. Source-contract wiring checks across RT cache, VulkanDevice,
//    VulkanCommandProcessor, and VulkanInstance.

#include "xenia/gpu/vulkan/rt_height_policy.h"
#include "xenia/gpu/vulkan/vulkan_driver_quirks.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

namespace {

using namespace xe::gpu::vulkan;

std::string ReadFileText(const std::string& path) {
  std::ifstream ifs(path);
  assert(ifs.is_open() && "Failed to open source file for contract test");
  std::ostringstream oss;
  oss << ifs.rdbuf();
  return oss.str();
}

void TestRtHeightPolicyInitialAndFloor() {
  // Tile pixel heights match Xenos EDRAM geometry: 16px for 1x, 8px for 2x/4x.
  assert(ComputeRtTileHeightPixels(/*1x=*/0) == 16u);
  assert(ComputeRtTileHeightPixels(/*2x=*/1) == 8u);
  assert(ComputeRtTileHeightPixels(/*4x=*/2) == 8u);

  // Disabled cvar always returns max_guest_height untouched.
  assert(ComputeInitialRenderTargetHeight(8192u, 0u, 0u, false, false) == 8192u);
  assert(ComputeInitialRenderTargetHeight(2048u, 128u, 1u, true, false) == 2048u);
  assert(ComputeInitialRenderTargetHeight(0u, 128u, 0u, false, true) == 0u);

  // Enabled cvar: floor 128 px when requested <= 128 and no merged tiling bands.
  assert(ComputeInitialRenderTargetHeight(8192u, 0u, 0u, false, true) == 128u);
  assert(ComputeInitialRenderTargetHeight(8192u, 64u, 0u, false, true) == 128u);
  assert(ComputeInitialRenderTargetHeight(8192u, 128u, 0u, false, true) == 128u);
  assert(ComputeInitialRenderTargetHeight(8192u, 0u, 1u, false, true) == 128u);
  assert(ComputeInitialRenderTargetHeight(8192u, 0u, 2u, false, true) == 128u);

  // Alignment to tile_h (16 for 1x, 8 for 2x/4x).
  assert(ComputeInitialRenderTargetHeight(8192u, 129u, 0u, false, true) == 144u);
  assert(ComputeInitialRenderTargetHeight(8192u, 130u, 1u, false, true) == 136u);
  assert(ComputeInitialRenderTargetHeight(8192u, 130u, 2u, false, true) == 136u);
  assert(ComputeInitialRenderTargetHeight(2048u, 720u, 0u, false, true) == 720u);

  // merge_tiling_bands && msaa >= 2x enforces >= 720 px floor.
  assert(ComputeInitialRenderTargetHeight(2048u, 0u, 1u, true, true) == 720u);
  assert(ComputeInitialRenderTargetHeight(2048u, 128u, 2u, true, true) == 720u);
  assert(ComputeInitialRenderTargetHeight(2048u, 721u, 1u, true, true) == 728u);
  // 1x MSAA is NOT affected by merge_tiling_bands floor.
  assert(ComputeInitialRenderTargetHeight(2048u, 0u, 0u, true, true) == 128u);

  // Clamped to max_guest_height when max < floor or aligned target.
  assert(ComputeInitialRenderTargetHeight(64u, 0u, 0u, false, true) == 64u);
  assert(ComputeInitialRenderTargetHeight(512u, 0u, 1u, true, true) == 512u);
  assert(ComputeInitialRenderTargetHeight(5464u, 5460u, 1u, false, true) ==
         5464u);
}

void TestRtHeightPolicyGeometricGrowth() {
  // No-op when needed <= current.
  assert(ComputeGrownRenderTargetHeight(128u, 128u, 8192u, 0u) == 128u);
  assert(ComputeGrownRenderTargetHeight(512u, 256u, 8192u, 0u) == 512u);
  assert(ComputeGrownRenderTargetHeight(128u, 128u, 0u, 0u) == 0u);

  // Geometric growth >= 50%, aligned to tile_h, clamped to max_guest_height.
  uint32_t h = ComputeInitialRenderTargetHeight(8192u, 0u, 0u, false, true);
  assert(h == 128u);
  h = ComputeGrownRenderTargetHeight(h, 140u, 8192u, 0u);
  assert(h == 192u);  // 128 + 64 = 192 (aligned to 16)
  h = ComputeGrownRenderTargetHeight(h, 256u, 8192u, 0u);
  assert(h == 288u);  // 192 + 96 = 288 >= 256
  h = ComputeGrownRenderTargetHeight(h, 512u, 8192u, 0u);
  assert(h == 512u);  // needed 512 > 432
  h = ComputeGrownRenderTargetHeight(h, 600u, 8192u, 0u);
  assert(h == 768u);  // 512 + 256 = 768
  h = ComputeGrownRenderTargetHeight(h, 800u, 8192u, 0u);
  assert(h == 1152u);  // 768 + 384 = 1152
  h = ComputeGrownRenderTargetHeight(h, 9000u, 8192u, 0u);
  assert(h == 8192u);  // clamped to max_guest_height

  // Verify bounded reallocation count from 128 -> 8192 even with +1px increments.
  uint32_t cur = 128u;
  uint32_t growths = 0u;
  while (cur < 8192u) {
    const uint32_t next = ComputeGrownRenderTargetHeight(cur, cur + 1u, 8192u, 0u);
    assert(next > cur);
    assert(next == 8192u || next >= cur + cur / 2u);
    assert((next % 16u) == 0u);
    cur = next;
    ++growths;
  }
  assert(cur == 8192u);
  assert(growths <= 12u);

  // 4x MSAA tile alignment (8 px).
  const uint32_t grown_4x = ComputeGrownRenderTargetHeight(136u, 140u, 4096u, 2u);
  assert(grown_4x == 208u);  // 136 + 68 = 204 -> align8(204) = 208
  assert((grown_4x % 8u) == 0u);
}

void TestFramebufferHeightAndInvalidation() {
  // No attachments -> fallback height.
  assert(ComputeFramebufferGuestHeight(2048u, nullptr, 0u) == 2048u);

  // Single attachment.
  const uint32_t single[1] = {720u};
  assert(ComputeFramebufferGuestHeight(2048u, single, 1u) == 720u);

  // Multiple bound RTs -> minimum allocated height (VUID attachment >= fb).
  const uint32_t multi[3] = {1024u, 720u, 1152u};
  assert(ComputeFramebufferGuestHeight(2048u, multi, 3u) == 720u);

  // Never exceeds fallback max height.
  const uint32_t oversized[1] = {4096u};
  assert(ComputeFramebufferGuestHeight(2048u, oversized, 1u) == 2048u);

  // Framebuffer invalidation matcher checks matching kind & used mask only.
  const uint32_t color_bases[4] = {0u, 512u, 1024u, 0u};
  // Depth used (bit 0) at base 100, color0 used (bit 1) at base 0, color1 used (bit 2) at base 512.
  const uint32_t used_mask = (1u << 0) | (1u << 1) | (1u << 2);
  assert(FramebufferReferencesRenderTargetBase(used_mask, 100u, color_bases, true, 100u));
  assert(!FramebufferReferencesRenderTargetBase(used_mask, 100u, color_bases, true, 0u));
  assert(!FramebufferReferencesRenderTargetBase(used_mask, 100u, color_bases, false, 100u));
  assert(FramebufferReferencesRenderTargetBase(used_mask, 100u, color_bases, false, 0u));
  assert(FramebufferReferencesRenderTargetBase(used_mask, 100u, color_bases, false, 512u));
  // color2 is NOT in used_mask -> base 1024 must NOT trigger match.
  assert(!FramebufferReferencesRenderTargetBase(used_mask, 100u, color_bases, false, 1024u));
}

void TestDriverClassificationAndMaliQuirksMatrix() {
  // Preserve existing Xclipse & non-Turnip mobile tiler helpers.
  assert(IsSamsungXclipseProprietary(kVkDriverIdSamsungProprietary, kVendorIdSamsung, "Xclipse 940"));
  assert(!IsSamsungXclipseProprietary(kVkDriverIdMesaRadv, kVendorIdSamsung, "Xclipse 940"));
  assert(IsNonTurnipMobileTiler(kVkDriverIdArmProprietary, kVendorIdArm));
  assert(IsNonTurnipMobileTiler(kVkDriverIdMesaPanvk, kVendorIdArm));
  assert(!IsNonTurnipMobileTiler(kVkDriverIdMesaTurnip, kVendorIdQualcomm));

  // Version parser.
  uint32_t maj = 99u, min = 99u;
  assert(ParseArmMaliDriverVersion("r34p0-01eac0", maj, min));
  assert(maj == 34u && min == 0u);
  assert(ParseArmMaliDriverVersion("Arm Mali r40p0", maj, min));
  assert(maj == 40u && min == 0u);
  assert(ParseArmMaliDriverVersion("r54p1-00rel0", maj, min));
  assert(maj == 54u && min == 1u);
  assert(!ParseArmMaliDriverVersion("Mesa 25.3.0 (malisx2)", maj, min));
  assert(maj == 0u && min == 0u);
  assert(!ParseArmMaliDriverVersion(nullptr, maj, min));

  // Case 1: Arm Mali proprietary r34p0 (< r38p1).
  {
    const DriverClassification dc = ClassifyVulkanDriver(
        kVendorIdArm, kVkDriverIdArmProprietary, "ARM", "r34p0-01eac0");
    assert(dc.is_arm_proprietary);
    assert(!dc.is_panvk);
    assert(!dc.is_malisx2);
    assert(!dc.is_armsx2_turnip);
    assert(dc.mali_driver_major == 34u && dc.mali_driver_minor == 0u);

    const MaliDriverQuirks q = ComputeMaliQuirks(
        dc.is_arm_proprietary, dc.is_panvk, dc.mali_driver_major,
        dc.mali_driver_minor, /*is_job_manager=*/false);
    assert(q.cap_spirv_13);
    assert(q.rewrite_scalar_select);
    assert(!q.allow_dynamic_rendering);
    assert(!q.allow_extended_dynamic_state);
    assert(q.normalize_mixed_load_ops);
    assert(!q.allow_push_descriptor);
    assert(q.enforce_host_coherent_readback);
  }

  // Case 2: Arm Mali proprietary r40p0 (>= r38p1, < r44p1).
  {
    const DriverClassification dc = ClassifyVulkanDriver(
        kVendorIdArm, /*pre-1.2 driverID=*/0u, "Mali", "r40p0-00rel0");
    assert(dc.is_arm_proprietary);
    assert(!dc.is_panvk);
    assert(dc.mali_driver_major == 40u && dc.mali_driver_minor == 0u);

    const MaliDriverQuirks q = ComputeMaliQuirks(
        dc.is_arm_proprietary, dc.is_panvk, dc.mali_driver_major,
        dc.mali_driver_minor, /*is_job_manager=*/false);
    assert(q.cap_spirv_13);
    assert(q.rewrite_scalar_select);
    assert(!q.allow_dynamic_rendering);
    assert(!q.allow_extended_dynamic_state);
    assert(!q.normalize_mixed_load_ops);
    assert(!q.allow_push_descriptor);
    assert(q.enforce_host_coherent_readback);
  }

  // Case 3: Arm Mali proprietary r47p0 CSF vs Job Manager (JM blocks EDS on r46..r50).
  {
    const MaliDriverQuirks csf = ComputeMaliQuirks(true, false, 47u, 0u, false);
    assert(!csf.cap_spirv_13);
    assert(!csf.rewrite_scalar_select);
    assert(!csf.allow_dynamic_rendering);
    assert(csf.allow_extended_dynamic_state);
    assert(!csf.normalize_mixed_load_ops);
    assert(!csf.allow_push_descriptor);

    const MaliDriverQuirks jm = ComputeMaliQuirks(true, false, 47u, 0u, true);
    assert(!jm.allow_extended_dynamic_state);
  }

  // Case 4: Arm Mali proprietary r54p1 (>= r52p0).
  {
    const DriverClassification dc = ClassifyVulkanDriver(
        kVendorIdArm, kVkDriverIdArmProprietary, "ARM", "r54p1-00rel0");
    assert(dc.is_arm_proprietary);
    assert(!dc.is_panvk);
    assert(dc.mali_driver_major == 54u && dc.mali_driver_minor == 1u);

    const MaliDriverQuirks q = ComputeMaliQuirks(
        dc.is_arm_proprietary, dc.is_panvk, dc.mali_driver_major,
        dc.mali_driver_minor, /*is_job_manager=*/true);
    assert(!q.cap_spirv_13);
    assert(!q.rewrite_scalar_select);
    assert(q.allow_dynamic_rendering);
    assert(q.allow_extended_dynamic_state);
    assert(!q.normalize_mixed_load_ops);
    assert(!q.allow_push_descriptor);
    assert(q.enforce_host_coherent_readback);
  }

  // Case 5: malisx2 with VK_DRIVER_ID_MESA_PANVK (20).
  {
    const DriverClassification dc = ClassifyVulkanDriver(
        kVendorIdArm, kVkDriverIdMesaPanvk, "panvk", "Mesa 25.3 (malisx2)");
    assert(dc.is_panvk);
    assert(dc.is_malisx2);
    assert(!dc.is_arm_proprietary);
    assert(!dc.is_armsx2_turnip);

    const MaliDriverQuirks q = ComputeMaliQuirks(
        dc.is_arm_proprietary, dc.is_panvk, dc.mali_driver_major,
        dc.mali_driver_minor, /*is_job_manager=*/true);
    assert(!q.cap_spirv_13);
    assert(!q.rewrite_scalar_select);
    assert(q.allow_dynamic_rendering);
    assert(q.allow_extended_dynamic_state);
    assert(!q.normalize_mixed_load_ops);
    assert(q.allow_push_descriptor);
    assert(!q.enforce_host_coherent_readback);
  }

  // Case 6: malisx2 wrapper reporting driverID == 0 + driverName="PanVK" (must NOT be Arm proprietary!).
  {
    const DriverClassification dc = ClassifyVulkanDriver(
        kVendorIdArm, 0u, "PanVK", "Mesa 25.3-devel kbase");
    assert(dc.is_panvk);
    assert(dc.is_malisx2);
    assert(!dc.is_arm_proprietary);

    const MaliDriverQuirks q = ComputeMaliQuirks(
        dc.is_arm_proprietary, dc.is_panvk, dc.mali_driver_major,
        dc.mali_driver_minor, /*is_job_manager=*/false);
    assert(!q.cap_spirv_13);
    assert(!q.rewrite_scalar_select);
    assert(q.allow_dynamic_rendering);
    assert(q.allow_extended_dynamic_state);
    assert(!q.normalize_mixed_load_ops);
    assert(q.allow_push_descriptor);
  }

  // Case 7: armsx2-turnip vs generic Turnip vs Qualcomm proprietary.
  {
    const DriverClassification armsx2 = ClassifyVulkanDriver(
        kVendorIdQualcomm, kVkDriverIdMesaTurnip, "turnip Mesa driver",
        "Mesa 25.3.0-armsx2 (git-deadbeef)");
    assert(armsx2.is_armsx2_turnip);
    assert(!armsx2.is_panvk);
    assert(!armsx2.is_malisx2);
    assert(!armsx2.is_arm_proprietary);
    assert(!armsx2.is_qualcomm_proprietary);

    const DriverClassification turnip = ClassifyVulkanDriver(
        kVendorIdQualcomm, kVkDriverIdMesaTurnip, "turnip Mesa driver",
        "Mesa 25.3.0-devel");
    assert(!turnip.is_armsx2_turnip);
    assert(!turnip.is_qualcomm_proprietary);

    const DriverClassification qcom = ClassifyVulkanDriver(
        kVendorIdQualcomm, kVkDriverIdQualcommProprietary, "Qualcomm",
        "512.762.0");
    assert(qcom.is_qualcomm_proprietary);
    assert(!qcom.is_armsx2_turnip);
    assert(!qcom.is_arm_proprietary);
    assert(!qcom.is_panvk);

    const DriverClassification qcom0 = ClassifyVulkanDriver(
        kVendorIdQualcomm, 0u, "Qualcomm", "512.502.0");
    assert(qcom0.is_qualcomm_proprietary);
    assert(!qcom0.is_armsx2_turnip);
  }

  // Case 8: Diagnostic mali_driver_override forces proprietary parsing.
  {
    const DriverClassification ov = ClassifyVulkanDriver(
        kVendorIdQualcomm, kVkDriverIdMesaTurnip, "turnip", "Mesa", "r43p0");
    assert(ov.is_arm_proprietary);
    assert(!ov.is_panvk);
    assert(ov.mali_driver_major == 43u && ov.mali_driver_minor == 0u);
  }
}

void TestTurnipOptInEnvHelpers() {
  // Whitelist for TU_AUTOTUNE_ALGO.
  assert(IsValidTurnipAutotuneAlgo("gmem"));
  assert(IsValidTurnipAutotuneAlgo("sysmem"));
  assert(IsValidTurnipAutotuneAlgo("auto"));
  assert(!IsValidTurnipAutotuneAlgo(""));
  assert(!IsValidTurnipAutotuneAlgo("GMEM"));
  assert(!IsValidTurnipAutotuneAlgo("invalid"));
  assert(!IsValidTurnipAutotuneAlgo(nullptr));

  // FD_DEV_FEATURES append preserves existing flags and deduplicates.
  assert(AppendFdDevFeature("", "tu_override_uncached_as_cache_coherent=1") ==
         "tu_override_uncached_as_cache_coherent=1");
  assert(AppendFdDevFeature("enable_tp_ubwc_flag_hint=1",
                            "tu_override_uncached_as_cache_coherent=1") ==
         "enable_tp_ubwc_flag_hint=1,tu_override_uncached_as_cache_coherent=1");
  assert(AppendFdDevFeature(
             "enable_tp_ubwc_flag_hint=1,tu_override_uncached_as_cache_coherent=1",
             "tu_override_uncached_as_cache_coherent=1") ==
         "enable_tp_ubwc_flag_hint=1,tu_override_uncached_as_cache_coherent=1");
  assert(AppendFdDevFeature("none", "tu_override_uncached_as_cache_coherent=1") ==
         "tu_override_uncached_as_cache_coherent=1");

  // Strict '0'|'1' Android property parser.
  bool out = true;
  assert(ParseStrictBinaryBoolProp("0", out) && !out);
  assert(ParseStrictBinaryBoolProp("1", out) && out);
  out = false;
  assert(!ParseStrictBinaryBoolProp("", out) && !out);
  assert(!ParseStrictBinaryBoolProp("true", out) && !out);
  assert(!ParseStrictBinaryBoolProp("false", out) && !out);
  assert(!ParseStrictBinaryBoolProp("01", out) && !out);
  assert(!ParseStrictBinaryBoolProp(nullptr, out) && !out);
}

void TestSourceContracts(const std::string& cpp_root) {
  const std::string xenia_src = cpp_root + "/xenia/src/xenia";

  const std::string rt_h =
      ReadFileText(xenia_src + "/gpu/render_target_cache.h");
  assert(rt_h.find("EnsureRenderTargetHeight") != std::string::npos);

  const std::string rt_cc =
      ReadFileText(xenia_src + "/gpu/render_target_cache.cc");
  assert(rt_cc.find("EnsureRenderTargetHeight") != std::string::npos);

  const std::string vk_rt_h =
      ReadFileText(xenia_src + "/gpu/vulkan/vulkan_render_target_cache.h");
  assert(vk_rt_h.find("allocated_guest_height_") != std::string::npos);
  assert(vk_rt_h.find("max_guest_height_") != std::string::npos);
  assert(vk_rt_h.find("deferred_rt_destructions_") != std::string::npos);
  assert(vk_rt_h.find("EnsureRenderTargetHeight") != std::string::npos);

  const std::string vk_rt_cc =
      ReadFileText(xenia_src + "/gpu/vulkan/vulkan_render_target_cache.cc");
  assert(vk_rt_cc.find("DEFINE_bool(\n    vulkan_rt_rows_on_demand, true") !=
             std::string::npos ||
         vk_rt_cc.find("DEFINE_bool(vulkan_rt_rows_on_demand, true") !=
             std::string::npos);
  assert(vk_rt_cc.find("debug.xendroid.rt_rows_on_demand") != std::string::npos);
  assert(vk_rt_cc.find("GrowRenderTarget") != std::string::npos);
  assert(vk_rt_cc.find("ComputeFramebufferGuestHeight") != std::string::npos);
  assert(vk_rt_cc.find("DestroyDeferredRtDestructions") != std::string::npos);

  // P0 regression (#121): growing a tiling-band target inside
  // NoteTilingBandDraw or ReplicateTilingBandRenderTargetRows invalidates
  // last_update_framebuffer_, so both paths MUST rebuild it before IssueDraw
  // enters the pass (and NoteTilingBandDraw must re-stage deferred clears).
  {
    const size_t note_pos =
        vk_rt_cc.find("uint32_t VulkanRenderTargetCache::NoteTilingBandDraw(");
    const size_t rep_pos = vk_rt_cc.find(
        "void VulkanRenderTargetCache::ReplicateTilingBandRenderTargetRows(");
    assert(note_pos != std::string::npos);
    assert(rep_pos != std::string::npos && rep_pos > note_pos);
    const std::string note_body = vk_rt_cc.substr(note_pos, rep_pos - note_pos);
    assert(note_body.find("!last_update_framebuffer_") != std::string::npos);
    assert(note_body.find("GetHostRenderTargetsFramebuffer(") !=
           std::string::npos);
    assert(note_body.find("StageDeferredResolveClearsForLastUpdate()") !=
           std::string::npos);

    const size_t rep_end = vk_rt_cc.find("\n}\n", rep_pos);
    assert(rep_end != std::string::npos);
    const std::string rep_body = vk_rt_cc.substr(rep_pos, rep_end - rep_pos);
    assert(rep_body.find("!last_update_framebuffer_") != std::string::npos);
    assert(rep_body.find("GetHostRenderTargetsFramebuffer(") !=
           std::string::npos);
  }

  const std::string vk_dev_h =
      ReadFileText(xenia_src + "/ui/vulkan/vulkan_device.h");
  assert(vk_dev_h.find("driverName[VK_MAX_DRIVER_NAME_SIZE]") != std::string::npos);
  assert(vk_dev_h.find("driverInfo[VK_MAX_DRIVER_INFO_SIZE]") != std::string::npos);
  assert(vk_dev_h.find("maliDriverMajor") != std::string::npos);
  assert(vk_dev_h.find("maliDriverMinor") != std::string::npos);
  assert(vk_dev_h.find("isPanvk") != std::string::npos);
  assert(vk_dev_h.find("isMalisx2") != std::string::npos);
  assert(vk_dev_h.find("isArmsx2Turnip") != std::string::npos);

  const std::string vk_dev_cc =
      ReadFileText(xenia_src + "/ui/vulkan/vulkan_device.cc");
  assert(vk_dev_cc.find("ClassifyVulkanDriver") != std::string::npos);
  assert(vk_dev_cc.find("ComputeMaliQuirks") != std::string::npos);

  const std::string vk_cp_cc =
      ReadFileText(xenia_src + "/gpu/vulkan/vulkan_command_processor.cc");
  assert(vk_cp_cc.find("device_properties.isPanvk") != std::string::npos);

  const std::string vk_inst_cc =
      ReadFileText(xenia_src + "/ui/vulkan/vulkan_instance.cc");
  // turnip_debug MUST remain "sysmem" by default!
  assert(vk_inst_cc.find("turnip_debug, \"sysmem\"") != std::string::npos);
  assert(vk_inst_cc.find("vulkan_turnip_swapchain_ubwc") != std::string::npos);
  assert(vk_inst_cc.find("vulkan_turnip_cached_coherent") != std::string::npos);
  assert(vk_inst_cc.find("vulkan_turnip_autotune_algo") != std::string::npos);
  assert(vk_inst_cc.find("TU_SWAPCHAIN_UBWC") != std::string::npos);
  assert(vk_inst_cc.find("tu_override_uncached_as_cache_coherent=1") !=
         std::string::npos);
  assert(vk_inst_cc.find("TU_AUTOTUNE_ALGO") != std::string::npos);
  assert(vk_inst_cc.find("debug.xendroid.turnip_swapchain_ubwc") !=
         std::string::npos);
  assert(vk_inst_cc.find("debug.xendroid.turnip_cached_coherent") !=
         std::string::npos);
  assert(vk_inst_cc.find("debug.xendroid.turnip_autotune_algo") !=
         std::string::npos);
}

}  // namespace

int main(int argc, char** argv) {
  TestRtHeightPolicyInitialAndFloor();
  TestRtHeightPolicyGeometricGrowth();
  TestFramebufferHeightAndInvalidation();
  TestDriverClassificationAndMaliQuirksMatrix();
  TestTurnipOptInEnvHelpers();
  if (argc >= 2 && argv[1] && argv[1][0] != '\0') {
    TestSourceContracts(argv[1]);
  }
  std::printf("wave6_onda4a_rt_quirks_test: ALL PASSED\n");
  return 0;
}
