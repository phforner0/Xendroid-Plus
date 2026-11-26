/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_VULKAN_VULKAN_PIPELINE_STATE_CACHE_H_
#define XENIA_GPU_VULKAN_VULKAN_PIPELINE_STATE_CACHE_H_

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "xenia/base/hash.h"
#include "xenia/base/platform.h"
#include "xenia/base/threading.h"
#include "xenia/base/xxhash.h"
#include "xenia/gpu/guest_spirv_shader_cache.h"
#include "xenia/gpu/primitive_processor.h"
#include "xenia/gpu/register_file.h"
#include "xenia/gpu/registers.h"
#include "xenia/gpu/shader_storage.h"
#include "xenia/gpu/spirv_shader_translator.h"
#include "xenia/gpu/vulkan/vulkan_pipeline_cache_file.h"
#include "xenia/gpu/vulkan/vulkan_render_target_cache.h"
#include "xenia/gpu/vulkan/vulkan_shader.h"
#include "xenia/gpu/xenos.h"
#include "xenia/ui/vulkan/vulkan_api.h"

namespace xe {
namespace gpu {
namespace vulkan {

class VulkanCommandProcessor;

// TODO(Triang3l): Create a common base for both the Vulkan and the Direct3D
// implementations.
class VulkanPipelineCache : public GuestSpirvShaderCache::Host {
 public:
  class PipelineLayoutProvider {
   public:
    virtual ~PipelineLayoutProvider() {}
    virtual VkPipelineLayout GetPipelineLayout() const = 0;
    virtual uint32_t GetDescriptorSetLayouts(
        VkDescriptorSetLayout
            layouts_out[SpirvShaderTranslator::kDescriptorSetCount]) const {
      return 0;
    }

   protected:
    PipelineLayoutProvider() = default;
  };

  // Pre-mapped dynamic-state values for one pipeline, derived from the same
  // (non-canonicalized) state the static pipeline would have baked. The command
  // processor reads these to emit the extended-dynamic-state setters per draw
  // with dirty tracking, guaranteeing the dynamic values match the static
  // encoding exactly. Only fields whose capability bit is set are dynamic; the
  // rest are baked and their values here are ignored.
  struct DynamicState {
    VkPrimitiveTopology primitive_topology;
    VkBool32 primitive_restart_enable;
    VkCullModeFlags cull_mode;
    VkFrontFace front_face;
    VkBool32 depth_test_enable;
    VkBool32 depth_write_enable;
    VkCompareOp depth_compare_op;
    VkBool32 stencil_test_enable;
    VkStencilOpState stencil_front;
    VkStencilOpState stencil_back;
    VkBool32 depth_clamp_enable;
    VkPolygonMode polygon_mode;
    // Per color render target. color_rts_used is the bitmask of valid indices
    // (matching render_pass_key.depth_and_color_used >> 1).
    uint32_t color_rts_used;
    VkBool32 color_blend_enable[xenos::kMaxColorRenderTargets];
    VkColorBlendEquationEXT color_blend_equation[xenos::kMaxColorRenderTargets];
    VkColorComponentFlags color_write_mask[xenos::kMaxColorRenderTargets];
  };

  struct Pipeline {
    // VK_NULL_HANDLE while asynchronous creation is pending (or after it
    // failed). The slot address is STABLE for the cache's lifetime
    // (unordered_map nodes never relocate, and entries are never erased while
    // running), so the deferred command buffer records &pipeline and loads it
    // at replay.
    std::atomic<VkPipeline> pipeline{VK_NULL_HANDLE};
    // The layouts are owned by the VulkanCommandProcessor, and must not be
    // destroyed by it while the pipeline cache is active. Atomic because an
    // interpreter placeholder is created with a minimal (no-texture) layout on
    // the draw thread, then upgraded to the real layout by the creation thread
    // once the deferred shaders are translated (before the pipeline hot-swap).
    // It is also the field the fork's deferred-bind model reads at draw time
    // (starts at the minimal layout from untranslated shaders, upgraded to the
    // real one by the creation thread) - the atomic serves both async models.
    std::atomic<const PipelineLayoutProvider*> pipeline_layout;
    // True while pipeline_layout still holds the minimal layout computed from
    // shaders whose bindings weren't ready at insertion. Shaders can become
    // ready before this entry's own job runs, so cache hits re-check and
    // upgrade on the draw thread - otherwise a draw in that window allocates
    // texture descriptors from a zero-binding layout.
    std::atomic<bool> pipeline_layout_stale{false};
    // Pre-mapped dynamic-state values for this pipeline (see DynamicState).
    DynamicState dynamic_state;

    // Placeholder pipeline support for reduced stutter (upstream model, used
    // when the vulkan_placeholder_pipelines cvar is on; all stay inert - false /
    // VK_NULL_HANDLE - when it is off and the deferred-bind model is used).
    // When true, the current pipeline uses a placeholder pixel shader and
    // the real pipeline is being compiled in the background.
    std::atomic<bool> is_placeholder{false};
    // When true, the placeholder rasterizes with the ucode interpreter VS (so
    // the draw must feed it full float constants and the ucode location). Set
    // once when the interpreter placeholder is built, gated by is_placeholder.
    std::atomic<bool> uses_interpreter{false};
    // The placeholder VkPipeline handle (VK_NULL_HANDLE if none). A draw
    // compares the pipeline handle it bound against this to know whether it
    // bound the placeholder - consistent with the single `pipeline` load,
    // unlike the separate is_placeholder flag which is cleared a few
    // instructions after the real pipeline is swapped in.
    std::atomic<VkPipeline> placeholder_pipeline{VK_NULL_HANDLE};
    // Set while a creation thread builds the entry, under
    // creation_request_lock_, for ExpeditePipeline.
    std::atomic<bool> creation_in_flight{false};
    // Set when translating its shaders or creating it failed: the slot stays
    // VK_NULL_HANDLE (or the placeholder), and draws don't wait for it.
    std::atomic<bool> creation_failed{false};
    // Whether the placeholder writes color (its pixel shader isn't the no-op
    // one), so the dynamic color write mask matches the static one.
    std::atomic<bool> placeholder_writes_color{false};
    // Poisoning guard: true until EnsurePipelineCreated observes VK_SUCCESS for
    // the real pipeline, preventing broken PSO descriptions from being written
    // to persistent .vk.xpso storage.
    std::atomic<bool> storage_write_pending{false};
    // Set once a fast-linked GPL pipeline has queued its background LTO pass.
    std::atomic<bool> lto_queued{false};

    Pipeline(const PipelineLayoutProvider* pipeline_layout_provider)
        : pipeline_layout(pipeline_layout_provider), dynamic_state{} {}

    // Copy constructor needed for unordered_map
    Pipeline(const Pipeline& other)
        : pipeline(other.pipeline.load(std::memory_order_acquire)),
          pipeline_layout(
              other.pipeline_layout.load(std::memory_order_acquire)),
          pipeline_layout_stale(
              other.pipeline_layout_stale.load(std::memory_order_acquire)),
          dynamic_state(other.dynamic_state),
          is_placeholder(other.is_placeholder.load(std::memory_order_acquire)),
          uses_interpreter(
              other.uses_interpreter.load(std::memory_order_acquire)),
          placeholder_pipeline(
              other.placeholder_pipeline.load(std::memory_order_acquire)),
          creation_in_flight(
              other.creation_in_flight.load(std::memory_order_acquire)),
          creation_failed(other.creation_failed.load(std::memory_order_acquire)),
          placeholder_writes_color(
              other.placeholder_writes_color.load(std::memory_order_acquire)),
          storage_write_pending(
              other.storage_write_pending.load(std::memory_order_acquire)),
          lto_queued(other.lto_queued.load(std::memory_order_acquire)) {}

    // Move constructor
    Pipeline(Pipeline&& other) noexcept
        : pipeline(other.pipeline.load(std::memory_order_acquire)),
          pipeline_layout(
              other.pipeline_layout.load(std::memory_order_acquire)),
          pipeline_layout_stale(
              other.pipeline_layout_stale.load(std::memory_order_acquire)),
          dynamic_state(other.dynamic_state),
          is_placeholder(other.is_placeholder.load(std::memory_order_acquire)),
          uses_interpreter(
              other.uses_interpreter.load(std::memory_order_acquire)),
          placeholder_pipeline(
              other.placeholder_pipeline.load(std::memory_order_acquire)),
          creation_in_flight(
              other.creation_in_flight.load(std::memory_order_acquire)),
          creation_failed(other.creation_failed.load(std::memory_order_acquire)),
          placeholder_writes_color(
              other.placeholder_writes_color.load(std::memory_order_acquire)),
          storage_write_pending(
              other.storage_write_pending.load(std::memory_order_acquire)),
          lto_queued(other.lto_queued.load(std::memory_order_acquire)) {}

    // Deleted copy assignment to prevent accidental copying
    Pipeline& operator=(const Pipeline&) = delete;

    // Deleted move assignment
    Pipeline& operator=(Pipeline&&) = delete;
  };

  // Resolved extended-dynamic-state capabilities for the current device and the
  // vulkan_dynamic_pipeline_state cvar. A field is dynamic only when its bit is
  // set here; otherwise it is baked into the pipeline key as before. EDS1/EDS2
  // fields share `extended_dynamic_state` (core in Vulkan 1.3); each EDS3
  // sub-feature has its own bit. Computed once in Initialize().
  struct DynamicStateCapabilities {
    // EDS1/EDS2 (core in 1.3): cull mode, front face, primitive topology,
    // primitive restart enable, depth test/write/compare op, stencil test
    // enable, stencil op.
    bool extended_dynamic_state = false;
    // EDS3 sub-features.
    bool depth_clamp_enable = false;
    bool polygon_mode = false;
    bool color_blend_enable = false;
    bool color_blend_equation = false;
    bool color_write_mask = false;
    // When false, dynamic topology must stay within its class, so the topology
    // class is kept in the pipeline key.
    bool primitive_topology_unrestricted = false;
  };
  const DynamicStateCapabilities& dynamic_state_capabilities() const {
    return dynamic_state_capabilities_;
  }

  static constexpr size_t kLayoutUIDEmpty = 0;

  VulkanPipelineCache(VulkanCommandProcessor& command_processor,
                      const RegisterFile& register_file,
                      VulkanRenderTargetCache& render_target_cache,
                      VkShaderStageFlags guest_shader_vertex_stages);
  ~VulkanPipelineCache();

  bool Initialize();
  void Shutdown();

  // Shader and pipeline storage.
  void InitializeShaderStorage(
      const std::filesystem::path& cache_root, uint32_t title_id, bool blocking,
      std::function<void()> completion_callback = nullptr);
  void ShutdownShaderStorage();

  // Submission-boundary hook; the command processor calls this on the GPU
  // thread right before replaying the deferred command buffer. Persists the
  // VkPipelineCache blob (throttled) and storage files, then waits until all
  // queued pipeline creations complete so every deferred pipeline bind in the
  // stream resolves to a created handle - unless vulkan_async_skip_draws is
  // enabled or the startup storage preload is running, in which case it
  // doesn't block and unready pipelines have their draws dropped at replay.
  void EndSubmission();
  // Async Submit Worker split of EndSubmission() (Onda 3A): runs Recorder-only
  // bookkeeping (MaybeSaveVkPipelineCache, storage_writer_.RequestFlush,
  // ProcessDeferredDestructions) on the GPU Commands thread and returns the
  // required primary pipeline creation ticket for the chunk (0 if no wait is
  // needed, e.g. under vulkan_async_skip_draws / startup_loading or when all
  // previously enqueued pipelines have already finished).
  uint64_t PrepareAsyncSubmissionOnRecorder();
  // Worker-thread safe: waits on creation_threads_ until every primary
  // pipeline creation with creation_ticket <= required_ticket has finished (or
  // shutdown is requested), without blocking on newer tickets queued for
  // subsequent chunks.
  void AwaitPipelineTicket(uint64_t required_ticket);
  // RAII scope marking the current thread as the asynchronous Submit & Replay
  // Worker during DeferredCommandBuffer::Execute() so ResolvePipelineForReplay
  // never calls ExpeditePipeline() / CreateQueuedPipeline(..., nullptr) with
  // the Recorder's single-threaded translator.
  class ScopedWorkerReplay {
   public:
    ScopedWorkerReplay();
    ~ScopedWorkerReplay();
    ScopedWorkerReplay(const ScopedWorkerReplay&) = delete;
    ScopedWorkerReplay& operator=(const ScopedWorkerReplay&) = delete;
  };
  static bool IsInAsyncReplayThread();
  // Worker-thread safe replacement for ExpeditePipeline during replay:
  // promotes `pipeline` in creation_queue_ and waits for creation_threads_ to
  // finish compiling it without touching guest_shader_cache_.translator().
  bool WaitForPipelineCompletionFromWorker(Pipeline* pipeline);
  // Throttled persist of the VkPipelineCache blob. Cheap no-op unless new
  // pipelines were created and the save interval elapsed. Called from
  // EndSubmission().
  void MaybeSaveVkPipelineCache();
  bool IsCreatingPipelines();
  // Waits for any pipeline creation needed by the current draw path to finish
  // before state is consumed. This was added so strict ZPD query paths stop
  // racing pipeline compilation and then blocking work on incomplete state.
  void AwaitPipelineCompletion();
  // For a draw that can't be dropped while its pipeline is still being created:
  // creates the pipeline on this thread if it is still queued, or waits for the
  // creation thread that took it. Other queued pipelines keep building in the
  // background. Returns whether the real pipeline (not a placeholder) exists
  // now. Draw thread only: it translates with the shared translator, and a
  // creation on this thread doesn't set creation_in_flight, so another caller
  // wouldn't wait for it.
  bool ExpeditePipeline(Pipeline* pipeline);
  // Bounded wait (up to timeout_ms) on background creation threads for
  // solid-color / visible-RT draws before dropping or standing in (adapted from
  // XDroid360 33623e916, BSD-3-Clause).
  bool WaitForPipeline(Pipeline* pipeline, uint32_t timeout_ms);
  // Resolves a deferred pipeline slot at DeferredCommandBuffer::Execute() time.
  // Returns VK_NULL_HANDLE if creation failed or did not finish within the
  // bounded wait so replay safely drops the draws.
  VkPipeline ResolvePipelineForReplay(Pipeline* pipeline, uint32_t timeout_ms,
                                      bool expedite_on_replay);

  VulkanShader* LoadShader(xenos::ShaderType shader_type,
                           const uint32_t* host_address, uint32_t dword_count);
  // Write-watch cached shader lookup for PM4_IM_LOAD (guest physical address).
  VulkanShader* LoadShaderFromGuestAddress(xenos::ShaderType shader_type,
                                           uint32_t guest_physical_address,
                                           const uint32_t* host_address,
                                           uint32_t dword_count);
  void InvalidateShaderLoadWatchCache();
  void OnShaderMemoryInvalidated(uint32_t physical_address, uint32_t length);
  // Analyze shader microcode on the translator thread.
  void AnalyzeShaderUcode(Shader& shader) {
    shader.AnalyzeUcode(ucode_disasm_buffer_);
  }

  // Retrieves the shader modification for the current state. The shader must
  // have microcode analyzed.
  SpirvShaderTranslator::Modification GetCurrentVertexShaderModification(
      const Shader& shader,
      Shader::HostVertexShaderType host_vertex_shader_type,
      uint32_t interpolator_mask, bool ps_param_gen_used) const;
  SpirvShaderTranslator::Modification GetCurrentPixelShaderModification(
      const Shader& shader, uint32_t interpolator_mask, uint32_t param_gen_pos,
      reg::RB_DEPTHCONTROL normalized_depth_control,
      uint32_t normalized_color_mask,
      bool apply_polygon_offset_in_shader) const;

  // The ucode interpreter placeholder defers translation by not calling this on
  // the draw thread; the creation thread calls it (passing its own worker
  // translator, since the shared one is single-threaded). nullptr uses the
  // shared translator.
  bool EnsureShadersTranslated(VulkanShader::VulkanTranslation* vertex_shader,
                               VulkanShader::VulkanTranslation* pixel_shader,
                               SpirvShaderTranslator* translator = nullptr);
  // use_interpreter requests the ucode interpreter VS placeholder for a new,
  // untranslated vertex shader (falls back to normal translation if the async
  // placeholder path can't be taken for this draw).
  bool ConfigurePipeline(
      VulkanShader::VulkanTranslation* vertex_shader,
      VulkanShader::VulkanTranslation* pixel_shader,
      const PrimitiveProcessor::ProcessingResult& primitive_processing_result,
      reg::RB_DEPTHCONTROL normalized_depth_control,
      uint32_t normalized_color_mask,
      VulkanRenderTargetCache::RenderPassKey render_pass_key,
      bool use_interpreter, Pipeline** pipeline_out);
  // The texture sign classes of the pixel shader of the next ConfigurePipeline
  // calls (PipelineDescription::texture_sign_classes).
  void SetTextureSignClasses(uint32_t texture_sign_classes) {
    texture_sign_classes_ = texture_sign_classes;
  }
  // PipelineDescription::texture_exp_adjust_zero of the next ConfigurePipeline
  // calls.
  void SetTextureExpAdjustZero(bool texture_exp_adjust_zero) {
    texture_exp_adjust_zero_ = texture_exp_adjust_zero;
  }
  // PipelineDescription::texture_integer_scale_used of the next
  // ConfigurePipeline calls (spirv_texture_iscale_specialization).
  void SetTextureIntegerScaleUsed(uint32_t texture_integer_scale_used) {
    texture_integer_scale_used_ = texture_integer_scale_used;
  }

  // VK_EXT_shader_object clean-room path (gated by vulkan_shader_object and
  // device support, with automatic fallback to ConfigurePipeline).
  struct ShaderObjectDrawConfig {
    VkShaderEXT vertex_shader = VK_NULL_HANDLE;
    VkShaderEXT fragment_shader = VK_NULL_HANDLE;
    const PipelineLayoutProvider* pipeline_layout = nullptr;
    DynamicState dynamic_state{};
    VkSampleCountFlagBits rasterization_samples = VK_SAMPLE_COUNT_1_BIT;
    VkSampleMask sample_mask = UINT32_MAX;
    VkBool32 alpha_to_coverage_enable = VK_FALSE;
    VkBool32 depth_bias_enable = VK_FALSE;
  };
  bool CanUseShaderObjects(
      const PrimitiveProcessor::ProcessingResult& primitive_processing_result)
      const;
  bool ConfigureShaderObjects(
      VulkanShader::VulkanTranslation* vertex_shader,
      VulkanShader::VulkanTranslation* pixel_shader,
      const PrimitiveProcessor::ProcessingResult& primitive_processing_result,
      reg::RB_DEPTHCONTROL normalized_depth_control,
      uint32_t normalized_color_mask,
      VulkanRenderTargetCache::RenderPassKey render_pass_key,
      ShaderObjectDrawConfig& config_out);

  // True while this draw must be fed the ucode interpreter's inputs (full float
  // constants + ucode location). False once hot-swapped to the real VS.
  static bool IsInterpreterPlaceholder(const Pipeline* pipeline) {
    return pipeline->uses_interpreter.load(std::memory_order_acquire) &&
           pipeline->is_placeholder.load(std::memory_order_acquire);
  }

  // Whether ConfigurePipeline will create the pipeline asynchronously (so the
  // draw thread must not translate shaders itself). Matches the use_async
  // condition inside ConfigurePipeline. has_pixel_shader because the async
  // placeholder path needs a pixel shader.
  bool CanCreatePipelineAsync(bool has_pixel_shader) const;

 private:
  // PipelineGeometryShader and GeometryShaderKey come from
  // GuestSpirvShaderCache.
  using GeometryShaderKey = GuestSpirvShaderCache::GeometryShaderKey;

  enum class PipelinePrimitiveTopology : uint32_t {
    kPointList,
    kLineList,
    kLineStrip,
    kTriangleList,
    kTriangleStrip,
    kTriangleFan,
    kLineListWithAdjacency,
    kPatchList,
  };

  enum class PipelinePolygonMode : uint32_t {
    kFill,
    kLine,
    kPoint,
  };

  // Tessellation mode for pipeline creation.
  // Must match the TCS (hull shader) selection logic.
  enum class PipelineTessellationMode : uint32_t {
    kNone,
    kDiscrete,    // Integer tessellation factors.
    kContinuous,  // Fractional (fractional_even) tessellation factors.
    kAdaptive,    // Per-edge factors from index buffer.
  };

  // Tessellation primitive type.
  enum class PipelineTessellationPatchType : uint32_t {
    kNone,
    kTriangle,
    kQuad,
  };

  enum class PipelineBlendFactor : uint32_t {
    kZero,
    kOne,
    kSrcColor,
    kOneMinusSrcColor,
    kDstColor,
    kOneMinusDstColor,
    kSrcAlpha,
    kOneMinusSrcAlpha,
    kDstAlpha,
    kOneMinusDstAlpha,
    kConstantColor,
    kOneMinusConstantColor,
    kConstantAlpha,
    kOneMinusConstantAlpha,
    kSrcAlphaSaturate,
  };

  // Update PipelineDescription::kVersion if anything is changed!
  XEPACKEDSTRUCT(PipelineRenderTarget, {
    PipelineBlendFactor src_color_blend_factor : 4;  // 4
    PipelineBlendFactor dst_color_blend_factor : 4;  // 8
    xenos::BlendOp color_blend_op : 3;               // 11
    PipelineBlendFactor src_alpha_blend_factor : 4;  // 15
    PipelineBlendFactor dst_alpha_blend_factor : 4;  // 19
    xenos::BlendOp alpha_blend_op : 3;               // 22
    uint32_t color_write_mask : 4;                   // 26
  });

  XEPACKEDSTRUCT(PipelineDescription, {
    uint64_t vertex_shader_hash;
    uint64_t vertex_shader_modification;
    // 0 if no pixel shader.
    uint64_t pixel_shader_hash;
    uint64_t pixel_shader_modification;
    VulkanRenderTargetCache::RenderPassKey render_pass_key;

    // Shader stages.
    PipelineGeometryShader geometry_shader : 2;            // 2
    PipelineTessellationMode tessellation_mode : 2;        // 4
    PipelineTessellationPatchType tessellation_patch : 2;  // 6
    // Input assembly.
    PipelinePrimitiveTopology primitive_topology : 3;  // 9
    uint32_t primitive_restart : 1;                    // 10
    // Rasterization.
    uint32_t depth_clamp_enable : 1;       // 7
    PipelinePolygonMode polygon_mode : 2;  // 9
    uint32_t cull_front : 1;               // 10
    uint32_t cull_back : 1;                // 11
    uint32_t front_face_clockwise : 1;     // 12
    // Depth / stencil.
    uint32_t depth_write_enable : 1;                      // 13
    xenos::CompareFunction depth_compare_op : 3;          // 15
    uint32_t stencil_test_enable : 1;                     // 17
    xenos::StencilOp stencil_front_fail_op : 3;           // 20
    xenos::StencilOp stencil_front_pass_op : 3;           // 23
    xenos::StencilOp stencil_front_depth_fail_op : 3;     // 26
    xenos::CompareFunction stencil_front_compare_op : 3;  // 29
    xenos::StencilOp stencil_back_fail_op : 3;            // 32

    xenos::StencilOp stencil_back_pass_op : 3;           // 3
    xenos::StencilOp stencil_back_depth_fail_op : 3;     // 6
    xenos::CompareFunction stencil_back_compare_op : 3;  // 9

    // Topology class (0=point, 1=line, 2=triangle, 3=patch). Only kept in the
    // key when dynamic primitive topology is enabled but topology may not cross
    // classes (dynamicPrimitiveTopologyUnrestricted absent). Zero otherwise.
    uint32_t topology_class : 2;  // 11
    // host_alpha_to_coverage: the guest's alpha to coverage done by the host's
    // fixed function (the pixel shaders don't emulate it then). Zero in
    // pipeline storages written before it, which is the emulated behavior.
    uint32_t alpha_to_coverage : 1;  // 12
    // spirv_texture_sign_specialization: the sign class
    // (SpirvShaderTranslator::TextureSignClass) of the texture of each of the
    // 8 slots of the pixel shader (GetTextureSignClassSlots), 2 bits each - its
    // specialization constants. Zero (handled at runtime) in pipeline storages
    // written before it.
    uint32_t texture_sign_classes : 16;  // 28
    // spirv_texture_exp_adjust_specialization: every texture the pixel shader
    // fetches has a zero exponent adjustment (its specialization constant).
    // Zero (handled at runtime) in pipeline storages written before it.
    uint32_t texture_exp_adjust_zero : 1;  // 29
    // spirv_texture_iscale_specialization: 32-bit per-slot bitmask of fetch
    // constants with non-zero integer resolution scale / point-snap bits.
    uint32_t texture_integer_scale_used;

    // Filled only for the attachments present in the render pass object.
    PipelineRenderTarget render_targets[xenos::kMaxColorRenderTargets];

    // Including all the padding, for a stable hash.
    PipelineDescription() { Reset(); }
    PipelineDescription(const PipelineDescription& description) {
      std::memcpy(this, &description, sizeof(*this));
    }
    PipelineDescription& operator=(const PipelineDescription& description) {
      std::memcpy(this, &description, sizeof(*this));
      return *this;
    }
    bool operator==(const PipelineDescription& description) const {
      return std::memcmp(this, &description, sizeof(*this)) == 0;
    }
    void Reset() { std::memset(this, 0, sizeof(*this)); }
    uint64_t GetHash() const { return XXH3_64bits(this, sizeof(*this)); }
    struct Hasher {
      size_t operator()(const PipelineDescription& description) const {
        return size_t(description.GetHash());
      }
    };

    static constexpr uint32_t kVersion = 0x20261118;
  });

  // Pipeline storage constants.
  static constexpr uint32_t kPipelineStorageVersionWithoutAPI = 0x20201219;
  static constexpr uint32_t kPipelineStorageAPIMagicVulkan = 'VLKN';

  // Pipeline storage description.
  XEPACKEDSTRUCT(PipelineStoredDescription, {
    uint64_t description_hash;
    PipelineDescription description;
  });

  // creation threads, with everything needed from caches pre-looked-up.
  struct PipelineCreationArguments {
    std::pair<const PipelineDescription, Pipeline>* pipeline;
    VulkanShader::VulkanTranslation* vertex_shader;
    VulkanShader::VulkanTranslation* pixel_shader;
    VkShaderModule geometry_shader;
    // Tessellation shaders (only used when tessellation is active).
    VkShaderModule tessellation_vertex_shader;   // VS for passing data to TCS.
    VkShaderModule tessellation_control_shader;  // TCS (hull shader).
    VkRenderPass render_pass;
    // For dynamic rendering (VK_KHR_dynamic_rendering / Vulkan 1.3).
    VulkanRenderTargetCache::RenderPassKey render_pass_key;
    // Priority for async compilation (higher = compiled sooner).
    // Pipelines that write to visible render targets get higher priority.
    uint8_t priority = 0;
    // True when queued solely to upgrade a fast-linked GPL pipeline to a
    // link-time-optimized (LTO) pipeline in the background.
    bool lto_link_only = false;
    // Monotonic ticket for primary pipelines queued from ConfigurePipeline
    // (0 for background LTO re-links and startup storage preload).
    uint64_t creation_ticket = 0;
  };

  // Comparator for priority queue - higher priority first.
  struct PipelineCreationPriorityCompare {
    bool operator()(const PipelineCreationArguments& a,
                    const PipelineCreationArguments& b) const {
      return a.priority < b.priority;  // max-heap: lower priority at bottom
    }
  };

  // A priority queue (a heap in a vector) that can also give up a given
  // pipeline's request, for ExpeditePipeline.
  class CreationRequestHeap {
   public:
    bool empty() const { return heap_.empty(); }
    const PipelineCreationArguments& top() const { return heap_.front(); }
    void push(const PipelineCreationArguments& creation_arguments) {
      heap_.push_back(creation_arguments);
      std::push_heap(heap_.begin(), heap_.end(),
                     PipelineCreationPriorityCompare());
    }
    void pop() {
      std::pop_heap(heap_.begin(), heap_.end(),
                    PipelineCreationPriorityCompare());
      heap_.pop_back();
    }
    // Removes the queued request of `pipeline` into creation_arguments_out.
    bool Take(const Pipeline* pipeline,
              PipelineCreationArguments& creation_arguments_out) {
      auto it = std::find_if(heap_.begin(), heap_.end(),
                             [pipeline](const PipelineCreationArguments& a) {
                               return &a.pipeline->second == pipeline;
                             });
      if (it == heap_.end()) {
        return false;
      }
      creation_arguments_out = *it;
      *it = heap_.back();
      heap_.pop_back();
      std::make_heap(heap_.begin(), heap_.end(),
                     PipelineCreationPriorityCompare());
      return true;
    }
    bool Contains(const Pipeline* pipeline) const {
      return std::any_of(heap_.begin(), heap_.end(),
                         [pipeline](const PipelineCreationArguments& a) {
                           return &a.pipeline->second == pipeline;
                         });
    }
    bool HasTicketAtOrBefore(uint64_t required_ticket) const {
      if (required_ticket == 0) {
        return false;
      }
      return std::any_of(
          heap_.begin(), heap_.end(),
          [required_ticket](const PipelineCreationArguments& a) {
            return a.creation_ticket != 0 &&
                   a.creation_ticket <= required_ticket;
          });
    }
    bool PromoteToMaxPriority(const Pipeline* pipeline) {
      auto it = std::find_if(heap_.begin(), heap_.end(),
                             [pipeline](const PipelineCreationArguments& a) {
                               return &a.pipeline->second == pipeline;
                             });
      if (it == heap_.end()) {
        return false;
      }
      if (it->priority < 255) {
        it->priority = 255;
        std::make_heap(heap_.begin(), heap_.end(),
                       PipelineCreationPriorityCompare());
      }
      return true;
    }

   private:
    std::vector<PipelineCreationArguments> heap_;
  };

  // Can be called from multiple threads. use_try_claim atomically claims the
  // translation so concurrent callers (draw thread + creation threads)
  // translate it exactly once, the losers waiting for the winner.
  bool TranslateAnalyzedShader(SpirvShaderTranslator& translator,
                               VulkanShader::VulkanTranslation& translation,
                               bool use_try_claim = false);

  // Translates shaders in parallel for storage loading.
  void TranslateShadersForStorage(
      const std::set<std::pair<uint64_t, uint64_t>>& translations_needed,
      bool edram_fsi_used);

  // Guest graphics pipeline layout for the given (translated) shaders. Binding
  // counts come from translation, so untranslated shaders yield the minimal
  // no-texture layout used by the interpreter placeholder. Thread-safe.
  // counts_complete_out (optional) reports whether every consulted shader had
  // its bindings ready, i.e. whether the returned layout is the real one.
  const PipelineLayoutProvider* GetGuestGraphicsPipelineLayout(
      const VulkanShader::VulkanTranslation* vertex_shader,
      const VulkanShader::VulkanTranslation* pixel_shader,
      bool* counts_complete_out = nullptr);
  // Upgrades a cache-hit pipeline's layout on the draw thread once its
  // shaders' bindings are ready, closing the window between translation
  // finishing and the entry's own creation job running.
  void RefreshPipelineLayoutIfStale(
      Pipeline& pipeline, const VulkanShader::VulkanTranslation* vertex_shader,
      const VulkanShader::VulkanTranslation* pixel_shader);

  void WritePipelineRenderTargetDescription(
      reg::RB_BLENDCONTROL blend_control, uint32_t write_mask,
      PipelineRenderTarget& render_target_out) const;
  bool GetCurrentStateDescription(
      const VulkanShader::VulkanTranslation* vertex_shader,
      const VulkanShader::VulkanTranslation* pixel_shader,
      const PrimitiveProcessor::ProcessingResult& primitive_processing_result,
      reg::RB_DEPTHCONTROL normalized_depth_control,
      uint32_t normalized_color_mask,
      VulkanRenderTargetCache::RenderPassKey render_pass_key,
      PipelineDescription& description_out) const;

  // Whether the pipeline for the given description is supported by the device.
  bool ArePipelineRequirementsMet(const PipelineDescription& description) const;

  VkShaderModule GetGeometryShader(GeometryShaderKey key);

  // GuestSpirvShaderCache::Host.
  std::unique_ptr<SpirvShaderTranslator> CreateTranslator() const override;
  bool precise_interpolation_supported() const override;
  bool depth_float24_round() const override {
    return render_target_cache_.depth_float24_round();
  }
  bool depth_float24_convert_in_pixel_shader() const override;

  // Get the appropriate tessellation control shader (hull shader) module.
  VkShaderModule GetTessellationControlShader(
      PipelineTessellationMode mode, PipelineTessellationPatchType patch_type,
      bool use_control_point_count) const;

  // Get the appropriate tessellation vertex shader module.
  VkShaderModule GetTessellationVertexShader(
      PipelineTessellationMode mode) const;

  // Can be called from creation threads - all needed data must be fully set up
  // at the point of the call: shaders must be translated, pipeline layout and
  // render pass objects must be available.
  // If fragment_shader_override is not VK_NULL_HANDLE, it is used instead of
  // the pixel shader from creation_arguments (for placeholder pipelines).
  // vertex_shader_override, if not VK_NULL_HANDLE, is used instead of the
  // translated vertex shader module (for the ucode interpreter placeholder,
  // whose guest VS is intentionally not translated yet).
  bool EnsurePipelineCreated(
      const PipelineCreationArguments& creation_arguments,
      VkShaderModule fragment_shader_override = VK_NULL_HANDLE,
      VkShaderModule vertex_shader_override = VK_NULL_HANDLE);

  // Creates a placeholder pipeline using the placeholder pixel shader.
  // Used for pipeline hot-swap to reduce stutter.
  bool EnsurePipelineCreatedWithPlaceholder(
      const PipelineCreationArguments& creation_arguments) {
    return EnsurePipelineCreated(creation_arguments, placeholder_pixel_shader_);
  }

  // Creates a placeholder pipeline that rasterizes the guest geometry via the
  // ucode interpreter VS while the real VS+PS compile in the background.
  bool EnsurePipelineCreatedWithInterpreterPlaceholder(
      const PipelineCreationArguments& creation_arguments);

  VulkanCommandProcessor& command_processor_;
  const RegisterFile& register_file_;
  VulkanRenderTargetCache& render_target_cache_;
  VkShaderStageFlags guest_shader_vertex_stages_;

  // SetTextureSignClasses.
  uint32_t texture_sign_classes_ = 0;
  // SetTextureExpAdjustZero.
  bool texture_exp_adjust_zero_ = false;
  // SetTextureIntegerScaleUsed.
  uint32_t texture_integer_scale_used_ = 0;

  //Substitute SPIR-V bytecode retained for VK_EXT_shader_object depth-only
  // float24 draws.
  std::vector<uint8_t> float24_truncate_fragment_shader_code_;
  std::vector<uint8_t> float24_round_fragment_shader_code_;

  // VK_EXT_graphics_pipeline_library state and 4-part library caches.
  bool use_graphics_pipeline_library_ = false;
  std::mutex gpl_mutex_;
  std::unordered_map<uint64_t, VkPipeline, xe::hash::IdentityHasher<uint64_t>>
      gpl_vertex_input_libs_;
  std::unordered_map<uint64_t, VkPipeline, xe::hash::IdentityHasher<uint64_t>>
      gpl_pre_raster_libs_;
  std::unordered_map<uint64_t, VkPipeline, xe::hash::IdentityHasher<uint64_t>>
      gpl_fragment_shader_libs_;
  std::unordered_map<uint64_t, VkPipeline, xe::hash::IdentityHasher<uint64_t>>
      gpl_fragment_output_libs_;

  // VK_EXT_shader_object state and per-stage shader object cache.
  std::mutex shader_objects_mutex_;
  std::unordered_map<uint64_t, VkShaderEXT, xe::hash::IdentityHasher<uint64_t>>
      shader_objects_;
  VkShaderEXT GetOrCreateShaderObject(
      VkShaderStageFlagBits stage, VkShaderStageFlags next_stage,
      uint64_t shader_hash, uint64_t modification,
      const uint8_t* spirv_bytes, size_t spirv_size,
      const PipelineLayoutProvider* pipeline_layout,
      const VkSpecializationInfo* specialization_info,
      uint64_t specialization_key);

  // Cached device features for geometry shader creation.
  unsigned int spirv_version_;
  bool signed_zero_inf_nan_preserve_float32_;
  bool denorm_flush_to_zero_float32_;
  // Already combined with the spirv_disable_rounding_mode_rte cvar.
  bool rounding_mode_rte_float32_;

  // Resolved once in Initialize() from the device features and the
  // vulkan_dynamic_pipeline_state cvar.
  DynamicStateCapabilities dynamic_state_capabilities_;

  // Zeroes the fields of `description` that are made dynamic by
  // dynamic_state_capabilities_ so that draws differing only in dynamic state
  // collapse onto one pipeline. Keeps the topology class when topology is not
  // unrestricted. Must be called right before hashing / cache lookup.
  void CanonicalizePipelineDescription(PipelineDescription& description) const;

  // Maps a (non-canonicalized) description to the pre-resolved Vulkan dynamic
  // state values the command processor emits per draw. Always computes all
  // fields; the command processor only emits the ones whose capability bit is
  // set.
  void FillDynamicState(const PipelineDescription& description,
                        DynamicState& dynamic_state) const;

  // Temporary storage for AnalyzeUcode calls on the processor thread.
  StringBuffer ucode_disasm_buffer_;
  // Shared guest SPIR-V translator, modification derivation and geometry keys.
  GuestSpirvShaderCache guest_shader_cache_;

  struct LayoutUID {
    size_t uid;
    size_t vector_span_offset;
    size_t vector_span_length;
  };
  std::mutex layouts_mutex_;
  // Texture binding layouts of different shaders, for obtaining layout UIDs.
  std::vector<VulkanShader::TextureBinding> texture_binding_layouts_;
  // Map of texture binding layouts used by shaders, for obtaining UIDs. Keys
  // are XXH3 hashes of layouts, values need manual collision resolution using
  // layout_vector_offset:layout_length of texture_binding_layouts_.
  std::unordered_multimap<uint64_t, LayoutUID,
                          xe::hash::IdentityHasher<uint64_t>>
      texture_binding_layout_map_;

  // Ucode hash -> shader.
  std::unordered_map<uint64_t, VulkanShader*,
                     xe::hash::IdentityHasher<uint64_t>>
      shaders_;

  // Write-watch cache for PM4_IM_LOAD (lazily allocated when
  // vulkan_shader_load_watch_cache is enabled).
  struct ShaderLoadWatchCacheState;
  std::unique_ptr<ShaderLoadWatchCacheState> shader_load_watch_cache_;
  bool shader_load_watch_cache_was_enabled_ = false;
  static std::pair<uint32_t, uint32_t> ShaderWatchInvalidationThunk(
      void* context_ptr, uint32_t physical_address_start, uint32_t length,
      bool exact_range);
  void ShutdownShaderLoadWatchCache();

  // Geometry shaders for Xenos primitive types not supported by Vulkan.
  // Stores VK_NULL_HANDLE if failed to create.
  std::unordered_map<GeometryShaderKey, VkShaderModule,
                     GeometryShaderKey::Hasher>
      geometry_shaders_;

  // Empty depth-only pixel shader for writing to depth buffer using fragment
  // shader interlock when no Xenos pixel shader provided.
  VkShaderModule depth_only_fragment_shader_ = VK_NULL_HANDLE;

  // Substitute depth-only pixel shaders that perform float24 conversion of the
  // rasterizer's depth, bound for guest depth-only draws when in-PS float24
  // conversion is active and the depth buffer is D24FS8. Mirrors the DXBC
  // backend's float24_{truncate,round}_ps.
  VkShaderModule float24_truncate_fragment_shader_ = VK_NULL_HANDLE;
  VkShaderModule float24_round_fragment_shader_ = VK_NULL_HANDLE;

  // Placeholder pixel shader for pipeline hot-swap to reduce stutter.
  // Outputs transparent black while the real shader compiles in background.
  VkShaderModule placeholder_pixel_shader_ = VK_NULL_HANDLE;

  // Ucode interpreter vertex shader - interprets a guest VS's ucode to render
  // its real geometry while the real VS translates+compiles in the background.
  VkShaderModule ucode_interpreter_vs_ = VK_NULL_HANDLE;
  // Flat grey debug pixel shader for interpreter placeholders, so the interim
  // geometry is visible (host-render-target path only).
  VkShaderModule placeholder_color_pixel_shader_ = VK_NULL_HANDLE;

  // Tessellation shaders.
  // Vertex shaders for tessellation - pass indices/factors to TCS.
  VkShaderModule tessellation_indexed_vs_ = VK_NULL_HANDLE;
  VkShaderModule tessellation_adaptive_vs_ = VK_NULL_HANDLE;
  // Tessellation control shaders (hull shaders) for different modes and
  // primitive types.
  // Discrete mode (integer tessellation factors).
  VkShaderModule discrete_triangle_1cp_hs_ = VK_NULL_HANDLE;
  VkShaderModule discrete_triangle_3cp_hs_ = VK_NULL_HANDLE;
  VkShaderModule discrete_quad_1cp_hs_ = VK_NULL_HANDLE;
  VkShaderModule discrete_quad_4cp_hs_ = VK_NULL_HANDLE;
  // Continuous mode (fractional_even tessellation factors).
  VkShaderModule continuous_triangle_1cp_hs_ = VK_NULL_HANDLE;
  VkShaderModule continuous_triangle_3cp_hs_ = VK_NULL_HANDLE;
  VkShaderModule continuous_quad_1cp_hs_ = VK_NULL_HANDLE;
  VkShaderModule continuous_quad_4cp_hs_ = VK_NULL_HANDLE;
  // Adaptive mode (per-edge factors from index buffer).
  VkShaderModule adaptive_triangle_hs_ = VK_NULL_HANDLE;
  VkShaderModule adaptive_quad_hs_ = VK_NULL_HANDLE;

  // Vulkan pipeline cache for faster pipeline creation.
  VkPipelineCache vk_pipeline_cache_ = VK_NULL_HANDLE;
  PFN_vkMergePipelineCaches vk_merge_pipeline_caches_ = nullptr;

  std::unordered_map<PipelineDescription, Pipeline, PipelineDescription::Hasher>
      pipelines_;

  // Previously used pipeline, to avoid lookups if the state wasn't changed.
  std::pair<const PipelineDescription, Pipeline>* last_pipeline_ = nullptr;

  void CreationThread();
  // Translates the shaders of a queued request if needed, upgrades its layout
  // and creates it, on a creation thread (with its own translator) or on the
  // draw thread (ExpeditePipeline, with the shared one).
  void CreateQueuedPipeline(PipelineCreationArguments& creation_arguments,
                            SpirvShaderTranslator* translator);
  // With creation_request_lock_ held: signals whatever waits for the creation
  // threads to run out of work, once they have.
  void NotifyCreationIdleLocked(std::unique_lock<std::mutex>& lock);

  // For asynchronous creation.
  std::vector<std::unique_ptr<xe::threading::Thread>> creation_threads_;
  std::atomic<bool> creation_threads_shutdown_{false};
  std::atomic<size_t> creation_threads_busy_{0};
  // Priority queue contains pointers to map entries. Pipelines are never
  // evicted as games have a finite set that should all remain cached for
  // performance. Higher priority pipelines (those writing to visible RTs)
  // are compiled first.
  CreationRequestHeap creation_queue_;
  uint64_t next_creation_ticket_ = 0;
  std::vector<uint64_t> in_flight_primary_tickets_;
  bool HasPendingCreationAtOrBeforeTicketLocked(uint64_t required_ticket) const;
  std::mutex creation_request_lock_;
  std::condition_variable creation_request_cond_;
  // Notified when a creation thread is done with a pipeline it took.
  std::condition_variable creation_done_cond_;
  std::unique_ptr<xe::threading::Event> creation_completion_event_ = nullptr;
  std::atomic<bool> creation_completion_set_event_{false};
  std::function<void()> creation_completion_callback_;
  // Set by ExpeditePipeline when it leaves the creation idle: a creation thread
  // signals that then, as the completion callback may block (on Android it
  // waits for the UI thread) and the draw thread must not.
  bool creation_idle_notify_pending_ = false;
  // During startup loading, don't block on pipeline creation to allow game
  // boot. Cleared by the completion callback: on a creation thread, or on
  // the thread calling InitializeShaderStorage when it queued nothing.
  std::atomic<bool> startup_loading_{false};

  // Deferred destruction of pipelines.
  // Pipelines are only destroyed after the GPU submission that might reference
  // them has completed (tracked via submission numbers from command processor).
  void ProcessDeferredDestructions();
  std::vector<std::pair<VkPipeline, uint64_t>> deferred_destroy_pipelines_;
  std::mutex deferred_destroy_mutex_;

  // Shader and pipeline storage.
  uint32_t shader_storage_title_id_ = 0;
  std::atomic<bool> shader_storage_file_flush_needed_{false};
  std::atomic<bool> pipeline_storage_file_flush_needed_{false};

  // Storage writer for shaders and pipelines (owns file handles and storage
  // index).
  ShaderStorageWriter<PipelineStoredDescription> storage_writer_;

  // VkPipelineCache persistence path.
  std::filesystem::path vk_pipeline_cache_path_;
  // Verified, size-bounded, atomically replaced file per driver identity (see
  // vulkan_pipeline_cache_file.h). Set while vk_pipeline_cache_path_ is.
  std::unique_ptr<pipeline_cache_file::Store> vk_pipeline_cache_store_;
  // Serializes SaveVkPipelineCacheFromHandle() and store teardown across threads.
  std::mutex vk_pipeline_cache_save_mutex_;
  // Non-blocking snapshot save: merges vk_pipeline_cache_ into a temporary
  // VkPipelineCache and dispatches vkGetPipelineCacheData + disk I/O onto
  // storage_writer_'s low-priority WriteThread (coalescing superseded saves).
  void SaveVkPipelineCacheAsync();
  // Synchronous fallback / final shutdown persist on the caller thread.
  void SaveVkPipelineCacheSync();
  void SaveVkPipelineCacheFromHandle(VkPipelineCache cache_handle);
  // Periodically persist vk_pipeline_cache_ on the GPU thread so the cache
  // survives Android process kills (upstream only saves at clean shutdown,
  // which Android frequently skips when terminating the process).
  static constexpr uint64_t kVkPipelineCacheSaveIntervalMs = 20000;
  static constexpr uint64_t kVkPipelineCachePrecreateSaveIntervalMs = 5000;
  static constexpr size_t kVkPipelineCachePrecreateSaveBatch = 256;
  std::atomic<bool> vk_pipeline_cache_active_{false};
  std::atomic<bool> vk_pipeline_cache_dirty_{false};
  std::atomic<bool> vk_pipeline_cache_saved_since_init_{false};
  uint64_t vk_pipeline_cache_last_save_ms_ = 0;
};

}  // namespace vulkan
}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_VULKAN_VULKAN_PIPELINE_STATE_CACHE_H_
