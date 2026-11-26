/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2020 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/gpu/vulkan/deferred_command_buffer.h"

#include <algorithm>
#include <cstring>

#include "xenia/base/assert.h"
#include "xenia/base/cvar.h"
#include "xenia/base/math.h"
#include "xenia/base/profiling.h"
#include "xenia/gpu/gpu_flags.h"
#include "xenia/gpu/shader_storage.h"
#include "xenia/gpu/vulkan/vulkan_command_processor.h"
#include "xenia/gpu/vulkan/vulkan_pipeline_cache.h"
#include "xenia/ui/vulkan/vulkan_instance.h"

#include "xenia/base/cvar.h"
#include "xenia/base/logging.h"

DECLARE_bool(log_gpu_frame_time_breakdown);

DEFINE_bool(vulkan_replay_stats, false,
            "Diagnostics - count the commands the deferred command buffers "
            "replay to the driver by type, and how many binds and dynamic "
            "states repeat what is already set in the same command buffer "
            "(VkReplay lines with log_gpu_frame_time_breakdown).",
            "Vulkan");

DEFINE_bool(vulkan_deferred_cmd_size_cursor, true,
            "Track the deferred command buffer's recorded length with a size "
            "cursor over a geometrically grown buffer instead of resizing the "
            "vector per recorded command (vector::resize zero-fill was ~2% of "
            "the GPU command processor thread).",
            "Vulkan");

namespace xe {
namespace gpu {
namespace vulkan {

DeferredCommandBuffer::DeferredCommandBuffer(
    const VulkanCommandProcessor& command_processor, size_t initial_size)
    : command_processor_(command_processor),
      use_size_cursor_(cvars::vulkan_deferred_cmd_size_cursor) {
  command_stream_.reserve(initial_size / sizeof(uintmax_t));
}

void DeferredCommandBuffer::Reset() {
  command_stream_size_ = 0;
  segment_stream_size_ = 0;
  if (!use_size_cursor_) {
    command_stream_.clear();
    segment_stream_.clear();
  }
  in_segment_ = false;
  active_segment_placeholder_ = kNoSegmentPlaceholder;
  // The recorded arguments are gone; a stale patch handle would write into a
  // future command's storage.
  render_area_patch_offset_ = kNoRenderAreaPatch;
  pass_drawn_width_ = 0;
  pass_drawn_height_ = 0;
  pass_recorded_draws_ = 0;
  current_scissor_ = {};
  current_viewport_ = {};
  has_current_viewport_ = false;
  viewport_clip_enabled_ = false;
}

size_t DeferredCommandBuffer::CmdExecuteSegmentPlaceholder() {
  assert_false(in_segment_);
  auto* args = reinterpret_cast<ArgsExecuteSegment*>(
      WriteCommand(Command::kExecuteSegment, sizeof(ArgsExecuteSegment)));
  args->start_elements = 0;
  args->size_elements = 0;
  return static_cast<size_t>(reinterpret_cast<uintmax_t*>(args) -
                             command_stream_.data());
}

void DeferredCommandBuffer::BeginSegment(size_t placeholder_offset) {
  assert_false(in_segment_);
  assert_true(placeholder_offset != kNoSegmentPlaceholder &&
              placeholder_offset < command_stream_size_);
  auto& args = *reinterpret_cast<ArgsExecuteSegment*>(command_stream_.data() +
                                                      placeholder_offset);
  if (args.size_elements == 0) {
    args.start_elements = segment_stream_size_;
  } else {
    assert_true(args.start_elements + args.size_elements ==
                segment_stream_size_);
  }
  in_segment_ = true;
  active_segment_placeholder_ = placeholder_offset;
}

void DeferredCommandBuffer::EndSegment() {
  assert_true(in_segment_);
  assert_true(active_segment_placeholder_ != kNoSegmentPlaceholder &&
              active_segment_placeholder_ < command_stream_size_);
  auto& args = *reinterpret_cast<ArgsExecuteSegment*>(
      command_stream_.data() + active_segment_placeholder_);
  args.size_elements = segment_stream_size_ - args.start_elements;
  in_segment_ = false;
  active_segment_placeholder_ = kNoSegmentPlaceholder;
}

void DeferredCommandBuffer::Execute(VkCommandBuffer command_buffer,
                                    VulkanPipelineCache* pipeline_cache) {
#if XE_GPU_FINE_GRAINED_DRAW_SCOPES
  SCOPE_profile_cpu_f("gpu");
#endif  // XE_GPU_FINE_GRAINED_DRAW_SCOPES
  assert_false(in_segment_);
  if (cvars::vulkan_replay_stats) {
    replay_last_args_.clear();
  }
  bool guest_graphics_pipeline_unready = false;
  ExecuteStream(command_buffer, command_stream_.data(), command_stream_size_,
                guest_graphics_pipeline_unready, pipeline_cache);
}

void DeferredCommandBuffer::ExecuteStream(
    VkCommandBuffer command_buffer, const uintmax_t* stream,
    size_t stream_remaining, bool& guest_graphics_pipeline_unready,
    VulkanPipelineCache* pipeline_cache) {
  const ui::vulkan::VulkanDevice::Functions& dfn =
      command_processor_.GetVulkanDevice()->functions();
  const bool count_replay = cvars::vulkan_replay_stats;
  while (stream_remaining) {
    const CommandHeader& header =
        *reinterpret_cast<const CommandHeader*>(stream);
    stream += kCommandHeaderSizeElements;
    stream_remaining -= kCommandHeaderSizeElements;

    if (count_replay) {
      // What counts as the same state slot for the redundancy check.
      uint64_t key = uint64_t(header.command) << 32;
      switch (header.command) {
        case Command::kVkBindDescriptorSets: {
          auto& args =
              *reinterpret_cast<const ArgsVkBindDescriptorSets*>(stream);
          key |= (uint64_t(args.pipeline_bind_point) << 8) | args.first_set;
          replay_stats_.descriptor_sets += args.descriptor_set_count;
        } break;
        case Command::kVkPushConstants: {
          auto& args = *reinterpret_cast<const ArgsVkPushConstants*>(stream);
          key |= (uint64_t(args.offset) << 16) | args.size;
          replay_stats_.push_constant_bytes += args.size;
        } break;
        case Command::kVkBindPipeline:
        case Command::kVkBindPipelineDeferred:
          // Both bind the graphics or compute pipeline slot.
          key = (uint64_t(Command::kVkBindPipeline) << 32) |
                uint64_t(reinterpret_cast<const ArgsVkBindPipeline*>(stream)
                             ->pipeline_bind_point);
          break;
        default:
          break;
      }
      CountReplayedCommand(uint32_t(header.command), stream,
                           header.arguments_size_elements * sizeof(uintmax_t),
                           key);
    }

    switch (header.command) {
      case Command::kVkBeginRenderPass: {
        auto& args = *reinterpret_cast<const ArgsVkBeginRenderPass*>(stream);
        size_t offset_bytes = sizeof(ArgsVkBeginRenderPass);
        VkRenderPassBeginInfo render_pass_begin_info;
        render_pass_begin_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        render_pass_begin_info.pNext = nullptr;
        render_pass_begin_info.renderPass = args.render_pass;
        render_pass_begin_info.framebuffer = args.framebuffer;
        render_pass_begin_info.renderArea = args.render_area;
        render_pass_begin_info.clearValueCount = args.clear_value_count;
        if (render_pass_begin_info.clearValueCount) {
          offset_bytes = xe::align(offset_bytes, alignof(VkClearValue));
          render_pass_begin_info.pClearValues =
              reinterpret_cast<const VkClearValue*>(
                  reinterpret_cast<const uint8_t*>(stream) + offset_bytes);
          offset_bytes +=
              sizeof(VkClearValue) * render_pass_begin_info.clearValueCount;
        } else {
          render_pass_begin_info.pClearValues = nullptr;
        }
        dfn.vkCmdBeginRenderPass(command_buffer, &render_pass_begin_info,
                                 args.contents);
      } break;

      case Command::kVkBindDescriptorSets: {
        auto& args = *reinterpret_cast<const ArgsVkBindDescriptorSets*>(stream);
        size_t offset_bytes = xe::align(sizeof(ArgsVkBindDescriptorSets),
                                        alignof(VkDescriptorSet));
        const VkDescriptorSet* descriptor_sets =
            reinterpret_cast<const VkDescriptorSet*>(
                reinterpret_cast<const uint8_t*>(stream) + offset_bytes);
        offset_bytes += sizeof(VkDescriptorSet) * args.descriptor_set_count;
        const uint32_t* dynamic_offsets = nullptr;
        if (args.dynamic_offset_count) {
          offset_bytes = xe::align(offset_bytes, alignof(uint32_t));
          dynamic_offsets = reinterpret_cast<const uint32_t*>(
              reinterpret_cast<const uint8_t*>(stream) + offset_bytes);
          offset_bytes += sizeof(uint32_t) * args.dynamic_offset_count;
        }
        dfn.vkCmdBindDescriptorSets(command_buffer, args.pipeline_bind_point,
                                    args.layout, args.first_set,
                                    args.descriptor_set_count, descriptor_sets,
                                    args.dynamic_offset_count, dynamic_offsets);
      } break;

      case Command::kVkBindIndexBuffer: {
        auto& args = *reinterpret_cast<const ArgsVkBindIndexBuffer*>(stream);
        dfn.vkCmdBindIndexBuffer(command_buffer, args.buffer, args.offset,
                                 args.index_type);
      } break;

      case Command::kVkBindPipeline: {
        auto& args = *reinterpret_cast<const ArgsVkBindPipeline*>(stream);
        dfn.vkCmdBindPipeline(command_buffer, args.pipeline_bind_point,
                              args.pipeline);
        if (args.pipeline_bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS) {
          guest_graphics_pipeline_unready = false;
        }
      } break;

      case Command::kVkBindPipelineDeferred: {
        auto& args =
            *reinterpret_cast<const ArgsVkBindPipelineDeferred*>(stream);
        VkPipeline resolved_pipeline = args.pipeline;
        if (args.pipeline_slot != nullptr && pipeline_cache != nullptr) {
          resolved_pipeline = pipeline_cache->ResolvePipelineForReplay(
              static_cast<VulkanPipelineCache::Pipeline*>(args.pipeline_slot),
              args.replay_wait_ms, args.expedite_on_replay);
        }
        if (EvaluateReplayPipelineBind(args.pipeline_slot != nullptr,
                                       resolved_pipeline != VK_NULL_HANDLE,
                                       guest_graphics_pipeline_unready) ==
            ReplayPipelineBindOutcome::kBind) {
          dfn.vkCmdBindPipeline(command_buffer, args.pipeline_bind_point,
                                resolved_pipeline);
        }
      } break;

      case Command::kVkBindVertexBuffers: {
        auto& args = *reinterpret_cast<const ArgsVkBindVertexBuffers*>(stream);
        size_t offset_bytes =
            xe::align(sizeof(ArgsVkBindVertexBuffers), alignof(VkBuffer));
        const VkBuffer* buffers = reinterpret_cast<const VkBuffer*>(
            reinterpret_cast<const uint8_t*>(stream) + offset_bytes);
        offset_bytes =
            xe::align(offset_bytes + sizeof(VkBuffer) * args.binding_count,
                      alignof(VkDeviceSize));
        const VkDeviceSize* offsets = reinterpret_cast<const VkDeviceSize*>(
            reinterpret_cast<const uint8_t*>(stream) + offset_bytes);
        dfn.vkCmdBindVertexBuffers(command_buffer, args.first_binding,
                                   args.binding_count, buffers, offsets);
      } break;
      case Command::kVkBeginQuery: {
        auto& args = *reinterpret_cast<const ArgsVkBeginQuery*>(stream);
        dfn.vkCmdBeginQuery(command_buffer, args.query_pool, args.query,
                            args.flags);
      } break;
      case Command::kVkEndQuery: {
        auto& args = *reinterpret_cast<const ArgsVkEndQuery*>(stream);
        dfn.vkCmdEndQuery(command_buffer, args.query_pool, args.query);
      } break;
      case Command::kVkCopyQueryPoolResults: {
        auto& args =
            *reinterpret_cast<const ArgsVkCopyQueryPoolResults*>(stream);
        dfn.vkCmdCopyQueryPoolResults(
            command_buffer, args.query_pool, args.first_query, args.query_count,
            args.dst_buffer, args.dst_offset, args.stride, args.flags);
      } break;
      case Command::kVkResetQueryPool: {
        auto& args = *reinterpret_cast<const ArgsVkResetQueryPool*>(stream);
        dfn.vkCmdResetQueryPool(command_buffer, args.query_pool,
                                args.first_query, args.query_count);
      } break;

      case Command::kVkWriteTimestamp: {
        auto& args = *reinterpret_cast<const ArgsVkWriteTimestamp*>(stream);
        dfn.vkCmdWriteTimestamp(command_buffer, args.pipeline_stage,
                                args.query_pool, args.query);
      } break;

      case Command::kVkClearAttachments: {
        auto& args = *reinterpret_cast<const ArgsVkClearAttachments*>(stream);
        size_t offset_bytes = xe::align(sizeof(ArgsVkClearAttachments),
                                        alignof(VkClearAttachment));
        const VkClearAttachment* attachments =
            reinterpret_cast<const VkClearAttachment*>(
                reinterpret_cast<const uint8_t*>(stream) + offset_bytes);
        offset_bytes = xe::align(
            offset_bytes + sizeof(VkClearAttachment) * args.attachment_count,
            alignof(VkClearRect));
        const VkClearRect* rects = reinterpret_cast<const VkClearRect*>(
            reinterpret_cast<const uint8_t*>(stream) + offset_bytes);
        dfn.vkCmdClearAttachments(command_buffer, args.attachment_count,
                                  attachments, args.rect_count, rects);
      } break;

      case Command::kVkClearColorImage: {
        auto& args = *reinterpret_cast<const ArgsVkClearColorImage*>(stream);
        dfn.vkCmdClearColorImage(
            command_buffer, args.image, args.image_layout, &args.color,
            args.range_count,
            reinterpret_cast<const VkImageSubresourceRange*>(
                reinterpret_cast<const uint8_t*>(stream) +
                xe::align(sizeof(ArgsVkClearColorImage),
                          alignof(VkImageSubresourceRange))));
      } break;

      case Command::kVkCopyBuffer: {
        auto& args = *reinterpret_cast<const ArgsVkCopyBuffer*>(stream);
        dfn.vkCmdCopyBuffer(
            command_buffer, args.src_buffer, args.dst_buffer, args.region_count,
            reinterpret_cast<const VkBufferCopy*>(
                reinterpret_cast<const uint8_t*>(stream) +
                xe::align(sizeof(ArgsVkCopyBuffer), alignof(VkBufferCopy))));
      } break;

      case Command::kVkFillBuffer: {
        auto& args = *reinterpret_cast<const ArgsVkFillBuffer*>(stream);
        dfn.vkCmdFillBuffer(command_buffer, args.dst_buffer, args.dst_offset,
                            args.size, args.data);
      } break;

      case Command::kVkCopyBufferToImage: {
        auto& args = *reinterpret_cast<const ArgsVkCopyBufferToImage*>(stream);
        dfn.vkCmdCopyBufferToImage(
            command_buffer, args.src_buffer, args.dst_image,
            args.dst_image_layout, args.region_count,
            reinterpret_cast<const VkBufferImageCopy*>(
                reinterpret_cast<const uint8_t*>(stream) +
                xe::align(sizeof(ArgsVkCopyBufferToImage),
                          alignof(VkBufferImageCopy))));
      } break;

      case Command::kVkBlitImage: {
        auto& args = *reinterpret_cast<const ArgsVkBlitImage*>(stream);
        dfn.vkCmdBlitImage(
            command_buffer, args.src_image, args.src_image_layout,
            args.dst_image, args.dst_image_layout, args.region_count,
            reinterpret_cast<const VkImageBlit*>(
                reinterpret_cast<const uint8_t*>(stream) +
                xe::align(sizeof(ArgsVkBlitImage), alignof(VkImageBlit))),
            args.filter);
      } break;

      case Command::kVkCopyImage: {
        auto& args = *reinterpret_cast<const ArgsVkCopyImage*>(stream);
        dfn.vkCmdCopyImage(
            command_buffer, args.src_image, args.src_image_layout,
            args.dst_image, args.dst_image_layout, args.region_count,
            reinterpret_cast<const VkImageCopy*>(
                reinterpret_cast<const uint8_t*>(stream) +
                xe::align(sizeof(ArgsVkCopyImage), alignof(VkImageCopy))));
      } break;

      case Command::kVkDispatch: {
        auto& args = *reinterpret_cast<const ArgsVkDispatch*>(stream);
        dfn.vkCmdDispatch(command_buffer, args.group_count_x,
                          args.group_count_y, args.group_count_z);
      } break;

      case Command::kVkDraw: {
        if (guest_graphics_pipeline_unready) {
          break;
        }
        auto& args = *reinterpret_cast<const ArgsVkDraw*>(stream);
        dfn.vkCmdDraw(command_buffer, args.vertex_count, args.instance_count,
                      args.first_vertex, args.first_instance);
      } break;

      case Command::kVkDrawIndexed: {
        if (guest_graphics_pipeline_unready) {
          break;
        }
        auto& args = *reinterpret_cast<const ArgsVkDrawIndexed*>(stream);
        dfn.vkCmdDrawIndexed(command_buffer, args.index_count,
                             args.instance_count, args.first_index,
                             args.vertex_offset, args.first_instance);
      } break;

      case Command::kVkEndRenderPass:
        dfn.vkCmdEndRenderPass(command_buffer);
        break;

      case Command::kVkBeginRendering: {
        auto& args = *reinterpret_cast<const ArgsVkBeginRendering*>(stream);
        size_t offset_bytes = xe::align(sizeof(ArgsVkBeginRendering),
                                        alignof(VkRenderingAttachmentInfo));
        const VkRenderingAttachmentInfo* color_attachments =
            args.color_attachment_count
                ? reinterpret_cast<const VkRenderingAttachmentInfo*>(
                      reinterpret_cast<const uint8_t*>(stream) + offset_bytes)
                : nullptr;
        offset_bytes +=
            sizeof(VkRenderingAttachmentInfo) * args.color_attachment_count;
        const VkRenderingAttachmentInfo* depth_attachment =
            args.has_depth_attachment
                ? reinterpret_cast<const VkRenderingAttachmentInfo*>(
                      reinterpret_cast<const uint8_t*>(stream) + offset_bytes)
                : nullptr;
        if (args.has_depth_attachment) {
          offset_bytes += sizeof(VkRenderingAttachmentInfo);
        }
        const VkRenderingAttachmentInfo* stencil_attachment =
            args.has_stencil_attachment
                ? reinterpret_cast<const VkRenderingAttachmentInfo*>(
                      reinterpret_cast<const uint8_t*>(stream) + offset_bytes)
                : nullptr;
        VkRenderingInfo rendering_info = {};
        rendering_info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        rendering_info.flags = args.flags;
        rendering_info.renderArea = args.render_area;
        rendering_info.layerCount = args.layer_count;
        rendering_info.viewMask = args.view_mask;
        rendering_info.colorAttachmentCount = args.color_attachment_count;
        rendering_info.pColorAttachments = color_attachments;
        rendering_info.pDepthAttachment = depth_attachment;
        rendering_info.pStencilAttachment = stencil_attachment;
        dfn.vkCmdBeginRendering(command_buffer, &rendering_info);
      } break;

      case Command::kVkEndRendering:
        dfn.vkCmdEndRendering(command_buffer);
        break;

      case Command::kVkPipelineBarrier: {
        auto& args = *reinterpret_cast<const ArgsVkPipelineBarrier*>(stream);
        size_t barrier_offset_bytes = sizeof(ArgsVkPipelineBarrier);
        const VkMemoryBarrier* memory_barriers = nullptr;
        if (args.memory_barrier_count) {
          barrier_offset_bytes =
              xe::align(barrier_offset_bytes, alignof(VkMemoryBarrier));
          memory_barriers = reinterpret_cast<const VkMemoryBarrier*>(
              reinterpret_cast<const uint8_t*>(stream) + barrier_offset_bytes);
          barrier_offset_bytes +=
              sizeof(VkMemoryBarrier) * args.memory_barrier_count;
        }
        const VkBufferMemoryBarrier* buffer_memory_barriers = nullptr;
        if (args.buffer_memory_barrier_count) {
          barrier_offset_bytes =
              xe::align(barrier_offset_bytes, alignof(VkBufferMemoryBarrier));
          buffer_memory_barriers =
              reinterpret_cast<const VkBufferMemoryBarrier*>(
                  reinterpret_cast<const uint8_t*>(stream) +
                  barrier_offset_bytes);
          barrier_offset_bytes +=
              sizeof(VkBufferMemoryBarrier) * args.buffer_memory_barrier_count;
        }
        const VkImageMemoryBarrier* image_memory_barriers = nullptr;
        if (args.image_memory_barrier_count) {
          barrier_offset_bytes =
              xe::align(barrier_offset_bytes, alignof(VkImageMemoryBarrier));
          image_memory_barriers = reinterpret_cast<const VkImageMemoryBarrier*>(
              reinterpret_cast<const uint8_t*>(stream) + barrier_offset_bytes);
          barrier_offset_bytes +=
              sizeof(VkImageMemoryBarrier) * args.image_memory_barrier_count;
        }
        dfn.vkCmdPipelineBarrier(
            command_buffer, args.src_stage_mask, args.dst_stage_mask,
            args.dependency_flags, args.memory_barrier_count, memory_barriers,
            args.buffer_memory_barrier_count, buffer_memory_barriers,
            args.image_memory_barrier_count, image_memory_barriers);
      } break;

      case Command::kVkPushConstants: {
        auto& args = *reinterpret_cast<const ArgsVkPushConstants*>(stream);
        dfn.vkCmdPushConstants(command_buffer, args.layout, args.stage_flags,
                               args.offset, args.size,
                               reinterpret_cast<const uint8_t*>(stream) +
                                   sizeof(ArgsVkPushConstants));
      } break;

      case Command::kVkSetRenderingInputAttachmentIndices: {
        auto& args = *reinterpret_cast<
            const ArgsVkSetRenderingInputAttachmentIndices*>(stream);
        VkRenderingInputAttachmentIndexInfo input_attachment_index_info;
        input_attachment_index_info.sType =
            VK_STRUCTURE_TYPE_RENDERING_INPUT_ATTACHMENT_INDEX_INFO;
        input_attachment_index_info.pNext = nullptr;
        input_attachment_index_info.colorAttachmentCount =
            args.color_attachment_count;
        input_attachment_index_info.pColorAttachmentInputIndices =
            args.color_attachment_input_indices;
        input_attachment_index_info.pDepthInputAttachmentIndex = nullptr;
        input_attachment_index_info.pStencilInputAttachmentIndex = nullptr;
        dfn.vkCmdSetRenderingInputAttachmentIndices(
            command_buffer, &input_attachment_index_info);
      } break;

      case Command::kVkSetBlendConstants: {
        auto& args = *reinterpret_cast<const ArgsVkSetBlendConstants*>(stream);
        dfn.vkCmdSetBlendConstants(command_buffer, args.blend_constants);
      } break;

      case Command::kVkSetDepthBias: {
        auto& args = *reinterpret_cast<const ArgsVkSetDepthBias*>(stream);
        dfn.vkCmdSetDepthBias(command_buffer, args.depth_bias_constant_factor,
                              args.depth_bias_clamp,
                              args.depth_bias_slope_factor);
      } break;

      case Command::kVkSetScissor: {
        auto& args = *reinterpret_cast<const ArgsVkSetScissor*>(stream);
        dfn.vkCmdSetScissor(
            command_buffer, args.first_scissor, args.scissor_count,
            reinterpret_cast<const VkRect2D*>(
                reinterpret_cast<const uint8_t*>(stream) +
                xe::align(sizeof(ArgsVkSetScissor), alignof(VkRect2D))));
      } break;

      case Command::kVkSetStencilCompareMask: {
        auto& args =
            *reinterpret_cast<const ArgsSetStencilMaskReference*>(stream);
        dfn.vkCmdSetStencilCompareMask(command_buffer, args.face_mask,
                                       args.mask_reference);
      } break;

      case Command::kVkSetStencilReference: {
        auto& args =
            *reinterpret_cast<const ArgsSetStencilMaskReference*>(stream);
        dfn.vkCmdSetStencilReference(command_buffer, args.face_mask,
                                     args.mask_reference);
      } break;

      case Command::kVkSetStencilWriteMask: {
        auto& args =
            *reinterpret_cast<const ArgsSetStencilMaskReference*>(stream);
        dfn.vkCmdSetStencilWriteMask(command_buffer, args.face_mask,
                                     args.mask_reference);
      } break;

      case Command::kVkSetViewport: {
        auto& args = *reinterpret_cast<const ArgsVkSetViewport*>(stream);
        dfn.vkCmdSetViewport(
            command_buffer, args.first_viewport, args.viewport_count,
            reinterpret_cast<const VkViewport*>(
                reinterpret_cast<const uint8_t*>(stream) +
                xe::align(sizeof(ArgsVkSetViewport), alignof(VkViewport))));
      } break;

      case Command::kVkSetCullMode: {
        auto& args = *reinterpret_cast<const ArgsVkSetCullMode*>(stream);
        dfn.vkCmdSetCullMode(command_buffer, args.cull_mode);
      } break;

      case Command::kVkSetFrontFace: {
        auto& args = *reinterpret_cast<const ArgsVkSetFrontFace*>(stream);
        dfn.vkCmdSetFrontFace(command_buffer, args.front_face);
      } break;

      case Command::kVkSetPrimitiveTopology: {
        auto& args =
            *reinterpret_cast<const ArgsVkSetPrimitiveTopology*>(stream);
        dfn.vkCmdSetPrimitiveTopology(command_buffer, args.primitive_topology);
      } break;

      case Command::kVkSetPrimitiveRestartEnable: {
        auto& args = *reinterpret_cast<const ArgsVkSetBool*>(stream);
        dfn.vkCmdSetPrimitiveRestartEnable(command_buffer, args.value);
      } break;

      case Command::kVkSetDepthTestEnable: {
        auto& args = *reinterpret_cast<const ArgsVkSetBool*>(stream);
        dfn.vkCmdSetDepthTestEnable(command_buffer, args.value);
      } break;

      case Command::kVkSetDepthWriteEnable: {
        auto& args = *reinterpret_cast<const ArgsVkSetBool*>(stream);
        dfn.vkCmdSetDepthWriteEnable(command_buffer, args.value);
      } break;

      case Command::kVkSetDepthCompareOp: {
        auto& args = *reinterpret_cast<const ArgsVkSetDepthCompareOp*>(stream);
        dfn.vkCmdSetDepthCompareOp(command_buffer, args.depth_compare_op);
      } break;

      case Command::kVkSetStencilTestEnable: {
        auto& args = *reinterpret_cast<const ArgsVkSetBool*>(stream);
        dfn.vkCmdSetStencilTestEnable(command_buffer, args.value);
      } break;

      case Command::kVkSetStencilOp: {
        auto& args = *reinterpret_cast<const ArgsVkSetStencilOp*>(stream);
        dfn.vkCmdSetStencilOp(command_buffer, args.face_mask, args.fail_op,
                              args.pass_op, args.depth_fail_op,
                              args.compare_op);
      } break;

      case Command::kVkSetDepthClampEnableEXT: {
        auto& args = *reinterpret_cast<const ArgsVkSetBool*>(stream);
        dfn.vkCmdSetDepthClampEnableEXT(command_buffer, args.value);
      } break;

      case Command::kVkSetPolygonModeEXT: {
        auto& args = *reinterpret_cast<const ArgsVkSetPolygonModeEXT*>(stream);
        dfn.vkCmdSetPolygonModeEXT(command_buffer, args.polygon_mode);
      } break;

      case Command::kVkSetColorBlendEnableEXT: {
        auto& args =
            *reinterpret_cast<const ArgsVkSetColorBlendEnableEXT*>(stream);
        dfn.vkCmdSetColorBlendEnableEXT(
            command_buffer, args.first_attachment, args.attachment_count,
            reinterpret_cast<const VkBool32*>(
                reinterpret_cast<const uint8_t*>(stream) +
                xe::align(sizeof(ArgsVkSetColorBlendEnableEXT),
                          alignof(VkBool32))));
      } break;

      case Command::kVkSetColorBlendEquationEXT: {
        auto& args =
            *reinterpret_cast<const ArgsVkSetColorBlendEquationEXT*>(stream);
        dfn.vkCmdSetColorBlendEquationEXT(
            command_buffer, args.first_attachment, args.attachment_count,
            reinterpret_cast<const VkColorBlendEquationEXT*>(
                reinterpret_cast<const uint8_t*>(stream) +
                xe::align(sizeof(ArgsVkSetColorBlendEquationEXT),
                          alignof(VkColorBlendEquationEXT))));
      } break;

      case Command::kVkSetColorWriteMaskEXT: {
        auto& args =
            *reinterpret_cast<const ArgsVkSetColorWriteMaskEXT*>(stream);
        dfn.vkCmdSetColorWriteMaskEXT(
            command_buffer, args.first_attachment, args.attachment_count,
            reinterpret_cast<const VkColorComponentFlags*>(
                reinterpret_cast<const uint8_t*>(stream) +
                xe::align(sizeof(ArgsVkSetColorWriteMaskEXT),
                          alignof(VkColorComponentFlags))));
      } break;

      case Command::kVkSetFragmentShadingRateKHR: {
        auto& args =
            *reinterpret_cast<const ArgsVkSetFragmentShadingRateKHR*>(stream);
        PFN_vkCmdSetFragmentShadingRateKHR set_fragment_shading_rate =
            command_processor_.GetVulkanDevice()
                ->vkCmdSetFragmentShadingRateKHR();
        if (set_fragment_shading_rate) {
          const VkExtent2D fragment_size = {args.width, args.height};
          // The pipeline's rate alone: keep it through both combiners.
          const VkFragmentShadingRateCombinerOpKHR combiner_ops[2] = {
              VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR,
              VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR};
          set_fragment_shading_rate(command_buffer, &fragment_size,
                                    combiner_ops);
        }
      } break;

      case Command::kVkBeginDebugUtilsLabelEXT: {
        const ui::vulkan::VulkanInstance::Functions& ifn =
            command_processor_.GetVulkanDevice()
                ->vulkan_instance()
                ->functions();
        if (ifn.vkCmdBeginDebugUtilsLabelEXT) {
          auto& args = *reinterpret_cast<const ArgsVkDebugUtilsLabel*>(stream);
          const char* label_name = reinterpret_cast<const char*>(
              reinterpret_cast<const uint8_t*>(stream) +
              sizeof(ArgsVkDebugUtilsLabel));
          VkDebugUtilsLabelEXT label_info = {};
          label_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
          label_info.pLabelName = label_name;
          ifn.vkCmdBeginDebugUtilsLabelEXT(command_buffer, &label_info);
        }
      } break;

      case Command::kVkEndDebugUtilsLabelEXT: {
        const ui::vulkan::VulkanInstance::Functions& ifn =
            command_processor_.GetVulkanDevice()
                ->vulkan_instance()
                ->functions();
        if (ifn.vkCmdEndDebugUtilsLabelEXT) {
          ifn.vkCmdEndDebugUtilsLabelEXT(command_buffer);
        }
      } break;

      case Command::kVkInsertDebugUtilsLabelEXT: {
        const ui::vulkan::VulkanInstance::Functions& ifn =
            command_processor_.GetVulkanDevice()
                ->vulkan_instance()
                ->functions();
        if (ifn.vkCmdInsertDebugUtilsLabelEXT) {
          auto& args = *reinterpret_cast<const ArgsVkDebugUtilsLabel*>(stream);
          const char* label_name = reinterpret_cast<const char*>(
              reinterpret_cast<const uint8_t*>(stream) +
              sizeof(ArgsVkDebugUtilsLabel));
          VkDebugUtilsLabelEXT label_info = {};
          label_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
          label_info.pLabelName = label_name;
          ifn.vkCmdInsertDebugUtilsLabelEXT(command_buffer, &label_info);
        }
      } break;

      case Command::kExecuteSegment: {
        auto& args = *reinterpret_cast<const ArgsExecuteSegment*>(stream);
        if (args.size_elements != 0) {
          assert_true(args.start_elements + args.size_elements <=
                      segment_stream_size_);
          ExecuteStream(command_buffer,
                        segment_stream_.data() + args.start_elements,
                        args.size_elements, guest_graphics_pipeline_unready,
                        pipeline_cache);
        }
      } break;

      case Command::kVkSetDepthBias2EXT: {
        auto& args = *reinterpret_cast<const ArgsVkSetDepthBias2EXT*>(stream);
        if (dfn.vkCmdSetDepthBias2EXT) {
          VkDepthBiasRepresentationInfoEXT rep_info = {};
          rep_info.sType =
              VK_STRUCTURE_TYPE_DEPTH_BIAS_REPRESENTATION_INFO_EXT;
          rep_info.depthBiasRepresentation = args.depth_bias_representation;
          rep_info.depthBiasExact = args.depth_bias_exact;
          VkDepthBiasInfoEXT bias_info = {};
          bias_info.sType = VK_STRUCTURE_TYPE_DEPTH_BIAS_INFO_EXT;
          bias_info.pNext = args.has_representation ? &rep_info : nullptr;
          bias_info.depthBiasConstantFactor = args.depth_bias_constant_factor;
          bias_info.depthBiasClamp = args.depth_bias_clamp;
          bias_info.depthBiasSlopeFactor = args.depth_bias_slope_factor;
          dfn.vkCmdSetDepthBias2EXT(command_buffer, &bias_info);
        } else {
          dfn.vkCmdSetDepthBias(command_buffer, args.depth_bias_constant_factor,
                                args.depth_bias_clamp,
                                args.depth_bias_slope_factor);
        }
      } break;

      case Command::kVkBindShadersEXT: {
        auto& args = *reinterpret_cast<const ArgsVkBindShadersEXT*>(stream);
        if (dfn.vkCmdBindShadersEXT && args.stage_count != 0) {
          size_t offset_bytes = xe::align(sizeof(ArgsVkBindShadersEXT),
                                          alignof(VkShaderStageFlagBits));
          const VkShaderStageFlagBits* stages =
              reinterpret_cast<const VkShaderStageFlagBits*>(
                  reinterpret_cast<const uint8_t*>(stream) + offset_bytes);
          offset_bytes = xe::align(
              offset_bytes +
                  sizeof(VkShaderStageFlagBits) * args.stage_count,
              alignof(VkShaderEXT));
          const VkShaderEXT* shaders = reinterpret_cast<const VkShaderEXT*>(
              reinterpret_cast<const uint8_t*>(stream) + offset_bytes);
          dfn.vkCmdBindShadersEXT(command_buffer, args.stage_count, stages,
                                  shaders);
          for (uint32_t i = 0; i < args.stage_count; ++i) {
            if (stages[i] == VK_SHADER_STAGE_VERTEX_BIT &&
                shaders[i] != VK_NULL_HANDLE) {
              guest_graphics_pipeline_unready = false;
              break;
            }
          }
        }
      } break;

      case Command::kVkSetVertexInputEXT: {
        auto& args = *reinterpret_cast<const ArgsVkSetVertexInputEXT*>(stream);
        if (dfn.vkCmdSetVertexInputEXT) {
          size_t offset_bytes =
              xe::align(sizeof(ArgsVkSetVertexInputEXT),
                        alignof(VkVertexInputBindingDescription2EXT));
          const VkVertexInputBindingDescription2EXT* bindings =
              args.binding_count
                  ? reinterpret_cast<
                        const VkVertexInputBindingDescription2EXT*>(
                        reinterpret_cast<const uint8_t*>(stream) + offset_bytes)
                  : nullptr;
          offset_bytes = xe::align(
              offset_bytes + sizeof(VkVertexInputBindingDescription2EXT) *
                                 args.binding_count,
              alignof(VkVertexInputAttributeDescription2EXT));
          const VkVertexInputAttributeDescription2EXT* attributes =
              args.attribute_count
                  ? reinterpret_cast<
                        const VkVertexInputAttributeDescription2EXT*>(
                        reinterpret_cast<const uint8_t*>(stream) + offset_bytes)
                  : nullptr;
          dfn.vkCmdSetVertexInputEXT(command_buffer, args.binding_count,
                                     bindings, args.attribute_count,
                                     attributes);
        }
      } break;

      case Command::kVkSetViewportWithCount: {
        auto& args =
            *reinterpret_cast<const ArgsVkSetViewportWithCount*>(stream);
        if (dfn.vkCmdSetViewportWithCount) {
          const VkViewport* viewports =
              args.viewport_count
                  ? reinterpret_cast<const VkViewport*>(
                        reinterpret_cast<const uint8_t*>(stream) +
                        xe::align(sizeof(ArgsVkSetViewportWithCount),
                                  alignof(VkViewport)))
                  : nullptr;
          dfn.vkCmdSetViewportWithCount(command_buffer, args.viewport_count,
                                        viewports);
        }
      } break;

      case Command::kVkSetScissorWithCount: {
        auto& args =
            *reinterpret_cast<const ArgsVkSetScissorWithCount*>(stream);
        if (dfn.vkCmdSetScissorWithCount) {
          const VkRect2D* scissors =
              args.scissor_count
                  ? reinterpret_cast<const VkRect2D*>(
                        reinterpret_cast<const uint8_t*>(stream) +
                        xe::align(sizeof(ArgsVkSetScissorWithCount),
                                  alignof(VkRect2D)))
                  : nullptr;
          dfn.vkCmdSetScissorWithCount(command_buffer, args.scissor_count,
                                       scissors);
        }
      } break;

      case Command::kVkSetRasterizerDiscardEnable: {
        auto& args = *reinterpret_cast<const ArgsVkSetBool*>(stream);
        if (dfn.vkCmdSetRasterizerDiscardEnable) {
          dfn.vkCmdSetRasterizerDiscardEnable(command_buffer, args.value);
        }
      } break;

      case Command::kVkSetDepthBiasEnable: {
        auto& args = *reinterpret_cast<const ArgsVkSetBool*>(stream);
        if (dfn.vkCmdSetDepthBiasEnable) {
          dfn.vkCmdSetDepthBiasEnable(command_buffer, args.value);
        }
      } break;

      case Command::kVkSetRasterizationSamplesEXT: {
        auto& args =
            *reinterpret_cast<const ArgsVkSetRasterizationSamplesEXT*>(stream);
        if (dfn.vkCmdSetRasterizationSamplesEXT) {
          dfn.vkCmdSetRasterizationSamplesEXT(command_buffer, args.samples);
        }
      } break;

      case Command::kVkSetSampleMaskEXT: {
        auto& args = *reinterpret_cast<const ArgsVkSetSampleMaskEXT*>(stream);
        if (dfn.vkCmdSetSampleMaskEXT) {
          dfn.vkCmdSetSampleMaskEXT(command_buffer, args.samples,
                                    &args.sample_mask);
        }
      } break;

      case Command::kVkSetAlphaToCoverageEnableEXT: {
        auto& args = *reinterpret_cast<const ArgsVkSetBool*>(stream);
        if (dfn.vkCmdSetAlphaToCoverageEnableEXT) {
          dfn.vkCmdSetAlphaToCoverageEnableEXT(command_buffer, args.value);
        }
      } break;

      case Command::kVkSetAlphaToOneEnableEXT: {
        auto& args = *reinterpret_cast<const ArgsVkSetBool*>(stream);
        if (dfn.vkCmdSetAlphaToOneEnableEXT) {
          dfn.vkCmdSetAlphaToOneEnableEXT(command_buffer, args.value);
        }
      } break;

      case Command::kVkSetLogicOpEnableEXT: {
        auto& args = *reinterpret_cast<const ArgsVkSetBool*>(stream);
        if (dfn.vkCmdSetLogicOpEnableEXT) {
          dfn.vkCmdSetLogicOpEnableEXT(command_buffer, args.value);
        }
      } break;

      case Command::kVkSetDepthBoundsTestEnable: {
        auto& args = *reinterpret_cast<const ArgsVkSetBool*>(stream);
        if (dfn.vkCmdSetDepthBoundsTestEnable) {
          dfn.vkCmdSetDepthBoundsTestEnable(command_buffer, args.value);
        }
      } break;

      default:
        assert_unhandled_case(header.command);
        break;
    }

    stream += header.arguments_size_elements;
    stream_remaining -= header.arguments_size_elements;
  }
}

void DeferredCommandBuffer::CmdVkPipelineBarrier(
    VkPipelineStageFlags src_stage_mask, VkPipelineStageFlags dst_stage_mask,
    VkDependencyFlags dependency_flags, uint32_t memory_barrier_count,
    const VkMemoryBarrier* memory_barriers,
    uint32_t buffer_memory_barrier_count,
    const VkBufferMemoryBarrier* buffer_memory_barriers,
    uint32_t image_memory_barrier_count,
    const VkImageMemoryBarrier* image_memory_barriers) {
  size_t arguments_size = sizeof(ArgsVkPipelineBarrier);
  size_t memory_barriers_offset = 0;
  if (memory_barrier_count) {
    arguments_size = xe::align(arguments_size, alignof(VkMemoryBarrier));
    memory_barriers_offset = arguments_size;
    arguments_size += sizeof(VkMemoryBarrier) * memory_barrier_count;
  }
  size_t buffer_memory_barriers_offset = 0;
  if (buffer_memory_barrier_count) {
    arguments_size = xe::align(arguments_size, alignof(VkBufferMemoryBarrier));
    buffer_memory_barriers_offset = arguments_size;
    arguments_size +=
        sizeof(VkBufferMemoryBarrier) * buffer_memory_barrier_count;
  }
  size_t image_memory_barriers_offset = 0;
  if (image_memory_barrier_count) {
    arguments_size = xe::align(arguments_size, alignof(VkImageMemoryBarrier));
    image_memory_barriers_offset = arguments_size;
    arguments_size += sizeof(VkImageMemoryBarrier) * image_memory_barrier_count;
  }
  uint8_t* args_ptr = reinterpret_cast<uint8_t*>(
      WriteCommand(Command::kVkPipelineBarrier, arguments_size));
  auto& args = *reinterpret_cast<ArgsVkPipelineBarrier*>(args_ptr);
  args.src_stage_mask = src_stage_mask;
  args.dst_stage_mask = dst_stage_mask;
  args.dependency_flags = dependency_flags;
  args.memory_barrier_count = memory_barrier_count;
  args.buffer_memory_barrier_count = buffer_memory_barrier_count;
  args.image_memory_barrier_count = image_memory_barrier_count;
  if (memory_barrier_count) {
    std::memcpy(args_ptr + memory_barriers_offset, memory_barriers,
                sizeof(VkMemoryBarrier) * memory_barrier_count);
  }
  if (buffer_memory_barrier_count) {
    std::memcpy(args_ptr + buffer_memory_barriers_offset,
                buffer_memory_barriers,
                sizeof(VkBufferMemoryBarrier) * buffer_memory_barrier_count);
  }
  if (image_memory_barrier_count) {
    std::memcpy(args_ptr + image_memory_barriers_offset, image_memory_barriers,
                sizeof(VkImageMemoryBarrier) * image_memory_barrier_count);
  }
}

void DeferredCommandBuffer::CmdVkBeginRendering(
    const VkRenderingInfo* rendering_info) {
  assert_null(rendering_info->pNext);

  size_t arguments_size = xe::align(sizeof(ArgsVkBeginRendering),
                                    alignof(VkRenderingAttachmentInfo));
  size_t color_attachments_offset = arguments_size;
  arguments_size +=
      sizeof(VkRenderingAttachmentInfo) * rendering_info->colorAttachmentCount;
  size_t depth_attachment_offset = arguments_size;
  if (rendering_info->pDepthAttachment) {
    arguments_size += sizeof(VkRenderingAttachmentInfo);
  }
  size_t stencil_attachment_offset = arguments_size;
  if (rendering_info->pStencilAttachment) {
    arguments_size += sizeof(VkRenderingAttachmentInfo);
  }

  uint8_t* args_ptr = reinterpret_cast<uint8_t*>(
      WriteCommand(Command::kVkBeginRendering, arguments_size));
  auto& args = *reinterpret_cast<ArgsVkBeginRendering*>(args_ptr);
  args.flags = rendering_info->flags;
  args.render_area = rendering_info->renderArea;
  args.layer_count = rendering_info->layerCount;
  args.view_mask = rendering_info->viewMask;
  args.color_attachment_count = rendering_info->colorAttachmentCount;
  args.has_depth_attachment = rendering_info->pDepthAttachment != nullptr;
  args.has_stencil_attachment = rendering_info->pStencilAttachment != nullptr;
  BeginRenderAreaTracking(arguments_size, /*dynamic_rendering=*/true);

  if (rendering_info->colorAttachmentCount) {
    std::memcpy(args_ptr + color_attachments_offset,
                rendering_info->pColorAttachments,
                sizeof(VkRenderingAttachmentInfo) *
                    rendering_info->colorAttachmentCount);
  }
  if (rendering_info->pDepthAttachment) {
    std::memcpy(args_ptr + depth_attachment_offset,
                rendering_info->pDepthAttachment,
                sizeof(VkRenderingAttachmentInfo));
  }
  if (rendering_info->pStencilAttachment) {
    std::memcpy(args_ptr + stencil_attachment_offset,
                rendering_info->pStencilAttachment,
                sizeof(VkRenderingAttachmentInfo));
  }
}

static uint32_t xe_shrink_logged = 0;

void DeferredCommandBuffer::ShrinkRenderAreaToDrawn(
    uint32_t granularity_width, uint32_t granularity_height) {
  if (render_area_patch_offset_ == kNoRenderAreaPatch) {
    return;
  }
  const size_t patch_offset = render_area_patch_offset_;
  // One-shot: a pass is finalized once, and the handle must not survive into
  // the next pass or across a Reset().
  render_area_patch_offset_ = kNoRenderAreaPatch;
  if (!pass_recorded_draws_ || !pass_drawn_width_ || !pass_drawn_height_) {
    // Nothing rasterized into the pass (or only load/store work) - the full
    // extent is the only safe answer.
    if (cvars::log_gpu_frame_time_breakdown && xe_shrink_logged < 200) {
      ++xe_shrink_logged;
      XELOGI("VkShrink: kept full extent (draws={} w={} h={})",
             pass_recorded_draws_, pass_drawn_width_, pass_drawn_height_);
    }
    return;
  }
  assert_true(patch_offset < command_stream_size_);
  VkRect2D* render_area;
  if (render_area_patch_dynamic_) {
    render_area = &reinterpret_cast<ArgsVkBeginRendering*>(
                       command_stream_.data() + patch_offset)
                       ->render_area;
  } else {
    render_area = &reinterpret_cast<ArgsVkBeginRenderPass*>(
                       command_stream_.data() + patch_offset)
                       ->render_area;
  }
  uint32_t width = pass_drawn_width_;
  uint32_t height = pass_drawn_height_;
  if (granularity_width > 1) {
    width = xe::round_up(width, granularity_width);
  }
  if (granularity_height > 1) {
    height = xe::round_up(height, granularity_height);
  }
  // Offsets stay at the recorded origin; only ever shrink.
  const uint32_t was_w = render_area->extent.width;
  const uint32_t was_h = render_area->extent.height;
  render_area->extent.width = std::min(width, render_area->extent.width);
  render_area->extent.height = std::min(height, render_area->extent.height);
  // Only the large passes matter here; startup floods the log otherwise.
  if (cvars::log_gpu_frame_time_breakdown && was_h >= 2000 &&
      xe_shrink_logged < 200) {
    ++xe_shrink_logged;
    XELOGI("VkShrink: {}x{} -> {}x{} ({} draws, dynamic={})", was_w, was_h,
           render_area->extent.width, render_area->extent.height,
           pass_recorded_draws_, render_area_patch_dynamic_ ? 1 : 0);
  }
}

void* DeferredCommandBuffer::WriteCommand(Command command,
                                          size_t arguments_size_bytes) {
  std::vector<uintmax_t>& target_stream =
      in_segment_ ? segment_stream_ : command_stream_;
  size_t& target_size =
      in_segment_ ? segment_stream_size_ : command_stream_size_;
  size_t arguments_size_elements =
      (arguments_size_bytes + sizeof(uintmax_t) - 1) / sizeof(uintmax_t);
  size_t offset = target_size;
  size_t new_size = offset + kCommandHeaderSizeElements + arguments_size_elements;
  if (use_size_cursor_) {
    // Grow geometrically and keep the arena allocated across Reset() — the
    // bytes past the cursor are dead, so resize()'s zero-fill per command is
    // pure waste. std::max ordering matters: size() * 2 is 0 right after
    // construction (the constructor only reserves).
    if (new_size > target_stream.size()) {
      target_stream.resize(std::max(new_size, target_stream.size() * 2));
    }
  } else {
    target_stream.resize(new_size);
  }
  target_size = new_size;
  CommandHeader& header =
      *reinterpret_cast<CommandHeader*>(target_stream.data() + offset);
  header.command = command;
  header.arguments_size_elements = uint32_t(arguments_size_elements);
  uintmax_t* arguments =
      target_stream.data() + (offset + kCommandHeaderSizeElements);
  if (cvars::vulkan_replay_stats) {
    // The replay statistics compare whole argument blocks, padding included.
    std::memset(arguments, 0, arguments_size_elements * sizeof(uintmax_t));
  }
  return arguments;
}

void DeferredCommandBuffer::CountReplayedCommand(uint32_t command,
                                                 const void* args,
                                                 size_t args_size,
                                                 uint64_t key) {
  static_assert(size_t(Command::kCount) == kReplayStatCommandCount);
  ++replay_stats_.commands[command];
  // Only state setting can be redundant.
  switch (Command(command)) {
    case Command::kVkBindDescriptorSets:
    case Command::kVkBindIndexBuffer:
    case Command::kVkBindPipeline:
    case Command::kVkBindPipelineDeferred:
    case Command::kVkBindVertexBuffers:
    case Command::kVkPushConstants:
    case Command::kVkSetBlendConstants:
    case Command::kVkSetDepthBias:
    case Command::kVkSetDepthBias2EXT:
    case Command::kVkSetScissor:
    case Command::kVkSetStencilCompareMask:
    case Command::kVkSetStencilReference:
    case Command::kVkSetStencilWriteMask:
    case Command::kVkSetViewport:
    case Command::kVkSetCullMode:
    case Command::kVkSetFrontFace:
    case Command::kVkSetPrimitiveTopology:
    case Command::kVkSetPrimitiveRestartEnable:
    case Command::kVkSetDepthTestEnable:
    case Command::kVkSetDepthWriteEnable:
    case Command::kVkSetDepthCompareOp:
    case Command::kVkSetStencilTestEnable:
    case Command::kVkSetStencilOp:
    case Command::kVkSetDepthClampEnableEXT:
    case Command::kVkSetPolygonModeEXT:
    case Command::kVkSetColorBlendEnableEXT:
    case Command::kVkSetColorBlendEquationEXT:
    case Command::kVkSetColorWriteMaskEXT:
    case Command::kVkBindShadersEXT:
    case Command::kVkSetVertexInputEXT:
    case Command::kVkSetViewportWithCount:
    case Command::kVkSetScissorWithCount:
    case Command::kVkSetRasterizerDiscardEnable:
    case Command::kVkSetDepthBiasEnable:
    case Command::kVkSetRasterizationSamplesEXT:
    case Command::kVkSetSampleMaskEXT:
    case Command::kVkSetAlphaToCoverageEnableEXT:
    case Command::kVkSetAlphaToOneEnableEXT:
    case Command::kVkSetLogicOpEnableEXT:
    case Command::kVkSetDepthBoundsTestEnable:
      break;
    default:
      return;
  }
  std::vector<uint8_t>& last = replay_last_args_[key];
  const uint8_t* bytes = static_cast<const uint8_t*>(args);
  if (last.size() == args_size &&
      std::memcmp(last.data(), bytes, args_size) == 0) {
    ++replay_stats_.redundant[command];
    return;
  }
  last.assign(bytes, bytes + args_size);
}

void DeferredCommandBuffer::TakeReplayStats(ReplayStats& stats) {
  for (size_t i = 0; i < kReplayStatCommandCount; ++i) {
    stats.commands[i] += replay_stats_.commands[i];
    stats.redundant[i] += replay_stats_.redundant[i];
  }
  stats.descriptor_sets += replay_stats_.descriptor_sets;
  stats.push_constant_bytes += replay_stats_.push_constant_bytes;
  replay_stats_ = ReplayStats();
}

const char* DeferredCommandBuffer::GetReplayStatCommandName(size_t index) {
  static const char* const kNames[] = {
      "BeginRenderPass", "BindDescriptorSets", "BindIndexBuffer",
      "BindPipeline", "BindPipelineDeferred", "BindVertexBuffers",
      "BeginQuery", "EndQuery", "CopyQueryPoolResults", "ResetQueryPool",
      "WriteTimestamp", "ClearAttachments", "ClearColorImage", "CopyBuffer",
      "CopyBufferToImage", "FillBuffer", "BlitImage", "CopyImage", "Dispatch",
      "Draw",
      "DrawIndexed", "EndRenderPass", "BeginRendering", "EndRendering",
      "PipelineBarrier", "PushConstants", "SetBlendConstants",
      "SetRenderingInputAttachmentIndices", "SetDepthBias", "SetScissor",
      "SetStencilCompareMask", "SetStencilReference", "SetStencilWriteMask",
      "SetViewport", "SetCullMode", "SetFrontFace", "SetPrimitiveTopology",
      "SetPrimitiveRestartEnable", "SetDepthTestEnable", "SetDepthWriteEnable",
      "SetDepthCompareOp", "SetStencilTestEnable", "SetStencilOp",
      "SetDepthClampEnable", "SetPolygonMode", "SetColorBlendEnable",
      "SetColorBlendEquation", "SetColorWriteMask", "SetFragmentShadingRate",
      "BeginDebugUtilsLabel", "EndDebugUtilsLabel", "InsertDebugUtilsLabel",
      "ExecuteSegment", "SetDepthBias2EXT", "BindShadersEXT",
      "SetVertexInputEXT", "SetViewportWithCount", "SetScissorWithCount",
      "SetRasterizerDiscardEnable", "SetDepthBiasEnable",
      "SetRasterizationSamplesEXT", "SetSampleMaskEXT",
      "SetAlphaToCoverageEnableEXT", "SetAlphaToOneEnableEXT",
      "SetLogicOpEnableEXT", "SetDepthBoundsTestEnable",
  };
  static_assert(xe::countof(kNames) == kReplayStatCommandCount);
  return index < kReplayStatCommandCount ? kNames[index] : "?";
}

}  // namespace vulkan
}  // namespace gpu
}  // namespace xe
