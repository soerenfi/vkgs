#include <vkgs/engine/engine.h>

#include <atomic>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <thread>
#include <mutex>
#include <map>

#include <filesystem>
#include <fstream>

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <vk_radix_sort.h>

#include <vkgs/scene/camera.h>
#include <vkgs/util/clock.h>

#include "vkgs/engine/splat_load_thread.h"
#include "vkgs/engine/vulkan/context.h"
#include "vkgs/engine/vulkan/swapchain.h"
#include "vkgs/engine/vulkan/attachment.h"
#include "vkgs/engine/vulkan/descriptor_layout.h"
#include "vkgs/engine/vulkan/pipeline_layout.h"
#include "vkgs/engine/vulkan/compute_pipeline.h"
#include "vkgs/engine/vulkan/graphics_pipeline.h"
#include "vkgs/engine/vulkan/render_pass.h"
#include "vkgs/engine/vulkan/framebuffer.h"
#include "vkgs/engine/vulkan/descriptor.h"
#include "vkgs/engine/vulkan/buffer.h"
#include "vkgs/engine/vulkan/cpu_buffer.h"
#include "vkgs/engine/vulkan/uniform_buffer.h"
#include "vkgs/engine/vulkan/shader/uniforms.h"

#include "generated/parse_ply_comp.h"
#include "generated/projection_comp.h"
#include "generated/rank_comp.h"
#include "generated/inverse_index_comp.h"
#include "generated/splat_vert.h"
#include "generated/splat_frag.h"
#include "generated/splat_geom_vert.h"
#include "generated/splat_geom_geom.h"
#include "generated/color_vert.h"
#include "generated/color_frag.h"

namespace vkgs {
namespace {

void check_vk_result(VkResult err) {
  if (err == 0) return;
  std::cerr << "[imgui vulkan] Error: VkResult = " << err << std::endl;
  if (err < 0) abort();
}

glm::mat4 ToScaleMatrix4(float s) {
  glm::mat4 m(1.f);
  m[0][0] = s;
  m[1][1] = s;
  m[2][2] = s;
  return m;
}

glm::mat4 ToTranslationMatrix4(const glm::vec3& t) {
  glm::mat4 m(1.f);
  m[3][0] = t[0];
  m[3][1] = t[1];
  m[3][2] = t[2];
  return m;
}

struct RenderPassKey {
  VkSampleCountFlagBits samples;
  VkFormat depth_format;

  bool operator<(const RenderPassKey& rhs) const noexcept {
    return samples != rhs.samples ? samples < rhs.samples
                                  : depth_format < rhs.depth_format;
  }
};

}  // namespace

class Engine::Impl {
 public:
  static void DropCallback(GLFWwindow* window, int count, const char** paths) {
    // use first file with .ply extension
    for (int i = 0; i < count; ++i) {
      std::string path = paths[i];
      if (path.length() > 4 && path.substr(path.length() - 4) == ".ply") {
        std::cout << "loading " << path << std::endl;
        auto* impl = reinterpret_cast<Impl*>(glfwGetWindowUserPointer(window));
        impl->LoadSplats(path);
      }
    }
  }

 private:
  enum class SplatRenderMode {
    TriangleList,
    GeometryShader,
  };
  std::vector<glm::mat4> trajectory_matrices_;

 public:
  Impl() {
    if (glfwInit() == GLFW_FALSE)
      throw std::runtime_error("Failed to initialize glfw.");

    context_ = vk::Context(0);

    samples_ = VK_SAMPLE_COUNT_1_BIT;

    depth_format_ = VK_FORMAT_D32_SFLOAT;

    // render pass
    render_pass_ = vk::RenderPass(context_, samples_, depth_format_);

    {
      vk::DescriptorLayoutCreateInfo descriptor_layout_info = {};
      descriptor_layout_info.bindings.resize(1);
      descriptor_layout_info.bindings[0] = {};
      descriptor_layout_info.bindings[0].binding = 0;
      descriptor_layout_info.bindings[0].descriptor_type =
          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      descriptor_layout_info.bindings[0].stage_flags =
          VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT;
      camera_descriptor_layout_ =
          vk::DescriptorLayout(context_, descriptor_layout_info);
    }

    {
      vk::DescriptorLayoutCreateInfo descriptor_layout_info = {};
      descriptor_layout_info.bindings.resize(5);
      descriptor_layout_info.bindings[0] = {};
      descriptor_layout_info.bindings[0].binding = 0;
      descriptor_layout_info.bindings[0].descriptor_type =
          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      descriptor_layout_info.bindings[0].stage_flags =
          VK_SHADER_STAGE_COMPUTE_BIT;

      descriptor_layout_info.bindings[1] = {};
      descriptor_layout_info.bindings[1].binding = 1;
      descriptor_layout_info.bindings[1].descriptor_type =
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      descriptor_layout_info.bindings[1].stage_flags =
          VK_SHADER_STAGE_COMPUTE_BIT;

      descriptor_layout_info.bindings[2] = {};
      descriptor_layout_info.bindings[2].binding = 2;
      descriptor_layout_info.bindings[2].descriptor_type =
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      descriptor_layout_info.bindings[2].stage_flags =
          VK_SHADER_STAGE_COMPUTE_BIT;

      descriptor_layout_info.bindings[3] = {};
      descriptor_layout_info.bindings[3].binding = 3;
      descriptor_layout_info.bindings[3].descriptor_type =
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      descriptor_layout_info.bindings[3].stage_flags =
          VK_SHADER_STAGE_COMPUTE_BIT;

      descriptor_layout_info.bindings[4] = {};
      descriptor_layout_info.bindings[4].binding = 4;
      descriptor_layout_info.bindings[4].descriptor_type =
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      descriptor_layout_info.bindings[4].stage_flags =
          VK_SHADER_STAGE_COMPUTE_BIT;

      gaussian_descriptor_layout_ =
          vk::DescriptorLayout(context_, descriptor_layout_info);
    }

    {
      vk::DescriptorLayoutCreateInfo descriptor_layout_info = {};
      descriptor_layout_info.bindings.resize(6);
      descriptor_layout_info.bindings[0] = {};
      descriptor_layout_info.bindings[0].binding = 0;
      descriptor_layout_info.bindings[0].descriptor_type =
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      descriptor_layout_info.bindings[0].stage_flags =
          VK_SHADER_STAGE_COMPUTE_BIT;

      descriptor_layout_info.bindings[1] = {};
      descriptor_layout_info.bindings[1].binding = 1;
      descriptor_layout_info.bindings[1].descriptor_type =
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      descriptor_layout_info.bindings[1].stage_flags =
          VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT;

      descriptor_layout_info.bindings[2] = {};
      descriptor_layout_info.bindings[2].binding = 2;
      descriptor_layout_info.bindings[2].descriptor_type =
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      descriptor_layout_info.bindings[2].stage_flags =
          VK_SHADER_STAGE_COMPUTE_BIT;

      descriptor_layout_info.bindings[3] = {};
      descriptor_layout_info.bindings[3].binding = 3;
      descriptor_layout_info.bindings[3].descriptor_type =
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      descriptor_layout_info.bindings[3].stage_flags =
          VK_SHADER_STAGE_COMPUTE_BIT;

      descriptor_layout_info.bindings[4] = {};
      descriptor_layout_info.bindings[4].binding = 4;
      descriptor_layout_info.bindings[4].descriptor_type =
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      descriptor_layout_info.bindings[4].stage_flags =
          VK_SHADER_STAGE_COMPUTE_BIT;

      descriptor_layout_info.bindings[5] = {};
      descriptor_layout_info.bindings[5].binding = 5;
      descriptor_layout_info.bindings[5].descriptor_type =
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      descriptor_layout_info.bindings[5].stage_flags =
          VK_SHADER_STAGE_COMPUTE_BIT;

      instance_layout_ = vk::DescriptorLayout(context_, descriptor_layout_info);
    }

    {
      vk::DescriptorLayoutCreateInfo descriptor_layout_info = {};
      descriptor_layout_info.bindings.resize(1);
      descriptor_layout_info.bindings[0] = {};
      descriptor_layout_info.bindings[0].binding = 0;
      descriptor_layout_info.bindings[0].descriptor_type =
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      descriptor_layout_info.bindings[0].stage_flags =
          VK_SHADER_STAGE_COMPUTE_BIT;

      ply_descriptor_layout_ =
          vk::DescriptorLayout(context_, descriptor_layout_info);
    }

    // compute pipeline layout
    {
      vk::PipelineLayoutCreateInfo pipeline_layout_info = {};
      pipeline_layout_info.layouts = {
          camera_descriptor_layout_,
          gaussian_descriptor_layout_,
          instance_layout_,
          ply_descriptor_layout_,
      };

      pipeline_layout_info.push_constants.resize(1);
      pipeline_layout_info.push_constants[0].stageFlags =
          VK_SHADER_STAGE_COMPUTE_BIT;
      pipeline_layout_info.push_constants[0].offset = 0;
      pipeline_layout_info.push_constants[0].size = sizeof(glm::mat4);

      compute_pipeline_layout_ =
          vk::PipelineLayout(context_, pipeline_layout_info);
    }

    // graphics pipeline layout
    {
      vk::PipelineLayoutCreateInfo pipeline_layout_info = {};
      pipeline_layout_info.layouts = {camera_descriptor_layout_,
                                      instance_layout_};

      pipeline_layout_info.push_constants.resize(1);
      pipeline_layout_info.push_constants[0].stageFlags =
          VK_SHADER_STAGE_VERTEX_BIT;
      pipeline_layout_info.push_constants[0].offset = 0;
      pipeline_layout_info.push_constants[0].size = sizeof(glm::mat4);

      graphics_pipeline_layout_ =
          vk::PipelineLayout(context_, pipeline_layout_info);
    }

    // parse ply pipeline
    {
      vk::ComputePipelineCreateInfo pipeline_info = {};
      pipeline_info.layout = compute_pipeline_layout_;
      pipeline_info.source = parse_ply_comp;
      parse_ply_pipeline_ = vk::ComputePipeline(context_, pipeline_info);
    }

    // rank pipeline
    {
      vk::ComputePipelineCreateInfo pipeline_info = {};
      pipeline_info.layout = compute_pipeline_layout_;
      pipeline_info.source = rank_comp;
      rank_pipeline_ = vk::ComputePipeline(context_, pipeline_info);
    }

    // inverse index pipeline
    {
      vk::ComputePipelineCreateInfo pipeline_info = {};
      pipeline_info.layout = compute_pipeline_layout_;
      pipeline_info.source = inverse_index_comp;
      inverse_index_pipeline_ = vk::ComputePipeline(context_, pipeline_info);
    }

    // projection pipeline
    {
      vk::ComputePipelineCreateInfo pipeline_info = {};
      pipeline_info.layout = compute_pipeline_layout_;
      pipeline_info.source = projection_comp;
      projection_pipeline_ = vk::ComputePipeline(context_, pipeline_info);
    }

    // splat pipeline
    {
      std::vector<VkPipelineColorBlendAttachmentState> color_blend_attachments(
          1);
      color_blend_attachments[0] = {};
      color_blend_attachments[0].blendEnable = VK_TRUE;
      color_blend_attachments[0].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
      color_blend_attachments[0].dstColorBlendFactor =
          VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      color_blend_attachments[0].colorBlendOp = VK_BLEND_OP_ADD;
      color_blend_attachments[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
      color_blend_attachments[0].dstAlphaBlendFactor =
          VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      color_blend_attachments[0].alphaBlendOp = VK_BLEND_OP_ADD;
      color_blend_attachments[0].colorWriteMask =
          VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

      vk::GraphicsPipelineCreateInfo pipeline_info = {};
      pipeline_info.layout = graphics_pipeline_layout_;
      pipeline_info.vertex_shader = splat_vert;
      pipeline_info.fragment_shader = splat_frag;
      pipeline_info.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
      pipeline_info.depth_test = true;
      pipeline_info.depth_write = false;
      pipeline_info.color_blend_attachments =
          std::move(color_blend_attachments);

      pipeline_info.render_pass = render_pass_;
      pipeline_info.samples = samples_;
      splat_pipeline_ = vk::GraphicsPipeline(context_, pipeline_info);
    }

    // splat geom pipeline
    if (context_.geometry_shader_available()) {
      std::vector<VkPipelineColorBlendAttachmentState> color_blend_attachments(
          1);
      color_blend_attachments[0] = {};
      color_blend_attachments[0].blendEnable = VK_TRUE;
      color_blend_attachments[0].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
      color_blend_attachments[0].dstColorBlendFactor =
          VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      color_blend_attachments[0].colorBlendOp = VK_BLEND_OP_ADD;
      color_blend_attachments[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
      color_blend_attachments[0].dstAlphaBlendFactor =
          VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      color_blend_attachments[0].alphaBlendOp = VK_BLEND_OP_ADD;
      color_blend_attachments[0].colorWriteMask =
          VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

      std::vector<VkVertexInputBindingDescription> input_bindings(1);
      input_bindings[0].binding = 0;
      input_bindings[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
      input_bindings[0].stride = sizeof(float) * 10;

      std::vector<VkVertexInputAttributeDescription> input_attributes(3);
      input_attributes[0].location = 0;
      input_attributes[0].binding = 0;
      input_attributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
      input_attributes[0].offset = 0;

      input_attributes[1].location = 1;
      input_attributes[1].binding = 0;
      input_attributes[1].format = VK_FORMAT_R32G32B32_SFLOAT;
      input_attributes[1].offset = sizeof(float) * 3;

      input_attributes[2].location = 2;
      input_attributes[2].binding = 0;
      input_attributes[2].format = VK_FORMAT_R32G32B32A32_SFLOAT;
      input_attributes[2].offset = sizeof(float) * 6;

      vk::GraphicsPipelineCreateInfo pipeline_info = {};
      pipeline_info.layout = graphics_pipeline_layout_;
      pipeline_info.vertex_shader = splat_geom_vert;
      pipeline_info.geometry_shader = splat_geom_geom;
      pipeline_info.fragment_shader = splat_frag;
      pipeline_info.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
      pipeline_info.depth_test = true;
      pipeline_info.depth_write = false;
      pipeline_info.input_bindings = std::move(input_bindings);
      pipeline_info.input_attributes = std::move(input_attributes);
      pipeline_info.color_blend_attachments =
          std::move(color_blend_attachments);

      pipeline_info.render_pass = render_pass_;
      pipeline_info.samples = samples_;
      splat_geom_pipeline_ = vk::GraphicsPipeline(context_, pipeline_info);
    }

    // color pipeline
    {
      std::vector<VkVertexInputBindingDescription> input_bindings(2);
      // xyz
      input_bindings[0].binding = 0;
      input_bindings[0].stride = sizeof(float) * 3;
      input_bindings[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

      // rgba
      input_bindings[1].binding = 1;
      input_bindings[1].stride = sizeof(float) * 4;
      input_bindings[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

      std::vector<VkVertexInputAttributeDescription> input_attributes(2);
      // xyz
      input_attributes[0].location = 0;
      input_attributes[0].binding = 0;
      input_attributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
      input_attributes[0].offset = 0;

      // rgba
      input_attributes[1].location = 1;
      input_attributes[1].binding = 1;
      input_attributes[1].format = VK_FORMAT_R32G32B32A32_SFLOAT;
      input_attributes[1].offset = 0;

      std::vector<VkPipelineColorBlendAttachmentState> color_blend_attachments(
          1);
      color_blend_attachments[0] = {};
      color_blend_attachments[0].blendEnable = VK_TRUE;
      color_blend_attachments[0].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
      color_blend_attachments[0].dstColorBlendFactor =
          VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      color_blend_attachments[0].colorBlendOp = VK_BLEND_OP_ADD;
      color_blend_attachments[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
      color_blend_attachments[0].dstAlphaBlendFactor =
          VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      color_blend_attachments[0].alphaBlendOp = VK_BLEND_OP_ADD;
      color_blend_attachments[0].colorWriteMask =
          VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

      vk::GraphicsPipelineCreateInfo pipeline_info = {};
      pipeline_info.layout = graphics_pipeline_layout_;
      pipeline_info.vertex_shader = color_vert;
      pipeline_info.fragment_shader = color_frag;
      pipeline_info.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
      pipeline_info.input_bindings = std::move(input_bindings);
      pipeline_info.input_attributes = std::move(input_attributes);
      pipeline_info.depth_test = true;
      pipeline_info.depth_write = true;
      pipeline_info.color_blend_attachments =
          std::move(color_blend_attachments);

      pipeline_info.render_pass = render_pass_;
      pipeline_info.samples = samples_;
      color_line_pipeline_ = vk::GraphicsPipeline(context_, pipeline_info);
    }

    // uniforms and descriptors
    camera_buffer_ = vk::UniformBuffer<vk::shader::Camera>(context_, 2);
    visible_point_count_cpu_buffer_ =
        vk::CpuBuffer(context_, 2 * sizeof(uint32_t));
    descriptors_.resize(2);
    for (int i = 0; i < 2; ++i) {
      descriptors_[i].camera =
          vk::Descriptor(context_, camera_descriptor_layout_);
      descriptors_[i].camera.Update(0, camera_buffer_, camera_buffer_.offset(i),
                                    camera_buffer_.element_size());

      descriptors_[i].gaussian =
          vk::Descriptor(context_, gaussian_descriptor_layout_);
      descriptors_[i].splat_instance =
          vk::Descriptor(context_, instance_layout_);
      descriptors_[i].ply = vk::Descriptor(context_, ply_descriptor_layout_);
    }

    splat_info_buffer_ = vk::UniformBuffer<vk::shader::SplatInfo>(context_, 2);
    splat_visible_point_count_ = vk::Buffer(
        context_, sizeof(uint32_t),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    splat_draw_indirect_ = vk::Buffer(context_, 12 * sizeof(uint32_t),
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                          VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);

    // commands and synchronizations
    draw_command_buffers_.resize(3);
    VkCommandBufferAllocateInfo command_buffer_info = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    command_buffer_info.commandPool = context_.command_pool();
    command_buffer_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_buffer_info.commandBufferCount = draw_command_buffers_.size();
    vkAllocateCommandBuffers(context_.device(), &command_buffer_info,
                             draw_command_buffers_.data());

    VkSemaphoreCreateInfo semaphore_info = {
        VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkFenceCreateInfo fence_info = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    render_finished_semaphores_.resize(2);
    render_finished_fences_.resize(2);
    for (int i = 0; i < 2; ++i) {
      // vkCreateSemaphore(context_.device(), &semaphore_info, NULL,
      //                   &render_finished_semaphores_[i]);
      vkCreateFence(context_.device(), &fence_info, NULL,
                    &render_finished_fences_[i]);
    }

    image_acquired_semaphores_.resize(3);
    for (int i = 0; i < 3; ++i) {
      vkCreateSemaphore(context_.device(), &semaphore_info, NULL,
                        &image_acquired_semaphores_[i]);
    }

    {
      VkSemaphoreTypeCreateInfo semaphore_type_info = {
          VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
      semaphore_type_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
      VkSemaphoreCreateInfo semaphore_info = {
          VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
      semaphore_info.pNext = &semaphore_type_info;
      vkCreateSemaphore(context_.device(), &semaphore_info, NULL,
                        &transfer_semaphore_);
    }

    // create query pools
    timestamp_query_pools_.resize(2);
    for (int i = 0; i < 2; ++i) {
      VkQueryPoolCreateInfo query_pool_info = {
          VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
      query_pool_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
      query_pool_info.queryCount = timestamp_count_;
      vkCreateQueryPool(context_.device(), &query_pool_info, NULL,
                        &timestamp_query_pools_[i]);
    }

    // frame info
    frame_infos_.resize(2);

    // preallocate splat storage
    {
      splat_storage_.position =
          vk::Buffer(context_, MAX_SPLAT_COUNT * 3 * sizeof(float),
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
      splat_storage_.cov3d =
          vk::Buffer(context_, MAX_SPLAT_COUNT * 6 * sizeof(float),
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
      splat_storage_.opacity =
          vk::Buffer(context_, MAX_SPLAT_COUNT * sizeof(float),
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
      splat_storage_.sh =
          vk::Buffer(context_, MAX_SPLAT_COUNT * 48 * sizeof(float),
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

      splat_storage_.key =
          vk::Buffer(context_, MAX_SPLAT_COUNT * sizeof(uint32_t),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
      splat_storage_.index =
          vk::Buffer(context_, MAX_SPLAT_COUNT * sizeof(uint32_t),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
      splat_storage_.inverse_index =
          vk::Buffer(context_, MAX_SPLAT_COUNT * sizeof(uint32_t),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT);

      splat_storage_.instance =
          vk::Buffer(context_, MAX_SPLAT_COUNT * 10 * sizeof(float),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                         VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    }

    {
      // create splat load thread
      splat_load_thread_ = SplatLoadThread(context_);
    }

    {
      // create sorter
      VrdxSorterCreateInfo sorter_info = {};
      sorter_info.physicalDevice = context_.physical_device();
      sorter_info.device = context_.device();
      sorter_info.pipelineCache = context_.pipeline_cache();
      vrdxCreateSorter(&sorter_info, &sorter_);

      // preallocate sorter storage
      VrdxSorterStorageRequirements requirements;
      vrdxGetSorterKeyValueStorageRequirements(sorter_, MAX_SPLAT_COUNT,
                                               &requirements);
      sort_storage_ =
          vk::Buffer(context_, requirements.size, requirements.usage);
    }

    PreparePrimitives();

    // // Setup Dear ImGui
    // IMGUI_CHECKVERSION();
    // ImGui::CreateContext();
    // ImGui::StyleColorsDark();
    //     ImGui::GetIO().ConfigWindowsMoveFromTitleBarOnly = (m_pCaptureParam
    //     != nullptr);
    // ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    // ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
  }

  ~Impl() {
    splat_load_thread_ = {};

    vkDeviceWaitIdle(context_.device());

    vrdxDestroySorter(sorter_);

    for (auto semaphore : image_acquired_semaphores_)
      vkDestroySemaphore(context_.device(), semaphore, NULL);
    for (auto semaphore : render_finished_semaphores_)
      vkDestroySemaphore(context_.device(), semaphore, NULL);
    for (auto fence : render_finished_fences_)
      vkDestroyFence(context_.device(), fence, NULL);
    vkDestroySemaphore(context_.device(), transfer_semaphore_, NULL);

    for (auto query_pool : timestamp_query_pools_)
      vkDestroyQueryPool(context_.device(), query_pool, NULL);
    // ImGui::DestroyContext();

    glfwTerminate();
  }

  void LoadSplats(const std::string& ply_filepath) {
    splat_load_thread_.Cancel();
    splat_load_thread_.Start(ply_filepath);
  }

  void LoadSplatsAsync(const std::string& ply_filepath) {
    std::unique_lock<std::mutex> guard{mutex_};
    pending_ply_filepath_ = ply_filepath;
  }

  void LoadTrajectory(const std::string& trajectory_path) {
    namespace fs = std::filesystem;
    std::map<int, fs::path> sorted_files;
    for (const auto& entry : fs::directory_iterator(trajectory_path)) {
      if (entry.path().extension() == ".txt") {
        std::string filename = entry.path().stem().string();
        int file_number = std::stoi(filename);
        sorted_files[file_number] = entry.path();
      }
    }

    for (const auto& [file_number, path] : sorted_files) {
      std::ifstream file(path);
      if (file.is_open()) {
        std::string line;
        glm::mat4 matrix(1.0f);
        for (int i = 0; i < 4; ++i) {
          if (std::getline(file, line)) {
            std::istringstream iss(line);
            iss >> matrix[i][0] >> matrix[i][1] >> matrix[i][2] >> matrix[i][3];
          }
        }
        trajectory_matrices_.push_back(matrix);
        std::cout << "Loaded matrix from " << path << ":" << std::endl;
        for (int i = 0; i < 4; ++i) {
          for (int j = 0; j < 4; ++j) {
            std::cout << matrix[i][j] << " ";
          }
          std::cout << std::endl;
        }
        file.close();
      }
    }
  }

  void TransitionImageLayout(VkCommandBuffer command_buffer, VkImage image,
                             VkImageLayout old_layout,
                             VkImageLayout new_layout) {
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = old_layout;
    barrier.newLayout = new_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkPipelineStageFlags source_stage;
    VkPipelineStageFlags destination_stage;

    if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED &&
        new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
      barrier.srcAccessMask = 0;
      barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

      source_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
      destination_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
               new_layout == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR) {
      barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;

      source_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
      destination_stage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    } else if (old_layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL &&
               new_layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
      barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

      source_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
      destination_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else {
      throw std::invalid_argument("unsupported layout transition!");
    }

    vkCmdPipelineBarrier(command_buffer, source_stage, destination_stage, 0, 0,
                         nullptr, 0, nullptr, 1, &barrier);
  }
  void SetupVulkanWindow(ImGui_ImplVulkanH_Window* wd, VkSurfaceKHR surface,
                         int width, int height) {
    wd->Surface = surface;

    // Check for WSI support
    VkBool32 res;
    vkGetPhysicalDeviceSurfaceSupportKHR(context_.physical_device(),
                                         context_.graphics_queue_family_index(),
                                         wd->Surface, &res);
    if (res != VK_TRUE) {
      fprintf(stderr, "Error no WSI support on physical device 0\n");
      exit(-1);
    }

    // Select Surface Format
    const VkFormat requestSurfaceImageFormat[] = {
        VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM,
        VK_FORMAT_B8G8R8_UNORM, VK_FORMAT_R8G8B8_UNORM};
    const VkColorSpaceKHR requestSurfaceColorSpace =
        VK_COLORSPACE_SRGB_NONLINEAR_KHR;
    wd->SurfaceFormat = ImGui_ImplVulkanH_SelectSurfaceFormat(
        context_.physical_device(), wd->Surface, requestSurfaceImageFormat,
        (size_t)IM_ARRAYSIZE(requestSurfaceImageFormat),
        requestSurfaceColorSpace);

    // Select Present Mode
#ifdef APP_USE_UNLIMITED_FRAME_RATE
    VkPresentModeKHR present_modes[] = {VK_PRESENT_MODE_MAILBOX_KHR,
                                        VK_PRESENT_MODE_IMMEDIATE_KHR,
                                        VK_PRESENT_MODE_FIFO_KHR};
#else
    VkPresentModeKHR present_modes[] = {VK_PRESENT_MODE_FIFO_KHR};
#endif
    wd->PresentMode = ImGui_ImplVulkanH_SelectPresentMode(
        context_.physical_device(), wd->Surface, &present_modes[0],
        IM_ARRAYSIZE(present_modes));
    // printf("[vulkan] Selected PresentMode = %d\n", wd->PresentMode);

    // Create SwapChain, RenderPass, Framebuffer, etc.
    ImGui_ImplVulkanH_CreateOrResizeWindow(
        context_.instance(), context_.physical_device(), context_.device(), wd,
        context_.graphics_queue_family_index(), NULL, width, height, 3);
  }

  void Run() {
    // create window
    width_ = 1920;
    height_ = 1080;
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    window_ = glfwCreateWindow(width_, height_, "vkgs", NULL, NULL);

    // ImVector<const char*> extensions;
    // uint32_t extensions_count = 0;
    // const char** glfw_extensions =
    // glfwGetRequiredInstanceExtensions(&extensions_count); for (uint32_t i =
    // 0; i < extensions_count; i++)
    //     extensions.push_back(glfw_extensions[i]);
    // SetupVulkan(extensions);

    // Create Window Surface
    VkSurfaceKHR surface;
    VkResult err =
        glfwCreateWindowSurface(context_.instance(), window_, NULL, &surface);
    check_vk_result(err);

    // Create Framebuffers
    int w, h;
    glfwGetFramebufferSize(window_, &w, &h);
    ImGui_ImplVulkanH_Window* wd = &main_window_;
    SetupVulkanWindow(wd, surface, w, h);

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;
    // ImGui::GetIO().ConfigWindowsMoveFromTitleBarOnly = (m_pCaptureParam !=
    // nullptr);
    ImGui::GetIO().ConfigWindowsMoveFromTitleBarOnly = true;
    ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    // io.ConfigFlags |=
    //     ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
    // io.ConfigFlags |=
    //     ImGuiConfigFlags_NavEnableGamepad;  // Enable Gamepad Controls
    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // file drop callback
    glfwSetWindowUserPointer(window_, this);
    glfwSetDropCallback(window_, DropCallback);

    ImGui_ImplGlfw_InitForVulkan(window_, true);
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = context_.instance();
    init_info.PhysicalDevice = context_.physical_device();
    init_info.Device = context_.device();
    init_info.QueueFamily = context_.graphics_queue_family_index();
    init_info.Queue = context_.graphics_queue();
    init_info.PipelineCache = VK_NULL_HANDLE;
    init_info.DescriptorPool = context_.descriptor_pool();
    init_info.Subpass = 0;
    init_info.MinImageCount = 3;
    init_info.ImageCount = wd->ImageCount;
    init_info.RenderPass = wd->RenderPass;
    init_info.MSAASamples = samples_;
    init_info.Allocator = VK_NULL_HANDLE;
    init_info.CheckVkResultFn = check_vk_result;
    ImGui_ImplVulkan_Init(&init_info);

    // Upload fonts.
    // {
    //     // Use any command queue
    //     vk::CommandPool command_pool =
    //     wd->Frames[wd->FrameIndex].CommandPool; vk::CommandBuffer
    //     command_buffer = wd->Frames[wd->FrameIndex].CommandBuffer;
    //     VC->Device->resetCommandPool(command_pool,
    //     vk::CommandPoolResetFlags());
    //     command_buffer.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    //     ImGui_ImplVulkan_CreateFontsTexture(command_buffer);
    //     command_buffer.end();

    //     vk::SubmitInfo submit;
    //     submit.setCommandBuffers(command_buffer);
    //     VC->Queue.submit(submit);
    //     VC->Device->waitIdle();
    //     ImGui_ImplVulkan_DestroyFontUploadObjects();
    // }

    // swapchain_ = vk::Swapchain(context_, surface);

    createFramebuffer();

    glfwShowWindow(window_);
    terminate_ = false;

    // main loop
    while (!glfwWindowShouldClose(window_) && !terminate_) {
      auto frame_start_time = std::chrono::high_resolution_clock::now();
      glfwPollEvents();

      // load pending file from async request
      {
        std::unique_lock<std::mutex> guard{mutex_};
        if (!pending_ply_filepath_.empty()) {
          LoadSplats(pending_ply_filepath_);
          pending_ply_filepath_.clear();
        }
      }

      int width, height;
      glfwGetFramebufferSize(window_, &width, &height);
      camera_.SetWindowSize(width, height);

      if (follow_trajectory_ && frame_counter_ % 100 == 0 &&
          !trajectory_matrices_.empty()) {
        static size_t trajectory_index = 0;
        camera_.SetPosition(trajectory_matrices_[trajectory_index]);
        trajectory_index = (trajectory_index + 1) % trajectory_matrices_.size();
        trajectory_index++;
      }

      // if (drive_) {
      //   // Read WASD keys and use as input for driving
      //   float steering_angle = 0.0f;
      //   float velocity = 0.0f;
      //   const float max_velocity = 10.0f;  // Maximum velocity
      //   const float acceleration = 5.0f;   // Acceleration rate
      //   const float deceleration = 5.0f;   // Deceleration rate
      //   const float steering_rate =
      //       glm::radians(30.0f);  // Steering rate in radians per second

      //   if (ImGui::IsKeyDown(ImGuiKey_W)) {
      //     velocity += acceleration * io.DeltaTime;
      //   }
      //   if (ImGui::IsKeyDown(ImGuiKey_S)) {
      //     velocity -= deceleration * io.DeltaTime;
      //   }
      //   if (ImGui::IsKeyDown(ImGuiKey_A)) {
      //     steering_angle += steering_rate * io.DeltaTime;
      //   }
      //   if (ImGui::IsKeyDown(ImGuiKey_D)) {
      //     steering_angle -= steering_rate * io.DeltaTime;
      //   }
      //   float delta_time = 0.01;
      //   // Clamp the velocity to the maximum allowed value
      //   velocity = glm::clamp(velocity, -max_velocity, max_velocity);

      //   // Vehicle parameters
      //   const float wheelbase = 2.5f;  // Distance between front and rear
      //   axles const float max_steering_angle =
      //       glm::radians(30.0f);  // Maximum steering angle in radians

      //   // Clamp the steering angle to the maximum allowed value
      //   steering_angle =
      //       glm::clamp(steering_angle, -max_steering_angle,
      //       max_steering_angle);

      //   // Calculate the turning radius
      //   float turning_radius = wheelbase / glm::tan(steering_angle);

      //   // Calculate the angular velocity
      //   float angular_velocity = velocity / turning_radius;

      //   // Update the vehicle's position and orientation
      //   glm::vec3 position = camera_.Eye();
      //   glm::quat orientation = camera_.Orientation();

      //   // Calculate the change in orientation
      //   glm::quat delta_orientation = glm::angleAxis(
      //       angular_velocity * delta_time, glm::vec3(0.0f, 0.0f, 1.0f));

      //   // Update the orientation
      //   orientation = delta_orientation * orientation;

      //   // Calculate the forward direction
      //   glm::vec3 forward = orientation * glm::vec3(0.0f, 1.0f, 0.0f);

      //   // Update the position
      //   position += forward * velocity * delta_time;

      //   // Set the new position and orientation
      //   camera_.SetPosition(position);
      //   camera_.SetOrientation(orientation);

      RenderFrame();
      RenderUI();

      // Present();
      // Calculate frame duration and sleep if necessary to limit to 30 FPS
      // auto frame_end_time = std::chrono::high_resolution_clock::now();
      // std::chrono::duration<double, std::milli> frame_duration =
      //     frame_end_time - frame_start_time;
      // double frame_time_ms = frame_duration.count();
      // double target_frame_time_ms = 1000.0 / 30.0;  // 30 FPS target

      // if (frame_time_ms < target_frame_time_ms) {
      //   std::this_thread::sleep_for(std::chrono::milliseconds(
      //       static_cast<int>(target_frame_time_ms - frame_time_ms)));
      // }
    }

    vkDeviceWaitIdle(context_.device());

    glfwDestroyWindow(window_);

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();

    terminate_ = false;
  }

  void Close() { terminate_ = true; }

 private:
  void PreparePrimitives() {
    std::vector<uint32_t> splat_index;
    splat_index.reserve(MAX_SPLAT_COUNT * 6);
    for (int i = 0; i < MAX_SPLAT_COUNT; ++i) {
      splat_index.push_back(4 * i + 0);
      splat_index.push_back(4 * i + 1);
      splat_index.push_back(4 * i + 2);
      splat_index.push_back(4 * i + 2);
      splat_index.push_back(4 * i + 1);
      splat_index.push_back(4 * i + 3);
    }

    std::vector<float> axis_position = {
        0.f, 0.f, 0.f, 1.f, 0.f, 0.f,  // x
        0.f, 0.f, 0.f, 0.f, 1.f, 0.f,  // y
        0.f, 0.f, 0.f, 0.f, 0.f, 1.f,  // z
    };
    std::vector<float> axis_color = {
        1.f, 0.f, 0.f, 1.f, 1.f, 0.f, 0.f, 1.f,  // x
        0.f, 1.f, 0.f, 1.f, 0.f, 1.f, 0.f, 1.f,  // y
        0.f, 0.f, 1.f, 1.f, 0.f, 0.f, 1.f, 1.f,  // z
    };
    std::vector<uint32_t> axis_index = {
        0, 1, 2, 3, 4, 5,
    };

    std::vector<float> grid_position;
    std::vector<float> grid_color;
    std::vector<uint32_t> grid_index;
    constexpr int grid_size = 10;
    for (int i = 0; i < grid_size * 2 + 1; ++i) {
      grid_index.push_back(4 * i + 0);
      grid_index.push_back(4 * i + 1);
      grid_index.push_back(4 * i + 2);
      grid_index.push_back(4 * i + 3);
    }
    for (int i = -grid_size; i <= grid_size; ++i) {
      float t = static_cast<float>(i) / grid_size;
      // old:  (x=-1, y=0, z=t), (x=1, y=0, z=t), (x=t, y=0, z=-1), (x=t, y=0,
      // z=1) new:  (x=-1, y=t, z=0), (x=1, y=t, z=0), (x=t, y=-1, z=0), (x=t,
      // y=1, z=0)

      // horizontal line in X direction at Y=t, Z=0
      grid_position.push_back(-1.f);
      grid_position.push_back(t);
      grid_position.push_back(0.f);
      grid_color.push_back(0.5f);
      grid_color.push_back(0.5f);
      grid_color.push_back(0.5f);
      grid_color.push_back(1.f);

      grid_position.push_back(1.f);
      grid_position.push_back(t);
      grid_position.push_back(0.f);
      grid_color.push_back(0.5f);
      grid_color.push_back(0.5f);
      grid_color.push_back(0.5f);
      grid_color.push_back(1.f);

      // vertical line in Y direction at X=t, Z=0
      grid_position.push_back(t);
      grid_position.push_back(-1.f);
      grid_position.push_back(0.f);
      grid_color.push_back(0.5f);
      grid_color.push_back(0.5f);
      grid_color.push_back(0.5f);
      grid_color.push_back(1.f);

      grid_position.push_back(t);
      grid_position.push_back(1.f);
      grid_position.push_back(0.f);
      grid_color.push_back(0.5f);
      grid_color.push_back(0.5f);
      grid_color.push_back(0.5f);
      grid_color.push_back(1.f);
    }
    splat_index_buffer_ = vk::Buffer(
        context_, splat_index.size() * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

    axis_.position_buffer = vk::Buffer(
        context_, axis_position.size() * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    axis_.color_buffer = vk::Buffer(
        context_, axis_color.size() * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    axis_.index_buffer = vk::Buffer(
        context_, axis_index.size() * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

    grid_.position_buffer = vk::Buffer(
        context_, grid_position.size() * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    grid_.color_buffer = vk::Buffer(
        context_, grid_color.size() * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    grid_.index_buffer = vk::Buffer(
        context_, grid_index.size() * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

    VkCommandBufferAllocateInfo command_buffer_info = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    command_buffer_info.commandPool = context_.command_pool();
    command_buffer_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_buffer_info.commandBufferCount = 1;
    VkCommandBuffer cb;
    vkAllocateCommandBuffers(context_.device(), &command_buffer_info, &cb);

    VkCommandBufferBeginInfo begin_info = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &begin_info);

    splat_index_buffer_.FromCpu(cb, splat_index);

    axis_.position_buffer.FromCpu(cb, axis_position);
    axis_.color_buffer.FromCpu(cb, axis_color);
    axis_.index_buffer.FromCpu(cb, axis_index);
    axis_.index_count = axis_index.size();

    grid_.position_buffer.FromCpu(cb, grid_position);
    grid_.color_buffer.FromCpu(cb, grid_color);
    grid_.index_buffer.FromCpu(cb, grid_index);
    grid_.index_count = grid_index.size();

    vkEndCommandBuffer(cb);

    uint64_t wait_value = transfer_timeline_;
    uint64_t signal_value = transfer_timeline_ + 1;
    VkTimelineSemaphoreSubmitInfo timeline_semaphore_submit_info = {
        VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    timeline_semaphore_submit_info.waitSemaphoreValueCount = 1;
    timeline_semaphore_submit_info.pWaitSemaphoreValues = &wait_value;
    timeline_semaphore_submit_info.signalSemaphoreValueCount = 1;
    timeline_semaphore_submit_info.pSignalSemaphoreValues = &signal_value;

    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo submit_info = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit_info.pNext = &timeline_semaphore_submit_info;
    submit_info.waitSemaphoreCount = 1;
    submit_info.pWaitSemaphores = &transfer_semaphore_;
    submit_info.pWaitDstStageMask = &wait_stage;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cb;
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = &transfer_semaphore_;
    vkQueueSubmit(context_.graphics_queue(), 1, &submit_info, NULL);

    transfer_timeline_++;
  }

  void RenderFrame() {
    // // recreate swapchain if need resize
    // if (swapchain_.ShouldRecreate()) {
    // vkWaitForFences(context_.device(), render_finished_fences_.size(),
    //                 render_finished_fences_.data(), VK_TRUE, UINT64_MAX);
    //   swapchain_.Recreate();
    //   RecreateFramebuffer();
    // }

    int32_t acquire_index = frame_counter_ % 3;
    int32_t frame_index = frame_counter_ % 2;
    // VkSemaphore image_acquired_semaphore =
    //     image_acquired_semaphores_[acquire_index];
    // VkSemaphore render_finished_semaphore =
    //     render_finished_semaphores_[frame_index];
    VkFence render_finished_fence = render_finished_fences_[frame_index];
    VkCommandBuffer cb = draw_command_buffers_[frame_index];
    VkQueryPool timestamp_query_pool = timestamp_query_pools_[frame_index];
    auto& frame_info = frame_infos_[frame_index];

    uint32_t image_index;
    // if (swapchain_.AcquireNextImage(image_acquired_semaphore, &image_index))
    // {
    static glm::vec3 lt(0.f);
    static glm::vec3 gt(0.f);
    static glm::vec3 lr(0.f);
    static glm::quat lq;
    static glm::vec3 gr(0.f);
    static glm::quat gq;
    static float scale = 1.f;
    glm::mat4 model(1.f);

    bool msaa_changed = false;
    static int msaa = 0;

    bool depth_format_changed = false;
    static int depth_format = 1;

    model = ToScaleMatrix4(scale_ * scale) * glm::toMat4(gq) *
            ToTranslationMatrix4(translation_ + gt) *
            glm::toMat4(rotation_ * lq) * ToTranslationMatrix4(lt);

    // // record command buffer
    vkWaitForFences(context_.device(), 1, &render_finished_fence, VK_TRUE,
                    UINT64_MAX);
    vkResetFences(context_.device(), 1, &render_finished_fence);

    // // get timestamps
    // if (frame_info.drew_splats) {
    //   std::cout << "Rendering 3fence " << frame_counter_ << std::endl;

    //   std::vector<uint64_t> timestamps(timestamp_count_);
    //   vkGetQueryPoolResults(
    //       context_.device(), timestamp_query_pool, 0, timestamps.size(),
    //       timestamps.size() * sizeof(uint64_t), timestamps.data(),
    //       sizeof(uint64_t), VK_QUERY_RESULT_64_BIT |
    //       VK_QUERY_RESULT_WAIT_BIT);

    // frame_info.rank_time = timestamps[2] - timestamps[1];
    // frame_info.sort_time = timestamps[4] - timestamps[3];
    // frame_info.inverse_time = timestamps[6] - timestamps[5];
    // frame_info.projection_time = timestamps[8] - timestamps[7];
    // frame_info.rendering_time = timestamps[10] - timestamps[9];
    // frame_info.end_to_end_time = timestamps[11] - timestamps[0];
    // }

    camera_buffer_[frame_index].projection = camera_.ProjectionMatrix();
    camera_buffer_[frame_index].view = camera_.ViewMatrix();
    camera_buffer_[frame_index].camera_position = camera_.Eye();
    camera_buffer_[frame_index].screen_size = {camera_.width(),
                                               camera_.height()};

    VkCommandBufferBeginInfo command_begin_info = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    command_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(cb, &command_begin_info);

    vkCmdResetQueryPool(cb, timestamp_query_pool, 0, timestamp_count_);

    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        timestamp_query_pool, 0);

    // check loading status
    auto progress = splat_load_thread_.GetProgress();
    frame_info.total_point_count = progress.total_point_count;
    frame_info.loaded_point_count = progress.loaded_point_count;
    frame_info.ply_buffer = progress.ply_buffer;

    if (!progress.buffer_barriers.empty()) {
      loaded_point_count_ = progress.loaded_point_count;
    }

    // update descriptor
    descriptors_[frame_index].gaussian.Update(
        0, splat_info_buffer_, splat_info_buffer_.offset(frame_index),
        splat_info_buffer_.element_size());

    descriptors_[frame_index].splat_instance.Update(
        0, splat_draw_indirect_, 0, splat_draw_indirect_.size());

    if (loaded_point_count_ != 0) {
      descriptors_[frame_index].gaussian.Update(
          1, splat_storage_.position, 0,
          loaded_point_count_ * 3 * sizeof(float));
      descriptors_[frame_index].gaussian.Update(
          2, splat_storage_.cov3d, 0, loaded_point_count_ * 6 * sizeof(float));
      descriptors_[frame_index].gaussian.Update(
          3, splat_storage_.opacity, 0,
          loaded_point_count_ * 1 * sizeof(float));
      descriptors_[frame_index].gaussian.Update(
          4, splat_storage_.sh, 0, loaded_point_count_ * 48 * sizeof(float));

      descriptors_[frame_index].splat_instance.Update(
          1, splat_storage_.instance, 0,
          loaded_point_count_ * 10 * sizeof(float));
      descriptors_[frame_index].splat_instance.Update(
          2, splat_visible_point_count_, 0, splat_visible_point_count_.size());
      descriptors_[frame_index].splat_instance.Update(
          3, splat_storage_.key, 0, loaded_point_count_ * sizeof(uint32_t));
      descriptors_[frame_index].splat_instance.Update(
          4, splat_storage_.index, 0, loaded_point_count_ * sizeof(uint32_t));
      descriptors_[frame_index].splat_instance.Update(
          5, splat_storage_.inverse_index, 0,
          loaded_point_count_ * sizeof(uint32_t));
    }

    // update uniform buffer
    splat_info_buffer_[frame_index].point_count = loaded_point_count_;

    VkMemoryBarrier barrier;

    // acquire ownership
    // according to spec:
    //   The buffer range or image subresource range specified in an
    //   acquireoperation must match exactly that of a previous release
    //   operation.
    if (!progress.buffer_barriers.empty()) {
      std::vector<VkBufferMemoryBarrier> buffer_barriers =
          std::move(progress.buffer_barriers);

      // change src/dst synchronization scope
      for (auto& buffer_barrier : buffer_barriers) {
        buffer_barrier.srcAccessMask = 0;
        buffer_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      }

      vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL,
                           buffer_barriers.size(), buffer_barriers.data(), 0,
                           NULL);

      // parse ply file
      // TODO: make parse async
      descriptors_[frame_index].ply.Update(0, progress.ply_buffer, 0);

      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                        parse_ply_pipeline_);

      VkDescriptorSet descriptor = descriptors_[frame_index].gaussian;
      vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                              compute_pipeline_layout_, 1, 1, &descriptor, 0,
                              NULL);

      descriptor = descriptors_[frame_index].ply;
      vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                              compute_pipeline_layout_, 3, 1, &descriptor, 0,
                              NULL);

      constexpr int local_size = 256;
      vkCmdDispatch(cb, (loaded_point_count_ + local_size - 1) / local_size, 1,
                    1);

      barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier,
                           0, NULL, 0, NULL);

      // hold buffer until the end of frame
      frame_info.ply_buffer = progress.ply_buffer;
    }

    if (loaded_point_count_ != 0) {
      // rank
      {
        vkCmdFillBuffer(cb, splat_visible_point_count_, 0, sizeof(uint32_t), 0);

        barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                             &barrier, 0, NULL, 0, NULL);

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, rank_pipeline_);

        std::vector<VkDescriptorSet> descriptors = {
            descriptors_[frame_index].camera,
            descriptors_[frame_index].gaussian,
            descriptors_[frame_index].splat_instance,
        };
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                compute_pipeline_layout_, 0, descriptors.size(),
                                descriptors.data(), 0, nullptr);

        vkCmdPushConstants(cb, compute_pipeline_layout_,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(model),
                           glm::value_ptr(model));

        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                            timestamp_query_pool, 1);

        constexpr int local_size = 256;
        vkCmdDispatch(cb, (loaded_point_count_ + local_size - 1) / local_size,
                      1, 1);

        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            timestamp_query_pool, 2);
      }

      // make visiblePointCount available for next transfer commands
      {
        barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0,
                             NULL, 0, NULL);
      }

      // visible point count to CPU
      {
        VkBufferCopy region = {};
        region.srcOffset = 0;
        region.dstOffset = sizeof(uint32_t) * frame_index;
        region.size = sizeof(uint32_t);
        vkCmdCopyBuffer(cb, splat_visible_point_count_,
                        visible_point_count_cpu_buffer_, 1, &region);
      }

      // radix sort
      {
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                            timestamp_query_pool, 3);

        vrdxCmdSortKeyValueIndirect(cb, sorter_, loaded_point_count_,
                                    splat_visible_point_count_, 0,
                                    splat_storage_.key, 0, splat_storage_.index,
                                    0, sort_storage_, 0, NULL, 0);

        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            timestamp_query_pool, 4);
      }

      // inverse map
      {
        vkCmdFillBuffer(cb, splat_storage_.inverse_index, 0,
                        loaded_point_count_ * sizeof(uint32_t), -1);

        barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask =
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask =
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cb,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                             &barrier, 0, NULL, 0, NULL);

        std::vector<VkDescriptorSet> descriptors = {
            descriptors_[frame_index].camera,
            descriptors_[frame_index].gaussian,
            descriptors_[frame_index].splat_instance,
        };
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                compute_pipeline_layout_, 0, descriptors.size(),
                                descriptors.data(), 0, nullptr);

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                          inverse_index_pipeline_);

        vkCmdPushConstants(cb, compute_pipeline_layout_,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(model),
                           glm::value_ptr(model));

        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            timestamp_query_pool, 5);

        constexpr int local_size = 256;
        vkCmdDispatch(cb, (loaded_point_count_ + local_size - 1) / local_size,
                      1, 1);

        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            timestamp_query_pool, 6);
      }

      // projection
      {
        barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                             &barrier, 0, NULL, 0, NULL);

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                          projection_pipeline_);

        vkCmdPushConstants(cb, compute_pipeline_layout_,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(model),
                           glm::value_ptr(model));

        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            timestamp_query_pool, 7);

        constexpr int local_size = 256;
        vkCmdDispatch(cb, (loaded_point_count_ + local_size - 1) / local_size,
                      1, 1);

        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            timestamp_query_pool, 8);
      }

      // draw
      {
        barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT |
                                VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT |
                                 VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                             0, 1, &barrier, 0, NULL, 0, NULL);

        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            timestamp_query_pool, 9);

        // DrawNormalPass(cb, frame_index, swapchain_.width(),
        //                swapchain_.height(),
        //                swapchain_.image_view(image_index));
        DrawNormalPass(cb, frame_index, width_, height_, offscreen_image_view_);

        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,
                            timestamp_query_pool, 10);
      }
      frame_info.drew_splats = true;
    } else {
      vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          timestamp_query_pool, 9);

      DrawNormalPass(cb, frame_index, width_, height_, offscreen_image_view_);

      vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,
                          timestamp_query_pool, 10);
      frame_info.drew_splats = false;
    }
    // // Transition the offscreen image layout to
    // // // VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
    // TransitionImageLayout(cb, offscreen_image_,
    //                       VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    //                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    // // Transition the swapchain image layout to
    // // // VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
    // TransitionImageLayout(cb, swapchain_.image(image_index),
    //                       VK_IMAGE_LAYOUT_UNDEFINED,
    //                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    // // Copy the offscreen image to the swapchain image
    // VkImageCopy copy_region = {};
    // copy_region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    // copy_region.srcSubresource.mipLevel = 0;
    // copy_region.srcSubresource.baseArrayLayer = 0;
    // copy_region.srcSubresource.layerCount = 1;
    // copy_region.srcOffset = {0, 0, 0};
    // copy_region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    // copy_region.dstSubresource.mipLevel = 0;
    // copy_region.dstSubresource.baseArrayLayer = 0;
    // copy_region.dstSubresource.layerCount = 1;
    // copy_region.dstOffset = {0, 0, 0};
    // copy_region.extent.width = width_;
    // copy_region.extent.height = height_;
    // copy_region.extent.depth = 1;

    // vkCmdCopyImage(cb, offscreen_image_,
    // VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
    //                swapchain_.image(image_index),
    //                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy_region);

    // // Transition the swapchain image layout to
    // // VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
    // TransitionImageLayout(cb, swapchain_.image(image_index),
    //                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    //                       VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    // vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
    //                     timestamp_query_pool, 11);

    vkEndCommandBuffer(cb);

    std::vector<VkSemaphore> wait_semaphores = {transfer_semaphore_};
    std::vector<VkPipelineStageFlags> wait_stages = {
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT};
    std::vector<uint64_t> wait_values = {0, transfer_timeline_};

    VkTimelineSemaphoreSubmitInfo timeline_semaphore_submit_info = {
        VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    timeline_semaphore_submit_info.waitSemaphoreValueCount = wait_values.size();
    timeline_semaphore_submit_info.pWaitSemaphoreValues = wait_values.data();

    VkSubmitInfo submit_info = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit_info.pNext = &timeline_semaphore_submit_info;
    submit_info.waitSemaphoreCount = 0;
    submit_info.pWaitSemaphores = NULL;
    submit_info.pWaitDstStageMask = wait_stages.data();
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cb;
    submit_info.signalSemaphoreCount = 0;
    submit_info.pSignalSemaphores = NULL;  //&render_finished_semaphore;
    vkQueueSubmit(context_.graphics_queue(), 1, &submit_info,
                  render_finished_fence);

    // VkSwapchainKHR swapchain_handle = swapchain_;
    // VkPresentInfoKHR present_info = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    // present_info.waitSemaphoreCount = 1;
    // present_info.pWaitSemaphores = &render_finished_semaphore;
    // present_info.swapchainCount = 1;
    // present_info.pSwapchains = &swapchain_handle;
    // present_info.pImageIndices = &image_index;
    // frame_info.present_timestamp = Clock::timestamp();
    // vkQueuePresentKHR(context_.graphics_queue(), &present_info);
    frame_info.present_done_timestamp = Clock::timestamp();

    frame_counter_++;
    vkQueueWaitIdle(context_.graphics_queue());
  }

  void FrameRender(ImGui_ImplVulkanH_Window* wd, ImDrawData* draw_data) {
    VkResult err;

    VkSemaphore image_acquired_semaphore =
        wd->FrameSemaphores[wd->SemaphoreIndex].ImageAcquiredSemaphore;
    VkSemaphore render_complete_semaphore =
        wd->FrameSemaphores[wd->SemaphoreIndex].RenderCompleteSemaphore;
    err = vkAcquireNextImageKHR(context_.device(), wd->Swapchain, UINT64_MAX,
                                image_acquired_semaphore, VK_NULL_HANDLE,
                                &wd->FrameIndex);
    if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR) {
      swapchain_rebuild_ = true;

      return;
    }
    check_vk_result(err);

    ImGui_ImplVulkanH_Frame* fd = &wd->Frames[wd->FrameIndex];
    {
      err = vkWaitForFences(
          context_.device(), 1, &fd->Fence, VK_TRUE,
          UINT64_MAX);  // wait indefinitely instead of periodically checking
      check_vk_result(err);

      err = vkResetFences(context_.device(), 1, &fd->Fence);
      check_vk_result(err);
    }
    {
      err = vkResetCommandPool(context_.device(), fd->CommandPool, 0);
      check_vk_result(err);
      VkCommandBufferBeginInfo info = {};
      info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      err = vkBeginCommandBuffer(fd->CommandBuffer, &info);
      check_vk_result(err);
    }
    {
      VkRenderPassBeginInfo info = {};
      info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
      info.renderPass = wd->RenderPass;
      info.framebuffer = fd->Framebuffer;
      info.renderArea.extent.width = wd->Width;
      info.renderArea.extent.height = wd->Height;
      info.clearValueCount = 1;
      info.pClearValues = &wd->ClearValue;
      vkCmdBeginRenderPass(fd->CommandBuffer, &info,
                           VK_SUBPASS_CONTENTS_INLINE);
    }

    // Record dear imgui primitives into command buffer
    ImGui_ImplVulkan_RenderDrawData(draw_data, fd->CommandBuffer);

    // Submit command buffer
    vkCmdEndRenderPass(fd->CommandBuffer);
    {
      VkPipelineStageFlags wait_stage =
          VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
      VkSubmitInfo info = {};
      info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
      info.waitSemaphoreCount = 1;
      info.pWaitSemaphores = &image_acquired_semaphore;
      info.pWaitDstStageMask = &wait_stage;
      info.commandBufferCount = 1;
      info.pCommandBuffers = &fd->CommandBuffer;
      info.signalSemaphoreCount = 1;
      info.pSignalSemaphores = &render_complete_semaphore;

      err = vkEndCommandBuffer(fd->CommandBuffer);
      check_vk_result(err);
      err = vkQueueSubmit(context_.graphics_queue(), 1, &info, fd->Fence);
      check_vk_result(err);
    }
  }

  void FramePresent(ImGui_ImplVulkanH_Window* wd) {
    if (swapchain_rebuild_) return;
    VkSemaphore render_complete_semaphore =
        wd->FrameSemaphores[wd->SemaphoreIndex].RenderCompleteSemaphore;
    VkPresentInfoKHR info = {};
    info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    info.waitSemaphoreCount = 1;
    info.pWaitSemaphores = &render_complete_semaphore;
    info.swapchainCount = 1;
    info.pSwapchains = &wd->Swapchain;
    info.pImageIndices = &wd->FrameIndex;
    VkResult err = vkQueuePresentKHR(context_.graphics_queue(), &info);
    if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR) {
      swapchain_rebuild_ = true;
      return;
    }
    check_vk_result(err);
    wd->SemaphoreIndex =
        (wd->SemaphoreIndex + 1) %
        wd->SemaphoreCount;  // Now we can use the next set of semaphores
  }

  void RenderUI() {
    int fb_width, fb_height;
    glfwGetFramebufferSize(window_, &fb_width, &fb_height);
    if (fb_width > 0 && fb_height > 0 &&
        (swapchain_rebuild_ || main_window_.Width != fb_width ||
         main_window_.Height != fb_height)) {
      ImGui_ImplVulkan_SetMinImageCount(3);
      ImGui_ImplVulkanH_CreateOrResizeWindow(
          context_.instance(), context_.physical_device(), context_.device(),
          &main_window_, context_.graphics_queue_family_index(), NULL, width_,
          height_, 3);
      main_window_.FrameIndex = 0;
      swapchain_rebuild_ = false;
    }
    if (glfwGetWindowAttrib(window_, GLFW_ICONIFIED) != 0) {
      ImGui_ImplGlfw_Sleep(10);
      return;
    }

    // Start the Dear ImGui frame
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    const auto& io = ImGui::GetIO();
    if (!io.WantCaptureKeyboard) {
      constexpr float speed = 1000.f;
      float dt = io.DeltaTime;
      if (ImGui::IsKeyDown(ImGuiKey_W)) {
        camera_.Translate(0.f, 0.f, speed * dt);
      }
      if (ImGui::IsKeyDown(ImGuiKey_S)) {
        camera_.Translate(0.f, 0.f, -speed * dt);
      }
      if (ImGui::IsKeyDown(ImGuiKey_A)) {
        camera_.Translate(-speed * dt, 0.f);
      }
      if (ImGui::IsKeyDown(ImGuiKey_D)) {
        camera_.Translate(speed * dt, 0.f);
      }
      if (ImGui::IsKeyDown(ImGuiKey_Space)) {
        camera_.Translate(0.f, speed * dt);
      }
    }
    if (ImGui::BeginMainMenuBar()) {
      if (ImGui::BeginMenu("File")) {
        if (ImGui::MenuItem("Add")) {
        }
        if (ImGui::MenuItem("Open", "Ctrl+O")) {
        }
        if (ImGui::MenuItem("Load Trajectory")) {
          // loadTrajectory();
        }
        if (ImGui::MenuItem("Save", "Ctrl+S")) {
        }
        if (ImGui::MenuItem("Save as..")) {
        }
        ImGui::EndMenu();
      }
      ImGui::EndMainMenuBar();
    }
    // ImGui::DockSpaceOverViewport();
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGuiID dockspace_id = ImGui::GetID("MyDockspace");
    ImGui::DockSpaceOverViewport(dockspace_id, viewport,
                                 ImGuiDockNodeFlags_PassthruCentralNode);
    {
      ImGui::Begin("Hello, world!");  // Create a window called "Hello, world!"
                                      // and append into it.

      ImGui::Text("This is some useful text.");  // Display some text (you can
                                                 // use a format strings too)

      ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
                  1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
      ImGui::End();
    }
    {
      // // Set initial docking position
      // ImGuiID dockspace_id = ImGui::GetID("MyDockspace");
      // ImGui::DockSpace(dockspace_id);

      // ImGui::SetNextWindowDockID(dockspace_id, ImGuiCond_FirstUseEver);
      ImGui::Begin("Viewport");

      // handle events
      if (!io.WantCaptureMouse) {
        bool left = io.MouseDown[ImGuiMouseButton_Left];
        bool right = io.MouseDown[ImGuiMouseButton_Right];
        float dx = io.MouseDelta.x;
        float dy = io.MouseDelta.y;

        if (left && !right) {
          camera_.Rotate(dx, dy);
        } else if (!left && right) {
          camera_.Translate(dx, dy);
        } else if (left && right) {
          camera_.Zoom(dy);
        }

        if (io.MouseWheel != 0.f) {
          if (ImGui::IsKeyDown(ImGuiKey_LeftCtrl)) {
            camera_.DollyZoom(io.MouseWheel);
          } else {
            camera_.Zoom(io.MouseWheel * 10.f);
          }
        }
      }

      ImVec2 viewportPanelSize = ImGui::GetContentRegionAvail();
      float aspectRatio = static_cast<float>(width_) / height_;
      float newWidth = viewportPanelSize.x;
      float newHeight = viewportPanelSize.x / aspectRatio;

      if (newHeight > viewportPanelSize.y) {
        newHeight = viewportPanelSize.y;
        newWidth = viewportPanelSize.y * aspectRatio;
      }

      ImVec2 offset = {(viewportPanelSize.x - newWidth) * 0.5f,
                       (viewportPanelSize.y - newHeight) * 0.5f};
      ImGui::SetCursorPosX(ImGui::GetCursorPosX() + offset.x);
      ImGui::SetCursorPosY(ImGui::GetCursorPosY() + offset.y);

      ImGui::Image((ImTextureID)(intptr_t)offscreen_descriptor_set_,
                   ImVec2{newWidth, newHeight});
      ImGui::End();
    }

    ImGui::Render();
    ImDrawData* draw_data = ImGui::GetDrawData();
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    const bool is_minimized =
        (draw_data->DisplaySize.x <= 0.0f || draw_data->DisplaySize.y <= 0.0f);
    if (!is_minimized) {
      main_window_.ClearValue.color.float32[0] = clear_color.x * clear_color.w;
      main_window_.ClearValue.color.float32[1] = clear_color.y * clear_color.w;
      main_window_.ClearValue.color.float32[2] = clear_color.z * clear_color.w;
      main_window_.ClearValue.color.float32[3] = clear_color.w;
      FrameRender(&main_window_, draw_data);
      FramePresent(&main_window_);
    }
  }

  void DrawNormalPass(VkCommandBuffer cb, uint32_t frame_index, uint32_t width,
                      uint32_t height, VkImageView target_image_view) {
    if (target_image_view == nullptr) {
      throw std::runtime_error("target_image_view is null!");
    }
    std::vector<VkClearValue> clear_values(2);
    clear_values[0].color.float32[0] = 0.0f;
    clear_values[0].color.float32[1] = 0.0f;
    clear_values[0].color.float32[2] = 0.0f;
    clear_values[0].color.float32[3] = 1.f;
    clear_values[1].depthStencil.depth = 1.f;

    std::vector<VkImageView> render_pass_attachments;

    if (samples_ == VK_SAMPLE_COUNT_1_BIT) {
      render_pass_attachments = {
          target_image_view,
          depth_attachment_,
      };
    } else {
      render_pass_attachments = {
          color_attachment_,
          depth_attachment_,
          target_image_view,
      };
    }

    VkRenderPassAttachmentBeginInfo render_pass_attachments_info = {
        VK_STRUCTURE_TYPE_RENDER_PASS_ATTACHMENT_BEGIN_INFO};
    VkRenderPassBeginInfo render_pass_begin_info = {
        VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    render_pass_begin_info.pNext = &render_pass_attachments_info;
    render_pass_begin_info.framebuffer = framebuffer_;
    render_pass_begin_info.renderArea.offset = {0, 0};
    render_pass_begin_info.renderArea.extent = {width, height};
    render_pass_begin_info.clearValueCount = clear_values.size();
    render_pass_begin_info.pClearValues = clear_values.data();
    render_pass_begin_info.renderPass = render_pass_;
    render_pass_attachments_info.attachmentCount =
        render_pass_attachments.size();
    render_pass_attachments_info.pAttachments = render_pass_attachments.data();

    vkCmdBeginRenderPass(cb, &render_pass_begin_info,
                         VK_SUBPASS_CONTENTS_INLINE);

    VkViewport viewport = {};
    viewport.x = 0.f;
    viewport.y = 0.f;
    viewport.width = static_cast<float>(width);
    viewport.height = static_cast<float>(height);
    viewport.minDepth = 0.f;
    viewport.maxDepth = 1.f;
    vkCmdSetViewport(cb, 0, 1, &viewport);

    VkRect2D scissor = {};
    scissor.offset = {0, 0};
    scissor.extent = {width, height};
    vkCmdSetScissor(cb, 0, 1, &scissor);

    std::vector<VkDescriptorSet> descriptors = {
        descriptors_[frame_index].camera,
        descriptors_[frame_index].splat_instance,
    };
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            graphics_pipeline_layout_, 0, descriptors.size(),
                            descriptors.data(), 0, nullptr);

    // draw axis and grid
    {
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        color_line_pipeline_);

      glm::mat4 model(1.f);
      model[0][0] = 10.f;
      model[1][1] = 10.f;
      model[2][2] = 10.f;
      vkCmdPushConstants(cb, graphics_pipeline_layout_,
                         VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(model), &model);

      if (show_axis_) {
        std::vector<VkBuffer> vbs = {axis_.position_buffer, axis_.color_buffer};
        std::vector<VkDeviceSize> vb_offsets = {0, 0};
        vkCmdBindVertexBuffers(cb, 0, vbs.size(), vbs.data(),
                               vb_offsets.data());

        vkCmdBindIndexBuffer(cb, axis_.index_buffer, 0, VK_INDEX_TYPE_UINT32);

        vkCmdDrawIndexed(cb, axis_.index_count, 1, 0, 0, 0);
      }

      if (show_grid_) {
        std::vector<VkBuffer> vbs = {grid_.position_buffer, grid_.color_buffer};
        std::vector<VkDeviceSize> vb_offsets = {0, 0};
        vkCmdBindVertexBuffers(cb, 0, vbs.size(), vbs.data(),
                               vb_offsets.data());

        vkCmdBindIndexBuffer(cb, grid_.index_buffer, 0, VK_INDEX_TYPE_UINT32);

        vkCmdDrawIndexed(cb, grid_.index_count, 1, 0, 0, 0);
      }
    }

    // draw splat
    if (loaded_point_count_ != 0) {
      switch (splat_render_mode_) {
        case SplatRenderMode::TriangleList: {
          vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            splat_pipeline_);

          vkCmdBindIndexBuffer(cb, splat_index_buffer_, 0,
                               VK_INDEX_TYPE_UINT32);

          vkCmdDrawIndexedIndirect(cb, splat_draw_indirect_, 0, 1, 0);
        } break;

        case SplatRenderMode::GeometryShader: {
          vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            splat_geom_pipeline_);

          std::vector<VkBuffer> vbs = {splat_storage_.instance};
          std::vector<VkDeviceSize> vb_offsets = {0};
          vkCmdBindVertexBuffers(cb, 0, vbs.size(), vbs.data(),
                                 vb_offsets.data());

          vkCmdDrawIndirect(cb, splat_draw_indirect_, sizeof(float) * 8, 1, 0);
        } break;
      }
    }

    vkCmdEndRenderPass(cb);
  }

  void createFramebuffer() {
    color_attachment_ = vk::Attachment(
        context_, width_, height_, VK_FORMAT_B8G8R8A8_UNORM, samples_, false);
    depth_attachment_ = vk::Attachment(context_, width_, height_, depth_format_,
                                       samples_, false);

    vk::FramebufferCreateInfo framebuffer_info;
    framebuffer_info.width = width_;
    framebuffer_info.height = height_;
    framebuffer_info.render_pass = render_pass_;

    framebuffer_info.image_specs = {
        color_attachment_.image_spec(),
        depth_attachment_.image_spec(),
    };

    framebuffer_ = vk::Framebuffer(context_, framebuffer_info);

    // vk::FramebufferCreateInfo imgui_framebuffer_info;
    // imgui_framebuffer_info.width = widt;
    // imgui_framebuffer_info.height = swapchain_.height();
    // imgui_framebuffer_info.render_pass = imgui_render_pass_;

    // imgui_framebuffer_info.image_specs = {
    //     swapchain_.image_spec(),
    // };

    // imgui_framebuffer_ = vk::Framebuffer(context_, imgui_framebuffer_info);

    VkImageCreateInfo image_info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.format = VK_FORMAT_B8G8R8A8_UNORM;
    image_info.extent.width = width_;
    image_info.extent.height = height_;
    image_info.extent.depth = 1;
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.usage =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |  // so we can render to it
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo alloc_info = {};
    alloc_info.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
    if (vmaCreateImage(context_.allocator(), &image_info, &alloc_info,
                       &offscreen_image_, &offscreen_image_allocation_,
                       NULL) != VK_SUCCESS) {
      throw std::runtime_error("failed to create offscreen image!");
    }

    // Create image view for the offscreen image
    VkImageViewCreateInfo view_info = {};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = offscreen_image_;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = VK_FORMAT_B8G8R8A8_UNORM;
    view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    if (vkCreateImageView(context_.device(), &view_info, nullptr,
                          &offscreen_image_view_) != VK_SUCCESS) {
      throw std::runtime_error("failed to create offscreen image view!");
    }

    // Define the descriptor set layout bindings
    VkDescriptorSetLayoutBinding layout_binding = {};
    layout_binding.binding = 0;
    layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    layout_binding.descriptorCount = 1;
    layout_binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    layout_binding.pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutCreateInfo layout_info = {};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = 1;
    layout_info.pBindings = &layout_binding;

    if (vkCreateDescriptorSetLayout(context_.device(), &layout_info, nullptr,
                                    &descriptor_set_layout_) != VK_SUCCESS) {
      throw std::runtime_error("failed to create descriptor set layout!");
    }
    // Create the sampler
    VkSamplerCreateInfo sampler_info = {};
    sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_info.magFilter = VK_FILTER_LINEAR;
    sampler_info.minFilter = VK_FILTER_LINEAR;
    sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_info.anisotropyEnable = VK_TRUE;
    sampler_info.maxAnisotropy = 16;
    sampler_info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    sampler_info.unnormalizedCoordinates = VK_FALSE;
    sampler_info.compareEnable = VK_FALSE;
    sampler_info.compareOp = VK_COMPARE_OP_ALWAYS;
    sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sampler_info.mipLodBias = 0.0f;
    sampler_info.minLod = 0.0f;
    sampler_info.maxLod = 0.0f;

    if (vkCreateSampler(context_.device(), &sampler_info, nullptr,
                        &offscreen_sampler_) != VK_SUCCESS) {
      throw std::runtime_error("failed to create sampler!");
    }

    // Create the descriptor pool
    VkDescriptorPoolSize pool_size = {};
    pool_size.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    pool_size.descriptorCount = 1;

    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    pool_info.maxSets = 1;

    if (vkCreateDescriptorPool(context_.device(), &pool_info, nullptr,
                               &descriptor_pool_) != VK_SUCCESS) {
      throw std::runtime_error("failed to create descriptor pool!");
    }
    // Allocate the descriptor set
    VkDescriptorSetAllocateInfo descriptor_alloc_info = {};
    descriptor_alloc_info.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptor_alloc_info.descriptorPool = descriptor_pool_;
    descriptor_alloc_info.descriptorSetCount = 1;
    descriptor_alloc_info.pSetLayouts =
        &descriptor_set_layout_;  // Assuming you have a descriptor set layout

    if (vkAllocateDescriptorSets(context_.device(), &descriptor_alloc_info,
                                 &offscreen_descriptor_set_) != VK_SUCCESS) {
      throw std::runtime_error("failed to allocate descriptor set!");
    }

    // Create a Vulkan texture from the offscreen image
    VkDescriptorImageInfo descriptor_image_info = {};
    descriptor_image_info.imageLayout =
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    descriptor_image_info.imageView = offscreen_image_view_;
    descriptor_image_info.sampler =
        offscreen_sampler_;  // Assuming you have a sampler created

    VkWriteDescriptorSet write_descriptor = {};
    write_descriptor.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write_descriptor.dstSet =
        offscreen_descriptor_set_;  // Assuming you have an
                                    // ImGui descriptor set
    write_descriptor.dstBinding = 0;
    write_descriptor.dstArrayElement = 0;
    write_descriptor.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write_descriptor.descriptorCount = 1;
    write_descriptor.pImageInfo = &descriptor_image_info;

    vkUpdateDescriptorSets(context_.device(), 1, &write_descriptor, 0, nullptr);
  }

  std::atomic_bool terminate_ = false;

  std::mutex mutex_;
  std::string pending_ply_filepath_;

  GLFWwindow* window_ = nullptr;
  int width_ = 0;
  int height_ = 0;

  VkSampleCountFlagBits samples_ = VK_SAMPLE_COUNT_1_BIT;
  VkFormat depth_format_ = VK_FORMAT_D32_SFLOAT;
  SplatRenderMode splat_render_mode_ = SplatRenderMode::TriangleList;

  Camera camera_;
  ImGui_ImplVulkanH_Window main_window_;
  bool swapchain_rebuild_ = false;
  vk::Context context_;
  // vk::Swapchain swapchain_;

  VkImage offscreen_image_;
  VmaAllocation offscreen_image_allocation_ = VK_NULL_HANDLE;
  VkDeviceMemory offscreen_image_memory_;
  VkImageView offscreen_image_view_;
  VkDescriptorSet offscreen_descriptor_set_;
  VkDescriptorSetLayout descriptor_set_layout_;
  VkDescriptorPool descriptor_pool_;
  VkSampler offscreen_sampler_;

  std::vector<VkCommandBuffer> draw_command_buffers_;
  std::vector<VkSemaphore> image_acquired_semaphores_;
  std::vector<VkSemaphore> render_finished_semaphores_;
  std::vector<VkFence> render_finished_fences_;

  vk::DescriptorLayout camera_descriptor_layout_;
  vk::DescriptorLayout gaussian_descriptor_layout_;
  vk::DescriptorLayout instance_layout_;
  vk::DescriptorLayout ply_descriptor_layout_;
  vk::PipelineLayout compute_pipeline_layout_;
  vk::PipelineLayout graphics_pipeline_layout_;

  // preprocess
  vk::ComputePipeline parse_ply_pipeline_;
  vk::ComputePipeline rank_pipeline_;
  vk::ComputePipeline inverse_index_pipeline_;
  vk::ComputePipeline projection_pipeline_;

  // sorter
  VrdxSorter sorter_ = VK_NULL_HANDLE;

  // normal pass
  vk::Framebuffer framebuffer_;
  vk::RenderPass render_pass_;
  // VkRenderPass imgui_render_pass_;
  // VkFramebuffer imgui_framebuffer_;

  vk::GraphicsPipeline color_line_pipeline_;
  vk::GraphicsPipeline splat_pipeline_;
  vk::GraphicsPipeline splat_geom_pipeline_;

  vk::Attachment color_attachment_;
  vk::Attachment depth_attachment_;

  vk::UniformBuffer<vk::shader::Camera> camera_buffer_;

  struct ColorObject {
    vk::Buffer position_buffer;
    vk::Buffer color_buffer;
    vk::Buffer index_buffer;
    int index_count;
  };
  ColorObject axis_;
  ColorObject grid_;

  struct FrameDescriptor {
    vk::Descriptor camera;
    vk::Descriptor gaussian;
    vk::Descriptor splat_instance;
    vk::Descriptor ply;
  };
  std::vector<FrameDescriptor> descriptors_;

  struct FrameInfo {
    bool drew_splats = false;
    uint32_t total_point_count = 0;
    uint32_t loaded_point_count = 0;

    uint64_t rank_time = 0;
    uint64_t sort_time = 0;
    uint64_t inverse_time = 0;
    uint64_t projection_time = 0;
    uint64_t rendering_time = 0;
    uint64_t end_to_end_time = 0;

    uint64_t present_timestamp = 0;
    uint64_t present_done_timestamp = 0;

    vk::Buffer ply_buffer;
  };
  std::vector<FrameInfo> frame_infos_;

  struct SplatStorage {
    vk::Buffer position;  // (N, 3)
    vk::Buffer cov3d;     // (N, 6)
    vk::Buffer opacity;   // (N)
    vk::Buffer sh;        // (N, 3, 16)

    vk::Buffer key;            // (N)
    vk::Buffer index;          // (N)
    vk::Buffer inverse_index;  // (N)

    vk::Buffer instance;  // (N, 10)
  };
  SplatStorage splat_storage_;
  vk::Buffer sort_storage_;
  static constexpr uint32_t MAX_SPLAT_COUNT = 1 << 23;  // 2^23
  // 2^23 * 3 * 16 * sizeof(float) is already 1.6GB.

  vk::UniformBuffer<vk::shader::SplatInfo> splat_info_buffer_;  // (2)
  vk::Buffer splat_visible_point_count_;                        // (2)
  vk::Buffer splat_draw_indirect_;                              // (5)

  glm::vec3 translation_{0.f, 0.f, 0.f};
  glm::quat rotation_{1.f, 0.f, 0.f, 0.f};
  float scale_{1.f};

  bool show_axis_ = true;
  bool show_grid_ = true;
  bool follow_trajectory_ = false;
  bool drive_ = false;

  vk::CpuBuffer visible_point_count_cpu_buffer_;  // (2) for debug

  vk::Buffer splat_index_buffer_;  // gaussian2d quads

  VkSemaphore transfer_semaphore_ = VK_NULL_HANDLE;
  uint64_t transfer_timeline_ = 0;

  SplatLoadThread splat_load_thread_;
  uint32_t loaded_point_count_ = 0;

  // timestamp queries
  static constexpr uint32_t timestamp_count_ = 12;
  std::vector<VkQueryPool> timestamp_query_pools_;

  uint64_t frame_counter_ = 0;
};

Engine::Engine() : impl_(std::make_shared<Impl>()) {}

Engine::~Engine() = default;

void Engine::LoadSplats(const std::string& ply_filepath) {
  impl_->LoadSplats(ply_filepath);
}

void Engine::LoadSplatsAsync(const std::string& ply_filepath) {
  impl_->LoadSplatsAsync(ply_filepath);
}

void Engine::LoadTrajectory(const std::string& trajectory_filepath) {
  impl_->LoadTrajectory(trajectory_filepath);
}

void Engine::Run() { impl_->Run(); }

void Engine::Close() { impl_->Close(); }

}  // namespace vkgs
