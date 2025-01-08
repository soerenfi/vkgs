#include <stdexcept>

#include "vk_mem_alloc.h"

#include "vkgs/engine/vulkan/context.h"
#include "vkgs/engine/vulkan/attachment.h"

namespace vkgs {
namespace vk {

class Attachment::Impl {
   public:
    Impl(Context context, uint32_t width, uint32_t height, VkFormat format,
         VkSampleCountFlagBits samples, bool input_attachment)
        : context_(context), width_(width), height_(height), format_(format) {
        usage_ = 0;
        VkImageAspectFlags aspect = 0;
        switch (format) {
            case VK_FORMAT_D16_UNORM:
            case VK_FORMAT_D32_SFLOAT:
            case VK_FORMAT_D16_UNORM_S8_UINT:
            case VK_FORMAT_D24_UNORM_S8_UINT:
            case VK_FORMAT_D32_SFLOAT_S8_UINT:
                usage_ = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
                aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
                break;

            default:
                usage_ = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
                // |VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
                aspect = VK_IMAGE_ASPECT_COLOR_BIT;
                break;
        }

        if (input_attachment) {
            usage_ |= VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
        }

        VkImageCreateInfo image_info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        image_info.imageType = VK_IMAGE_TYPE_2D;
        image_info.format = format;
        image_info.extent = {width_, height_, 1};
        image_info.mipLevels = 1;
        image_info.arrayLayers = 1;
        image_info.samples = samples;
        image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        image_info.usage = usage_;
        image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo alloc_info = {};
        alloc_info.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
        alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
        vmaCreateImage(context.allocator(), &image_info, &alloc_info, &image_, &allocation_, NULL);

        VkImageViewCreateInfo image_view_info = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        image_view_info.image = image_;
        image_view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        image_view_info.format = format;
        image_view_info.subresourceRange = {aspect, 0, 1, 0, 1};
        vkCreateImageView(context.device(), &image_view_info, NULL, &image_view_);

        // Create sampler
        VkSamplerCreateInfo sampler_info = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
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

        if (vkCreateSampler(context_.device(), &sampler_info, nullptr, &sampler_) != VK_SUCCESS) {
            throw std::runtime_error("failed to create sampler!");
        }

        // Create descriptor set layout
        VkDescriptorSetLayoutBinding layout_binding = {};
        layout_binding.binding = 0;
        layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        layout_binding.descriptorCount = 1;
        layout_binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        layout_binding.pImmutableSamplers = nullptr;

        VkDescriptorSetLayoutCreateInfo layout_info = {
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layout_info.bindingCount = 1;
        layout_info.pBindings = &layout_binding;

        if (vkCreateDescriptorSetLayout(context_.device(), &layout_info, nullptr,
                                        &descriptor_set_layout_) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor set layout!");
        }

        // Create descriptor pool
        VkDescriptorPoolSize pool_size = {};
        pool_size.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        pool_size.descriptorCount = 1;

        VkDescriptorPoolCreateInfo pool_info = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &pool_size;
        pool_info.maxSets = 1;

        if (vkCreateDescriptorPool(context_.device(), &pool_info, nullptr, &descriptor_pool_) !=
            VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor pool!");
        }

        // Allocate descriptor set
        VkDescriptorSetAllocateInfo descriptor_alloc_info = {
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        descriptor_alloc_info.descriptorPool = descriptor_pool_;
        descriptor_alloc_info.descriptorSetCount = 1;
        descriptor_alloc_info.pSetLayouts = &descriptor_set_layout_;

        if (vkAllocateDescriptorSets(context_.device(), &descriptor_alloc_info, &descriptor_set_) !=
            VK_SUCCESS) {
            throw std::runtime_error("failed to allocate descriptor set!");
        }

        // Update descriptor set
        VkDescriptorImageInfo descriptor_image_info = {};
        descriptor_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        descriptor_image_info.imageView = image_view_;
        descriptor_image_info.sampler = sampler_;

        VkWriteDescriptorSet write_descriptor = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write_descriptor.dstSet = descriptor_set_;
        write_descriptor.dstBinding = 0;
        write_descriptor.dstArrayElement = 0;
        write_descriptor.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write_descriptor.descriptorCount = 1;
        write_descriptor.pImageInfo = &descriptor_image_info;

        vkUpdateDescriptorSets(context_.device(), 1, &write_descriptor, 0, nullptr);
    }

    ~Impl() {
        vkDestroyImageView(context_.device(), image_view_, NULL);
        vmaDestroyImage(context_.allocator(), image_, allocation_);
    }

    operator VkImageView() const noexcept { return image_view_; }

    VkImage image() const noexcept { return image_; }
    VkImageUsageFlags usage() const noexcept { return usage_; }
    VkFormat format() const noexcept { return format_; }
    ImageSpec image_spec() const noexcept { return ImageSpec{width_, height_, usage_, format_}; }

   private:
    Context context_;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    VkImageUsageFlags usage_ = 0;
    VkFormat format_ = VK_FORMAT_UNDEFINED;
    VkImage image_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = VK_NULL_HANDLE;
    VkImageView image_view_ = VK_NULL_HANDLE;
    VkSampler sampler_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;

   public:
    VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;
};

Attachment::Attachment() = default;

Attachment::Attachment(Context context, uint32_t width, uint32_t height, VkFormat format,
                       VkSampleCountFlagBits samples, bool input_attachment)
    : impl_(std::make_shared<Impl>(context, width, height, format, samples, input_attachment)) {}

Attachment::~Attachment() = default;

Attachment::operator VkImageView() const { return *impl_; }

VkDescriptorSet Attachment::getDescriptorSet() const { return impl_->descriptor_set_; }

VkImage Attachment::image() const { return impl_->image(); }

VkImageUsageFlags Attachment::usage() const { return impl_->usage(); }

VkFormat Attachment::format() const { return impl_->format(); }

ImageSpec Attachment::image_spec() const { return impl_->image_spec(); }

}  // namespace vk
}  // namespace vkgs
