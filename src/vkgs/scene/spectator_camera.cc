#include <vkgs/scene/spectator_camera.h>
#include <iostream>
#include <algorithm>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>
namespace vkgs {

SpectatorCamera::SpectatorCamera() {}
SpectatorCamera::~SpectatorCamera() {}

void SpectatorCamera::SetRenderSize(uint32_t width, uint32_t height) {
    width_ = width;
    height_ = height;
}

void SpectatorCamera::SetPosition(const glm::mat4& transform) {
    // Extract translation
    center_ = glm::vec3(transform[0][3], transform[1][3], transform[2][3]);
    std::cout << "Center: (" << center_.x << ", " << center_.y << ", " << center_.z << ")"
              << std::endl;
}

void SpectatorCamera::SetPosition(const glm::vec3& position) {
    center_ = position;
    std::cout << "Center: (" << center_.x << ", " << center_.y << ", " << center_.z << ")"
              << std::endl;
}

void SpectatorCamera::SetOrientation(const glm::quat& orientation) {
    glm::mat4 transform = glm::toMat4(orientation);

    // Extract translation
    center_ = glm::vec3(transform[3]);

    // Extract rotation
    glm::vec3 forward = glm::normalize(glm::vec3(transform[2]));
    glm::vec3 right = glm::normalize(glm::vec3(transform[0]));
    glm::vec3 up = glm::normalize(glm::vec3(transform[1]));

    // Calculate spherical coordinates from the forward vector
    phi_ = std::acos(forward.z);
    theta_ = std::atan2(forward.y, forward.x);

    std::cout << "Orientation set. Center: (" << center_.x << ", " << center_.y << ", " << center_.z
              << ")" << std::endl;
}
glm::quat SpectatorCamera::Orientation() const {
    glm::vec3 forward = glm::normalize(Eye() - center_);
    glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0.f, 0.f, 1.f)));
    glm::vec3 up = glm::cross(right, forward);

    return glm::quat_cast(glm::mat3(right, up, -forward));
}

void SpectatorCamera::SetFov(float fov) {
    // dolly zoom
    r_ *= std::tan(fovy_ / 2.f) / std::tan(fov / 2.f);
    fovy_ = fov;
}

glm::mat4 SpectatorCamera::ProjectionMatrix() const {
    float aspect = static_cast<float>(width_) / height_;
    glm::mat4 projection = glm::perspective(fovy_, aspect, near_, far_);

    // GL to Vulkan projection matrix
    glm::mat4 conversion = glm::mat4(1.f);
    conversion[1][1] = -1.f;
    conversion[2][2] = 0.5f;
    conversion[3][2] = 0.5f;
    return conversion * projection;
}

// --------------------------------------------------------------------
// 1) Change the up vector to (0, 0, 1)
// --------------------------------------------------------------------
glm::mat4 SpectatorCamera::ViewMatrix() const {
    return glm::lookAt(Eye(), center_, glm::vec3(0.f, 0.f, 1.f));
}

// --------------------------------------------------------------------
// 2) Rewrite Eye() so that Z is "up" in spherical coordinates
//
//    Original Y-up had:
//       Eye = center + r * ( sin(phi)*sin(theta),
//                            cos(phi),
//                            sin(phi)*cos(theta) )
//
//    For Z-up, a common spherical parameterization is:
//       x = r sin(phi) cos(theta)
//       y = r sin(phi) sin(theta)
//       z = r cos(phi)
// --------------------------------------------------------------------
glm::vec3 SpectatorCamera::Eye() const {
    float sin_phi = std::sin(phi_);
    float cos_phi = std::cos(phi_);
    float sin_theta = std::sin(theta_);
    float cos_theta = std::cos(theta_);

    // Now z = cos_phi, so that phi=0 means "looking from the top (Z+)"
    return center_ + r_ * glm::vec3(sin_phi * cos_theta,  // X
                                    sin_phi * sin_theta,  // Y
                                    cos_phi               // Z
                          );
}

// --------------------------------------------------------------------
// 3) Keep the same Rotate() logic, but realize you're now rotating
//    around Z if you move "theta_", etc.  You might invert signs
//    or reorder if you want different drag behavior.
// --------------------------------------------------------------------
void SpectatorCamera::Rotate(float x, float y) {
    theta_ -= rotation_sensitivity_ * x;

    float eps = glm::radians(0.1f);
    phi_ = std::clamp(phi_ - rotation_sensitivity_ * y, eps, glm::pi<float>() - eps);
}

// --------------------------------------------------------------------
// 4) Translate() logic: We define a local "forward" (f), "right" (r),
//    and "up" (u) based on the new Eye() orientation. Then we move
//    the 'center_' accordingly.  If directions feel inverted in your
//    app, flip the signs on x/y/z below.
// --------------------------------------------------------------------
void SpectatorCamera::Translate(float x, float y, float z) {
    // Local "forward"
    glm::vec3 f = glm::normalize(Eye() - center_);
    // Global "Z-up"
    glm::vec3 worldUp(0.f, 0.f, 1.f);

    // Right = forward x worldUp (normalized)
    glm::vec3 r = glm::normalize(glm::cross(f, worldUp));
    // Local SpectatorCamera up = right x forward
    glm::vec3 u = glm::cross(r, f);

    // Move center_ in local SpectatorCamera space
    center_ += translation_sensitivity_ * r_ *
               (-x * r +  // move left/right
                y * u +   // move up/down
                -z * f    // move forward/back
               );
}

// Zoom & Dolly Zoom remain unchanged
void SpectatorCamera::Zoom(float x) { r_ /= std::exp(zoom_sensitivity_ * x); }

void SpectatorCamera::DollyZoom(float scroll) {
    float new_fov = std::clamp(fovy_ - scroll * dolly_zoom_sensitivity_, min_fov(), max_fov());
    SetFov(new_fov);
}

}  // namespace vkgs
