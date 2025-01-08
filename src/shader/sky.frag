#version 450
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 1) uniform SkyColor {
    vec4 color;
} skyColor;

void main() {
    // outColor = skyColor.color;
    outColor = vec4(0.5, 0.7, 1.0, 1.0); // Hardcoded light blue color

}