#version 330 core

layout (location = 0) out vec4 fragColor;

in vec3 vColor;
in vec3 fragPos;


void main() {
    float gamma = 2.2;
    vec3 color = vColor;
    fragColor = vec4(color, 1.0);
}










