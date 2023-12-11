#version 330 core

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_color;


out vec3 vColor;
out vec3 fragPos;

uniform mat4 m_proj;
uniform mat4 m_view;
uniform mat4 m_model;

void main() {
    fragPos = vec3(m_model * vec4(in_position, 1.0));
    gl_Position = m_proj * m_view * m_model * vec4(in_position, 1.0);
    vColor = in_color;

}