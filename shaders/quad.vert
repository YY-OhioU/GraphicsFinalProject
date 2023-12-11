#version 330

in vec2 in_vert;
in vec2 in_texcoord;

out vec2 v_text;

void main() {
    v_text = in_texcoord;
    gl_Position = vec4(in_vert, 0.0, 1.0);
}