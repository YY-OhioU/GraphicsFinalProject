#version 330

// Will read from texture bound to channel / locaton 0 by default
uniform sampler2D Texture;

// Interpolated texture coordinate from vertex shader
in vec2 v_text;
// The fragment ending up on the screen
out vec4 f_color;

void main() {
    f_color = texture(Texture, v_text);
    //    f_color = (v_text)
}