#version 450
layout (push_constant) uniform PushConsts {
    float time;
} push;

layout(location = 0) out vec4 color;
layout(location = 1) in vec3 frag_color;

void main()
{
    float time01 = -0.9 * abs(sin(push.time * 0.9)) + 0.9;
    color = vec4(frag_color, 1.0) * vec4(time01, time01, time01, 1.0);
}