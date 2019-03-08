#version 450
layout (push_constant) uniform PushConsts {
    float time;
} push;

layout(set = 0, binding = 0) uniform texture2D tex;
layout(set = 0, binding = 1) uniform sampler samp;

layout(location = 0) out vec4 color;
layout(location = 1) in vec2 frag_uv;
void main()
{
    color = texture(sampler2D(tex, samp), frag_uv);
}