#version 450
layout (location = 0) in vec3 position;
layout (location = 1) in vec2 uv;
layout (push_constant) uniform PushConsts {
    mat4 mvp;
} push;
layout (binding = 2) uniform UniformBlock {
    mat4 projection;
} uniform_block;
layout (location = 0) out gl_PerVertex {
    vec4 gl_Position;
};
layout (location = 1) out vec2 frag_uv;

void main() {
    gl_Position =  push.mvp* uniform_block.projection * vec4(position, 1.0);
    frag_uv = uv;
}