#version 450
layout (push_constant) uniform PushConsts {
    float time;
} push;


layout(location = 0) out vec4 color;
void main()
{
    color = vec4(1.0,0.0,0.0,1.0);
}