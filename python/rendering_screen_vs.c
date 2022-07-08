#version 130
in vec3 aPos;
in vec2 tex_coord;

out vec2 TexCoords;

void main()
{
    gl_Position = vec4(aPos.x, aPos.y, 0.1, 1.0);
    TexCoords = tex_coord;
}