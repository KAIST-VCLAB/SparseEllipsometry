#version 130
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D screenTexture1;

void main()
{
    FragColor = texture(screenTexture1, vec2(TexCoords.x,TexCoords.y));

}