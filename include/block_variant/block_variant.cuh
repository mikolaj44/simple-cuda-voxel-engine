#pragma once

#include "block_texture.cuh"
#include "material.cuh"

class BlockVariant {
public:
    Material material;
    BlockTexture* texture;

    BlockVariant(Material material_, BlockTexture* texture_) : material(material_), texture(texture_) {};
};