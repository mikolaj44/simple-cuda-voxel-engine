#pragma once

#include "scve/internal/block_variant/block_texture.cuh"
#include "scve/internal/structure/material.h"

namespace scve {

class BlockVariant {
public:
    Material material;
    BlockTexture* texture;

    __host__ BlockVariant(Material material_, BlockTexture* texture_) : material(material_), texture(texture_) {};
};

}