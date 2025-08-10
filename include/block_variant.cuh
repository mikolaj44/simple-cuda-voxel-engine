#pragma once

#include <string>
#include "block_texture.cuh"

#include "material.cuh"

class BlockVariant {

public:
    Material material;
    BlockTexture* texture;

    __device__ BlockVariant(Material material, BlockTexture* texture);
};