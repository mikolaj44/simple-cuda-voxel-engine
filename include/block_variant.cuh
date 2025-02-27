#pragma once
#include <string>
#include "block_texture.cuh"

using namespace std;

enum BlockType {
    SOLID,
    AIR,
    LIQUID
};

class BlockVariant {

public:
    BlockType type = SOLID;
    BlockTexture* texture;

    __device__ BlockVariant(BlockType type, BlockTexture* texture);
};