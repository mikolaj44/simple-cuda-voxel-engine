#pragma once
#include <string>
#include "blocktexture.cuh"

using namespace std;

enum BlockType {
    SOLID,
    AIR,
    LIQUID
};

class Block {

public:
    BlockType type = SOLID;
    BlockTexture* texture;

    __device__ Block(BlockType type, BlockTexture* texture);
};