#pragma once
#include <string>
#include <vector>
#include "block_variant.cuh"
#include "pixel_drawing.cuh"
#include "cuda_math.cuh"

using namespace std;

extern __device__ BlockVariant** blockVariants;

__device__ void setPixelById(int sX, int sY, int blockX, int blockY, int blockZ, float x, float y, float z, unsigned char blockId, unsigned char* pixels);

__global__ void createBlocksData(BlockTexture** textures);