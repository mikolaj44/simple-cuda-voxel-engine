#pragma once
#include <string>
#include <vector>
#include "block.cuh"
#include "pixeldrawing.cuh"
#include "cudamath.cuh"

using namespace std;

extern __device__ Block** blocks;


__device__ void setPixelById(int sX, int sY, int blockX, int blockY, int blockZ, float x, float y, float z, unsigned char blockId, unsigned char* pixels);

__global__ void createBlocksData(BlockTexture** textures);