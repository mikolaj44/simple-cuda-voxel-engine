#pragma once

#include <string>
#include <vector>

#include "block_variant.cuh"
#include "pointlight.cuh"

extern __device__ BlockVariant** blockVariants;

__device__ void setPixelById(int sX, int sY, int blockX, int blockY, int blockZ, float x, float y, float z, unsigned char blockId, uchar4* pixels, Vector3 cameraPos, PointLight light, bool textureRenderingEnabled);

__global__ void createBlocksData(BlockTexture** textures);

__device__ void hueToRGB(float hue, int& r, int& g, int&b);

__device__ void getPhongIllumination(Vector3 pos, Vector3 cameraPos, Vector3 normal, Material material, PointLight light, int& r, int& g, int&b);