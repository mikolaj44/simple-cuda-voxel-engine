#pragma once

#include <string>
#include <vector>

#include "block_variant.cuh"
#include "point_light.cuh"

extern __device__ BlockVariant** blockVariants;

__global__ void createBlocksData(BlockTexture** textures);

__device__ void hueToRGB(float hue, int& r, int& g, int&b);

__device__ void getPhongIllumination(Vector3<> pos, Vector3<> cameraPos, Vector3<> normal, Material material, PointLight light, int& r, int& g, int&b);