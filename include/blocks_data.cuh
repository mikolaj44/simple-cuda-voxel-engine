#pragma once

#include <string>
#include <vector>

#include "block_variant.cuh"
#include "point_light.cuh"

extern __device__ BlockVariant** blockVariants;

__global__ void createBlocksData(BlockTexture** textures);