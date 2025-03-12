#pragma once

#include "chunk.cuh"
#include "octree.cuh"
#include "globals.cuh"
#include "cuda_noise.cuh"

#include <thrust/device_vector.h>

using namespace std;

void generateChunks(Octree* octree, Vector3 cameraPos, dim3 gridSize, dim3 blockSize);

__global__ void generateChunksKernel(Octree* octree, Vector3 cameraPos);