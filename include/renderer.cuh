#pragma once
#include <vector>

#include "globals.cuh"
#include "octree.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

class Renderer {
public:
    static void calculateFOV();

    static void renderScreenCuda(Octree* octree, float cameraAngleX, float cameraAngleY, float oX, float oY, float oZ, uchar4* pixels, unsigned int gridSize, unsigned int blockSize);
};

__global__ void renderScreenCudaKernel(Octree* octree, float cameraAngleX, float cameraAngleY, float oX, float oY, float oZ, uchar4* pixels);