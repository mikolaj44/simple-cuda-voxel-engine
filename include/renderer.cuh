#pragma once
#include <vector>

#include "octree.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

class Renderer {
public:
    //static void calculateFOV();

    static void renderScreenCuda(Octree* octree, float cameraAngleX, float cameraAngleY, float oX, float oY, float oZ, uchar4* pixels, unsigned int gridSize, unsigned int blockSize);
};

__device__ void setPixel(uchar4* pixels, int x, int y, int r, int g, int b, int a);

__global__ void renderScreenCudaKernel(Octree* octree, float cameraAngleX, float cameraAngleY, float oX, float oY, float oZ, uchar4* pixels);