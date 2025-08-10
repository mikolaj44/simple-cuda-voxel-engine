#pragma once

#include "octree/octree.cuh"

namespace cuda_renderer {
    void render(uchar4* pixels, Octree* octree, Vector3 cameraPos, Vector3 cameraAngle2d, int screenWidth, int screenHeight, unsigned int gridSize, unsigned int blockSize);

    __device__ void setPixel(uchar4* pixels, int x, int y, int r, int g, int b, int a);
}