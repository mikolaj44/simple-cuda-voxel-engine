#pragma once

#include "scve/internal/octree/octree.cuh"

namespace scve {
namespace cuda_renderer {
    extern uchar4* devicePixels;

    __managed__ extern float focalLength;

    cudaError_t init(int windowWidth, int windowHeight);

    cudaError_t cleanup();

    void render(Octree* octree, scve::Vector3<> cameraPos, scve::Vector3<> cameraAngle2d, int windowWidth, int windowHeight, bool isTextureRenderingEnabled, bool isPhongIlluminationEnabled, unsigned int gridSize, unsigned int blockSize);
}
}