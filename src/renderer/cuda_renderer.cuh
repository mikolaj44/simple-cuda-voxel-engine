#pragma once

#include "octree/octree.cuh"

namespace scve::cuda_renderer {
    cudaError_t init(int windowWidth, int windowHeight);

    cudaError_t cleanup();

    void render(Octree* octree, scve::Vector3<> cameraPos, scve::Vector3<> cameraAngle2d, int windowWidth, int windowHeight, bool isTextureRenderingEnabled, bool isMaterialColorOnlyEnabled, bool isPhongIlluminationEnabled, unsigned int gridSize, unsigned int blockSize);
}