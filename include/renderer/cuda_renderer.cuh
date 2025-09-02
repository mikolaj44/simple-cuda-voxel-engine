#pragma once

#include "octree/octree.cuh"

namespace cuda_renderer {
    cudaError_t init(int windowWidth, int windowHeight);

    cudaError_t cleanup();

    void render(Octree* octree, Vector3<> cameraPos, Vector3<> cameraAngle2d, int windowWidth, int windowHeight, bool isTextureRenderingEnabled, bool isMaterialColorOnlyEnabled, unsigned int gridSize, unsigned int blockSize);
}