#pragma once

#include "octree/octree.cuh"

namespace cuda_renderer {
    void init(int windowWidth, int windowHeight);

    void cleanup();

    void render(Octree* octree, Vector3<> cameraPos, Vector3<> cameraAngle2d, int windowWidth, int windowHeight, bool isTextureRenderingEnabled, unsigned int gridSize, unsigned int blockSize);
}