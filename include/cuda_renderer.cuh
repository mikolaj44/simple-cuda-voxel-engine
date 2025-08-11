#pragma once

#include "octree/octree.cuh"

namespace cuda_renderer {
    void render(uchar4* pixels, Octree* octree, Vector3<> cameraPos, Vector3<> cameraAngle2d, int screenWidth, int screenHeight, bool textureRenderingEnabled, unsigned int gridSize, unsigned int blockSize);

    __device__ void setPixelByHitInfo(uchar4* pixels, octree_utils::Pair<BlockInfo<>, BlockInfo<float>> intersectionData, Vector3<> cameraPos, int sX, int sY, bool textureRenderingEnabled);

    __device__ void setPixel(uchar4* pixels, int sX, int sY, int r, int g, int b, int a);
}