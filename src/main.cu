#include "voxel_engine.cuh"

#include <iostream>

int main() {
    VoxelEngine::init(1920, 1080);

    auto blockPosToIdFunction = [] __device__ (int x, int y, int z, uint64_t frameCount) {
        if(x*x + y*y + z*z <= 128 * 128)
            return (x*x + y*y + z*z) % 127 + 1;
        return -1;
    };

    VoxelEngine::insertVoxels(blockPosToIdFunction);

    VoxelEngine::cleanup();

    VoxelEngine::init(1920, 1080);

    VoxelEngine::cleanup();
}