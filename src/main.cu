#include "voxel_engine.h"

int main() {
    VoxelEngine::init(500, 700);

    auto blockPosToIdFunction = [] __device__ (int x, int y, int z, uint64_t frameCount){
        if(x*x + y*y + z*z <= 50 * 50 * 50)
            return int(x + y + z) % 127 + 1;
        return -1;
    };

    VoxelEngine::insertVoxels(blockPosToIdFunction);

    while(true){
        VoxelEngine::handleInput();
        VoxelEngine::displayFrame();
    }
}