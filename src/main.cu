#include "voxel_engine.cuh"
#include "cuda_noise.cuh"

#include <iostream>

void func() {
    VoxelEngine::clearVoxels();

    Vector3<> cameraPos = VoxelEngine::getCameraPos();

    auto blockPosToIdFunction = [=] __device__ (int x, int y, int z, uint64_t frameCount) {
        // float val = cudaNoise::perlinNoise(make_float3(float(x) / 1000.0, 1, float(z) / 1000.0), 1, 0);

        // printf("%f\n", val);
        
        // if(absv(val) >= 0.4 /*&& y <= val + 20.5 + 10*/){
        //     return 1;
        // }

        // if(val >= 0.05){
        //     return 2;
        // }

        if(y >= cameraPos.y + 5 && y <= cameraPos.y + 10)
            return (x+y+z) % 127 + 1;
        return 0;
    };

    VoxelEngine::insertVoxels(blockPosToIdFunction);

    Vector3<> pos = cameraPos.add(Vector3<>(-512,10,512));

    VoxelEngine::setOctreeMinPos(pos);
}

int main() {
    VoxelEngine::init(1920, 1080, 10);

    VoxelEngine::setCameraPos(Vector3<>(0, 0, -100));

    VoxelEngine::inputLoop(func);    
}