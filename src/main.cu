#include "voxel_engine.cuh"
#include "cuda_noise.cuh"

#include <iostream>

// void func() {
//     VoxelEngine::clearVoxels();

//     Vector3<> cameraPos = VoxelEngine::getCameraPos();

//     auto blockPosToIdFunction = [=] __device__ (int x, int y, int z, uint64_t frameCount) {
//         // float val = cudaNoise::perlinNoise(make_float3(float(x) / 1000.0, 1, float(z) / 1000.0), 1, 0);

//         // printf("%f\n", val);
        
//         // if(absv(val) >= 0.4 /*&& y <= val + 20.5 + 10*/){
//         //     return 1;
//         // }

//         // if(val >= 0.05){
//         //     return 2;
//         // }

//         if(y >= cameraPos.y + 5 && y <= cameraPos.y + 10)
//             return (x+y+z) % 127 + 1;
//         return 0;
//     };

//     VoxelEngine::insertVoxels(blockPosToIdFunction);

//     Vector3<> pos = cameraPos.add(Vector3<>(-512,10,512));

//     VoxelEngine::setOctreeMinPos(pos);
// }

int main() {
    VoxelEngine::init(1920, 1080, 10);

    VoxelEngine::setOctreeMinPos(Vector3<>(-512, -512, -512));

    auto blockPosFrameToIdFunction = [] __device__ (int x, int y, int z, uint64_t frameNumber) {
        int maxIterations = 2;

        float newX = float(x) / 450.0;
        float newY = float(y) / 450.0;
        float newZ = float(z) / 450.0;

        float wX = newX;
        float wY = newY;
        float wZ = newZ;

        int iterations = maxIterations;

        while(iterations--){
            float x_ = wX;
            float y_ = wY;
            float z_ = wZ;

            float x2 = x_*x_;
            float y2 = y_*y_;
            float z2 = z_*z_;

            float x4 = x2*x2;
            float y4 = y2*y2;
            float z4 = z2*z2;

            float k3 = x2 + z2;
            float k2 = 1.0 / sqrt(k3*k3*k3*k3*k3*k3*k3);
            float k1 = x4 + y4 + z4 - 6.0*y2*z2 - 6.0*x2*y2 + 2.0*z2*x2;
            float k4 = x2 - y2 + z2;

            wX =  64.0*x_*y_*z_*(x2-z2)*k4*(x4-6.0*x2*z2+z4)*k1*k2;
            wY = -16.0*y2*k3*k4*k4 + k1*k1;
            wZ = -8.0*y_*k4*(x4*x4 - 28.0*x4*x2*z2 + 70.0*x4*z4 - 28.0*x2*z2*z4 + z4*z4)*k1*k2;

            wX += newX;
            wY += newY;
            wZ += newZ;

            if(wX * wX + wY * wY + wZ * wZ > 4.0){
                return 0;
            }
        }

        //if(x < 100)
            return int(sqrtf(x*x + y*y + z*z)) % 127 + 1;
        //return 0;

        // return int(sqrtf(absv(x) * absv(x) * absv(x) + absv(x) + absv(y) + absv(z))) % 127 + 1; 
    };

    VoxelEngine::setTextureRenderingEnabled(false);

    VoxelEngine::insertVoxels(blockPosFrameToIdFunction);

    printf("done inserting\n");

    // VoxelEngine::setCameraPos(Vector3<>(0, 0, -100));

    VoxelEngine::inputLoop();    
}