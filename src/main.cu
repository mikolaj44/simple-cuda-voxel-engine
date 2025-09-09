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
    VoxelEngine::test(VoxelEngine::init(1920, 1080, 10));

    VoxelEngine::setOctreeMinPos(Vector3<>(-512, -512, -512));

    auto blockPosFrameToIdFunction = [] __device__ (int x, int y, int z, uint64_t frameNumber) {
        //if(x == 204 && y == 253 && z == 01)
        //if(x == y)
        // if(x*x + y*y + z*z <= 50000)
             //return (absv(x) + absv(y) + absv(z)) % 4 + 1;

             return 1;
        // return 0;
        
        int maxIterations = 4;

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

            if(wX * wX + wY * wY + wZ * wZ > 0.6) { // 4.0
                return 0;
            }
        }

        // if(x == 0 && y == 0 && z == 0)
        //     return 1;
        // if(x == 0 && y == 0 && z == 1)
        //     return 2;
        // if(x == 0 && y == 1 && z == 0)
        //     return 3;
        // if(x == 0 && y == 1 && z == 1)
        //     return 4;

        //if(x < 100)
            return int(sqrtf(x*x + y*y + z*z)) % 127 + 1;
        return 0;

        // return int(sqrtf(absv(x) * absv(x) * absv(x) + absv(x) + absv(y) + absv(z))) % 127 + 1; 
    };

    auto blockIdFrameToIdFunction = [] __device__ (uint8_t blockId, uint64_t frameNumber) {
        return Material(Vector3<>(255, 0, 0), 1.0, 0.0, 20.0);

        // return Material(Vector3<>(blockId * blockId, blockId * 3 * blockId, blockId * 5 + 100), 1.0, 0.0, 20.0);
    };

    VoxelEngine::setCameraPos(Vector3<>(0, 0, -10000));

    VoxelEngine::setTextureRenderingEnabled(true);

    VoxelEngine::setCalculatingInsertLODsEnabled(true);

    VoxelEngine::setMaterialColorOnlyEnabled(false);

    VoxelEngine::setMouseControlEnabled(false);

    VoxelEngine::setPhongIlluminationEnabled(true);



    VoxelEngine::setMaterials(blockIdFrameToIdFunction);

    VoxelEngine::setPointLights({PointLight(Vector3<>(0, 0, -300000), Vector3<>(255, 0, 0)), PointLight(Vector3<>(0, -300000, -300000), Vector3<>(0, 0, 255))});    

    VoxelEngine::setAmbientLightIntensity(0.5);


    size_t chunkWidth = 8;

    uint8_t* blockIdArray = new uint8_t[chunkWidth * chunkWidth * chunkWidth];

    for(int i = 0; i < chunkWidth * chunkWidth * chunkWidth; i++) {
        blockIdArray[i] = 1;
    }

    //VoxelEngine::insertVoxels(blockPosFrameToIdFunction);

    VoxelEngine::test(VoxelEngine::insertVoxels(blockIdArray, chunkWidth, Vector3<int>(0, 0, 0)));

    delete blockIdArray;



    uint8_t* outBlockIdArray;

    VoxelEngine::test(VoxelEngine::getVoxels(&outBlockIdArray, chunkWidth, Vector3<int>(0, 0, 0)));

    for(int i = 0; i < chunkWidth * chunkWidth * chunkWidth; i++) {
        std::cout << (int)outBlockIdArray[i] << " ";
    }

    delete outBlockIdArray;


    VoxelEngine::inputLoop();

    VoxelEngine::test(VoxelEngine::cleanup());
}