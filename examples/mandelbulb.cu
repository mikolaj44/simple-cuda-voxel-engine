#include <iostream>

#include "scve/voxel_engine.h"

class MyXYZFrameToIdFunctor : public scve::XYZFrameToIdFunctor {
    __host__ __device__ uint8_t operator()(int x, int y, int z, uint64_t frameNumber) override {
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

            if(wX * wX + wY * wY + wZ * wZ > 0.6) {
                return 0;
            }
        }
        
        return int(sqrtf(x*x + y*y + z*z)) % 127 + 1;
    }
};

int main() {
    using namespace scve;
    
    try {
       VoxelEngine::test(VoxelEngine::init(1920, 1080, "/home/mikolaj/Desktop/cuda-voxel-engine/res/textures/", 10));
    }
    catch (const std::exception& e) {
        std::cout << "Caught exception: " << e.what() << "\n";
        return 1;
    }

    VoxelEngine::setCameraPos(Vector3<>(0, 0, -10000));

    VoxelEngine::setTextureRenderingEnabled(false);

    VoxelEngine::setPropagatingInsertLODsEnabled(false);

    VoxelEngine::setKeyboardControlEnabled(true);

    VoxelEngine::setMouseControlEnabled(false);

    VoxelEngine::setPhongIlluminationEnabled(false);



    VoxelEngine::insertVoxels(MyXYZFrameToIdFunctor());



    VoxelEngine::inputLoop();

    VoxelEngine::test(VoxelEngine::cleanup());
}