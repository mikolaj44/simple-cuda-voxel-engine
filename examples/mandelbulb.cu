#include <iostream>

#include "scve/voxel_engine.h"

// This functor provides a device-side (GPU) function that generates the mandelbulb fractal
// I got this algorithm from here: https://iquilezles.org/articles/mandelbulb/
class MyXYZFrameToIdFunctor : public scve::XYZFrameToIdFunctor {
    __device__ uint8_t operator()(int x, int y, int z, uint64_t frameNumber) const override {
        //return int(sqrtf(x*x + y*y + z*z)) % 6 + 1;

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
        
        return int(sqrtf(x*x + y*y + z*z)) % 6 + 1;
    }
};

// This is the same function that the engine uses for converting hue to RGB, we will provide our own mapping with it later
scve::Vector3<> hueToRGB(float hue) {
    float kr = std::fmod(5.0f + hue * 6.0f, 6.0f);
    float kg = std::fmod(3.0f + hue * 6.0f, 6.0f);
    float kb = std::fmod(1.0f + hue * 6.0f, 6.0f);

    float r = (1.0f - max(min(min(kr, 4 - kr), 1.0f), 0.0f)) * 255.0f;
    float g = (1.0f - max(min(min(kg, 4 - kg), 1.0f), 0.0f)) * 255.0f;
    float b = (1.0f - max(min(min(kb, 4 - kb), 1.0f), 0.0f)) * 255.0f;

    return scve::Vector3<>(r, g, b);
}

int main() {
    using namespace scve;

    // Initialize the engine while checking for CUDA error messages with VoxelEngine::test and catching any exceptions with loading textures
    try {
       VoxelEngine::test(VoxelEngine::init(800, 800, "D:\\Pulpit\\simple-cuda-voxel-engine\\examples\\textures", 10));
    }
    catch (const std::exception& e) {
        std::cout << "Caught exception: " << e.what() << "\n";
        return 1;
    }



    // Set the position of the camera further back, so the fractal is visible
    VoxelEngine::setCameraPos(Vector3<>(0, 0, -10000));

    // Enable the texture rendering mode (disable it by pressing "c"), on by default
    VoxelEngine::setTextureRenderingEnabled(true);

    // Enable keyboard control - it's enabled by default as well
    VoxelEngine::setKeyboardControlEnabled(true);

    // Disable mouse control - it's disabled by default too
    VoxelEngine::setMouseControlEnabled(false);

    // Enable Phong illumination - this feature is still experimental in its accuracy
    VoxelEngine::setPhongIlluminationEnabled(true);

    // Disable calculating insert LODs - this feature is still being implemented (off by default)
    VoxelEngine::setPropagatingInsertLODsEnabled(false);



    // Add two point lights behind the camera - red and blue
    VoxelEngine::setPointLights({PointLight(Vector3<>(5000, 0, -12000), Vector3<>(255, 0, 0))}); // , PointLight(Vector3<>(-5000, -0, -11000), Vector3<>(0, 0, 255))



    // You can set materials with a host-allocated map (like below) or a functor (IdFrameToMaterialFunctor):
    std::unordered_map<uint8_t, Material> map;

    // This mimics the default material mapping but does it for the first 6 block ids instead of 127
    for(int i = 1; i <= 6; i++) {
        map[i] = Material(hueToRGB((i) * 60.0 / 360.0), 1.0, 0.0, 20.0);
    }

    VoxelEngine::setMaterials(map);



    // Insert the voxels using the XYZFrameToIdFunctor functor
    VoxelEngine::insertVoxels(MyXYZFrameToIdFunctor());


    // Start the input loop that also renders frames at the end of the program (displayFrame = true)
    VoxelEngine::test(VoxelEngine::inputLoop());
}