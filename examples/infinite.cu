#include <iostream>

#include "scve/voxel_engine.h"

// This is just for you to change if you want to have textures or not
constexpr bool TEXTURE_RENDERING_ENABLED = false;

// Assuming that there are 6 textures in your texture folder
constexpr uint8_t TEXTURE_COUNT = 6;

// A helper to know what id's we can insert
constexpr uint8_t MOD = TEXTURE_RENDERING_ENABLED ? TEXTURE_COUNT : 127;

// This functor provides a device-side (GPU) function that just fills the octree halfway up
class MyXYZFrameToIdFunctor : public scve::XYZFrameToIdFunctor {
    __device__ uint8_t operator()(int x, int y, int z, uint64_t frameNumber) const override {
        // The coordinate system puts 0 at the top, y coordinate grows going down

        if(y != 0) {
            return 0;
        }

        x = fabsf(x);
        z = fabsf(z);

        if(x % 10 <= 6 && z % 10 <= 6) {
            return int(sqrtf(x*x + z*z)) % MOD + 1;
        }
        if(x % 3 <= 1 && z % 3 == 1) {
            return int(sqrtf(x + z)) % MOD + 1;
        }

        return 60;
    }
};

// The callback that will move the octree to the camera position and insert the voxels
void myFunc() {
    using namespace scve;

    VoxelEngine::clearVoxels();

    Vector3<int> cameraPos = VoxelEngine::getCameraPos();

    // Move it slightly in front of the camera, you need to fly up with W
    VoxelEngine::setOctreeCenter(Vector3<int>(cameraPos.x, 500, cameraPos.z + 2000));

    VoxelEngine::insertVoxels(MyXYZFrameToIdFunctor());
}

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
       VoxelEngine::test(VoxelEngine::init(800, 800, "", 10)); // Provide "" if you don't want textures or a path like "D:\\Pulpit\\simple-cuda-voxel-engine\\examples\\textures"
    }
    catch (const std::exception& e) {
        std::cout << "Caught exception: " << e.what() << "\n";
        return 1;
    }



    // Set the position of the camera further back, so the fractal is visible
    VoxelEngine::setCameraPos(Vector3<>(0, 0, -10000));

    // Disable the texture rendering mode (disable it by pressing "c"), on by default
    VoxelEngine::setTextureRenderingEnabled(false);

    // Enable keyboard control - it's enabled by default as well
    VoxelEngine::setKeyboardControlEnabled(true);

    // Disable mouse control - it's disabled by default too
    VoxelEngine::setMouseControlEnabled(false);

    // Enable Phong illumination - this feature is still experimental in its accuracy
    VoxelEngine::setPhongIlluminationEnabled(false);

    // Disable calculating insert LODs - this feature is still being implemented (off by default)
    VoxelEngine::setPropagatingInsertLODsEnabled(false);



    // Add two point lights behind the camera - red and blue
    VoxelEngine::setPointLights({PointLight(Vector3<>(5000, 0, -12000), Vector3<>(255, 0, 0))}); // , PointLight(Vector3<>(-5000, -0, -11000), Vector3<>(0, 0, 255))



    // You can set materials with a host-allocated map (like below) or a functor (IdFrameToMaterialFunctor):
    std::unordered_map<uint8_t, Material> map;

    // This mimics the default material mapping but does it for the first TEXTURE_COUNT block ids instead of 127
    for(int i = 1; i <= TEXTURE_COUNT; i++) {
        map[i] = Material(hueToRGB((i) * 60.0 / 360.0), 1.0, 0.0, 20.0);
    }

    VoxelEngine::setMaterials(map);



    // Start the input loop that also renders frames at the end of the program (displayFrame = true)
    VoxelEngine::test(VoxelEngine::inputLoop(&myFunc));
}