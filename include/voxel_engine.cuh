#pragma once

#include "octree/octree.cuh"

#include <cuda_runtime.h>
#include "light/point_light.cuh"
#include "block_variant/block_variant_manager.cuh"

class VoxelEngine {
public:
    static cudaError_t init(unsigned int windowWidth, unsigned int windowHeight, unsigned int initialMaxOctreeDepth = 1, unsigned int initialNumLights = 1);
    
    static cudaError_t cleanup();

    static cudaError_t clearVoxels();

    static cudaError_t setPointLights(std::vector<PointLight> pointLights);

    static void test(cudaError_t error);

    static cudaError_t setMaxOctreeDepth(int depth);

    static void setCameraPos(Vector3<> pos);

    static void displayFrame();

    static void setOctreeMinPos(Vector3<> pos);

    static Vector3<> getCameraPos();
    
    static void setCameraAngle2D(Vector3<> angle);
    
    static Vector3<> getCameraAngle2D();

    static bool getIsInitialized();

    static void setTextureRenderingEnabled(bool isEnabled);

    static bool getTextureRenderingEnabled();

    static void setCalculatingInsertLODsEnabled(bool isEnabled);

    static bool getCalculatingInsertLODsEnabled();

    static void setMaterialColorOnlyEnabled(bool isEnabled);

    static bool getMaterialColorOnlyEnabled();

    static int getMaxOctreeLevelByGPU();

    static cudaError_t setNumLights(unsigned int numLights);

    static unsigned int getWindowWidth();

    static unsigned int getWindowHeight();

    static uint64_t getFrameNumber();

    static float getCameraSpeed();

    static void setCameraSpeed(float speed);

    static float getCameraTurnSpeed();

    static void setCameraTurnSpeed(float speed);

    static float getMouseSensitivity();

    static void setMouseSensitivity(float sensitivity);

    static bool getKeyboardControlEnabled();

    static void setKeyboardControlEnabled(bool isEnabled);

    static bool getMouseControlEnabled();

    static void setMouseControlEnabled(bool isEnabled);

    static bool getPhongIlluminationEnabled();

    static void setPhongIlluminationEnabled(bool isEnabled);

    static void setAmbientLightColor(Vector3<> color);

    static Vector3<> getAmbientLightColor();

    static void setAmbientLightIntensity(float intensity);

    static float getAmbientLightIntensity();

    template<typename XYZFrameToIdFunction>
    static cudaError_t insertVoxels(XYZFrameToIdFunction func) {
        uint64_t totalVoxels = octree->getMaxSize();

        totalVoxels = totalVoxels * totalVoxels * totalVoxels;

        return octree->insertBlockByXYZFrameFunction(func, frameNumber, isCalculatingInsertLODsEnabled, (totalVoxels + insertionBlockSize - 1) / insertionBlockSize,  insertionBlockSize);
    }

    template<typename IdFrameToMaterialFunction>
    static void setMaterials(IdFrameToMaterialFunction func) {
        int numVariants = block_variant_manager::blockVariants->size();
        block_variant_manager::setBlocksVariantMaterialsKernel<<<1, numVariants>>>(func, frameNumber);
    }

    template <bool displayFrame = true>
    static void inputLoop(void (*func)() = nullptr);
private:
    static bool isInitialized;

    static bool isTextureRenderingEnabled;

    static bool isCalculatingInsertLODsEnabled;

    static bool isMaterialColorOnlyEnabled;

    static bool isKeyboardControlEnabled;

    static bool isMouseControlEnabled;

    static bool isPhongIlluminationEnabled;

    static unsigned int windowWidth, windowHeight;

    static const unsigned int insertionBlockSize = 512;

    static const unsigned int renderThreadsPerBlock = 600;
    static unsigned int renderBlocksPerGrid;

    static int prevMouseX;
    static int prevMouseY;

    static Octree* octree;

    static uint64_t frameNumber;

    static dim3 maxGridSize;
    static dim3 blockSize;

    static Vector3<> cameraPos;
    static Vector3<> cameraAngle;

    static float cameraSpeed;
    static float cameraTurnSpeed;
    static float mouseSensitivity;

    static void initBlockTextures();

    static void handleCameraMovement(int mouseX, int mouseY);

    static cudaError_t initLights();
};