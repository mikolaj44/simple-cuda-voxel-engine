#pragma once

#include "octree/octree.cuh"

#include <cuda_runtime.h>
#include "block_variant/block_variant_manager.cuh"

class VoxelEngine {
public:
    static cudaError_t init(unsigned int windowWidth, unsigned int windowHeight, unsigned int initialMaxOctreeDepth = 1);
    
    static cudaError_t cleanup();

    static cudaError_t clearVoxels();

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

    template<typename XYZFrameToIdFunction>
    static cudaError_t insertVoxels(XYZFrameToIdFunction func) {
        uint64_t totalVoxels = octree->getMaxSize();
        totalVoxels = totalVoxels * totalVoxels * totalVoxels;

        return octree->insertBlockByXYZFrameFunction(func, frameNumber, isCalculatingInsertLODsEnabled, (totalVoxels + insertionBlockSize - 1) / insertionBlockSize,  insertionBlockSize);
    }

    template<typename IdFrameToMaterialFunction>
    static void setMaterials(IdFrameToMaterialFunction func) {
        block_variant_manager::setMaterials(func, frameNumber);
    }

    template <bool displayFrame = true>
    static void inputLoop(void (*func)() = nullptr);

    static int getWindowWidth();

    static int getWindowHeight();

    static uint64_t getFrameNumber();
private:
    static bool isInitialized;

    static bool isTextureRenderingEnabled;

    static bool isCalculatingInsertLODsEnabled;

    static bool isMaterialColorOnlyEnabled;

    static unsigned int windowWidth, windowHeight;

    static const unsigned int insertionBlockSize = 512;

    static Octree* octree;

    static uint64_t frameNumber;

    static dim3 maxGridSize;
    static dim3 blockSize;

    static void initBlockTextures();

    static void handleCameraMovement(int mouseX, int mouseY, int& prevMouseX, int& prevMouseY);
};