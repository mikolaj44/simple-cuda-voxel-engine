#pragma once

#include "octree/octree.cuh"
#include "octree/octree_insertion_methods.cuh"

#include <cuda_runtime.h>

#include "chunk_generation.cuh"

class VoxelEngine {
public:
    static cudaError_t init(unsigned int windowWidth, unsigned int windowHeight, unsigned int initialMaxOctreeDepth = 1);
    
    static cudaError_t cleanup();

    static cudaError_t clearVoxels();

    static cudaError_t setMaxOctreeDepth(int depth);

    static void displayFrame();

    static void setOctreeMinPos(Vector3<> pos);

    template<typename XYZFrameToIdFunction>
    static void insertVoxels(XYZFrameToIdFunction func) {
        octree_insertion_methods::insertBlockByXYZFrameFunction(octree, func, frameNumber, 65535, 900);
        frameNumber++;
        frameNumber %= UINT64_MAX;
    }

    template <bool displayFrame = true>
    static void inputLoop(void (*func)() = nullptr);


    static int getWindowWidth();

    static int getWindowHeight();

    static uint64_t getFrameNumber();
private:
    static bool wasInitialized;

    static unsigned int windowWidth, windowHeight;

    static Octree* octree;

    static uint64_t frameNumber;

    static dim3 maxGridSize;
    static dim3 blockSize;

    static void initBlockTextures();

    static void handleCameraMovement(int mouseX, int mouseY, int& prevMouseX, int& prevMouseY);
};