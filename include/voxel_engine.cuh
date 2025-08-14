#pragma once

#include "octree/octree.cuh"

#include <cuda_runtime.h>

#include "chunk_generation.cuh"

class VoxelEngine {
public:
    static cudaError_t init(int windowWidth, int windowHeight);
    
    static cudaError_t cleanup();

    static cudaError_t clearVoxels();

    static void displayFrame();

    template<typename XYZToIdFunction>
    static void insertVoxels(XYZToIdFunction func, Vector3<> octreeCenter = Vector3<>()) {
        generateChunks(octree, octreeCenter, func, maxGridSize, blockSize, frameNumber);
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