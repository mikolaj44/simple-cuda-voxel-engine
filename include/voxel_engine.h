#pragma once

#define SDL_MAIN_HANDLED

#include <SDL.h>
#include <SDL_ttf.h>

#include <GL/glew.h>

#include "octree.cuh"
#include "chunk_generation.cuh"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>

class VoxelEngine {
public:
    static void init(int WINDOW_WIDTH, int WINDOW_HEIGHT);

    static void cleanup();

    static void handleInput();

    static void displayFrame();

    static void clearVoxels();

    template<typename blockPosToIdFunction>
    static void insertVoxels(blockPosToIdFunction func, Vector3 octreeCenter = Vector3(0,0,0)) {
        generateChunks(octree, octreeCenter, func, maxGridSize, blockSize, frameCount);
        frameCount++;
        frameCount %= UINT64_MAX;
    }

private:
    static bool wasInitialized;

    static void handleCameraMovement(int mouseX, int mouseY, int& prevMouseX, int& prevMouseY);

    static void initBlockTextures();

    static void writeAndRenderTexture();

    static void createCudaTexture();

    static Octree* octree;

    static SDL_Window* window;
    static SDL_Renderer* renderer;
    static SDL_Texture* texture;

    static SDL_Surface* textSurface;
    static SDL_Texture* textTexture;

    static GLuint textureID;
    static cudaGraphicsResource *cudaResource;

    static uint64_t frameCount;

    static BlockTexture** blockTextures;

    static dim3 maxGridSize;
    static dim3 blockSize;

    static int prevMouseX;
    static int prevMouseY;

    static SDL_Event event_;
};