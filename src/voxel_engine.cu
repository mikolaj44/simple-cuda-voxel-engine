#include "voxel_engine.cuh"
#include "block_variant/block_texture.cuh"
#include "block_variant/block_variant_manager.cuh"

#include "renderer/cuda_renderer.cuh"
#include "renderer/cuda_renderer_utils.cuh"
#include "octree/octree.cuh"

#include <iostream>
#include <filesystem>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>

bool VoxelEngine::isInitialized = false;
bool VoxelEngine::isTextureRenderingEnabled = true;
bool VoxelEngine::isCalculatingInsertLODsEnabled = true;
bool VoxelEngine::isMaterialColorOnlyEnabled = false;

unsigned int VoxelEngine::windowWidth;
unsigned int VoxelEngine::windowHeight;

Octree* VoxelEngine::octree;

uint64_t VoxelEngine::frameNumber = 0;

dim3 VoxelEngine::maxGridSize(32768,32768,32768);
dim3 VoxelEngine::blockSize(9,9,9);

void VoxelEngine::handleCameraMovement(int mouseX, int mouseY, int& prevMouseX, int& prevMouseY) {
    mouseX -= windowWidth / 2;
    mouseY -= windowHeight / 2;

    cameraAngle.y -= (prevMouseX - mouseX) * MOUSE_SENSITIVITY;
    cameraAngle.x += (prevMouseY - mouseY) * MOUSE_SENSITIVITY;

    if (cameraAngle.x < -M_PI / 2.0) {
        cameraAngle.x = -M_PI / 2.0;
    }
    else if (cameraAngle.x > M_PI / 2.0) {
        cameraAngle.x = M_PI / 2.0;
    }

    //cout << cameraAngle.x << endl;

    prevMouseX = mouseX;
    prevMouseY = mouseY;
}

cudaError_t VoxelEngine::clearVoxels(){
    return octree->clear();
}

template <bool displayFrame>
void VoxelEngine::inputLoop(void (*func)()) {
    bool quit = false;

    int prevMouseX = 0, prevMouseY = 0;

    while (!quit) {
        SDL_Event event_;

        while (SDL_PollEvent(&event_)) {
            switch (event_.type) {
                int x, y, z;

                case SDL_MOUSEMOTION:
                    if (mouseControls) {
                        handleCameraMovement(event_.motion.x, event_.motion.y, prevMouseX, prevMouseY);
                    }
                    break;

                case SDL_WINDOWEVENT:
                    switch (event_.window.event) {
                        case SDL_WINDOWEVENT_CLOSE:   // exit game
                            return;
                        default:
                            break;
                    }
                    break;

                case SDL_QUIT:
                    return;

                case SDL_KEYDOWN:

                    switch (event_.key.keysym.sym) {

                        case SDLK_z:
                            PLAYER_SPEED /= 2;
                            break;
                        case SDLK_x:
                            PLAYER_SPEED *= 2;
                            break;
                        case SDLK_c:
                            // octree->isTextureRenderingEnabled = !octree->isTextureRenderingEnabled;
                            break; 

                        case SDLK_UP:
                            cameraPos.x += sin(cameraAngle.y) * PLAYER_SPEED;
                            cameraPos.z += cos(cameraAngle.y) * PLAYER_SPEED;
                            break;
                        case SDLK_DOWN:
                            cameraPos.x -= sin(cameraAngle.y) * PLAYER_SPEED;
                            cameraPos.z -= cos(cameraAngle.y) * PLAYER_SPEED;
                            break;

                        case SDLK_LEFT:
                            cameraPos.x -= sin(cameraAngle.y + M_PI/2) * PLAYER_SPEED;
                            cameraPos.z -= cos(cameraAngle.y + M_PI/2) * PLAYER_SPEED;
                            break;
                        case SDLK_RIGHT:
                            cameraPos.x += sin(cameraAngle.y + M_PI/2) * PLAYER_SPEED;
                            cameraPos.z += cos(cameraAngle.y + M_PI/2) * PLAYER_SPEED;
                            break;

                        case SDLK_s:
                            cameraPos.y += PLAYER_SPEED;
                            break;
                        case SDLK_w:
                            cameraPos.y -= PLAYER_SPEED;
                            break;

                        case SDLK_q:
                            cameraAngle.y -= PLAYER_TURN_Y_SPEED;
                            break;
                        case SDLK_e:
                            cameraAngle.y += PLAYER_TURN_Y_SPEED;
                            break;

                        case SDLK_r:
                            cameraAngle.x += 0.1;
                            if (cameraAngle.x > 2 * M_PI)
                                cameraAngle.x = 0;
                            break;

                        case SDLK_f:
                            cameraAngle.x -= 0.1;
                            if (cameraAngle.x < 0)
                                cameraAngle.x = 2 * M_PI;
                            break;

                        case SDLK_t:
                            doOldRendering = !doOldRendering;
                            break;
                            
                        default:
                            break;
                    }
            }
        }

        if constexpr (displayFrame) {
            cuda_renderer::render(octree, cameraPos, cameraAngle, windowWidth, windowHeight, isTextureRenderingEnabled, isMaterialColorOnlyEnabled, 4096, 512);
            SDL_GL_SwapWindow(window);

            frameNumber++;
            frameNumber %= UINT64_MAX;
        }

        if(func != nullptr) {
            func();
        }
    }
}


cudaError_t VoxelEngine::init(unsigned int windowWidth_, unsigned int windowHeight_, unsigned int initialMaxOctreeDepth) {
    size_t freeBytes, totalBytes;

	cudaError_t error = cudaMemGetInfo(&freeBytes, &totalBytes);

    if(error != cudaSuccess) {
        return error;
    }

	printf("\n%zu bytes free out of %zu\n", freeBytes, totalBytes);

    if(isInitialized) {
        throw "The engine has already been initialized.";
    }

    VoxelEngine::windowWidth = windowWidth_;
    VoxelEngine::windowHeight = windowHeight_;

    const int threadsPerBlock = 600;
    const int blocksPerGrid = (windowWidth_ * windowHeight_ + threadsPerBlock - 1) / threadsPerBlock;

    error = block_variant_manager::init();

    if(error != cudaSuccess) {
        return error;
    }

    error = cuda_renderer::init(windowWidth_, windowHeight_);

    if(error != cudaSuccess) {
        return error;
    }

    error = cudaMallocManaged(&octree, sizeof(Octree));

    if(error != cudaSuccess) {
        return error;
    }

    error = octree->create(initialMaxOctreeDepth);

    if(error != cudaSuccess) {
        return error;
    }

    isInitialized = true;

    return error;
}

cudaError_t VoxelEngine::cleanup() {
    cudaError_t error = octree->cleanup();

    if(error != cudaSuccess) {
        return error;
    }

    error = cudaFree(octree);

    if(error != cudaSuccess) {
        return error;
    }

    error = cuda_renderer::cleanup();

    if(error != cudaSuccess) {
        return error;
    }

    error = block_variant_manager::cleanup();

    if(error != cudaSuccess) {
        return error;
    }

    size_t freeBytes, totalBytes;

    error = cudaMemGetInfo(&freeBytes, &totalBytes);

    if(error != cudaSuccess) {
        return error;
    }

    printf("\n%zu bytes free out of %zu\n\n", freeBytes, totalBytes);

    isInitialized = false;

    return error;
}

bool VoxelEngine::getIsInitialized() {
    return isInitialized;
}

cudaError_t VoxelEngine::setMaxOctreeDepth(int depth) {
    return octree->setMaxLevel(depth);
}

void VoxelEngine::setOctreeMinPos(Vector3<> pos) {
    octree->setMinPos(pos);
}

int VoxelEngine::getWindowWidth() {
    return windowWidth;
}

int VoxelEngine::getWindowHeight() {
    return windowHeight;
}

uint64_t VoxelEngine::getFrameNumber() {
    return frameNumber;
}

void VoxelEngine::setCameraPos(Vector3<> pos) {
    cameraPos = pos;
}

Vector3<> VoxelEngine::getCameraPos() {
    return cameraPos;
}

void VoxelEngine::setCameraAngle2D(Vector3<> angle) {
    cameraAngle = angle;
}

Vector3<> VoxelEngine::getCameraAngle2D() {
    return cameraAngle;
}

void VoxelEngine::setTextureRenderingEnabled(bool isEnabled) {
    isTextureRenderingEnabled = isEnabled;
}

bool VoxelEngine::getTextureRenderingEnabled() {
    return isTextureRenderingEnabled;
}

void VoxelEngine::setCalculatingInsertLODsEnabled(bool isEnabled) {
    isCalculatingInsertLODsEnabled = isEnabled;
}

bool VoxelEngine::getCalculatingInsertLODsEnabled() {
    return isCalculatingInsertLODsEnabled;
}

int VoxelEngine::getMaxOctreeLevelByGPU() {
    return Octree::getMaxOctreeLevelByGPU();
}

void VoxelEngine::setMaterialColorOnlyEnabled(bool isEnabled) {
    isMaterialColorOnlyEnabled = isEnabled;
}

bool VoxelEngine::getMaterialColorOnlyEnabled() {
    return isMaterialColorOnlyEnabled;
}


template void VoxelEngine::inputLoop<true>(void (*func)());
template void VoxelEngine::inputLoop<false>(void (*func)());