#include "scve/voxel_engine.h"
#include "scve/internal/block_variant/block_texture.cuh"
#include "scve/internal/block_variant/block_variant_manager.cuh"
#include "scve/internal/light/point_light_manager.cuh"

#include "scve/internal/renderer/cuda_renderer.cuh"
#include "scve/internal/renderer/cuda_renderer_utils.cuh"
#include "scve/internal/octree/octree.cuh"

#include <iostream>
#include <filesystem>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>

namespace scve {

bool VoxelEngine::isInitialized = false;
bool VoxelEngine::isTextureRenderingEnabled = true;
bool VoxelEngine::isCalculatingInsertLODsEnabled = false;
bool VoxelEngine::isMouseControlEnabled = false;
bool VoxelEngine::isKeyboardControlEnabled = true;
bool VoxelEngine::isPhongIlluminationEnabled = true;
bool VoxelEngine::isDisplayingMemoryInfoEnabled = true;

unsigned int VoxelEngine::windowWidth;
unsigned int VoxelEngine::windowHeight;

Octree* VoxelEngine::octree;

uint64_t VoxelEngine::frameNumber = 0;

Vector3<> VoxelEngine::cameraPos = Vector3<>(0, 0, 0);
Vector3<> VoxelEngine::cameraAngle = Vector3<>(0, 0, 0);

float VoxelEngine::cameraSpeed = 1;
float VoxelEngine::cameraTurnSpeed = 0.1;
float VoxelEngine::mouseSensitivity = 0.0002;

int VoxelEngine::prevMouseX;
int VoxelEngine::prevMouseY;

unsigned int VoxelEngine::renderBlocksPerGrid;

namespace {
    // https://stackoverflow.com/questions/61277046/convert-just-a-hue-into-rgb
    __device__ __host__ scve::Vector3<> hueToRGB(float hue) {
        #ifdef __CUDA_ARCH__
            float kr = fmodf(5.0f + hue * 6.0f, 6.0f);
            float kg = fmodf(3.0f + hue * 6.0f, 6.0f);
            float kb = fmodf(1.0f + hue * 6.0f, 6.0f);
        #else
            float kr = std::fmod(5.0f + hue * 6.0f, 6.0f);
            float kg = std::fmod(3.0f + hue * 6.0f, 6.0f);
            float kb = std::fmod(1.0f + hue * 6.0f, 6.0f);
        #endif

        float r = (1.0f - maxv(minv(minv(kr, 4 - kr), 1.0f), 0.0f)) * 255.0f;
        float g = (1.0f - maxv(minv(minv(kg, 4 - kg), 1.0f), 0.0f)) * 255.0f;
        float b = (1.0f - maxv(minv(minv(kb, 4 - kb), 1.0f), 0.0f)) * 255.0f;

        return scve::Vector3<>(r, g, b);
    }
}

void VoxelEngine::handleCameraMovement(int mouseX, int mouseY) {
    mouseX -= windowWidth / 2;
    mouseY -= windowHeight / 2;

    cameraAngle.y -= (prevMouseX - mouseX) * mouseSensitivity;
    cameraAngle.x += (prevMouseY - mouseY) * mouseSensitivity;

    if (cameraAngle.x < -M_PI / 2.0) {
        cameraAngle.x = -M_PI / 2.0;
    }
    else if (cameraAngle.x > M_PI / 2.0) {
        cameraAngle.x = M_PI / 2.0;
    }

    if (cameraAngle.y < -M_PI / 2.0) {
        cameraAngle.y = -M_PI / 2.0;
    }
    else if (cameraAngle.y > M_PI / 2.0) {
        cameraAngle.y = M_PI / 2.0;
    }

    prevMouseX = mouseX;
    prevMouseY = mouseY;
}

cudaError_t VoxelEngine::clearVoxels(){
    return octree->clear();
}

void VoxelEngine::showCursorIfEnabled() {
    if(VoxelEngine::isMouseControlEnabled) {
        SDL_ShowCursor(SDL_DISABLE);
    }
    else {
        SDL_ShowCursor(SDL_ENABLE);
    }
}

void VoxelEngine::displayFrame() {
	SDL_ShowWindow(cuda_renderer_utils::window);
	
    cuda_renderer::render(octree, cameraPos, cameraAngle, windowWidth, windowHeight, isTextureRenderingEnabled, isPhongIlluminationEnabled, renderBlocksPerGrid, renderThreadsPerBlock);

    frameNumber++;
    frameNumber %= UINT64_MAX;
}

cudaError_t VoxelEngine::inputLoop(void (*func)(), bool displayFrame) {
    bool quit = false;

    SDL_ShowWindow(cuda_renderer_utils::window);

    while (!quit) {
        SDL_Event event_;

        while (SDL_PollEvent(&event_)) {
            switch (event_.type) {
                int x, y, z;

                case SDL_MOUSEMOTION:
                    if (isMouseControlEnabled) {
                        handleCameraMovement(event_.motion.x, event_.motion.y);
                    }
                    break;

                case SDL_WINDOWEVENT:
                    break;

                case SDL_QUIT:
                    break;

                case SDL_KEYDOWN:
                    if(!isKeyboardControlEnabled) {
                        break;
                    }
                    switch (event_.key.keysym.sym) {
                        case SDLK_z:
                            cameraSpeed /= 2;
                            break;
                        case SDLK_x:
                            cameraSpeed *= 2;
                            break;

                        case SDLK_b:
                            cameraTurnSpeed /= 2;
                            break;
                        case SDLK_n:
                            cameraTurnSpeed *= 2;
                            break;

                        case SDLK_c:
                            isTextureRenderingEnabled = !isTextureRenderingEnabled;
                            break;
                        case SDLK_v:
                            isPhongIlluminationEnabled = !isPhongIlluminationEnabled;
                            break;

                        case SDLK_UP:
                            cameraPos.x += sin(cameraAngle.y) * cameraSpeed;
                            cameraPos.z += cos(cameraAngle.y) * cameraSpeed;
                            break;
                        case SDLK_DOWN:
                            cameraPos.x -= sin(cameraAngle.y) * cameraSpeed;
                            cameraPos.z -= cos(cameraAngle.y) * cameraSpeed;
                            break;

                        case SDLK_LEFT:
                            cameraPos.x -= sin(cameraAngle.y + M_PI/2) * cameraSpeed;
                            cameraPos.z -= cos(cameraAngle.y + M_PI/2) * cameraSpeed;
                            break;
                        case SDLK_RIGHT:
                            cameraPos.x += sin(cameraAngle.y + M_PI/2) * cameraSpeed;
                            cameraPos.z += cos(cameraAngle.y + M_PI/2) * cameraSpeed;
                            break;

                        case SDLK_s:
                            cameraPos.y += cameraSpeed;
                            break;
                        case SDLK_w:
                            cameraPos.y -= cameraSpeed;
                            break;

                        case SDLK_q:
                            cameraAngle.y -= cameraTurnSpeed;
                            if (cameraAngle.y < 0)
                                cameraAngle.y = 2 * M_PI;
                            break;
                        case SDLK_e:
                            cameraAngle.y += cameraTurnSpeed;
                            if (cameraAngle.y > 2 * M_PI)
                                cameraAngle.y = 0;
                            break;

                        case SDLK_r:
                            cameraAngle.x += cameraTurnSpeed;
                            if (cameraAngle.x > 2 * M_PI)
                                cameraAngle.x = 0;
                            break;

                        case SDLK_f:
                            cameraAngle.x -= cameraTurnSpeed;
                            if (cameraAngle.x < 0)
                                cameraAngle.x = 2 * M_PI;
                            break;
                            
                        default:
                            break;
                    }
            }
        }

        if(displayFrame) {
            VoxelEngine::displayFrame();
        }

        if(func != nullptr) {
            func();
        }
    }

    return cleanup();
}

cudaError_t VoxelEngine::init(unsigned int windowWidth_, unsigned int windowHeight_, std::string texturesPath, unsigned int initialMaxOctreeDepth) {
    cudaError_t error = cudaSuccess;

    if(isDisplayingMemoryInfoEnabled) {
        size_t freeBytes, totalBytes;

        error = cudaMemGetInfo(&freeBytes, &totalBytes);

        if(error != cudaSuccess) {
            return error;
        }

	    printf("\ninit: %zu bytes free out of %zu\n", freeBytes, totalBytes);
    }

    if(isInitialized) {
        throw std::runtime_error("The engine has already been initialized.");
    }

    VoxelEngine::windowWidth = windowWidth_;
    VoxelEngine::windowHeight = windowHeight_;

    renderBlocksPerGrid = (windowWidth_ * windowHeight_ + renderThreadsPerBlock - 1) / renderThreadsPerBlock;

    error = cudaMallocManaged(&octree, sizeof(Octree));

    if(error != cudaSuccess) {
        return error;
    }

    error = octree->init(initialMaxOctreeDepth, isDisplayingMemoryInfoEnabled);

    if(error != cudaSuccess) {
        cleanup();
        return error;
    }

    setOctreeCenter(Vector3<int>(0, 0, 0));

    error = block_variant_manager::init(texturesPath, 127, texturesPath == "" ? true : false);

    if(error != cudaSuccess) {
        cleanup();
        return error;
    }

    error = point_light_manager::init(1);

    if(error != cudaSuccess) {
        cleanup();
        return error;
    }

    auto defaultIdFrameToMaterialFunction = [] __device__ (uint8_t blockId, uint64_t frameNumber) {
        return Material(hueToRGB((blockId) * 2.8346 / 360.0), 1.0, 0.0, 20.0);
    };

    setMaterials(defaultIdFrameToMaterialFunction);

    error = cuda_renderer::init(windowWidth_, windowHeight_);

    if(error != cudaSuccess) {
        cleanup();
        return error;
    }

    showCursorIfEnabled();

    isInitialized = true;

    return error;
}

cudaError_t VoxelEngine::cleanup() {
    cudaError_t lastError = cudaSuccess;
    
    cudaError_t error = octree->cleanup();

    if(error != cudaSuccess) {
        lastError = error;
    }

    error = cudaFree(octree);

    if(error != cudaSuccess) {
        lastError = error;
    }

    error = cuda_renderer::cleanup();

    if(error != cudaSuccess) {
        lastError = error;
    }

    error = block_variant_manager::cleanup();

    if(error != cudaSuccess) {
        lastError = error;
    }

    error = point_light_manager::cleanup();

    if(error != cudaSuccess) {
        lastError = error;
    }

    if(isDisplayingMemoryInfoEnabled) {
        size_t freeBytes, totalBytes;

        error = cudaMemGetInfo(&freeBytes, &totalBytes);

        if(error != cudaSuccess) {
            lastError = error;
        }

        printf("\ncleanup: %zu bytes free out of %zu\n\n", freeBytes, totalBytes);
    }

    if(error == cudaSuccess) {
        isInitialized = false;
    }

    return lastError;
}

void VoxelEngine::test(cudaError_t error) {
    if(error != cudaSuccess) {
        printf("Got CUDA error: %s", cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

cudaError_t VoxelEngine::setPointLights(const std::vector<PointLight>& pointLights) {
    PointLight ambientLight = *point_light_manager::ambientLight;
    PointLight backgroundLight = *point_light_manager::backgroundLight;
    
    cudaError_t error = point_light_manager::cleanup();

    if(error != cudaSuccess) {
        return error;
    }

    error = point_light_manager::init(pointLights.size());

    if(error != cudaSuccess) {
        return error;
    }

    for(int i = 0; i < pointLights.size(); i++) {
        (*(point_light_manager::pointLights))[i]->color = pointLights[i].color;
        (*(point_light_manager::pointLights))[i]->pos = pointLights[i].pos;
        (*(point_light_manager::pointLights))[i]->intensity = pointLights[i].intensity;
    }

    point_light_manager::ambientLight->color = ambientLight.color;
    point_light_manager::ambientLight->pos = ambientLight.pos;
    point_light_manager::ambientLight->intensity = ambientLight.intensity;

    point_light_manager::backgroundLight->color = backgroundLight.color;
    point_light_manager::backgroundLight->pos = backgroundLight.pos;
    point_light_manager::backgroundLight->intensity = backgroundLight.intensity;

    return error;
}

cudaError_t VoxelEngine::setPointLights(PointLight* pointLights, unsigned int numLights) {
    PointLight ambientLight = *point_light_manager::ambientLight;
    PointLight backgroundLight = *point_light_manager::backgroundLight;
    
    cudaError_t error = point_light_manager::cleanup();

    if(error != cudaSuccess) {
        return error;
    }

    error = point_light_manager::init(numLights);

    if(error != cudaSuccess) {
        return error;
    }

    for(int i = 0; i < numLights; i++) {
        (*(point_light_manager::pointLights))[i]->color = pointLights[i].color;
        (*(point_light_manager::pointLights))[i]->pos = pointLights[i].pos;
        (*(point_light_manager::pointLights))[i]->intensity = pointLights[i].intensity;
    }

    point_light_manager::ambientLight->color = ambientLight.color;
    point_light_manager::ambientLight->pos = ambientLight.pos;
    point_light_manager::ambientLight->intensity = ambientLight.intensity;

    point_light_manager::backgroundLight->color = backgroundLight.color;
    point_light_manager::backgroundLight->pos = backgroundLight.pos;
    point_light_manager::backgroundLight->intensity = backgroundLight.intensity;

    return error;
}

cudaError_t VoxelEngine::insertVoxels(uint8_t* hostBlockIdArray, unsigned int chunkWidth, Vector3<int> startOffset) {
    size_t totalVoxels = chunkWidth;

    totalVoxels = totalVoxels * totalVoxels * totalVoxels;

    uint8_t* deviceBlockIdArray;

    cudaError_t error = cudaMalloc(&deviceBlockIdArray, totalVoxels * sizeof(uint8_t));

    if(error != cudaSuccess) {
        return error;
    }

    error = cudaMemcpy(deviceBlockIdArray, hostBlockIdArray, totalVoxels * sizeof(uint8_t), cudaMemcpyHostToDevice);

    if(error != cudaSuccess) {
        cudaFree(deviceBlockIdArray);
        return error;
    }

    error = octree->insertBlocks(deviceBlockIdArray, startOffset, isCalculatingInsertLODsEnabled, chunkWidth, (totalVoxels + insertionBlockSize - 1) / insertionBlockSize, minv(static_cast<size_t>(insertionBlockSize), totalVoxels));

    if(error != cudaSuccess) {
        cudaFree(deviceBlockIdArray);
        return error;
    }

    return cudaFree(deviceBlockIdArray);
}

void VoxelEngine::setMaterials(const std::unordered_map<uint8_t, Material>& materialMap, bool setAbsentMaterialsToDefault) {
    for(int i = 1; i <= 127; i++) {
        if(materialMap.find(i) != materialMap.end()) {
            (*(block_variant_manager::blockVariants))[i - 1]->material = materialMap.at(i);
        }
        else if(setAbsentMaterialsToDefault) {
            (*(block_variant_manager::blockVariants))[i - 1]->material = Material(hueToRGB((i) * 2.8346 / 360.0), 1.0, 0.0, 20.0);
        }
    }
}

cudaError_t VoxelEngine::getVoxels(uint8_t** hostBlockIdArrayPtr, unsigned int chunkWidth, Vector3<int> startOffset) {
    size_t totalVoxels = chunkWidth;

    totalVoxels = totalVoxels * totalVoxels * totalVoxels;

    uint8_t* deviceBlockIdArray;

    cudaError_t error = cudaMalloc(&deviceBlockIdArray, totalVoxels * sizeof(uint8_t));

    if(error != cudaSuccess) {
        return error;
    }

    error = octree->getBlocks(deviceBlockIdArray, startOffset, chunkWidth, (totalVoxels + insertionBlockSize - 1) / insertionBlockSize, minv((size_t)insertionBlockSize, totalVoxels));

    if(error != cudaSuccess) {
        cudaFree(deviceBlockIdArray);
        return error;
    }

    *hostBlockIdArrayPtr = new uint8_t[totalVoxels];

    error = cudaMemcpy(*hostBlockIdArrayPtr, deviceBlockIdArray, totalVoxels * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    if(error != cudaSuccess) {
        cudaFree(deviceBlockIdArray);
        delete[] *hostBlockIdArrayPtr;
        return error;
    }

    error = cudaFree(deviceBlockIdArray);

    if(error != cudaSuccess) {
        delete[] *hostBlockIdArrayPtr;
        return error;
    }

    return error;
}

cudaError_t VoxelEngine::getPixels(uchar4** hostPixelArrayPtr) {
    size_t arrayLength = windowHeight * windowWidth * 4;

    *hostPixelArrayPtr = new uchar4[arrayLength];

    cudaError_t error = cudaMemcpy(*hostPixelArrayPtr, cuda_renderer::devicePixels, arrayLength * sizeof(uchar4), cudaMemcpyDeviceToHost);

    if(error != cudaSuccess) {
        delete[] *hostPixelArrayPtr;
        return error;
    }

    return error;
}

bool VoxelEngine::getIsInitialized() {
    return isInitialized;
}

cudaError_t VoxelEngine::setOctreeMaxDepth(int depth) {
    Vector3<int> minPos = octree->getMinPos();

    cudaError_t error = octree->cleanup();

    if(error != cudaSuccess) {
        return error;
    }

    return octree->init(minPos.x, minPos.y, minPos.z, depth, isDisplayingMemoryInfoEnabled);
}

void VoxelEngine::setOctreeMinPos(Vector3<int> pos) {
    octree->setMinPos(pos);
}

Vector3<int> VoxelEngine::getOctreeMinPos() {
    return octree->getMinPos();
}

Vector3<int> VoxelEngine::getOctreeCenter() {
    return Vector3<int>::add(octree->getMinPos(), Vector3<int>(getOctreeMaxSize()).div(2));
}

void VoxelEngine::setOctreeCenter(Vector3<int> pos) {
    octree->setMinPos(pos.sub(Vector3<int>(getOctreeMaxSize()).div(2)));
}

int VoxelEngine::getOctreeMaxSize() {
    return octree->getMaxSize();
}

unsigned int VoxelEngine::getWindowWidth() {
    return windowWidth;
}

unsigned int VoxelEngine::getWindowHeight() {
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

void VoxelEngine::setPropagatingInsertLODsEnabled(bool isEnabled) {
    isCalculatingInsertLODsEnabled = isEnabled;
}

bool VoxelEngine::getPropagatingInsertLODsEnabled() {
    return isCalculatingInsertLODsEnabled;
}

unsigned int VoxelEngine::getMaxOctreeLevelByGPU() {
    return Octree::getMaxOctreeLevelByGPU();
}

float VoxelEngine::getCameraMoveSpeed() {
    return cameraSpeed;
}

void VoxelEngine::setCameraMoveSpeed(float speed) {
    cameraSpeed = speed;
}

float VoxelEngine::getCameraTurnSpeed() {
    return cameraTurnSpeed;
}

void VoxelEngine::setCameraTurnSpeed(float speed) {
    cameraTurnSpeed = speed;
}

float VoxelEngine::getMouseSensitivity() {
    return mouseSensitivity;
}

void VoxelEngine::setMouseSensitivity(float sensitivity) {
    mouseSensitivity = sensitivity;
}

bool VoxelEngine::getKeyboardControlEnabled() {
    return isKeyboardControlEnabled;
}

void VoxelEngine::setKeyboardControlEnabled(bool isEnabled) {
    isKeyboardControlEnabled = isEnabled;
}

bool VoxelEngine::getMouseControlEnabled() {
    return isMouseControlEnabled;
}

void VoxelEngine::setMouseControlEnabled(bool isEnabled) {
    isMouseControlEnabled = isEnabled;

    showCursorIfEnabled();
}

bool VoxelEngine::getPhongIlluminationEnabled() {
    return isPhongIlluminationEnabled;
}

void VoxelEngine::setPhongIlluminationEnabled(bool isEnabled) {
    isPhongIlluminationEnabled = isEnabled;
}

bool VoxelEngine::getDisplayingMemoryInfoEnabled() {
    return isDisplayingMemoryInfoEnabled;
}

void VoxelEngine::setDisplayingMemoryInfoEnabled(bool isEnabled) {
    isDisplayingMemoryInfoEnabled = isEnabled;
}

void VoxelEngine::setAmbientLightColor(Vector3<> color) {
    point_light_manager::ambientLight->color = color;
}

Vector3<> VoxelEngine::getAmbientLightColor() {
    return point_light_manager::ambientLight->color;
}

void VoxelEngine::setAmbientLightIntensity(float intensity) {
    point_light_manager::ambientLight->intensity = intensity;
}

float VoxelEngine::getAmbientLightIntensity() {
    return point_light_manager::ambientLight->intensity;
}

void VoxelEngine::setBackgroundColor(Vector3<> color) {
    point_light_manager::backgroundLight->color = color;
}

Vector3<> VoxelEngine::getBackgroundColor() {
    return point_light_manager::backgroundLight->color;
}

void VoxelEngine::setFocalLength(float focalLength) {
    cuda_renderer::focalLength = focalLength;
}

float VoxelEngine::getFocalLength() {
    return cuda_renderer::focalLength;
}

}
