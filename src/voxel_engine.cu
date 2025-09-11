#include "voxel_engine.cuh"
#include "block_variant/block_texture.cuh"
#include "block_variant/block_variant_manager.cuh"
#include "light/point_light_manager.cuh"

#include "renderer/cuda_renderer.cuh"
#include "renderer/cuda_renderer_utils.cuh"
#include "octree/octree.cuh"

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

unsigned int VoxelEngine::windowWidth;
unsigned int VoxelEngine::windowHeight;

Octree* VoxelEngine::octree;

uint64_t VoxelEngine::frameNumber = 0;

dim3 VoxelEngine::maxGridSize(32768,32768,32768);
dim3 VoxelEngine::blockSize(9,9,9);

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
    __device__ scve::Vector3<> hueToRGB(float hue) {
        float kr = fmodf(5.0f + hue * 6.0f, 6.0f);
        float kg = fmodf(3.0f + hue * 6.0f, 6.0f);
        float kb = fmodf(1.0f + hue * 6.0f, 6.0f);

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

    prevMouseX = mouseX;
    prevMouseY = mouseY;
}

cudaError_t VoxelEngine::clearVoxels(){
    return octree->clear();
}

void VoxelEngine::inputLoop(void (*func)(), bool displayFrame) {
    bool quit = false;

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
                            break;
                        case SDLK_e:
                            cameraAngle.y += cameraTurnSpeed;
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
                            
                        default:
                            break;
                    }
            }
        }

        if(displayFrame) {
            cuda_renderer::render(octree, cameraPos, cameraAngle, windowWidth, windowHeight, isTextureRenderingEnabled, isPhongIlluminationEnabled, renderBlocksPerGrid, renderThreadsPerBlock);
            SDL_GL_SwapWindow(window);

            frameNumber++;
            frameNumber %= UINT64_MAX;
        }

        if(func != nullptr) {
            func();
        }
    }
}

cudaError_t VoxelEngine::init(unsigned int windowWidth_, unsigned int windowHeight_, std::string texturesPath, unsigned int initialMaxOctreeDepth) {
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

    renderBlocksPerGrid = (windowWidth_ * windowHeight_ + renderThreadsPerBlock - 1) / renderThreadsPerBlock;

    error = cudaMallocManaged(&octree, sizeof(Octree));

    if(error != cudaSuccess) {
        return error;
    }

    error = octree->init(initialMaxOctreeDepth);

    if(error != cudaSuccess) {
        return error;
    }

    if(texturesPath == "") {
        error = block_variant_manager::init(texturesPath, 127, true);
    }
    else {
        error = block_variant_manager::init(texturesPath, 127, false);
    }

    if(error != cudaSuccess) {
        return error;
    }

    error = point_light_manager::init(1);

    if(error != cudaSuccess) {
        return error;
    }

    auto defaultIdFrameToMaterialFunction = [] __device__ (uint8_t blockId, uint64_t frameNumber) {
        return Material(hueToRGB((blockId) * 2.8346 / 360.0), 1.0, 0.0, 20.0);
    };

    setMaterials(defaultIdFrameToMaterialFunction);

    error = cuda_renderer::init(windowWidth_, windowHeight_);

    if(error != cudaSuccess) {
        return error;
    }

    isInitialized = true;

    return error;
}

cudaError_t VoxelEngine::cleanup() {
    cudaError_t error = cudaSuccess;
    
    error = octree->cleanup();

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

    error = point_light_manager::cleanup();

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

    error = octree->insertBlocks(deviceBlockIdArray, startOffset, isCalculatingInsertLODsEnabled, chunkWidth, (totalVoxels + insertionBlockSize - 1) / insertionBlockSize, minv((size_t)insertionBlockSize, totalVoxels));

    if(error != cudaSuccess) {
        cudaFree(deviceBlockIdArray);
        return error;
    }

    return cudaFree(deviceBlockIdArray);
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

cudaError_t VoxelEngine::setMaxOctreeDepth(int depth) {
    return octree->setMaxLevel(depth);
}

void VoxelEngine::setOctreeMinPos(Vector3<> pos) {
    octree->setMinPos(pos);
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

void VoxelEngine::setCalculatingInsertLODsEnabled(bool isEnabled) {
    isCalculatingInsertLODsEnabled = isEnabled;
}

bool VoxelEngine::getCalculatingInsertLODsEnabled() {
    return isCalculatingInsertLODsEnabled;
}

unsigned int VoxelEngine::getMaxOctreeLevelByGPU() {
    return Octree::getMaxOctreeLevelByGPU();
}

float VoxelEngine::getCameraSpeed() {
    return cameraSpeed;
}

void VoxelEngine::setCameraSpeed(float speed) {
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
}

bool VoxelEngine::getPhongIlluminationEnabled() {
    return isPhongIlluminationEnabled;
}

void VoxelEngine::setPhongIlluminationEnabled(bool isEnabled) {
    isPhongIlluminationEnabled = isEnabled;
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