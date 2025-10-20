#include <iostream>
#include <vector>
#include <cmath>
#include <map>

#include "scve/internal/renderer/cuda_renderer.cuh"
#include "scve/internal/renderer/cuda_renderer_utils.cuh"
#include "scve/internal/block_variant/block_variant_manager.cuh"
#include "scve/internal/light/point_light_manager.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace scve {
namespace cuda_renderer {
    uchar4* devicePixels = nullptr;

    __managed__ float focalLength = 10000; // 350 // 1200 // 4000

    constexpr float epsilon = 0.01;

    namespace {
        __device__ void setPixel(uchar4* pixels, unsigned int windowWidth, unsigned int windowHeight, unsigned int sX, unsigned int sY, unsigned int r, unsigned int g, unsigned int b, unsigned int a) {
            pixels[(windowHeight - 1 - sY) * windowWidth + sX] = make_uchar4(r, g, b, a);
        }

        __device__ scve::Vector3<> getPhongIllumination(scve::Vector3<> startColor, scve::Vector3<> pos, scve::Vector3<> cameraPos, scve::Vector3<> normal, Material material, ManagedList<PointLight*>* pointLights, PointLight* ambientLight) {   
            scve::Vector3<> resultColor = scve::Vector3<>::mul(scve::Vector3<>::div(ambientLight->color, 255.0), ambientLight->intensity);

            startColor = startColor.div(255.0);

            scve::Vector3<> h = scve::Vector3<>(cameraPos.x - pos.x, cameraPos.y - pos.y, cameraPos.z - pos.z).norm();

            for(int i = 0; i < pointLights->size(); i++) {
                scve::Vector3<> ln = scve::Vector3<>(((*pointLights)[i])->pos.x - pos.x, ((*pointLights)[i])->pos.y - pos.y, ((*pointLights)[i])->pos.z - pos.z).norm();
            
                if (scve::Vector3<>::dot(ln, normal) < 0) {
                    continue;
                }  

                scve::Vector3<> dh = scve::Vector3<>::norm(scve::Vector3<>::sub(scve::Vector3<>::mul(normal, 2 * scve::Vector3<>::dot(ln, normal)), ln));
                
                scve::Vector3<> lightColor = scve::Vector3<>::div(((*pointLights)[i])->color, 255.0);

                float intensity = ((*pointLights)[i])->intensity;

                resultColor = resultColor.add(
                    scve::Vector3<>::add(
                        scve::Vector3<>::mul(
                            scve::Vector3<>::mul(startColor, lightColor), 
                            material.diffuse * scve::Vector3<>::dot(ln, normal) * intensity
                        ), 
                        scve::Vector3<>::mul(
                            lightColor, 
                            material.specular * powf(scve::Vector3<>::dot(h, dh), material.specularExponent) * intensity
                        )
                    )
                );
            }

            return resultColor.mul(255.0).clamp(255.0);
        }
        
        __device__ void setPixelByHitInfo(uchar4* pixels, int windowWidth, int windowHeight, Triple<scve::Vector3<int>, scve::Vector3<float>, uint8_t> intersectionData, scve::Vector3<> cameraPos, int sX, int sY, bool isTextureRenderingEnabled, bool isPhongIlluminationEnabled) {            
            using namespace block_variant_manager;

            uint8_t blockId = intersectionData.third - 1;

            if(isTextureRenderingEnabled && ((*blockVariants)[blockId])->texture == nullptr) {
                return;
            }
    
            int imgWidth;
            int imgHeight;
            int imgChannels;
                
            if(isTextureRenderingEnabled) {
                imgChannels = ((*blockVariants)[blockId])->texture->getChannels();
            }
    
            int blockX = intersectionData.first.x;
            int blockY = intersectionData.first.y;
            int blockZ = intersectionData.first.z;
    
            float x = intersectionData.second.x;
            float y = intersectionData.second.y;
            float z = intersectionData.second.z;
    
            scve::Vector3<> color;
            scve::Vector3<> normal;

            ImagePosition imagePosition;
            int imgX = 0;
            int imgY = 0;

            // check which side of the block we are on    
            if (equals(y, static_cast<float>(blockY), epsilon)) {  // top
                imagePosition = ImagePosition::TOP;

                if(isTextureRenderingEnabled) {
                    imgWidth  = ((*blockVariants)[blockId])->texture->getWidth(imagePosition);
                    imgHeight = ((*blockVariants)[blockId])->texture->getHeight(imagePosition);

                    imgX = static_cast<int>(absv(x - floorf(x)) * imgWidth);
                    imgY = static_cast<int>(absv(z - floorf(z)) * imgHeight);
                }
    
                normal = scve::Vector3<>(0, -1, 0);
            }
            else if (equals(y, static_cast<float>(blockY + 1.0), epsilon)) { // bottom
                imagePosition = ImagePosition::BOTTOM;

                if(isTextureRenderingEnabled) {
                    imgWidth  = ((*blockVariants)[blockId])->texture->getWidth(imagePosition);
                    imgHeight = ((*blockVariants)[blockId])->texture->getHeight(imagePosition);

                    imgX = static_cast<int>(absv(x - floorf(x)) * imgWidth);
                    imgY = static_cast<int>(absv(z - floorf(z)) * imgHeight);
                }
    
                normal = scve::Vector3<>(0, 1, 0);
            }
            else if (equals(x, static_cast<float>(blockX), epsilon)) { // left
                imagePosition = ImagePosition::LEFT;

                if(isTextureRenderingEnabled) {
                    imgWidth  = ((*blockVariants)[blockId])->texture->getWidth(imagePosition);
                    imgHeight = ((*blockVariants)[blockId])->texture->getHeight(imagePosition);

                    imgX = static_cast<int>(absv(z - floorf(z)) * imgWidth);
                    imgY = static_cast<int>(absv(y - floorf(y)) * imgHeight);
                }
    
                normal = scve::Vector3<>(-1, 0, 0);
            }
            else if (equals(x, static_cast<float>(blockX + 1.0), epsilon)) { // right
                imagePosition = ImagePosition::RIGHT;

                if(isTextureRenderingEnabled) {
                    imgWidth  = ((*blockVariants)[blockId])->texture->getWidth(imagePosition);
                    imgHeight = ((*blockVariants)[blockId])->texture->getHeight(imagePosition);

                    imgX = static_cast<int>(absv(z - floorf(z)) * imgWidth);
                    imgY = static_cast<int>(absv(y - floorf(y)) * imgHeight);
                }
    
                normal = scve::Vector3<>(1, 0, 0);
            }
            else if (equals(z, static_cast<float>(blockZ), epsilon)) { // front
                imagePosition = ImagePosition::FRONT;

                if(isTextureRenderingEnabled) {
                    imgWidth  = ((*blockVariants)[blockId])->texture->getWidth(imagePosition);
                    imgHeight = ((*blockVariants)[blockId])->texture->getHeight(imagePosition);

                    imgX = static_cast<int>(absv(x - floorf(x)) * imgWidth);
                    imgY = static_cast<int>(absv(y - floorf(y)) * imgHeight);
                }
    
                normal = scve::Vector3<>(0, 0, -1);
            }
            else if (equals(z, static_cast<float>(blockZ + 1.0), epsilon)) { // back
                imagePosition = ImagePosition::BACK;

                if(isTextureRenderingEnabled) {
                    imgWidth  = ((*blockVariants)[blockId])->texture->getWidth(imagePosition);
                    imgHeight = ((*blockVariants)[blockId])->texture->getHeight(imagePosition);

                    imgX = static_cast<int>(absv(x - floorf(x)) * imgWidth);
                    imgY = static_cast<int>(absv(y - floorf(y)) * imgHeight);
                }
    
                normal = scve::Vector3<>(0, 0, 1);
            }
            else {
                return;
            }

            if (isTextureRenderingEnabled) {
                color.x = ((*blockVariants)[blockId])->texture->getImage(imagePosition)[(imgY * imgWidth + imgX) * imgChannels];
                color.y = ((*blockVariants)[blockId])->texture->getImage(imagePosition)[(imgY * imgWidth + imgX) * imgChannels + 1];
                color.z = ((*blockVariants)[blockId])->texture->getImage(imagePosition)[(imgY * imgWidth + imgX) * imgChannels + 2];
            }
            else {
                color = ((*blockVariants)[blockId])->material.color;
            }
    
            if(isPhongIlluminationEnabled) {
                color = getPhongIllumination(color, scve::Vector3<>(x, y, z), cameraPos, normal, ((*blockVariants)[blockId])->material, point_light_manager::pointLights, point_light_manager::ambientLight);
            }

            setPixel(pixels, windowWidth, windowHeight, sX, sY, static_cast<int>(color.x), static_cast<int>(color.y), static_cast<int>(color.z), 255);
        }

        __global__ void renderKernel(uchar4* pixels, Octree* octree, scve::Vector3<> cameraPos, scve::Vector3<> cameraAngle2d, int windowWidth, int windowHeight, bool isTextureRenderingEnabled, bool isPhongIlluminationEnabled) {                        
            unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

            if (index >= windowWidth * windowHeight) {
                return;
            }

            int sX = index % windowWidth;
            int sY = index / windowWidth;

            float alpha = (atanf(-(sX - windowWidth  / 2) / focalLength) - cameraAngle2d.y + M_PI / 2); // horizontal angle
            float polar = (atanf(-(sY - windowHeight / 2) / focalLength) + cameraAngle2d.x + M_PI / 2); // vertical angle

            float dX = sin(polar) * cos(alpha);
            float dZ = sin(polar) * sin(alpha);
            float dY = cos(polar);

            Triple<scve::Vector3<int>, scve::Vector3<>, uint8_t> intersectionData = octree->getRayIntersectionData(cameraPos, scve::Vector3<>(dX, dY, dZ), sX, sY, 1);

            if(intersectionData.third == 0) {
                setPixel(pixels, windowWidth, windowHeight, sX, sY, point_light_manager::backgroundLight->color.x, point_light_manager::backgroundLight->color.y, point_light_manager::backgroundLight->color.z, 255);
            }
            else {
                setPixelByHitInfo(pixels, windowWidth, windowHeight, intersectionData, cameraPos, sX, sY, isTextureRenderingEnabled, isPhongIlluminationEnabled);
            }
        }
    }


    cudaError_t init(int windowWidth, int windowHeight) {
        return cuda_renderer_utils::initSDL(windowWidth, windowHeight);
    }

    cudaError_t cleanup() {
        cudaError_t lastError = cudaSuccess;

        cudaError_t error = cuda_renderer_utils::cleanupSDL();

        if(error != cudaSuccess) {
            lastError = error;
        }

        error = cudaFree(devicePixels);

        devicePixels = nullptr;

        if(error != cudaSuccess) {
            lastError = error;
        }

        return lastError;
    }

    // I'm not error checking for performance reasons
    void render(Octree* octree, scve::Vector3<> cameraPos, scve::Vector3<> cameraAngle2d, int windowWidth, int windowHeight, bool isTextureRenderingEnabled, bool isPhongIlluminationEnabled, unsigned int gridSize, unsigned int blockSize) {
        using namespace cuda_renderer_utils;

        cudaFree(devicePixels);

        cudaArray* cudaArrayPtr;
        cudaGraphicsMapResources(1, &cudaResource, 0);
        cudaGraphicsSubResourceGetMappedArray(&cudaArrayPtr, cudaResource, 0, 0);
    
        // Copy data to CUDA array
        cudaMalloc(&devicePixels, windowWidth * windowHeight * sizeof(uchar4));
    
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        
        renderKernel<<<gridSize,blockSize>>>(devicePixels, octree, cameraPos, cameraAngle2d, windowWidth, windowHeight, isTextureRenderingEnabled, isPhongIlluminationEnabled);

        cudaDeviceSynchronize();
    
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Kernel execution time: %f ms (%f fps)\n", milliseconds, 1.0 / (milliseconds / 1000.0));
        cudaDeviceSynchronize();
    
        // Copy memory from CUDA device to OpenGL texture
        cudaMemcpy2DToArray(cudaArrayPtr, 0, 0, devicePixels, windowWidth * sizeof(uchar4), windowWidth * sizeof(uchar4), windowHeight, cudaMemcpyDeviceToDevice);
    
        cudaGraphicsUnmapResources(1, &cudaResource, 0);
    
        glClear(GL_COLOR_BUFFER_BIT);
        
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, textureID);
        
        glBegin(GL_QUADS);
            glTexCoord2f(0, 1); glVertex2f(-1,  1);
            glTexCoord2f(0, 0); glVertex2f(-1, -1);
            glTexCoord2f(1, 0); glVertex2f( 1, -1);
            glTexCoord2f(1, 1); glVertex2f( 1,  1);
        glEnd();
        
        glDisable(GL_TEXTURE_2D);

        SDL_GL_SwapWindow(window);
    }
}
}