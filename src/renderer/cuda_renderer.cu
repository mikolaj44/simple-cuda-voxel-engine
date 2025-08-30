#include <iostream>
#include <vector>
#include <cmath>
#include <map>

#include "renderer/cuda_renderer.cuh"
#include "renderer/cuda_renderer_utils.cuh"
#include "chunk_generation.cuh"
#include "globals.cuh"
#include "blocks_data.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

constexpr int blocksAmount = 4;
constexpr float epsilon = 0.002;

namespace cuda_renderer {
    namespace {
        __device__ void setPixel(uchar4* pixels, int sX, int sY, int r, int g, int b, int a) {
            pixels[(SCREEN_HEIGHT - 1 - sY) * SCREEN_WIDTH + sX] = make_uchar4(r, g, b, a);
        }

        // https://stackoverflow.com/questions/61277046/convert-just-a-hue-into-rgb
        __device__ Vector3<> hueToRGB(float hue){
            float kr = remainderf(5 + hue * 6, 6);
            float kg = remainderf(3 + hue * 6, 6);
            float kb = remainderf(1 + hue * 6, 6);

            unsigned int r = (1 - maxv(minv(minv(kr, 4-kr), 1.0f), 0.0f)) * 255;
            unsigned int g = (1 - maxv(minv(minv(kg, 4-kg), 1.0f), 0.0f)) * 255;
            unsigned int b = (1 - maxv(minv(minv(kb, 4-kb), 1.0f), 0.0f)) * 255;

            return Vector3<>(r, g, b);
        }

        __device__ Vector3<> getPhongIllumination(Vector3<> startColor, Vector3<> pos, Vector3<> cameraPos, Vector3<> normal, Material material, PointLight light){        
            Vector3<> ln = Vector3<>(light.pos.x - pos.x, light.pos.y - pos.y, light.pos.z - pos.z).norm();
        
            if (normal.dot(ln) < 0) {
                return Vector3<>(0, 0, 0);
            }
                
            Vector3<> h = Vector3<>(cameraPos.x - pos.x, cameraPos.y - pos.y, cameraPos.z - pos.z).norm();

            Vector3<> dh = normal.mul(2 * ln.dot(normal)).sub(ln).norm();

            Vector3<> lighting = light.color.mul(material.diffuse * normal.dot(ln));
            
            return lighting.clamp(255).div(255.0).mul(startColor);
        }
        
        __device__ void setPixelByHitInfo(uchar4* pixels, Triple<Vector3<int>, Vector3<float>, uint8_t> intersectionData, Vector3<> cameraPos, int sX, int sY, bool isTextureRenderingEnabled) {            
            uint8_t blockId = intersectionData.third;
            
            if (blockId == 0 || (isTextureRenderingEnabled && blockId > blocksAmount)) {
                return;
            }
    
            int imgWidth, imgHeight, imgChannels;
    
            if(isTextureRenderingEnabled){
                blockId -= 1;
                imgWidth = blockVariants[blockId]->texture->width;
                imgHeight = blockVariants[blockId]->texture->height;
                imgChannels = blockVariants[blockId]->texture->channels;
            }
    
            int blockX = intersectionData.first.x;
            int blockY = intersectionData.first.y;
            int blockZ = intersectionData.first.z;
    
            float x = intersectionData.second.x;
            float y = intersectionData.second.y;
            float z = intersectionData.second.z;
    
            int imgX = 0, imgY = 0;
    
            int r, g, b;
            Vector3<> normal;

            // check which side of the block we are on    
            if (equals(y, (float)blockY, epsilon)) { // top
                if(isTextureRenderingEnabled){
                    imgX = (int)(absv(x - (int)x) * imgWidth);
                    imgY = (int)(absv(z - (int)z) * imgHeight);
    
                    r = blockVariants[blockId]->texture->topImage[(imgY * imgWidth + imgX) * imgChannels];
                    g = blockVariants[blockId]->texture->topImage[(imgY * imgWidth + imgX) * imgChannels + 1];
                    b = blockVariants[blockId]->texture->topImage[(imgY * imgWidth + imgX) * imgChannels + 2];
                }
    
                normal = Vector3<>(0, -1, 0);
            }
            else if (equals(y, (float)blockY + 1.0, epsilon)) { // bottom
                if(isTextureRenderingEnabled){
                    imgX = (int)(absv(x - (int)x) * imgWidth);
                    imgY = (int)(absv(z - (int)z) * imgHeight);
    
                    r = blockVariants[blockId]->texture->bottomImage[(imgY * imgWidth + imgX) * imgChannels];
                    g = blockVariants[blockId]->texture->bottomImage[(imgY * imgWidth + imgX) * imgChannels + 1];
                    b = blockVariants[blockId]->texture->bottomImage[(imgY * imgWidth + imgX) * imgChannels + 2];
                }
    
                normal = Vector3<>(0, 1, 0);
            }
            else if (equals(x, (float)blockX, epsilon)) { // left
                if(isTextureRenderingEnabled){
                    imgX = (int)(absv(z - (int)z) * imgWidth);
                    imgY = (int)(absv(y - (int)y) * imgHeight);
    
                    r = blockVariants[blockId]->texture->leftImage[(imgY * imgWidth + imgX) * imgChannels];
                    g = blockVariants[blockId]->texture->leftImage[(imgY * imgWidth + imgX) * imgChannels + 1];
                    b = blockVariants[blockId]->texture->leftImage[(imgY * imgWidth + imgX) * imgChannels + 2];
                }
    
                normal = Vector3<>(-1, 0, 0);
            }
            else if (equals(x, (float)blockX + 1.0, epsilon)) { // right
                if(isTextureRenderingEnabled){
                    imgX = (int)(absv(z - (int)z) * imgWidth);
                    imgY = (int)(absv(y - (int)y) * imgHeight);
    
                    r = blockVariants[blockId]->texture->rightImage[(imgY * imgWidth + imgX) * imgChannels];
                    g = blockVariants[blockId]->texture->rightImage[(imgY * imgWidth + imgX) * imgChannels + 1];
                    b = blockVariants[blockId]->texture->rightImage[(imgY * imgWidth + imgX) * imgChannels + 2];
                }
    
                normal = Vector3<>(1, 0, 0);
            }
            else if (equals(z, (float)blockZ, epsilon)) { // front
                if(isTextureRenderingEnabled){
                    imgX = (int)(absv(x - (int)x) * imgWidth);
                    imgY = (int)(absv(y - (int)y) * imgHeight);
    
                    r = blockVariants[blockId]->texture->frontImage[(imgY * imgWidth + imgX) * imgChannels];
                    g = blockVariants[blockId]->texture->frontImage[(imgY * imgWidth + imgX) * imgChannels + 1];
                    b = blockVariants[blockId]->texture->frontImage[(imgY * imgWidth + imgX) * imgChannels + 2];
                }
    
                normal = Vector3<>(0, 0, -1);
            }
            else if (equals(z, (float)blockZ + 1.0, epsilon)) { // back
                if(isTextureRenderingEnabled){
                    imgX = (int)(absv(x - (int)x) * imgWidth);
                    imgY = (int)(absv(y - (int)y) * imgHeight);
    
                    r = blockVariants[blockId]->texture->backImage[(imgY * imgWidth + imgX) * imgChannels];
                    g = blockVariants[blockId]->texture->backImage[(imgY * imgWidth + imgX) * imgChannels + 1];
                    b = blockVariants[blockId]->texture->backImage[(imgY * imgWidth + imgX) * imgChannels + 2];
                }
    
                normal = Vector3<>(0, 0, 1);
            }
            else {
                return;
            }

            Vector3<> color = Vector3<>(r, g, b);
            
            if(!isTextureRenderingEnabled) {
                color = hueToRGB(float(blockId) * 2.8125 / 360.0);
            }
    
            // color = getPhongIllumination(color, Vector3<>(x, y, z), cameraPos, normal, blockVariants[blockId]->material, PointLight(Vector3<>(0, 700, -1000), Vector3<>(255,255,255)));
    
            setPixel(pixels, sX, sY, (int)color.x, (int)color.y, (int)color.z, 255);
        }

        __global__ void renderKernel(uchar4* pixels, Octree* octree, Vector3<> cameraPos, Vector3<> cameraAngle2d, int screenWidth, int screenHeight, bool isTextureRenderingEnabled) {            
            unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;

            if (index >= screenWidth * screenHeight)
                return;

            int sX = index % screenWidth;
            int sY = index / screenWidth;

            float alpha = (atanf(-(sX - SCREEN_WIDTH  / 2) / FOCAL_LENGTH) - cameraAngle2d.y + M_PI / 2); // horizontal angle
            float polar = (atanf(-(sY - SCREEN_HEIGHT / 2) / FOCAL_LENGTH) + cameraAngle2d.x + M_PI / 2); // vertical angle

            float dX = sin(polar) * cos(alpha);
            float dZ = sin(polar) * sin(alpha);
            float dY = cos(polar);

            Triple<Vector3<int>, Vector3<>, uint8_t> intersectionData = octree->getRayIntersectionData(pixels, cameraPos, Vector3<>(dX, dY, dZ), sX, sY, 1);

            if(intersectionData.third == 0) {
                setPixel(pixels, sX, sY, 0, 0, 255, 255);
            }
            else {
                setPixelByHitInfo(pixels, intersectionData, cameraPos, sX, sY, isTextureRenderingEnabled);
            }
        }
    }

    void init(int windowWidth, int windowHeight) {
        cuda_renderer_utils::initSDL(windowWidth, windowHeight);
    }

    void cleanup() {
        cuda_renderer_utils::cleanupSDL();
    }

    void render(Octree* octree, Vector3<> cameraPos, Vector3<> cameraAngle2d, int windowWidth, int windowHeight, bool isTextureRenderingEnabled, unsigned int gridSize, unsigned int blockSize) {
        using namespace cuda_renderer_utils;

        cudaArray* cudaArrayPtr;
        cudaGraphicsMapResources(1, &cudaResource, 0);
        cudaGraphicsSubResourceGetMappedArray(&cudaArrayPtr, cudaResource, 0, 0);
    
        // Copy data to CUDA array
        uchar4* devPtr;
        size_t pitch;
        cudaMallocPitch(&devPtr, &pitch, windowWidth * sizeof(uchar4), windowHeight);
    
        // cudaEvent_t start, stop;
        // cudaEventCreate(&start);
        // cudaEventCreate(&stop);
        // cudaEventRecord(start);
    
        // cudaError_t error = cudaGetLastError();
                
        // if(error != cudaSuccess)
        //     printf("CUDA error: %s\n", cudaGetErrorString(error));
    
        renderKernel<<<gridSize,blockSize>>>(devPtr, octree, cameraPos, cameraAngle2d, windowWidth, windowHeight, isTextureRenderingEnabled);
    
        // cudaEventRecord(stop);
        // cudaEventSynchronize(stop);
        // float milliseconds = 0;
        // cudaEventElapsedTime(&milliseconds, start, stop);
        // printf("Kernel execution time: %f ms (%f fps)\n", milliseconds, 1.0 / (milliseconds / 1000.0));
        // cudaDeviceSynchronize();
    
        // Copy memory from CUDA device to OpenGL texture
        cudaMemcpy2DToArray(cudaArrayPtr, 0, 0, devPtr, pitch, SCREEN_WIDTH * sizeof(uchar4), SCREEN_HEIGHT, cudaMemcpyDeviceToDevice);
        cudaFree(devPtr);
    
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
    }
}