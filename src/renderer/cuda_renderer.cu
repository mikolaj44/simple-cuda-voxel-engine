#include <iostream>
#include <vector>
#include <cmath>
#include <map>

#include "cuda_renderer.cuh"
#include "cuda_renderer_utils.cuh"
#include "block_variant/block_variant_manager.cuh"
#include "light/point_light_manager.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

constexpr float FOCAL_LENGTH = 10000; //350 //1200 //4000
constexpr float SCALE_V = 1;

constexpr float epsilon = 0.01;

namespace cuda_renderer {
    namespace {
        __device__ void setPixel(uchar4* pixels, int windowWidth, int windowHeight, int sX, int sY, int r, int g, int b, int a) {
            pixels[(windowHeight - 1 - sY) * windowWidth + sX] = make_uchar4(r, g, b, a);
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

        __device__ Vector3<> getPhongIllumination(Vector3<> startColor, Vector3<> pos, Vector3<> cameraPos, Vector3<> normal, Material material, ManagedList<PointLight*>* pointLights, PointLight* ambientLight) {   
            Vector3<> resultColor = Vector3<>::mul(Vector3<>::div(ambientLight->color, 255.0), ambientLight->intensity);

            startColor = startColor.div(255.0);

            Vector3<> h = Vector3<>(cameraPos.x - pos.x, cameraPos.y - pos.y, cameraPos.z - pos.z).norm();

            for(int i = 0; i < pointLights->size(); i++) {
                Vector3<> ln = Vector3<>(((*pointLights)[i])->pos.x - pos.x, ((*pointLights)[i])->pos.y - pos.y, ((*pointLights)[i])->pos.z - pos.z).norm();
            
                if (normal.dot(ln) < 0) {
                    continue;
                }  

                Vector3<> dh = Vector3<>::norm(Vector3<>::sub(Vector3<>::mul(normal, 2 * Vector3<>::dot(ln, normal)), ln));
                
                Vector3<> lightColor = Vector3<>::div(((*pointLights)[i])->color, 255.0);

                float intensity = ((*pointLights)[i])->intensity;

                resultColor = resultColor.add(
                    Vector3<>::add(
                        Vector3<>::mul(
                            Vector3<>::mul(startColor, lightColor), 
                            material.diffuse * Vector3<>::dot(ln, normal) * intensity
                        ), 
                        Vector3<>::mul(
                            lightColor, 
                            material.specular * powf(Vector3<>::dot(h, dh), material.specularExponent) * intensity
                        )
                    )
                );
            }

            return resultColor.mul(255.0);
        }
        
        __device__ void setPixelByHitInfo(uchar4* pixels, int windowWidth, int windowHeight, Triple<Vector3<int>, Vector3<float>, uint8_t> intersectionData, Vector3<> cameraPos, int sX, int sY, bool isTextureRenderingEnabled, bool isMaterialColorOnlyEnabled, bool isPhongIlluminationEnabled) {            
            using namespace block_variant_manager;

            uint8_t blockId = (intersectionData.third - 1) % blockVariants->size();
    
            int imgWidth, imgHeight, imgChannels;
                
            if(isTextureRenderingEnabled && !isMaterialColorOnlyEnabled) {
                imgChannels = ((*blockVariants)[blockId])->texture->getChannels();
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

            // printf("%d %d %d -> %f %f %f\n", blockX, blockY, blockZ, x, y, z);

            // check which side of the block we are on    
            if (equals(y, (float)blockY, epsilon)) { // top
                if(isTextureRenderingEnabled && !isMaterialColorOnlyEnabled){
                    imgWidth  = ((*blockVariants)[blockId])->texture->getWidth(ImagePosition::TOP);
                    imgHeight = ((*blockVariants)[blockId])->texture->getHeight(ImagePosition::TOP);

                    imgX = (int)(absv(x - floorf(x)) * imgWidth);
                    imgY = (int)(absv(z - floorf(z)) * imgHeight);
    
                    r = ((*blockVariants)[blockId])->texture->getImage(ImagePosition::TOP)[(imgY * imgWidth + imgX) * imgChannels];
                    g = ((*blockVariants)[blockId])->texture->getImage(ImagePosition::TOP)[(imgY * imgWidth + imgX) * imgChannels + 1];
                    b = ((*blockVariants)[blockId])->texture->getImage(ImagePosition::TOP)[(imgY * imgWidth + imgX) * imgChannels + 2];
                }
    
                normal = Vector3<>(0, -1, 0);
            }
            else if (equals(y, (float)blockY + 1.0, epsilon)) { // bottom
                if(isTextureRenderingEnabled && !isMaterialColorOnlyEnabled) {
                    imgWidth  = ((*blockVariants)[blockId])->texture->getWidth(ImagePosition::BOTTOM);
                    imgHeight = ((*blockVariants)[blockId])->texture->getHeight(ImagePosition::BOTTOM);

                    imgX = (int)(absv(x - floorf(x)) * imgWidth);
                    imgY = (int)(absv(z - floorf(z)) * imgHeight);
    
                    r = ((*blockVariants)[blockId])->texture->getImage(ImagePosition::BOTTOM)[(imgY * imgWidth + imgX) * imgChannels];
                    g = ((*blockVariants)[blockId])->texture->getImage(ImagePosition::BOTTOM)[(imgY * imgWidth + imgX) * imgChannels + 1];
                    b = ((*blockVariants)[blockId])->texture->getImage(ImagePosition::BOTTOM)[(imgY * imgWidth + imgX) * imgChannels + 2];
                }
    
                normal = Vector3<>(0, 1, 0);
            }
            else if (equals(x, (float)blockX, epsilon)) { // left
                if(isTextureRenderingEnabled && !isMaterialColorOnlyEnabled){
                    imgWidth  = ((*blockVariants)[blockId])->texture->getWidth(ImagePosition::LEFT);
                    imgHeight = ((*blockVariants)[blockId])->texture->getHeight(ImagePosition::LEFT);

                    imgX = (int)(absv(z - floorf(z)) * imgWidth);
                    imgY = (int)(absv(y - floorf(y)) * imgHeight);
    
                    r = ((*blockVariants)[blockId])->texture->getImage(ImagePosition::LEFT)[(imgY * imgWidth + imgX) * imgChannels];
                    g = ((*blockVariants)[blockId])->texture->getImage(ImagePosition::LEFT)[(imgY * imgWidth + imgX) * imgChannels + 1];
                    b = ((*blockVariants)[blockId])->texture->getImage(ImagePosition::LEFT)[(imgY * imgWidth + imgX) * imgChannels + 2];
                }
    
                normal = Vector3<>(-1, 0, 0);
            }
            else if (equals(x, (float)blockX + 1.0, epsilon)) { // right
                if(isTextureRenderingEnabled && !isMaterialColorOnlyEnabled){
                    imgWidth  = ((*blockVariants)[blockId])->texture->getWidth(ImagePosition::RIGHT);
                    imgHeight = ((*blockVariants)[blockId])->texture->getHeight(ImagePosition::RIGHT);

                    imgX = (int)(absv(z - floorf(z)) * imgWidth);
                    imgY = (int)(absv(y - floorf(y)) * imgHeight);
    
                    r = ((*blockVariants)[blockId])->texture->getImage(ImagePosition::RIGHT)[(imgY * imgWidth + imgX) * imgChannels];
                    g = ((*blockVariants)[blockId])->texture->getImage(ImagePosition::RIGHT)[(imgY * imgWidth + imgX) * imgChannels + 1];
                    b = ((*blockVariants)[blockId])->texture->getImage(ImagePosition::RIGHT)[(imgY * imgWidth + imgX) * imgChannels + 2];
                }
    
                normal = Vector3<>(1, 0, 0);
            }
            else if (equals(z, (float)blockZ, epsilon)) { // front
                if(isTextureRenderingEnabled && !isMaterialColorOnlyEnabled){
                    imgWidth  = ((*blockVariants)[blockId])->texture->getWidth(ImagePosition::FRONT);
                    imgHeight = ((*blockVariants)[blockId])->texture->getHeight(ImagePosition::FRONT);

                    imgX = (int)(absv(x - floorf(x)) * imgWidth);
                    imgY = (int)(absv(y - floorf(y)) * imgHeight);
    
                    r = ((*blockVariants)[blockId])->texture->getImage(ImagePosition::FRONT)[(imgY * imgWidth + imgX) * imgChannels];
                    g = ((*blockVariants)[blockId])->texture->getImage(ImagePosition::FRONT)[(imgY * imgWidth + imgX) * imgChannels + 1];
                    b = ((*blockVariants)[blockId])->texture->getImage(ImagePosition::FRONT)[(imgY * imgWidth + imgX) * imgChannels + 2];
                }
    
                normal = Vector3<>(0, 0, -1);
            }
            else if (equals(z, (float)blockZ + 1.0, epsilon)) { // back
                if(isTextureRenderingEnabled && !isMaterialColorOnlyEnabled){
                    imgWidth  = ((*blockVariants)[blockId])->texture->getWidth(ImagePosition::BACK);
                    imgHeight = ((*blockVariants)[blockId])->texture->getHeight(ImagePosition::BACK);

                    imgX = (int)(absv(x - floorf(x)) * imgWidth);
                    imgY = (int)(absv(y - floorf(y)) * imgHeight);
    
                    r = ((*blockVariants)[blockId])->texture->getImage(ImagePosition::BACK)[(imgY * imgWidth + imgX) * imgChannels];
                    g = ((*blockVariants)[blockId])->texture->getImage(ImagePosition::BACK)[(imgY * imgWidth + imgX) * imgChannels + 1];
                    b = ((*blockVariants)[blockId])->texture->getImage(ImagePosition::BACK)[(imgY * imgWidth + imgX) * imgChannels + 2];
                }
    
                normal = Vector3<>(0, 0, 1);
            }
            else {
                // printf("black\n");
                // printf("%d %d %d -> %f %f %f\n", blockX, blockY, blockZ, x, y, z);
                return;
            }

            Vector3<> color = Vector3<>(r, g, b);

            if(isMaterialColorOnlyEnabled) {
                color = ((*blockVariants)[blockId])->material.color;
            }
            else if(!isTextureRenderingEnabled) {
                color = hueToRGB(float(blockId) * 2.8125 / 360.0);
            }
    
            if(isPhongIlluminationEnabled) {
                color = getPhongIllumination(color, Vector3<>(x, y, z), cameraPos, normal, ((*blockVariants)[blockId])->material, point_light_manager::pointLights, point_light_manager::ambientLight);
            }

            setPixel(pixels, windowWidth, windowHeight, sX, sY, (int)color.x, (int)color.y, (int)color.z, 255);
        }

        __global__ void renderKernel(uchar4* pixels, Octree* octree, Vector3<> cameraPos, Vector3<> cameraAngle2d, int windowWidth, int windowHeight, bool isTextureRenderingEnabled, bool isMaterialColorOnlyEnabled, bool isPhongIlluminationEnabled) {                        
            unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;

            if (index >= windowWidth * windowHeight) {
                return;
            }

            int sX = index % windowWidth;
            int sY = index / windowWidth;

            float alpha = (atanf(-(sX - windowWidth  / 2) / FOCAL_LENGTH) - cameraAngle2d.y + M_PI / 2); // horizontal angle
            float polar = (atanf(-(sY - windowHeight / 2) / FOCAL_LENGTH) + cameraAngle2d.x + M_PI / 2); // vertical angle

            float dX = sin(polar) * cos(alpha);
            float dZ = sin(polar) * sin(alpha);
            float dY = cos(polar);

            Triple<Vector3<int>, Vector3<>, uint8_t> intersectionData = octree->getRayIntersectionData(cameraPos, Vector3<>(dX, dY, dZ), sX, sY, 1);

            if(intersectionData.third == 0) {
                setPixel(pixels, windowWidth, windowHeight, sX, sY, 0, 0, 255, 255);
            }
            else {
                // setPixel(pixels, windowWidth, windowHeight, sX, sY, intersectionData.third, intersectionData.third, intersectionData.third, 255);

                setPixelByHitInfo(pixels, windowWidth, windowHeight, intersectionData, cameraPos, sX, sY, isTextureRenderingEnabled, isMaterialColorOnlyEnabled, isPhongIlluminationEnabled);
            }
        }
    }


    cudaError_t init(int windowWidth, int windowHeight) {
        return cuda_renderer_utils::initSDL(windowWidth, windowHeight);
    }

    cudaError_t cleanup() {
        return cuda_renderer_utils::cleanupSDL();
    }

    // I'm not error checking for performance reasons
    void render(Octree* octree, Vector3<> cameraPos, Vector3<> cameraAngle2d, int windowWidth, int windowHeight, bool isTextureRenderingEnabled, bool isMaterialColorOnlyEnabled, bool isPhongIlluminationEnabled, unsigned int gridSize, unsigned int blockSize) {
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
        
        renderKernel<<<gridSize,blockSize>>>(devPtr, octree, cameraPos, cameraAngle2d, windowWidth, windowHeight, isTextureRenderingEnabled, isMaterialColorOnlyEnabled, isPhongIlluminationEnabled);
    
        // cudaEventRecord(stop);
        // cudaEventSynchronize(stop);
        // float milliseconds = 0;
        // cudaEventElapsedTime(&milliseconds, start, stop);
        // printf("Kernel execution time: %f ms (%f fps)\n", milliseconds, 1.0 / (milliseconds / 1000.0));
        // cudaDeviceSynchronize();
    
        // Copy memory from CUDA device to OpenGL texture
        cudaMemcpy2DToArray(cudaArrayPtr, 0, 0, devPtr, pitch, windowWidth * sizeof(uchar4), windowHeight, cudaMemcpyDeviceToDevice);
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