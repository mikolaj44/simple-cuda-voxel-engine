#include <iostream>
#include <vector>
#include <cmath>
#include <map>

#include "cuda_renderer.cuh"
#include "chunk_generation.cuh"
#include "chunk.cuh"
#include "globals.cuh"
#include "blocks_data.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

constexpr int blocksAmount = 4;
constexpr float epsilon = 0.0001;

namespace cuda_renderer {
    namespace {
        __global__ void renderKernel(uchar4* pixels, Octree* octree, Vector3<> cameraPos, Vector3<> cameraAngle2d, int screenWidth, int screenHeight, bool textureRenderingEnabled) {            
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

            octree_utils::Pair<BlockInfo<>, BlockInfo<float>> intersectionData = octree->getRayIntersectionData(pixels, cameraPos, Vector3<>(dX, dY, dZ), sX, sY, 1);

            if(intersectionData.first.id == 0) {
                cuda_renderer::setPixel(pixels, sX, sY, 0, 0, 255, 255);
            }
            else {
                setPixelByHitInfo(pixels, intersectionData, cameraPos, sX, sY, textureRenderingEnabled);
            }
        }

        // https://stackoverflow.com/questions/61277046/convert-just-a-hue-into-rgb
        __device__ void hueToRGB(float hue, int& r, int& g, int&b){
            float kr = remainderf(5 + hue * 6, 6);
            float kg = remainderf(3 + hue * 6, 6);
            float kb = remainderf(1 + hue * 6, 6);

            r = (1 - maxv(minv(minv(kr, 4-kr), 1.0f), 0.0f)) * 255;
            g = (1 - maxv(minv(minv(kg, 4-kg), 1.0f), 0.0f)) * 255;
            b = (1 - maxv(minv(minv(kb, 4-kb), 1.0f), 0.0f)) * 255;
        }

        __device__ Vector3<int> getPhongIllumination(Vector3<> startColor, Vector3<> pos, Vector3<> cameraPos, Vector3<> normal, Material material, PointLight light){
            material.color = startColor;
        
            float r = 0;
            float g = 0;
            float b = 0;
        
            Vector3<> ln = Vector3<>(light.pos.x - pos.x, light.pos.y - pos.y, light.pos.z - pos.z).norm();
        
            if (normal.dot(ln) < 0) {
                return Vector3<int>(0, 0, 0);
            }
                
            Vector3<> h = Vector3<>(cameraPos.x - pos.x, cameraPos.y - pos.y, cameraPos.z - pos.z).norm();

            Vector3<> dh = normal.mul(2 * ln.dot(normal)).sub(ln).norm();

            Vector3<> lighting = light.color.mul(material.diffuse * normal.dot(ln));
            
            if (lighting.x > 255)
                lighting.x = 255;
            if (lighting.y > 255)
                lighting.y = 255;
            if (lighting.z > 255)
                lighting.z = 255;
        
            lighting = lighting.div(255.0);
            lighting = lighting.mul(material.color);
        
            return Vector3<int>((int)lighting.x, (int)lighting.y, (int)lighting.z);
        }
    }

    void render(uchar4* pixels, Octree* octree, Vector3<> cameraPos, Vector3<> cameraAngle2d, int screenWidth, int screenHeight, bool textureRenderingEnabled, unsigned int gridSize, unsigned int blockSize) {
        renderKernel<<<gridSize,blockSize>>>(pixels, octree, cameraPos, cameraAngle2d, screenWidth, screenHeight, textureRenderingEnabled);
    }

    __device__ void setPixelByHitInfo(uchar4* pixels, octree_utils::Pair<BlockInfo<>, BlockInfo<float>> intersectionData, Vector3<> cameraPos, int sX, int sY, bool textureRenderingEnabled) {
        uint8_t blockId = intersectionData.first.id;
        
        if (blockId == 0 || (textureRenderingEnabled && blockId > blocksAmount)) {
            printf("%d\n", blockId);
            return;
        }

        blockId -= 1;

        int imgWidth, imgHeight, imgChannels;

        if(textureRenderingEnabled){
            imgWidth = blockVariants[blockId]->texture->width;
            imgHeight = blockVariants[blockId]->texture->height;
            imgChannels = blockVariants[blockId]->texture->channels;
        }

        float x = intersectionData.second.pos.x;
        float y = intersectionData.second.pos.y;
        float z = intersectionData.second.pos.z;

        int blockX = intersectionData.first.pos.x;
        int blockY = intersectionData.first.pos.y;
        int blockZ = intersectionData.first.pos.z;

        int imgX = 0, imgY = 0;

        int r, g, b;
        Vector3<> normal;

        // check which side of the block we are on

        if (equals(y, (float)blockY, epsilon)) { // top
            if(textureRenderingEnabled){
                imgX = (int)(absv(x - (int)x) * imgWidth);
                imgY = (int)(absv(z - (int)z) * imgHeight);

                r = blockVariants[blockId]->texture->topImage[(imgY * imgWidth + imgX) * imgChannels];
                g = blockVariants[blockId]->texture->topImage[(imgY * imgWidth + imgX) * imgChannels + 1];
                b = blockVariants[blockId]->texture->topImage[(imgY * imgWidth + imgX) * imgChannels + 2];
            }

            normal = Vector3<>(0, -1, 0);
        }
        else if (equals(y, (float)blockY + 1.0, epsilon)) { // bottom
            if(textureRenderingEnabled){
                imgX = (int)(absv(x - (int)x) * imgWidth);
                imgY = (int)(absv(z - (int)z) * imgHeight);

                r = blockVariants[blockId]->texture->bottomImage[(imgY * imgWidth + imgX) * imgChannels];
                g = blockVariants[blockId]->texture->bottomImage[(imgY * imgWidth + imgX) * imgChannels + 1];
                b = blockVariants[blockId]->texture->bottomImage[(imgY * imgWidth + imgX) * imgChannels + 2];
            }

            normal = Vector3<>(0, 1, 0);
        }
        else if (equals(x, (float)blockX, epsilon)) { // left
            if(textureRenderingEnabled){
                imgX = (int)(absv(z - (int)z) * imgWidth);
                imgY = (int)(absv(y - (int)y) * imgHeight);

                r = blockVariants[blockId]->texture->leftImage[(imgY * imgWidth + imgX) * imgChannels];
                g = blockVariants[blockId]->texture->leftImage[(imgY * imgWidth + imgX) * imgChannels + 1];
                b = blockVariants[blockId]->texture->leftImage[(imgY * imgWidth + imgX) * imgChannels + 2];
            }

            normal = Vector3<>(-1, 0, 0);
        }
        else if (equals(x, (float)blockX + 1.0, epsilon)) { // right
            if(textureRenderingEnabled){
                imgX = (int)(absv(z - (int)z) * imgWidth);
                imgY = (int)(absv(y - (int)y) * imgHeight);

                r = blockVariants[blockId]->texture->rightImage[(imgY * imgWidth + imgX) * imgChannels];
                g = blockVariants[blockId]->texture->rightImage[(imgY * imgWidth + imgX) * imgChannels + 1];
                b = blockVariants[blockId]->texture->rightImage[(imgY * imgWidth + imgX) * imgChannels + 2];
            }

            normal = Vector3<>(1, 0, 0);
        }
        else if (equals(z, (float)blockZ, epsilon)) { // front
            if(textureRenderingEnabled){
                imgX = (int)(absv(x - (int)x) * imgWidth);
                imgY = (int)(absv(y - (int)y) * imgHeight);

                r = blockVariants[blockId]->texture->frontImage[(imgY * imgWidth + imgX) * imgChannels];
                g = blockVariants[blockId]->texture->frontImage[(imgY * imgWidth + imgX) * imgChannels + 1];
                b = blockVariants[blockId]->texture->frontImage[(imgY * imgWidth + imgX) * imgChannels + 2];
            }

            normal = Vector3<>(0, 0, -1);
        }
        else if (equals(z, (float)blockZ + 1.0, epsilon)) { // back
            if(textureRenderingEnabled){
                imgX = (int)(absv(x - (int)x) * imgWidth);
                imgY = (int)(absv(y - (int)y) * imgHeight);

                r = blockVariants[blockId]->texture->backImage[(imgY * imgWidth + imgX) * imgChannels];
                g = blockVariants[blockId]->texture->backImage[(imgY * imgWidth + imgX) * imgChannels + 1];
                b = blockVariants[blockId]->texture->backImage[(imgY * imgWidth + imgX) * imgChannels + 2];
            }

            normal = Vector3<>(0, 0, 1);
        }
        else{
            return;
        }
        
        if(!textureRenderingEnabled) {
            hueToRGB(float(blockId + 1) * 2.8125 / 360.0, r, g, b);
        }

        Vector3<int> color = getPhongIllumination(Vector3<>(r, g, b), Vector3<>(x, y, z), cameraPos, normal, blockVariants[blockId % blocksAmount]->material, PointLight(cameraPos, Vector3<>(255,255,255)));

        cuda_renderer::setPixel(pixels, sX, sY, color.x, color.y, color.z, 255);
    }

    __device__ void setPixel(uchar4* pixels, int sX, int sY, int r, int g, int b, int a) {
        pixels[(SCREEN_HEIGHT - 1 - sY) * SCREEN_WIDTH + sX] = make_uchar4(r, g, b, a);
    }
}