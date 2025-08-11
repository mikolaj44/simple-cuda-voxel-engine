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

            if(intersectionData == octree_utils::Pair<BlockInfo<int>, BlockInfo<float>>(BlockInfo<int>::invalidBlockInfo(), BlockInfo<float>::invalidBlockInfo())) {
                cuda_renderer::setPixel(pixels, sX, sY, 0, 0, 255, 255);
            }
            else {
                setPixelByHitInfo(pixels, intersectionData, cameraPos, sX, sY, textureRenderingEnabled);
            }
        }
    }

    void render(uchar4* pixels, Octree* octree, Vector3<> cameraPos, Vector3<> cameraAngle2d, int screenWidth, int screenHeight, bool textureRenderingEnabled, unsigned int gridSize, unsigned int blockSize) {
        renderKernel<<<gridSize,blockSize>>>(pixels, octree, cameraPos, cameraAngle2d, screenWidth, screenHeight, textureRenderingEnabled);
    }

    __device__ void setPixelByHitInfo(uchar4* pixels, octree_utils::Pair<BlockInfo<>, BlockInfo<float>> intersectionData, Vector3<> cameraPos, int sX, int sY, bool textureRenderingEnabled) {
        int blocksAmount = 4;
        float epsilon = 0.0001;

        uint8_t blockId = intersectionData.first.id;
        
        if (blockId == 0 || (textureRenderingEnabled && blockId > blocksAmount)) {
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
            //printf("%f\n", float(blockId));
            hueToRGB(float(blockId + 1) * 2.8125 / 360.0, r, g, b);
        }

        // getPhongIllumination(Vector3(x, y, z), cameraPos, normal, blockVariants[blockId % blocksAmount]->material, light, r, g, b);

        cuda_renderer::setPixel(pixels, sX, sY, r, g, b, 255);
    }

    __device__ void setPixel(uchar4* pixels, int sX, int sY, int r, int g, int b, int a) {
        pixels[(SCREEN_HEIGHT - 1 - sY) * SCREEN_WIDTH + sX] = make_uchar4(r, g, b, a);
    }
}