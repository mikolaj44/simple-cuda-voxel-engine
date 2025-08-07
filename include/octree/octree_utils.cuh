#pragma once

#include "globals.cuh"

namespace octree_utils {

    class Stack {
    public:
        struct Frame {
            float tx0, ty0, tz0, txm, tym, tzm, tx1, ty1, tz1;
            uint32_t mortonCode = 1;
            unsigned char nodeIndex;
        };

        Frame data[CUDA_STACK_SIZE];
        
        int topIndex = 0;

        __device__ inline void push(Frame&& frame) {
            data[topIndex++] = frame;
        }
        
        __device__ inline void pop() {
            topIndex--;
        }
        
        __device__ inline Frame* top() {
            return &data[topIndex - 1];
        }

        __device__ inline bool isEmpty() {
            return topIndex <= 0;
        }
    };

    

    __device__ unsigned int nodeLevel(uint32_t mortonCode, unsigned int octreeLevel){
        int depth = 0;

        for (uint32_t code = mortonCode; code != 1; code >>= 3, depth++);

        return octreeLevel - depth;
    }

    __device__ unsigned int nodeSize(uint32_t mortonCode, unsigned int octreeLevel){
        return 1 << nodeLevel(mortonCode, octreeLevel);
    }

    __device__ void drawTexturePixel(int blockX, int blockY, int blockZ, float oX, float oY, float oZ, float dX, float dY, float dZ, int sX, int sY, unsigned char blockId, uchar4* pixels, bool textureRenderingEnabled) {
        if (dX == 0 || dY == 0 || dZ == 0) { // for now
            return;
        }
        
        float tmin =  minv((float)((float)blockX - oX) / dX, (float)((float)blockX + 1.0 - oX) / dX);
        float tymin = minv((float)((float)blockY - oY) / dY, (float)((float)blockY + 1.0 - oY) / dY);
        float tzmin = minv((float)((float)blockZ - oZ) / dZ, (float)((float)blockZ + 1.0 - oZ) / dZ);
    
        tmin = maxv(maxv(tmin, tymin), tzmin);
    
        float x = oX + tmin * dX;
        float y = oY + tmin * dY;
        float z = oZ + tmin * dZ;
    
        //printf("%d %d %d\n", x, y, z);
    
        setPixelById(sX, sY, blockX, blockY, blockZ, x, y, z, blockId, pixels, Vector3(oX, oY, oZ), Material(Vector3(255,255,255), 0.3, 0.6, 35), PointLight(Vector3(oX, oY, oZ), Vector3(255, 255, 255)), false);
    }
}