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
        
        __device__ inline Frame& top() {
            return data[topIndex - 1];
        }

        __device__ inline bool isEmpty() {
            return topIndex <= 0;
        }
    };

    __device__ unsigned int nodeLevel(uint32_t mortonCode, unsigned int octreeLevel);

    __device__ unsigned int nodeSize(uint32_t mortonCode, unsigned int octreeLevel);

    __device__ Vector3<> getBlockHitPos(Vector3<int> blockPos, Vector3<> rayOrigin, Vector3<> rayDirection);

    namespace revelles {
        __device__ unsigned char firstNode(float tx0, float ty0, float tz0, float txm, float tym, float tzm, float epsilon);
    
        __device__ unsigned char newNode(float tx, unsigned char i1, float ty, unsigned char i2, float tz, unsigned char i3, float epsilon);
    
        __device__ uint32_t childMortonRevelles(uint32_t mortonCode, unsigned char revellesChildIndex);
    }
}