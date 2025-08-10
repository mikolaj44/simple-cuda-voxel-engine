#include <iostream>
#include <vector>
#include <cmath>
#include <map>

#include "cuda_renderer.cuh"
#include "chunk_generation.cuh"
#include "chunk.cuh"
#include "globals.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace cuda_renderer {
    namespace {
        __global__ void renderKernel(uchar4* pixels, Octree* octree, Vector3 cameraPos, Vector3 cameraAngle2d, int screenWidth, int screenHeight) {
            unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;

            if (index >= screenWidth * screenHeight)
                return;

            int pX = index % screenWidth;
            int pY = index / screenWidth;

            float alpha = (atanf(-(pX - SCREEN_WIDTH  / 2) / FOCAL_LENGTH) - cameraAngle2d.y + M_PI / 2); // horizontal angle
            float polar = (atanf(-(pY - SCREEN_HEIGHT / 2) / FOCAL_LENGTH) + cameraAngle2d.x + M_PI / 2); // vertical angle

            float sX = sin(polar) * cos(alpha);
            float sZ = sin(polar) * sin(alpha);
            float sY = cos(polar);

            octree->getRayIntersectionData(pixels, cameraPos, Vector3(sX, sY, sZ), pX, pY, 1);
        }
    }

    void render(uchar4* pixels, Octree* octree, Vector3 cameraPos, Vector3 cameraAngle2d, int screenWidth, int screenHeight, unsigned int gridSize, unsigned int blockSize) {
        renderKernel<<<gridSize,blockSize>>>(pixels, octree, cameraPos, cameraAngle2d, screenWidth, screenHeight);
    }

    __device__ void setPixel(uchar4* pixels, int x, int y, int r, int g, int b, int a) {
        pixels[(SCREEN_HEIGHT - 1 - y) * SCREEN_WIDTH + x] = make_uchar4(r, g, b, a);
    }
}