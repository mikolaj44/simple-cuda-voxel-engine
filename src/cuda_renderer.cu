#include <iostream>
#include <vector>
#include <cmath>
#include <map>

#include "cuda_renderer.cuh"
#include "chunk_generation.cuh"
#include "chunk.cuh"
#include "pixel_drawing.cuh"
#include "globals.cuh"

namespace cuda_renderer {
    namespace {
        __global__ void renderKernel(uchar4* pixels, Octree* octree, int screenWidth, int screenHeight, Vector3 cameraAngle2D, Vector3 rayOrigin) {
            unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;

            if (index >= screenWidth * screenHeight)
                return;

            int pX = index % screenWidth;
            int pY = index / screenWidth;

            float alpha = (atanf(-(pX - SCREEN_WIDTH  / 2) / FOCAL_LENGTH) - cameraAngle2D.y + M_PI / 2); // horizontal angle
            float polar = (atanf(-(pY - SCREEN_HEIGHT / 2) / FOCAL_LENGTH) + cameraAngle2D.x + M_PI / 2); // vertical angle

            float sX = sin(polar) * cos(alpha);
            float sZ = sin(polar) * sin(alpha);
            float sY = cos(polar);

            printf("%f %f %f\n", sX, sY, sZ);
            
            //performRaycast(octree, oX, oY, oZ, sX, sY, sZ, pX, pY, 1, pixels);
        }
    }

    void render(uchar4* pixels, Octree* octree, int screenWidth, int screenHeight, Vector3 cameraAngle2D, Vector3 rayOrigin, unsigned int gridSize, unsigned int blockSize) {
        renderKernel<<<gridSize,blockSize>>>(pixels, octree, screenWidth, screenHeight, cameraAngle2D, rayOrigin);
    }
}