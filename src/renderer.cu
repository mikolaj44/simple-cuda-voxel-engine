#include <iostream>
#include <vector>
#include <cmath>
#include <map>

#include "renderer.cuh"
#include "chunk_generation.cuh"
#include "chunk.cuh"
#include "pixel_drawing.cuh"

using namespace std;

void calculateFOV() {
    halfHorFOV = atanf(SCREEN_WIDTH / (2.0 * FOCAL_LENGTH));
    halfVerFOV = atanf(SCREEN_HEIGHT / (2.0 * FOCAL_LENGTH));
}

typedef void (*func_ptr)(Octree*, float, float, float, float, float, float, int, int, int, uchar4*);

__global__ void renderScreenCudaKernel(Octree* octree, int width, int height, float cameraAngleX, float cameraAngleY, float oX, float oY, float oZ, uchar4* pixels) {
    unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;

    if (index >= width * height)
        return;

    int pX = index % width;
    int pY = index / width;

    // setPixel(pixels, pX, pY, 255, 0, 0);
    // return;

    //if (!(pX == 0 && pY == 0))
    //    return;

    //printf("%f %f %f\n", oX, oY, oZ);

    float alpha;
    float polar;

    alpha = (atanf(-(pX - SCREEN_WIDTH  / 2) / FOCAL_LENGTH) - cameraAngleY + M_PI / 2); // horizontal angle
    polar = (atanf(-(pY - SCREEN_HEIGHT / 2) / FOCAL_LENGTH) + cameraAngleX + M_PI / 2); // vertical angle

    float sX = sin(polar) * cos(alpha);
    float sZ = sin(polar) * sin(alpha);
    float sY = cos(polar);

    performRaycast(octree, oX, oY, oZ, sX, sY, sZ, pX, pY, 1, pixels);
}

void renderScreenCuda(Octree* octree, int width, int height, float cameraAngleX, float cameraAngleY, float oX, float oY, float oZ, uchar4* pixels, unsigned int gridSize, unsigned int blockSize) {
    renderScreenCudaKernel<<<gridSize,blockSize>>>(octree, SCREEN_WIDTH, SCREEN_HEIGHT, cameraAngle.x, cameraAngle.y, cameraPos.x, cameraPos.y, cameraPos.z, pixels);
}