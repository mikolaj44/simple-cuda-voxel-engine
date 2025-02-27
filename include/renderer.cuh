#pragma once
#include <vector>

#include "globals.cuh"
#include "octree.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

void calculateFOV();

void DrawVisibleScreenSection(int x1, int x2, int y1, int y2, Octree* octree);

void DrawVisibleFaces(Octree* octree);

template<typename MapInsertRef, typename MapFindRef>
__global__ void renderScreenCudaKernel(Octree* octree, MapInsertRef insertRef, MapFindRef findRef, int width, int height, float cameraAngleX, float cameraAngleY, float oX, float oY, float oZ, unsigned char* pixels) {

    unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;

    if (index >= width * height)
        return;

    int pX = index % width;
    int pY = index / width;

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

    performRaycast(octree, insertRef, findRef, oX, oY, oZ, sX, sY, sZ, pX, pY, 1, pixels);
}

void renderScreenCuda(Octree* octree, int width, int height, float cameraAngleX, float cameraAngleY, float oX, float oY, float oZ, unsigned char* pixels, unsigned int gridSize, unsigned int blockSize);