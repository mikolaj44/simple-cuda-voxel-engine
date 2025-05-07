#include <iostream>
#include <vector>
#include <cmath>
#include <map>

#include "renderer.cuh"
#include "chunk_generation.cuh"
#include "chunk.cuh"
#include "pixel_drawing.cuh"

using namespace std;

void Renderer::calculateFOV() {
    float halfHorFOV_ = atanf(SCREEN_WIDTH_HOST / (2.0 * FOCAL_LENGTH));
    float halfVerFOV_ = atanf(SCREEN_HEIGHT_HOST / (2.0 * FOCAL_LENGTH));

    cudaMemcpyToSymbol(halfHorFOV, &halfHorFOV_, sizeof(float), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(halfVerFOV, &halfVerFOV_, sizeof(float), 0, cudaMemcpyHostToDevice);
}

__global__ void renderScreenCudaKernel(Octree* octree, float cameraAngleX, float cameraAngleY, float oX, float oY, float oZ, uchar4* pixels) {
    unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;

    if (index >= SCREEN_WIDTH_DEVICE * SCREEN_HEIGHT_DEVICE)
        return;

    int pX = index % SCREEN_WIDTH_DEVICE;
    int pY = index / SCREEN_HEIGHT_DEVICE;

    float alpha;
    float polar;

    alpha = (atanf(-(pX - SCREEN_WIDTH_DEVICE / 2) / FOCAL_LENGTH) - cameraAngleY + M_PI / 2); // horizontal angle
    polar = (atanf(-(pY - SCREEN_HEIGHT_DEVICE / 2) / FOCAL_LENGTH) + cameraAngleX + M_PI / 2); // vertical angle

    float sX = sin(polar) * cos(alpha);
    float sZ = sin(polar) * sin(alpha);
    float sY = cos(polar);

    performRaycast(octree, oX, oY, oZ, sX, sY, sZ, pX, pY, 1, pixels);
}

void Renderer::renderScreenCuda(Octree* octree, float cameraAngleX, float cameraAngleY, float oX, float oY, float oZ, uchar4* pixels, unsigned int gridSize, unsigned int blockSize) {
    renderScreenCudaKernel<<<gridSize,blockSize>>>(octree, cameraAngle.x, cameraAngle.y, cameraPos.x, cameraPos.y, cameraPos.z, pixels);
}