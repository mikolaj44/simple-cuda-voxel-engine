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

__global__ void renderScreenCudaKernel(Octree* octree, int width, int height, float cameraAngleX, float cameraAngleY, float oX, float oY, float oZ, uchar4* pixels);

void renderScreenCuda(Octree* octree, int width, int height, float cameraAngleX, float cameraAngleY, float oX, float oY, float oZ, uchar4* pixels, unsigned int gridSize, unsigned int blockSize);