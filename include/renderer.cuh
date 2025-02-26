#pragma once
#include <vector>
#include "globals.cuh"
#include "Octree.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

void calculateFOV();

void DrawVisibleScreenSection(int x1, int x2, int y1, int y2, Octree* octree);

__global__ void renderScreenCUDA(int width, int height, Octree* octree, float cameraAngleX, float cameraAngleY, float oX, float oY, float oZ, unsigned char* pixels);

void DrawVisibleFaces(Octree* octree);