#pragma once
#include <vector>

#include "globals.cuh"
#include "octree.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace cuda_renderer {
    void render(uchar4* pixels, Octree* octree, int screenWidth, int screenHeight, Vector3 cameraAngle2D, Vector3 rayOrigin, unsigned int gridSize, unsigned int blockSize);
}