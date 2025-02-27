#pragma once

#include "chunk.cuh"
#include "octree.cuh"

#include <thrust/device_vector.h>

using namespace std;

void generateChunk(Octree* octree, int x, int y, int z, unsigned int gridSize = 1, unsigned int blockSize = 1, int offsetX = 0, int offsetY = 0);

void generateVisibleChunks(Octree* octree);