#pragma once
#include "chunk.cuh"
#include "Octree.cuh"

using namespace std;

void GenerateChunk(int x, int z, Octree* octree, float offsetX = 0, float offsetZ = 0);

void GenerateVisibleChunks(Octree* octree);