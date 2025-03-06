#include "chunk_generation.cuh"

#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

void generateChunks(Octree* octree, Vector3 cameraPos, dim3 gridSize, dim3 blockSize){

    octree->xMin = cameraPos.x - CHUNK_W * RENDER_DISTANCE_CHUNKS / 2;
    octree->yMin = cameraPos.y - CHUNK_W * RENDER_DISTANCE_CHUNKS / 2;
    octree->zMin = cameraPos.z - CHUNK_W * RENDER_DISTANCE_CHUNKS / 2;
    octree->level = log2(RENDER_DISTANCE_CHUNKS * CHUNK_W * 2) - 1;

    generateChunksKernel<<<gridSize, blockSize>>>(octree, octree->nodeMap.ref(cuco::insert), cameraPos);
    //cudaDeviceSynchronize(); // maybe remove this later
}