#include "chunk_generation.cuh"

#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

void generateChunks(Octree* octree, Vector3 cameraPos, dim3 gridSize, dim3 blockSize){

    octree->xMin = cameraPos.x - CHUNK_W * RENDER_DISTANCE_CHUNKS / 2;
    octree->yMin = cameraPos.y - CHUNK_W * RENDER_DISTANCE_CHUNKS / 2;
    octree->zMin = cameraPos.z - CHUNK_W * RENDER_DISTANCE_CHUNKS / 2;
    octree->level = log2(RENDER_DISTANCE_CHUNKS * CHUNK_W * 2) - 1;

    generateChunksKernel<<<gridSize, blockSize>>>(octree, cameraPos);

    //cudaDeviceSynchronize(); // maybe remove this later
}

__global__ void generateChunksKernel(Octree* octree, Vector3 cameraPos){

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;

	if(x > RENDER_DISTANCE_CHUNKS * CHUNK_W || y > RENDER_DISTANCE_CHUNKS * CHUNK_W || z > RENDER_DISTANCE_CHUNKS * CHUNK_W){
		return;
	}

    x += cameraPos.x - CHUNK_W * RENDER_DISTANCE_CHUNKS / 2;
    y += cameraPos.y - CHUNK_W * RENDER_DISTANCE_CHUNKS / 2;
    z += cameraPos.z - CHUNK_W * RENDER_DISTANCE_CHUNKS / 2;

    float val = cudaNoise::perlinNoise(make_float3(float(x) / smoothing, 0, float(z) / smoothing), 1, 0) * amplify;

    // calculate blockId using perlin noise or some other user-defined lambda function
	
    if(y >= val + 30 /*&& y <= val + 20.5 + 10*/){
	    octree->insert(Block(x,y,z,1));
    }
}