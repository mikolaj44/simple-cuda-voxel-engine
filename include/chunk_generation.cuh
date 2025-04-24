#pragma once

#include "chunk.cuh"
#include "octree.cuh"
#include "globals.cuh"
#include "cuda_noise.cuh"

#include <thrust/device_vector.h>

using namespace std;

template<typename XYZtoIdFunction>
__global__ void generateChunksKernel(Octree* octree, Vector3 pos, XYZtoIdFunction blockPosToIdFunction, uint64_t frameCount){

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;

    // x %= RENDER_DISTANCE_CHUNKS * CHUNK_W * 2;
    // y %= RENDER_DISTANCE_CHUNKS * CHUNK_W * 2;
    // z %= RENDER_DISTANCE_CHUNKS * CHUNK_W * 2;

	if(x > RENDER_DISTANCE_CHUNKS * CHUNK_W * 2 || y > RENDER_DISTANCE_CHUNKS * CHUNK_W * 2 || z > RENDER_DISTANCE_CHUNKS * CHUNK_W * 2){
		return;
	}

    x += pos.x - CHUNK_W * RENDER_DISTANCE_CHUNKS;
    y += pos.y - CHUNK_W * RENDER_DISTANCE_CHUNKS;
    z += pos.z - CHUNK_W * RENDER_DISTANCE_CHUNKS;

    //printf("%d %d %d\n", x,y,z);

    char id = blockPosToIdFunction(x, y, z, frameCount);

    if(id != -1){
        octree->insert(Block(x, y, z, id));
    }

    // float val = pseudoRandom(x ^ y ^ z ^ frameCount | 88172645463325252LL);//cudaNoise::perlinNoise(make_float3(float(x) / smoothing, 0, float(z) / smoothing), 1, 0) * amplify;

    // // calculate blockId using perlin noise or some other user-defined lambda function

    // //printf("%f\n", val);
	
    // if(val >= 0.5 /*&& y <= val + 20.5 + 10*/){
	//     octree->insert(Block(x,y,z,1));
    // }
    // if(val >= 0.2){
	//     octree->insert(Block(x,y,z,2));
    // }
    // else{
    //     octree->insert(Block(x,y,z,3));
    // } 
}

template<typename XYZtoIdFunction>
void generateChunks(Octree* octree, Vector3 pos, XYZtoIdFunction blockPosToIdFunction, dim3 maxGridSize, dim3 blockSize, uint64_t frameCount){

    const int length = RENDER_DISTANCE_CHUNKS * CHUNK_W * 2;

    octree->xMin = pos.x - length / 2;
    octree->yMin = pos.y - length / 2;
    octree->zMin = pos.z - length / 2;
    octree->level = log2(length); // * 2) - 1

    printf("%d %d %d\n", octree->xMin, octree->yMin, octree->zMin);
    printf("%d %d %d\n", octree->xMin + (1 << octree->level), octree->yMin + (1 << octree->level), octree->zMin + (1 << octree->level));

    dim3 gridSize = dim3(min(maxGridSize.x, (length + blockSize.x - 1) / blockSize.x), min(maxGridSize.y, (length + blockSize.y - 1) / blockSize.y), min(maxGridSize.z, (length + blockSize.z - 1) / blockSize.z));

    generateChunksKernel<<<gridSize, blockSize>>>(octree, pos, blockPosToIdFunction, frameCount);

    //cudaDeviceSynchronize(); // maybe remove this later
}

// __device__ float pseudoRandom(unsigned long long seed) {
//     //static unsigned long long x = ;

//     seed ^= (seed<<13);
//     seed ^= (seed>>7);
//     seed ^= (seed<<17);

//     return (seed & 0xFFFFFF) / float(0x1000000);
// }