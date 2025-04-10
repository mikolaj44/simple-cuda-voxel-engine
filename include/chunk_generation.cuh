#pragma once

#include "chunk.cuh"
#include "octree.cuh"
#include "globals.cuh"
#include "cuda_noise.cuh"

#include <thrust/device_vector.h>

using namespace std;

template<typename XYZtoIdFunction>
__global__ void generateChunksKernel(Octree* octree, Vector3 cameraPos, XYZtoIdFunction blockPosToIdFunction, uint64_t frameCount){

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;

	if(x > RENDER_DISTANCE_CHUNKS * CHUNK_W || y > RENDER_DISTANCE_CHUNKS * CHUNK_W || z > RENDER_DISTANCE_CHUNKS * CHUNK_W){
		return;
	}

    x += cameraPos.x - CHUNK_W * RENDER_DISTANCE_CHUNKS / 2;
    y += cameraPos.y - CHUNK_W * RENDER_DISTANCE_CHUNKS / 2;
    z += cameraPos.z - CHUNK_W * RENDER_DISTANCE_CHUNKS / 2;

    char id = blockPosToIdFunction(x, y, z);

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
void generateChunks(Octree* octree, Vector3 cameraPos, XYZtoIdFunction blockPosToIdFunction, dim3 gridSize, dim3 blockSize, uint64_t frameCount){

    octree->xMin = cameraPos.x - CHUNK_W * RENDER_DISTANCE_CHUNKS / 2;
    octree->yMin = cameraPos.y - CHUNK_W * RENDER_DISTANCE_CHUNKS / 2;
    octree->zMin = cameraPos.z - CHUNK_W * RENDER_DISTANCE_CHUNKS / 2;
    octree->level = log2(RENDER_DISTANCE_CHUNKS * CHUNK_W * 2) - 1;

    generateChunksKernel<<<gridSize, blockSize>>>(octree, cameraPos, blockPosToIdFunction, frameCount);

    //cudaDeviceSynchronize(); // maybe remove this later
}

// __device__ float pseudoRandom(unsigned long long seed) {
//     //static unsigned long long x = ;

//     seed ^= (seed<<13);
//     seed ^= (seed>>7);
//     seed ^= (seed<<17);

//     return (seed & 0xFFFFFF) / float(0x1000000);
// }