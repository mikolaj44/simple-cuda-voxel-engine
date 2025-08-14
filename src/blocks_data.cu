#include "blocks_data.cuh"
#include "cuda_math.cuh"
#include "renderer/cuda_renderer.cuh"

__device__ BlockVariant** blockVariants = nullptr;
__constant__ int blocksAmount = 4;

__global__ void createBlocksData(BlockTexture** textures) {
    cudaMalloc(&blockVariants, sizeof(BlockVariant*) * blocksAmount);

    for (int i = 0; i < blocksAmount; i++) {
        blockVariants[i] = new BlockVariant(Material(Vector3<>(255,255,255), 1, 0, 20), textures[i]);
    }
}