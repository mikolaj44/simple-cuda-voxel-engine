#pragma once

#include "block_variant/block_variant.cuh"
#include <stdint.h>

namespace block_variant_manager {
    extern __managed__ BlockVariant** blockVariants;

    extern unsigned int numVariants;

    cudaError_t init(unsigned int numVariants);

    cudaError_t cleanup();

    template<typename IdFrameToMaterialFunction>
    __global__ void setBlocksVariantMaterialsKernel(IdFrameToMaterialFunction func, uint64_t frameNumber, unsigned int numVariants) {
        int index = threadIdx.x + blockIdx.x * blockDim.x;

        if(index < numVariants) {
            blockVariants[index]->material = func(uint8_t(index + 1), frameNumber);
        }
    }
}