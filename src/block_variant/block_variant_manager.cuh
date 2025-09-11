#pragma once

#include "block_variant/block_variant.cuh"
#include "managed_list.cuh"

#include <stdint.h>

namespace scve::block_variant_manager {
    extern __managed__ ManagedList<BlockVariant*>* blockVariants;

    extern __managed__ int numVariantsWithTextures;

    cudaError_t init(std::string texturesPath, unsigned int maxNumVariants, bool skipTextureLoading = false);

    cudaError_t cleanup();

    template<typename IdFrameToMaterialFunction>
    __global__ void setBlocksVariantMaterialsKernel(IdFrameToMaterialFunction func, uint64_t frameNumber) {
        int index = threadIdx.x + blockIdx.x * blockDim.x;

        if(index < blockVariants->size()) {
            ((*blockVariants)[index])->material = func(uint8_t(index + 1), frameNumber);
        }
    }
}