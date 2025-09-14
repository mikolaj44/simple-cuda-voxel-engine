#pragma once

#include "block_variant/block_variant.cuh"
#include "managed_list.cuh"
#include "functor.h"

#include <stdint.h>

namespace scve::block_variant_manager {
    extern __managed__ ManagedList<BlockVariant*>* blockVariants;

    cudaError_t init(std::string texturesPath, unsigned int maxNumVariants, bool skipTextureLoading = false);

    cudaError_t cleanup();

    template<typename IdFrameToMaterialFunctor>
    __global__ void setBlocksVariantMaterialsKernel(IdFrameToMaterialFunctor functor, uint64_t frameNumber) {
        int index = threadIdx.x + blockIdx.x * blockDim.x;

        if(index < blockVariants->size()) {
            ((*blockVariants)[index])->material = functor(uint8_t(index + 1), frameNumber);
        }
    }
}