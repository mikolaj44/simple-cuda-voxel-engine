#pragma once

#include "block_variant/block_variant.cuh"
#include <stdint.h>

namespace block_variant_manager {
    extern __device__ BlockVariant** blockVariants;

    cudaError_t init();

    cudaError_t cleanup();

    template<typename IdFrameToMaterialFunction>
    void setMaterials(IdFrameToMaterialFunction func, uint64_t frameNumber);
}