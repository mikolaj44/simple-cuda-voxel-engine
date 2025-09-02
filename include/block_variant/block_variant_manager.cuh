#pragma once

#include "block_variant.cuh"
#include <stdint.h>

namespace block_variant_manager {
    extern __device__ BlockVariant** blockVariants;

    void init();

    void cleanup();

    template<typename IdFrameToMaterialFunction>
    void setMaterials(IdFrameToMaterialFunction func, uint64_t frameNumber);
}