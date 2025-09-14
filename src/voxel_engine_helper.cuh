#pragma once

#include "block_variant/block_variant_manager.cuh"

class VoxelEngineHelper {
    template<typename IdFrameToMaterialFunctor>
    static void setMaterials(IdFrameToMaterialFunctor functor) {
        int numVariants = block_variant_manager::blockVariants->size();
        block_variant_manager::setBlocksVariantMaterialsKernel<<<1, numVariants>>>(functor, frameNumber);
    }
};