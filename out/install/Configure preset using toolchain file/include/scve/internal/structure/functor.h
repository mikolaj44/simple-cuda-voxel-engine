#pragma once

#include <cstdint>

#include "scve/internal/structure/material.h"

namespace scve {
    /**
    * @brief The functor you can pass to \ref insertVoxels, provides a position + frame number to block id mapping
    * */
    class XYZFrameToIdFunctor {
        public:
            virtual __host__ __device__ uint8_t operator()(int x, int y, int z, uint64_t frameNumber) = 0;
            virtual __host__ __device__ ~XYZFrameToIdFunctor() = default;
    };

    /**
    * @brief The functor you can pass to \ref setMaterials, provides a block id + frame number to material mapping
    * */
    class IdFrameToMaterialFunctor {
        public:
            virtual __host__ __device__ Material operator()(uint8_t blockId, uint64_t frameNumber) = 0;
            virtual __host__ __device__ ~IdFrameToMaterialFunctor() = default;
    };
}