#pragma once

#include <cstdint>

#include "scve/internal/structure/material.h"

namespace scve {

    class XYZFrameToIdFunctor {
        public:
            virtual __device__ uint8_t operator()(int x, int y, int z, uint64_t frameNumber) const = 0;
            virtual __host__ __device__ ~XYZFrameToIdFunctor() = default;
    };

    class IdFrameToMaterialFunctor {
        public:
            virtual __device__ Material operator()(uint8_t blockId, uint64_t frameNumber) const = 0;
            virtual __host__ __device__ ~IdFrameToMaterialFunctor() = default;
    };
}