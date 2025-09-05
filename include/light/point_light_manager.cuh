#pragma once

#include "light/point_light.cuh"
#include <stdint.h>

namespace point_light_manager {
    extern __device__ PointLight** pointLights;
    
    extern unsigned int numLights;

    cudaError_t init(unsigned int numLights);

    cudaError_t cleanup();
}