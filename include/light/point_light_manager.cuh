#pragma once

#include "light/point_light.cuh"
#include "managed_list.cuh"
#include <stdint.h>

namespace point_light_manager {
    extern __managed__ ManagedList<PointLight*>* pointLights;
    
    cudaError_t init(unsigned int numLights);

    cudaError_t cleanup();

    template<typename IndexFrameToPointLightFunction>
    __global__ void setPointLightsKernel(IndexFrameToPointLightFunction func, uint64_t frameNumber) {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
    
        if(index < pointLights->size()) {
            PointLight pointLight = func(index, frameNumber);

            (*pointLights)[index]->color = pointLight.color;
            (*pointLights)[index]->pos = pointLight.pos;
        }
    }
}