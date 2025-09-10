#pragma once

#include "light/point_light.cuh"
#include "managed_list.cuh"
#include <stdint.h>

namespace scve::point_light_manager {
    extern __managed__ ManagedList<PointLight*>* pointLights;

    extern __managed__ PointLight* ambientLight;
    
    cudaError_t init(unsigned int numLights, const PointLight& ambientLight);

    cudaError_t cleanup();
}