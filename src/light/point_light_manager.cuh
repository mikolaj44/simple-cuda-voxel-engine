#pragma once

#include "light/point_light.h"
#include "managed_list.cuh"
#include <stdint.h>

namespace scve::point_light_manager {
    extern __managed__ ManagedList<PointLight*>* pointLights;

    extern __managed__ PointLight* ambientLight;
    extern __managed__ PointLight* backgroundLight;
    
    cudaError_t init(unsigned int numLights);

    cudaError_t cleanup();
}