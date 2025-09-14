#pragma once

#include <stdint.h>

#include "scve/internal/light/point_light.h"
#include "scve/internal/structure/managed_list.cuh"

namespace scve::point_light_manager {
    extern __managed__ ManagedList<PointLight*>* pointLights;

    extern __managed__ PointLight* ambientLight;
    extern __managed__ PointLight* backgroundLight;
    
    cudaError_t init(unsigned int numLights);

    cudaError_t cleanup();
}