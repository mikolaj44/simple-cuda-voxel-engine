#pragma once

#include "light/point_light_manager.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

namespace point_light_manager {
    __managed__ ManagedList<PointLight*>* pointLights;

    cudaError_t init(unsigned int numLights_) {
        cudaError_t error = cudaMallocManaged(&pointLights, sizeof(ManagedList<PointLight*>));

        if(error != cudaSuccess) {
            return error;
        }

        new (pointLights) ManagedList<PointLight*>(); 

        error = pointLights->init(numLights_);

        if(error != cudaSuccess) {
            cudaFree(pointLights);
            return error;
        }

        for(int i = 0; i < numLights_; i++) {
            PointLight* pointLight;

            error = cudaMallocManaged(&pointLight, sizeof(PointLight));

            if(error != cudaSuccess) {
                return error;
            }

            new (pointLight) PointLight(Vector3<float>(0, -20000, -30000), Vector3<float>(255, 255, 255));

            error = pointLights->add(pointLight);

            if(error != cudaSuccess) {
                return error;
            }
        }

        return error;
    }

    cudaError_t cleanup() {
        cudaError_t error = cudaSuccess;

        for(int i = 0; i < pointLights->size(); i++) {
            error = cudaFree((*pointLights)[i]);
    
            if(error != cudaSuccess) {
                return error;
            }    
        }

        error = pointLights->cleanup();

        if(error != cudaSuccess) {
            return error;
        }

        return cudaFree(pointLights);
    }
}