#pragma once

#include "light/point_light_manager.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

namespace scve::point_light_manager {
    __managed__ ManagedList<PointLight*>* pointLights;

    __managed__ PointLight* ambientLight;
    __managed__ PointLight* backgroundLight;

    cudaError_t init(unsigned int numLights_) {
        cudaError_t error = cudaMallocManaged(&pointLights, sizeof(ManagedList<PointLight*>));

        if(error != cudaSuccess) {
            return error;
        }

        error = cudaMallocManaged(&ambientLight, sizeof(PointLight));

        if(error != cudaSuccess) {
            return error;
        }

        error = cudaMallocManaged(&backgroundLight, sizeof(PointLight));

        if(error != cudaSuccess) {
            return error;
        }

        *ambientLight = PointLight(Vector3<>(0, 0, 0), Vector3<>(0, 0, 0));
        *backgroundLight = PointLight(Vector3<>(0, 0, 0), Vector3<>(0, 0, 255));

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

            new (pointLight) PointLight();

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

        error = cudaFree(pointLights);

        if(error != cudaSuccess) {
            return error;
        }

        error = cudaFree(backgroundLight);

        if(error != cudaSuccess) {
            return error;
        }

        return cudaFree(ambientLight);
    }
}