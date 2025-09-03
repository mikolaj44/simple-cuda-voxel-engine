#pragma once

#include "light/point_light_manager.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace point_light_manager {
    PointLight** pointLightsHost = nullptr;

    __device__ PointLight** pointLights = nullptr;

    unsigned int numLights;

    template<typename IndexFrameToPointLightFunction>
    __global__ void setPointLightsKernel(IndexFrameToPointLightFunction func, uint64_t frameNumber) {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
    
        if(index < numLights) {
            PointLight pointLight = func(index, frameNumber);

            pointLights[index]->color = pointLight.color;
            pointLights[index]->pos = pointLight.pos;
        }
    }

    __global__ void createLightsKernel(PointLight** pointLights, int numLights) {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
    
        if(index < numLights) {
            new (pointLights[index]) PointLight(Vector3<float>(0, 0, 0), Vector3<float>(255, 255, 255));
        }
    }

    cudaError_t init(unsigned int numLights_) {
        numLights = numLights_;

        cudaError_t error = cudaSuccess;

        pointLightsHost = new PointLight*[numLights];

        PointLight** pointLightsDevice = nullptr;

        for(int i = 0; i < numLights; i++) {
            error = cudaMallocManaged(&pointLightsHost[i], sizeof(PointLight));

            if(error != cudaSuccess) {
                return error;
            }
        }

        error = cudaMallocManaged(&pointLightsDevice, sizeof(PointLight*) * numLights);

        if(error != cudaSuccess) {
            return error;
        }

        error = cudaMemcpy(pointLightsDevice, pointLightsHost, sizeof(PointLight*) * numLights, cudaMemcpyHostToDevice);

        if(error != cudaSuccess) {
            return error;
        }

        createLightsKernel<<<1, numLights>>>(pointLightsDevice, numLights);

        error = cudaDeviceSynchronize();

        if(error != cudaSuccess) {
            return error;
        }

        error = cudaMemcpyToSymbol(pointLights, &pointLightsDevice, sizeof(PointLight**));

        if(error != cudaSuccess) {
            return error;
        }

        return cudaFree(pointLightsDevice);
    }

    cudaError_t cleanup() {
        cudaError_t error = cudaSuccess;

        for(int i = 0; i < numLights; i++) {
            error = cudaFree(pointLightsHost[i]);
    
            if(error != cudaSuccess) {
                return error;
            }    
        }
    
        delete[] pointLightsHost;

        return error;
    }

    template<typename IndexFrameToPointLightFunction>
    void setPointLights(IndexFrameToPointLightFunction func, uint64_t frameNumber) {
        setPointLightsKernel<<<1, numLights>>>(func, frameNumber);
    }
}