#include "block_variant/block_variant_manager.cuh"
#include "cuda_math.cuh"
#include "renderer/cuda_renderer.cuh"

#include <filesystem>
#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace scve::block_variant_manager {
    __managed__ ManagedList<BlockVariant*>* blockVariants;
    __managed__ int numVariantsWithTextures = 0;

    BlockTexture** blockTextures = nullptr;

    cudaError_t init(unsigned int maxNumVariants) {
        cudaError_t error = cudaMallocManaged(&blockTextures, size_t(sizeof(BlockTexture*) * maxNumVariants));

        if(error != cudaSuccess) {
            return error;
        }
    
        std::filesystem::path textureDirPath = std::filesystem::current_path().parent_path() += "/res/textures/";
    
        if(!std::filesystem::exists(textureDirPath)) {
            return error;
        }
    
        int index = 0;
        int variantIndex = 0;
    
        while(index < maxNumVariants) {
            std::filesystem::path currentPath = textureDirPath;
    
            currentPath += std::to_string(index + 1);
    
            if(!std::filesystem::exists(currentPath)) {
                index++;
                continue;
            }
    
            error = cudaMallocManaged(&blockTextures[variantIndex], sizeof(BlockTexture));

            if(error != cudaSuccess) {
                return error;
            }
    
            std::string paths[6] = {currentPath.string() + "/top.png", currentPath.string() + "/bottom.png", currentPath.string() + "/left.png", currentPath.string() + "/right.png", currentPath.string() + "/front.png", currentPath.string() + "/back.png"};
                          
            new (blockTextures[variantIndex]) BlockTexture();

            error = blockTextures[variantIndex]->init(4, paths);

            if(error != cudaSuccess) {
                return error;
            }    
        
            index++;
            variantIndex++;
        }

        error = cudaMallocManaged(&blockVariants, sizeof(ManagedList<BlockVariant*>));

        if(error != cudaSuccess) {
            return error;
        }

        numVariantsWithTextures = variantIndex;

        new (blockVariants) ManagedList<BlockVariant*>();

        error = blockVariants->init(maxNumVariants);

        if(error != cudaSuccess) {
            cudaFree(blockVariants);
            return error;
        }

        for(int i = 0; i < maxNumVariants; i++) {
            BlockVariant* blockVariant;

            error = cudaMallocManaged(&blockVariant, sizeof(BlockVariant));

            if(error != cudaSuccess) {
                return error;
            }

            if(i < numVariantsWithTextures) {
                new (blockVariant) BlockVariant(Material(Vector3<>(255, 0, 255), 1, 0, 20), blockTextures[i]);
            }
            else {
                new (blockVariant) BlockVariant(Material(Vector3<>(255, 0, 255), 1, 0, 20), nullptr);
            }

            error = blockVariants->add(blockVariant);

            if(error != cudaSuccess) {
                return error;
            }
        }

        return error;
    }

    cudaError_t cleanup() {
        cudaError_t error = cudaSuccess;

        for(int i = 0; i < blockVariants->size(); i++) {
            if(i < numVariantsWithTextures) {
                error = ((*blockVariants)[i])->texture->cleanup();

                if(error != cudaSuccess) {
                    return error;
                }
            }

            error = cudaFree((*blockVariants)[i]);

            if(error != cudaSuccess) {
                return error;
            }
        }

        error = blockVariants->cleanup();

        if(error != cudaSuccess) {
            return error;
        }

        error = cudaFree(blockVariants);

        if(error != cudaSuccess) {
            return error;
        }

        numVariantsWithTextures = 0;

        return cudaFree(blockTextures);
    }
}