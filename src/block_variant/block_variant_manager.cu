#include "block_variant/block_variant_manager.cuh"
#include "cuda_math.cuh"
#include "renderer/cuda_renderer.cuh"

#include <filesystem>
#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace block_variant_manager {
    __managed__ BlockVariant** blockVariants = nullptr;

    unsigned int numVariants;

    BlockTexture** blockTextures = nullptr;

    cudaError_t init(unsigned int numVariants_) {
        numVariants = numVariants_;

        cudaError_t error = cudaMallocManaged(&blockTextures, size_t(sizeof(BlockTexture*) * numVariants));

        if(error != cudaSuccess) {
            return error;
        }
    
        std::filesystem::path textureDirPath = std::filesystem::current_path().parent_path() += "/res/textures/";
    
        if(!std::filesystem::exists(textureDirPath)) {
            return error;
        }
    
        int textureIndex = 0;
    
        while(textureIndex < numVariants) {
            std::filesystem::path currentPath = textureDirPath;
    
            currentPath += std::to_string(textureIndex + 1);
    
            if(!std::filesystem::exists(currentPath)) {
                textureIndex++;
                continue;
            }
    
            error = cudaMallocManaged(&blockTextures[textureIndex], sizeof(BlockTexture));

            if(error != cudaSuccess) {
                return error;
            }
    
            std::string paths[6] = {currentPath.string() + "/top.png", currentPath.string() + "/bottom.png", currentPath.string() + "/left.png", currentPath.string() + "/right.png", currentPath.string() + "/front.png", currentPath.string() + "/back.png"};
            
            try {
                new (blockTextures[textureIndex]) BlockTexture();

                error = blockTextures[textureIndex]->init(4, paths);

                if(error != cudaSuccess) {
                    return error;
                }    
            }
            catch (std::string exceptionMessage) {
                std::cerr << exceptionMessage << std::endl;
                abort();
            }
        
            textureIndex++;
        }

        error = cudaMallocManaged(&blockVariants, sizeof(BlockVariant*) * numVariants);

        if(error != cudaSuccess) {
            return error;
        }

        for(int i = 0; i < numVariants; i++) {
            error = cudaMallocManaged(&blockVariants[i], sizeof(BlockVariant));

            if(error != cudaSuccess) {
                return error;
            }

            new (blockVariants[i]) BlockVariant(Material(Vector3<>(255, 0, 255), 1, 0, 20), blockTextures[i]);
        }

        return error;
    }

    cudaError_t cleanup() {
        cudaError_t error = cudaSuccess;

        for(int i = 0; i < numVariants; i++) {
            error = blockVariants[i]->texture->cleanup();;

            if(error != cudaSuccess) {
                return error;
            }    
        }

        return error;
    }
}