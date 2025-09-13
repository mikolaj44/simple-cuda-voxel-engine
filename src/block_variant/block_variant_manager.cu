#include "block_variant/block_variant_manager.cuh"
#include "cuda_math.cuh"
#include "renderer/cuda_renderer.cuh"

#include <filesystem>
#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace scve::block_variant_manager {
    __managed__ ManagedList<BlockVariant*>* blockVariants;

    BlockTexture** blockTextures = nullptr;

    cudaError_t init(std::string texturesPath, unsigned int maxNumVariants, bool skipTextureLoading) {
        cudaError_t error = cudaMallocManaged(&blockTextures, size_t(sizeof(BlockTexture*) * maxNumVariants));

        if(error != cudaSuccess) {
            return error;
        }

        if(!skipTextureLoading) {
            if(texturesPath[texturesPath.length() - 1] != '/') {
                texturesPath += '/';
            }
        
            std::filesystem::path textureDirPath = texturesPath;
        
            if(!std::filesystem::exists(textureDirPath)) {
                throw std::runtime_error("texture path does not exist");
            }
        
            int index = 0;
        
            while(index < maxNumVariants) {            
                std::filesystem::path currentPath = textureDirPath;
        
                currentPath += std::to_string(index + 1);
        
                if(!std::filesystem::exists(currentPath)) {
                    index++;
                    continue;
                }
        
                error = cudaMallocManaged(&blockTextures[index], sizeof(BlockTexture));

                if(error != cudaSuccess) {
                    cleanup();
                    return error;
                }
        
                std::string paths[6] = {currentPath.string() + "/top.png", currentPath.string() + "/bottom.png", currentPath.string() + "/left.png", currentPath.string() + "/right.png", currentPath.string() + "/front.png", currentPath.string() + "/back.png"};
                            
                new (blockTextures[index]) BlockTexture();

                error = blockTextures[index]->init(4, paths);

                if(error != cudaSuccess) {
                    cleanup();
                    return error;
                }    
            
                index++;
            }
        }

        error = cudaMallocManaged(&blockVariants, sizeof(ManagedList<BlockVariant*>));

        if(error != cudaSuccess) {
            cleanup();
            return error;
        }

        new (blockVariants) ManagedList<BlockVariant*>();

        error = blockVariants->init(maxNumVariants);

        if(error != cudaSuccess) {
            cleanup();
            return error;
        }

        for(int i = 0; i < maxNumVariants; i++) {
            BlockVariant* blockVariant;

            error = cudaMallocManaged(&blockVariant, sizeof(BlockVariant));

            if(error != cudaSuccess) {
                cleanup();
                return error;
            }

            new (blockVariant) BlockVariant(Material(Vector3<>(255, 0, 255), 1, 0, 20), blockTextures[i]);

            error = blockVariants->add(blockVariant);

            if(error != cudaSuccess) {
                cleanup();
                return error;
            }
        }

        return error;
    }

    cudaError_t cleanup() {
        cudaError_t lastError = cudaSuccess;

        for(int i = 0; i < blockVariants->size(); i++) {
            if(((*blockVariants)[i])->texture != nullptr) {
                cudaError_t error = ((*blockVariants)[i])->texture->cleanup();

                if(error != cudaSuccess) {
                    lastError = error;
                }
            }

            cudaError_t error = cudaFree((*blockVariants)[i]);

            if(error != cudaSuccess) {
                lastError = error;
            }
        }

        cudaError_t error = blockVariants->cleanup();

        if(error != cudaSuccess) {
            lastError = error;
        }

        error = cudaFree(blockVariants);

        if(error != cudaSuccess) {
            lastError = error;
        }

        error = cudaFree(blockTextures);

        if(error != cudaSuccess) {
            lastError = error;
        }

        return lastError;
    }
}