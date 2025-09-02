#include "block_variant/block_variant_manager.cuh"
#include "cuda_math.cuh"
#include "renderer/cuda_renderer.cuh"

#include <filesystem>
#include <iostream>

namespace block_variant_manager {
    __device__ BlockVariant** blockVariants = nullptr;

    BlockVariant* blockVariantsHost[127];
    BlockVariant* blockVariantsDevice = nullptr;

    BlockTexture** blockTextures;

    __global__ void initializeBlocksVariantsKernel(BlockTexture** blockTextures) {
        int index = threadIdx.x + blockIdx.x * blockDim.x;

        if(index < 127) {
            new (blockVariants[index]) BlockVariant(Material(Vector3<>(255,255,255), 1, 0, 20), blockTextures[index]);
        }
    }

    template<typename IdFrameToMaterialFunction>
    __global__ void setBlocksVariantMaterialsKernel(IdFrameToMaterialFunction func, uint64_t frameNumber, BlockTexture** blockTextures) {
        int index = threadIdx.x + blockIdx.x * blockDim.x;

        if(index < 127) {
            blockVariants[index]->material = func(index + 1, frameNumber);
        }
    }

    template<typename IdFrameToMaterialFunction>
    void setMaterials(IdFrameToMaterialFunction func, uint64_t frameNumber) {
        setBlocksVariantMaterialsKernel<<<127,1>>>(func, frameNumber, blockTextures);
    }

    cudaError_t init() {
        cudaError_t error = cudaMallocManaged(&blockTextures, size_t(127 * sizeof(BlockTexture*)));

        if(error != cudaSuccess) {
            return error;
        }
    
        std::filesystem::path textureDirPath = std::filesystem::current_path().parent_path() += "/res/textures/";
    
        if(!std::filesystem::exists(textureDirPath)) {
            return error;
        }
    
        int textureIndex = 0;
    
        while(textureIndex < 127) {
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

                error = blockTextures[textureIndex]->create(4, paths);

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

        for(int i = 0; i < 127; i++) {
            error = cudaMalloc(&blockVariantsHost[i], sizeof(BlockVariant));

            if(error != cudaSuccess) {
                return error;
            }
        }

        error = cudaMalloc(&blockVariantsDevice, sizeof(BlockVariant*) * 127);

        if(error != cudaSuccess) {
            return error;
        }

        error = cudaMemcpy(blockVariantsDevice, blockVariantsHost, sizeof(BlockVariant*) * 127, cudaMemcpyHostToDevice);

        if(error != cudaSuccess) {
            return error;
        }

        error = cudaMemcpyToSymbol(blockVariants, &blockVariantsDevice, sizeof(BlockVariant**));

        if(error != cudaSuccess) {
            return error;
        }

        error = cudaFree(blockVariantsDevice);

        if(error != cudaSuccess) {
            return error;
        }

        initializeBlocksVariantsKernel<<<127, 1>>>(blockTextures);

        error = cudaDeviceSynchronize();

        return error;
    }

    cudaError_t cleanup() {
        cudaError_t error = cudaSuccess;

        for(int i = 0; i < 127; i++) {
            error = cudaFree(blockVariantsHost[i]);

            if(error != cudaSuccess) {
                return error;
            }    
        }

        return error;
    }
}