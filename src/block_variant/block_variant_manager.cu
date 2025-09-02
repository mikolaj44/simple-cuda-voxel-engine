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

    template<typename IdFrameToMaterialFunction>
    __global__ void setBlocksDataKernel(IdFrameToMaterialFunction func, uint64_t frameNumber, BlockTexture** blockTextures) {
        int index = threadIdx.x + blockIdx.x * blockDim.x;

        if(index < 127) {
            new (blockVariants[index]) BlockVariant(func(index + 1, frameNumber), blockTextures[index]);
        }
    }

    template<typename IdFrameToMaterialFunction>
    void setMaterials(IdFrameToMaterialFunction func, uint64_t frameNumber) {
        setBlocksDataKernel<<<127,1>>>(func, frameNumber, blockTextures);
    }

    void init() {
        cudaMallocManaged(&blockTextures, size_t(127 * sizeof(BlockTexture*)));
    
        std::filesystem::path textureDirPath = std::filesystem::current_path().parent_path() += "/res/textures/";
    
        if(!std::filesystem::exists(textureDirPath)) {
            return;
        }
    
        int textureIndex = 0;
    
        while(textureIndex < 127) {
            std::filesystem::path currentPath = textureDirPath;
    
            currentPath += std::to_string(textureIndex + 1);
    
            if(!std::filesystem::exists(currentPath)) {
                textureIndex++;
                continue;
            }
    
            cudaMallocManaged(&blockTextures[textureIndex], sizeof(BlockTexture));
    
            std::string paths[6] = {currentPath.string() + "/top.png", currentPath.string() + "/bottom.png", currentPath.string() + "/left.png", currentPath.string() + "/right.png", currentPath.string() + "/front.png", currentPath.string() + "/back.png"};
            
            try {
                new (blockTextures[textureIndex]) BlockTexture(4, paths);
            }
            catch (std::string exceptionMessage) {
                std::cerr << exceptionMessage << std::endl;
                abort();
            }
        
            textureIndex++;
        }

        for(int i = 0; i < 127; i++) {
            cudaMalloc(&blockVariantsHost[i], sizeof(BlockVariant));
        }

        cudaMalloc(&blockVariantsDevice, sizeof(BlockVariant*) * 127);

        cudaMemcpy(blockVariantsDevice, blockVariantsHost, sizeof(BlockVariant*) * 127, cudaMemcpyHostToDevice);

        cudaMemcpyToSymbol(blockVariants, &blockVariantsDevice, sizeof(BlockVariant**));

        cudaFree(blockVariantsDevice);

        auto idFrameToMaterialFunction = [] __device__ (uint8_t id, uint64_t frameNumber) {
            return Material(Vector3<>(255,255,255), 1, 0, 20);
        };

        setMaterials(idFrameToMaterialFunction, 0);
    
        cudaDeviceSynchronize();
    }

    void cleanup() {
        for(int i = 0; i < 127; i++) {
            cudaFree(blockVariantsHost[i]);
        }
    }
}