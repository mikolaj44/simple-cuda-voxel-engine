#pragma once

#include <stdint.h>

#include "vector3.cuh"
#include "octree/octree_utils.cuh"

#include "cuda_math.cuh"
#include "octree/octree_utils.cuh"
#include "pair.cuh"
#include "block_info.cuh"

class Octree {
public:
	__host__ cudaError_t init(int xMin, int yMin, int zMin, unsigned int maxLevel);

	__host__ cudaError_t init(unsigned int maxLevel);

	__host__ cudaError_t cleanup();

	__host__ cudaError_t clear();

	template<typename XYZFrametoIdFunction>
    cudaError_t insertBlocksByXYZFrameFunction(XYZFrametoIdFunction blockPosToIdFunction, uint64_t frameNumber, bool isCalculatingInsertLODsEnabled, unsigned int gridSize, unsigned int blockSize) {
        insertBlockByXYZFrameFunctionKernel<<<gridSize, blockSize>>>(this, blockPosToIdFunction, frameNumber);

		if(isCalculatingInsertLODsEnabled) {
			for(int level = 0; level <= maxLevel; level++) {
				// printf("grid size: %d, total: %d, side length: %d\n", gridSize, gridSize * blockSize, (int)cbrtf(float(gridSize) * float(blockSize)));

				insertBlocksByXYZFrameFunctionFixLODKernel<<<gridSize, blockSize>>>(this, blockPosToIdFunction, frameNumber, level);

				if(blockSize >= 8) {
					blockSize /= 8;
				}
				else if(gridSize >= 8) {
					gridSize /= 8;
				}
			}
		}

		return cudaDeviceSynchronize();
    }

    cudaError_t insertBlocks(uint8_t* blockIdArray, scve::Vector3<int> startOffset, bool isCalculatingInsertLODsEnabled, unsigned int chunkWidth, unsigned int gridSize, unsigned int blockSize);

	cudaError_t getBlocks(uint8_t* outBlockIdArray, scve::Vector3<int> startOffset, unsigned int chunkWidth, unsigned int gridSize, unsigned int blockSize);

	__device__ Triple<scve::Vector3<int>, scve::Vector3<float>, uint8_t> getRayIntersectionData(scve::Vector3<> rayOrigin, scve::Vector3<> rayDirection, int sX, int sY, int minNodeSize);

	__device__ __host__ void setMinPos(scve::Vector3<> minPos);

	__host__ cudaError_t setMaxLevel(unsigned int maxLevel);

	__device__ __host__ scve::Vector3<int> getMinPos() const;

	__device__ __host__ unsigned int getMaxLevel() const;

	__device__ __host__ unsigned int getMaxSize() const;

	__host__ static unsigned int getMaxOctreeLevelByGPU();
private:
	struct alignas(uint8_t) Node {
		// most significant bit is 1 if the node is not solid - solid (0) means either leaf or all children with id of one type
		// the rest of the bits are for block id (0 reserved for air / empty node)
		uint8_t id;

		__device__ Node() {};

		__device__ Node(uint8_t id_) : id(id_) {};

		__device__ inline bool isMixed() const {
			return id & 128;
		}

		__device__ inline unsigned char blockId() const {
			return id & 127;
		}
	};

	Node* nodes = nullptr;

	int xMin, yMin, zMin;

	unsigned int maxLevel = 0; // level 0 is a terminal node

	unsigned int maxSize;

	size_t allocatedMemoryInBytes;

	constexpr static unsigned int maxPossibleLevel = 10;


	__device__ void insert(const BlockInfo<>& block);

	__device__ void insertFixLOD(uint32_t mortonCode, uint8_t blockId);



	__host__ cudaError_t allocateByMaxLevel(unsigned int newMaxLevel);



	__device__ scve::Vector3<int> morton3Ddecode(uint32_t mortonCode);

	__device__ uint32_t morton3Dencode(scve::Vector3<int> pos);



	__device__ void traverseNewNode(bool& foundSolid, Triple<scve::Vector3<int>, scve::Vector3<float>, uint8_t>& intersectionData, octree_utils::Stack& stack, scve::Vector3<> origRayOrigin, scve::Vector3<> origRayDirection, float tx0, float ty0, float tz0, float tx1, float ty1, float tz1, unsigned int nodeIdx, int minNodeSize, int sX, int sY);

	__device__ void traverseChildNodes(bool& foundSolid, Triple<scve::Vector3<int>, scve::Vector3<float>, uint8_t>& intersectionData, octree_utils::Stack& stack, octree_utils::Stack::Frame& data, scve::Vector3<> origRayOrigin, scve::Vector3<> origRayDirection, unsigned char a, int minNodeSize, int sX, int sY);

	template<typename XYZFrametoIdFunction>
	friend __global__ void insertBlockByXYZFrameFunctionKernel(Octree* octree, XYZFrametoIdFunction blockPosToIdFunction, uint64_t frameNumber);

	template<typename XYZFrametoIdFunction>
	friend __global__ void insertBlocksByXYZFrameFunctionFixLODKernel(Octree* octree, XYZFrametoIdFunction blockPosToIdFunction, uint64_t frameNumber, unsigned int level);

	friend __global__ void insertBlocksKernel(Octree* octree, uint8_t* blockIdArray, unsigned int chunkWidth, scve::Vector3<int> startOffset);

	friend __global__ void getBlocksKernel(Octree* octree, uint8_t* outBlockIdArray, unsigned int chunkWidth, scve::Vector3<int> startOffset);
};

template<typename XYZFrametoIdFunction>
__global__ void insertBlockByXYZFrameFunctionKernel(Octree* octree, XYZFrametoIdFunction blockPosToIdFunction, uint64_t frameNumber) {            
    size_t size = octree->getMaxSize();

	size_t index = threadIdx.x + blockIdx.x * blockDim.x;

    size_t x = index % size;
    size_t y = (index / size) % size;
    size_t z = index / (size * size);

    if(x >= size || y >= size || z >= size) {
        return;
    }

    scve::Vector3<int> minPos = octree->getMinPos();

    uint8_t id = blockPosToIdFunction(x + minPos.x, y + minPos.y, z + minPos.z, frameNumber);
	
    if(id != 0) {
        octree->insert(BlockInfo<>(scve::Vector3<int>(x, y, z), id));
    }
}

template<typename XYZFrametoIdFunction>
__global__ void insertBlocksByXYZFrameFunctionFixLODKernel(Octree* octree, XYZFrametoIdFunction blockPosToIdFunction, uint64_t frameNumber, unsigned int level) {
	unsigned int size = octree->getMaxSize();

	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;

    int x = (index % size)		    * (1 << level);
    int y = ((index / size) % size) * (1 << level);
    int z = (index / (size * size)) * (1 << level);

    if(x >= size || y >= size || z >= size) {
        return;
    }

	scve::Vector3<int> pos = scve::Vector3<int>::add(scve::Vector3<int>(x, y, z), octree->getMinPos());

    uint8_t id = blockPosToIdFunction(pos.x, pos.y, pos.z, frameNumber);



	// scve::Vector3<int> original = scve::Vector3<int>(x + minPos.x, y + minPos.y, z + minPos.z);

	// scve::Vector3<int> decoded = octree->morton3Ddecode(octree->morton3Dencode(original));

	// if(original != decoded)
	// 	printf("%d %d %d -> %d %d %d\n", original.x, original.y, original.z, decoded.x, decoded.y, decoded.z);


	if(id != 0) {	
    	octree->insertFixLOD(octree->morton3Dencode(pos) >> (level * 3), id);
	}

	// printf("1\n");
}