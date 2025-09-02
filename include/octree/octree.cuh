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
	__host__ cudaError_t create(int xMin, int yMin, int zMin, unsigned int maxLevel);

	__host__ cudaError_t create(unsigned int maxLevel);

	__host__ cudaError_t cleanup();

	__host__ cudaError_t clear();

	template<typename XYZFrametoIdFunction>
    cudaError_t insertBlockByXYZFrameFunction(XYZFrametoIdFunction blockPosToIdFunction, uint64_t frameNumber, bool isCalculatingInsertLODsEnabled, unsigned int gridSize, unsigned int blockSize) {
        insertBlockByXYZFrameFunctionKernel<<<gridSize, blockSize>>>(this, blockPosToIdFunction, frameNumber);

		if(isCalculatingInsertLODsEnabled) {
			for(int level = 0; level <= maxLevel; level++) {
				// printf("grid size: %d, total: %d, side length: %d\n", gridSize, gridSize * blockSize, (int)cbrtf(float(gridSize) * float(blockSize)));

				insertFixLODKernel<<<gridSize, blockSize>>>(this, blockPosToIdFunction, frameNumber, level);

				if(blockSize >= 8) {
					blockSize /= 8;
				}
				else if(gridSize >= 8) {
					gridSize /= 8;
				}
			}
		}

		return cudaDeviceSynchronize();

		// printf("synchronized\n");
    }

	__device__ Triple<Vector3<int>, Vector3<float>, uint8_t> getRayIntersectionData(uchar4* pixels, Vector3<> rayOrigin, Vector3<> rayDirection, int sX, int sY, int minNodeSize);

	__device__ __host__ void setMinPos(Vector3<> minPos);

	__host__ cudaError_t setMaxLevel(unsigned int maxLevel);

	__device__ __host__ Vector3<int> getMinPos() const;

	__device__ __host__ unsigned int getMaxLevel() const;

	__device__ __host__ unsigned int getMaxSize() const;

	__host__ static int getMaxOctreeLevelByGPU();
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

	Node* nodes;

	int xMin, yMin, zMin;

	unsigned int maxLevel = 0; // level 0 is a terminal node

	unsigned int maxSize;

	size_t allocatedMemoryInBytes;



	__device__ void insert(BlockInfo<>& block);

	__device__ void insertFixLOD(uint32_t mortonCode, uint8_t blockId);



	__host__ cudaError_t allocateByMaxLevel(unsigned int newMaxLevel);



	__device__ Vector3<int> morton3Ddecode(uint32_t mortonCode);

	__device__ uint32_t morton3Dencode(Vector3<int> pos);



	__device__ void traverseNewNode(bool& foundSolid, Triple<Vector3<int>, Vector3<float>, uint8_t>& intersectionData, octree_utils::Stack& stack, uchar4* pixels, Vector3<> origRayOrigin, Vector3<> origRayDirection, float tx0, float ty0, float tz0, float tx1, float ty1, float tz1, unsigned int nodeIdx, int minNodeSize, int sX, int sY);

	__device__ void traverseChildNodes(bool& foundSolid, Triple<Vector3<int>, Vector3<float>, uint8_t>& intersectionData, octree_utils::Stack& stack, octree_utils::Stack::Frame& data, uchar4* pixels, Vector3<> origRayOrigin, Vector3<> origRayDirection, unsigned char a, int minNodeSize, int sX, int sY);

	template<typename XYZFrametoIdFunction>
	friend __global__ void insertBlockByXYZFrameFunctionKernel(Octree* octree, XYZFrametoIdFunction blockPosToIdFunction, uint64_t frameNumber);

	template<typename XYZFrametoIdFunction>
	friend __global__ void insertFixLODKernel(Octree* octree, XYZFrametoIdFunction blockPosToIdFunction, uint64_t frameNumber, unsigned int level);
};

template<typename XYZFrametoIdFunction>
__global__ void insertBlockByXYZFrameFunctionKernel(Octree* octree, XYZFrametoIdFunction blockPosToIdFunction, uint64_t frameNumber) {            
    unsigned int size = octree->getMaxSize();

	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;

    int x = index % size;
    int y = (index / size) % size;
    int z = index / (size * size);

    if(x >= size || y >= size || z >= size) {
        return;
    }

    Vector3<int> minPos = octree->getMinPos();

    uint8_t id = blockPosToIdFunction(x + minPos.x, y + minPos.y, z + minPos.z, frameNumber);

    BlockInfo<> blockInfo = BlockInfo<>(Vector3<int>(x, y, z), id);
	
    if(id != 0) {
        octree->insert(blockInfo);
    }
}


template<typename XYZFrametoIdFunction>
__global__ void insertFixLODKernel(Octree* octree, XYZFrametoIdFunction blockPosToIdFunction, uint64_t frameNumber, unsigned int level) {
	unsigned int size = octree->getMaxSize();

	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;

    int x = index % size;
    int y = (index / size) % size;
    int z = index / (size * size);

    if(x >= size || y >= size || z >= size) {
        return;
    }

	Vector3<int> minPos = octree->getMinPos().add(Vector3<int>(x, y, z));

    uint8_t id = blockPosToIdFunction(minPos.x, minPos.y, minPos.z, frameNumber);



	// Vector3<int> original = Vector3<int>(x + minPos.x, y + minPos.y, z + minPos.z);

	// Vector3<int> decoded = octree->morton3Ddecode(octree->morton3Dencode(original));

	// if(original != decoded)
	// 	printf("%d %d %d -> %d %d %d\n", original.x, original.y, original.z, decoded.x, decoded.y, decoded.z);


	
    octree->insertFixLOD(octree->morton3Dencode(minPos) >> (level * 3), id);

	// printf("1\n");
}