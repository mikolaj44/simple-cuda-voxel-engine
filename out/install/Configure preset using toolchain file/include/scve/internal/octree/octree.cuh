#pragma once

#include <stdint.h>

#include "scve/internal/octree/octree_utils.cuh"

#include "scve/internal/structure/vector3.h"
#include "scve/internal/structure/functor.h"
#include "scve/internal/structure/tuple.cuh"
#include "scve/internal/structure/block_info.cuh"
#include "scve/internal/cuda_math.cuh"


namespace scve {

class Octree {
public:
	__host__ cudaError_t init(int xMin, int yMin, int zMin, unsigned int maxLevel, bool displayMemoryInfo = false);

	__host__ cudaError_t init(unsigned int maxLevel, bool displayMemoryInfo = false);

	__host__ cudaError_t cleanup();

	__host__ cudaError_t clear();

	template<typename XYZFrameToIdFunctor>
    cudaError_t insertBlocksByXYZFrameFunctor(XYZFrameToIdFunctor blockPosToIdFunctor, uint64_t frameNumber, bool isCalculatingInsertLODsEnabled, unsigned int gridSize, unsigned int blockSize) {		
		insertBlockByXYZFrameFunctorKernel<<<gridSize, blockSize>>>(this, blockPosToIdFunctor, frameNumber);

		// if(isCalculatingInsertLODsEnabled) {
		// 	for(int level = 0; level <= maxLevel; level++) {
		// 		// printf("grid size: %d, total: %d, side length: %d\n", gridSize, gridSize * blockSize, (int)cbrtf(float(gridSize) * float(blockSize)));

		// 		insertBlocksByXYZFrameFunctorFixLODKernel<<<gridSize, blockSize>>>(this, blockPosToIdFunctor, frameNumber, level);

		// 		if(blockSize >= 8) {
		// 			blockSize /= 8;
		// 		}
		// 		else if(gridSize >= 8) {
		// 			gridSize /= 8;
		// 		}
		// 	}
		// }

		return cudaDeviceSynchronize();
    }

    cudaError_t insertBlocks(uint8_t* blockIdArray, scve::Vector3<int> startOffset, bool isCalculatingInsertLODsEnabled, unsigned int chunkWidth, unsigned int gridSize, unsigned int blockSize);

	cudaError_t getBlocks(uint8_t* outBlockIdArray, scve::Vector3<int> startOffset, unsigned int chunkWidth, unsigned int gridSize, unsigned int blockSize);

	__device__ Triple<scve::Vector3<int>, scve::Vector3<float>, uint8_t> getRayIntersectionData(scve::Vector3<> rayOrigin, scve::Vector3<> rayDirection, int sX, int sY, int minNodeSize);

	__device__ __host__ void setMinPos(scve::Vector3<int> minPos);

	__device__ __host__ scve::Vector3<int> getMinPos() const {
		return scve::Vector3<int>(xMin, yMin, zMin);
	};

	__device__ __host__ unsigned int getMaxLevel() const {
		return maxLevel;
	}

	__device__ __host__ unsigned int getMaxSize() const {
		return maxSize;
	}

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

	size_t allocatedMemoryInBytes = 0;

	constexpr static unsigned int MAX_POSSIBLE_LEVEL = 10;

	// First part of the insertion (top-down): inserting the correct leaf-voxel data
	__device__ void insert(const BlockInfo<>& block) {
		int x = block.pos.x;
		int y = block.pos.y;
		int z = block.pos.z;
	
		int size = maxSize;
	
		int xMin = 0;
		int yMin = 0;
		int zMin = 0;
	
		int xM;
		int yM;
		int zM;
	
		uint32_t index = 1; // root node index
	
		// Iterate over all node levels up until the leaf node
		do {
			// Get the node at index (to insert the right block data)
			if (size == 1) {
				nodes[index].id = block.id;
				return;
			}
	
			nodes[index].id = 128;
		
			// Get the midpoint
			xM = (2 * xMin + size) / 2;
			yM = (2 * yMin + size) / 2;
			zM = (2 * zMin + size) / 2;
	
			index <<= 3;
	
			// Compute the coordinates and morton code of the child node
			if (x >= xM) {
				xMin += size / 2;
				index |= 1;
			}
			if (y >= yM) {
				yMin += size / 2;
				index |= 2;
			}
	
			if (z >= zM) {	
				zMin += size / 2;
				index |= 4;
			}
		
			size /= 2;
	
		} while (size >= 1);
	}

	// Second and final part of the insertion (bottom-up): fixing the LOD data (determining if nodes are solid)
	__device__ void insertFixLOD(uint32_t mortonCode, uint8_t blockId);



	__host__ cudaError_t allocateByMaxLevel(unsigned int newMaxLevel, bool displayMemoryInfo);



	__device__ scve::Vector3<int> morton3Ddecode(uint32_t mortonCode);

	__device__ uint32_t morton3Dencode(scve::Vector3<int> pos);



	__device__ void traverseNewNode(bool& foundSolid, Triple<scve::Vector3<int>, scve::Vector3<float>, uint8_t>& intersectionData, octree_utils::Stack& stack, scve::Vector3<> origRayOrigin, scve::Vector3<> origRayDirection, float tx0, float ty0, float tz0, float tx1, float ty1, float tz1, unsigned int nodeIdx, int minNodeSize, int sX, int sY);

	__device__ void traverseChildNodes(bool& foundSolid, Triple<scve::Vector3<int>, scve::Vector3<float>, uint8_t>& intersectionData, octree_utils::Stack& stack, octree_utils::Stack::Frame& data, scve::Vector3<> origRayOrigin, scve::Vector3<> origRayDirection, unsigned char a, int minNodeSize, int sX, int sY);

	template<typename XYZFrameToIdFunctor>
	friend __global__ void insertBlockByXYZFrameFunctorKernel(Octree* octree, XYZFrameToIdFunctor blockPosToIdFunctor, uint64_t frameNumber);

	template<typename XYZFrameToIdFunctor>
	friend __global__ void insertBlocksByXYZFrameFunctorFixLODKernel(Octree* octree, XYZFrameToIdFunctor blockPosToIdFunctor, uint64_t frameNumber, unsigned int level);

	friend __global__ void insertBlocksKernel(Octree* octree, uint8_t* blockIdArray, unsigned int chunkWidth, scve::Vector3<int> startOffset);

	friend __global__ void getBlocksKernel(Octree* octree, uint8_t* outBlockIdArray, unsigned int chunkWidth, scve::Vector3<int> startOffset);
};

template<typename XYZFrameToIdFunctor>
__global__ void insertBlockByXYZFrameFunctorKernel(Octree* octree, XYZFrameToIdFunctor blockPosToIdFunctor, uint64_t frameNumber) {            
    size_t size = octree->getMaxSize();

	size_t index = threadIdx.x + blockIdx.x * blockDim.x;

    size_t x = index % size;
    size_t y = (index / size) % size;
    size_t z = index / (size * size);

    if(x >= size || y >= size || z >= size) {
        return;
    }

    scve::Vector3<int> minPos = octree->getMinPos();

    uint8_t id = blockPosToIdFunctor(x + minPos.x, y + minPos.y, z + minPos.z, frameNumber);
	
    if(id != 0) {
        octree->insert(BlockInfo<>(scve::Vector3<int>(x, y, z), id));
    }
}

template<typename XYZFrameToIdFunctor>
__global__ void insertBlocksByXYZFrameFunctorFixLODKernel(Octree* octree, XYZFrameToIdFunctor blockPosToIdFunctor, uint64_t frameNumber, unsigned int level) {
	unsigned int size = octree->getMaxSize();

	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;

    int x = (index % size)		    * (1 << level);
    int y = ((index / size) % size) * (1 << level);
    int z = (index / (size * size)) * (1 << level);

    if(x >= size || y >= size || z >= size) {
        return;
    }

	scve::Vector3<int> pos = scve::Vector3<int>::add(scve::Vector3<int>(x, y, z), octree->getMinPos());

    uint8_t id = blockPosToIdFunctor(pos.x, pos.y, pos.z, frameNumber);



	// scve::Vector3<int> original = scve::Vector3<int>(x + minPos.x, y + minPos.y, z + minPos.z);

	// scve::Vector3<int> decoded = octree->morton3Ddecode(octree->morton3Dencode(original));

	// if(original != decoded)
	// 	printf("%d %d %d -> %d %d %d\n", original.x, original.y, original.z, decoded.x, decoded.y, decoded.z);


	if(id != 0) {	
    	octree->insertFixLOD(octree->morton3Dencode(pos) >> (level * 3), id);
	}

	// printf("1\n");
}

}