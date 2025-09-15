#if defined(__GNUC__) || defined(__clang__)
	#include <cxxabi.h>
#endif

#include <bitset>
#include <cstddef>
#include <iostream>
#include <limits>
#include <string>
#include <cmath>

#include "scve/internal/octree/octree.cuh"
#include "scve/internal/block_variant/block_variant_manager.cuh"
#include "scve/internal/cuda_math.cuh"
#include "scve/internal/renderer/cuda_renderer.cuh"
#include "scve/internal/light/point_light_manager.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace scve {

constexpr float epsilon = 0.00001;
constexpr float blockHitEpsilon = 0.00001;

namespace {
	size_t getAllocatedMemoryInBytes(unsigned int maxLevel) {
		return size_t(2) << (3 * maxLevel + 1);
	}
}

__host__  cudaError_t Octree::allocateByMaxLevel(unsigned int newMaxLevel, bool displayMemoryInfo) {
	if(newMaxLevel == maxLevel) {
		return cudaSuccess;
	}

	newMaxLevel = minv(MAX_POSSIBLE_LEVEL, newMaxLevel);

	allocatedMemoryInBytes = getAllocatedMemoryInBytes(newMaxLevel);
	maxLevel = newMaxLevel;
	maxSize = 1 << maxLevel;

	cudaError_t error = cudaSuccess;

	if(displayMemoryInfo) {
		size_t freeBytes, totalBytes;

		error = cudaMemGetInfo(&freeBytes, &totalBytes);

		if(error != cudaSuccess) {
			return error;
		}

		printf("\noctree: %zu bytes free out of %zu before allocation\n", freeBytes, totalBytes);

		printf("octree: allocating %zu bytes (%d levels)\n", allocatedMemoryInBytes, maxLevel);
	}

	error = cudaMalloc(&nodes, allocatedMemoryInBytes);

	if(error != cudaSuccess) {
		return error;
	}

	error = cudaMemset(nodes, 0, allocatedMemoryInBytes);

	if(error != cudaSuccess) {
		cudaFree(nodes);
		return error;
	}

	return cudaDeviceSynchronize();

	if(displayMemoryInfo) {
		size_t freeBytes, totalBytes;

		error = cudaMemGetInfo(&freeBytes, &totalBytes);

		if(error != cudaSuccess) {
			cudaFree(nodes);
			return error;
		}

		printf("octree: %zu bytes free out of %zu after allocation\n", freeBytes, totalBytes);
	}

	return error;
}

__host__ cudaError_t Octree::init(int xMin_, int yMin_, int zMin_, unsigned int maxLevel_, bool displayMemoryInfo) {
	xMin = xMin_;
	yMin = yMin_;
	zMin = zMin_;
	return allocateByMaxLevel(maxLevel_, displayMemoryInfo);
}

__host__ cudaError_t Octree::init(unsigned int maxLevel, bool displayMemoryInfo) {
	return init(0, 0, 0, maxLevel, displayMemoryInfo);
}

__host__ cudaError_t Octree::cleanup() {
	return cudaFree(nodes);
}

__host__ cudaError_t Octree::clear() {
	cudaError_t error = cudaMemset(nodes, 0, allocatedMemoryInBytes);

	if (error != cudaSuccess) {
		return error;
	}

	return cudaDeviceSynchronize();
}

__device__ scve::Vector3<int> Octree::morton3Ddecode(uint32_t mortonCode) {
	static const uint32_t mostSignificant1 = uint32_t(1) << 31;

	int index = 0;
	uint32_t code = mortonCode;

	while(code >>= 1) {
		index++;
	}

	mortonCode <<= (32 - index);

	int x = xMin;
	int y = yMin;
	int z = zMin;
	
	int size = maxSize;

	while(size > 1) {
		if(mortonCode & mostSignificant1) {
			z += size / 2;
		}
		if(mortonCode & (mostSignificant1 >> 1)){
			y += size / 2;
		}
		if(mortonCode & (mostSignificant1 >> 2)){
			x += size / 2;
		}

		mortonCode <<= 3;
		size /= 2;
	}

	return scve::Vector3<int>(x, y, z);
}

__device__ uint32_t Octree::morton3Dencode(scve::Vector3<int> pos) {
	uint32_t mortonCode = 1;

	int x = pos.x - xMin;
	int y = pos.y - yMin;
	int z = pos.z - zMin;

	int xMinCopy = 0;
	int yMinCopy = 0;
	int zMinCopy = 0;
	
	int size = maxSize;

	while(size > 1) {
		mortonCode <<= 3;

		if(x >= xMinCopy + size / 2) {
			mortonCode |= 0b001;
			xMinCopy += size / 2;
		}

		if(y >= yMinCopy + size / 2) {
			mortonCode |= 0b010;
			yMinCopy += size / 2;
		}

		if(z >= zMinCopy + size / 2) {
			mortonCode |= 0b100;
			zMinCopy += size / 2;
		}

		size /= 2;
	}

	return mortonCode;
}

// Second and final part of the insertion (bottom-up): fixing the LOD data (determining if nodes are solid)
__device__ void Octree::insertFixLOD(uint32_t mortonCode, uint8_t blockId) {
	static uint8_t combinations[8] = {0b000, 0b001, 0b010, 0b011, 0b100, 0b101, 0b110, 0b111};

	for(int i = 0; i < 8; i++) {
		mortonCode &= ~combinations[7]; // clear the last 3 bits
		mortonCode |= combinations[i];  // set them to the correct combination

		if(nodes[mortonCode].id != blockId) {
			return;
		}
	}

	mortonCode >>= 3;

	nodes[mortonCode].id = blockId;
}

__device__ void Octree::traverseNewNode(bool& foundSolid, Triple<scve::Vector3<int>, scve::Vector3<>, uint8_t>& intersectionData, octree_utils::Stack& stack, scve::Vector3<> origRayOrigin, scve::Vector3<> origRayDirection, float tx0, float ty0, float tz0, float tx1, float ty1, float tz1, unsigned int nodeIdx, int minNodeSize, int sX, int sY) {
	using namespace octree_utils::revelles;
	using namespace octree_utils;
	
	if(stack.topIndex >= maxLevel + 1) {
		return;
	}

	if (!nodes[nodeIdx].isMixed() && nodes[nodeIdx].blockId() != 0) { // nodeLevel(nodeIdx, maxLevel) == 0 && nodes[nodeIdx].blockId() != 0
		intersectionData.first = morton3Ddecode(nodeIdx);
		intersectionData.second = getBlockHitPos(intersectionData.first, origRayOrigin, origRayDirection, blockHitEpsilon);
		intersectionData.third = nodes[nodeIdx].blockId();
		
		foundSolid = true;
		return;
	}

	// (!nodes[nodeIdx].isMixed() && nodes[nodeIdx].blockId() == 0)
	if (nodes[nodeIdx].id == 0 || tx1 < 0.0f || ty1 < 0.0f || tz1 < 0.0f) { // !nodes[nodeIdx].isNotSolid()
		return;
	}

	const float txm = 0.5f * (tx0 + tx1);
	const float tym = 0.5f * (ty0 + ty1);
	const float tzm = 0.5f * (tz0 + tz1);

	stack.push({
		tx0, ty0, tz0,
		txm, tym, tzm,
		tx1, ty1, tz1,
		nodeIdx,
		firstNode(tx0, ty0, tz0, txm, tym, tzm, epsilon),
	});
}

__device__ void Octree::traverseChildNodes(bool& foundSolid, Triple<scve::Vector3<int>, scve::Vector3<>, uint8_t>& intersectionData, octree_utils::Stack& stack, octree_utils::Stack::Frame& data, scve::Vector3<> origRayOrigin, scve::Vector3<> origRayDirection, unsigned char a, int minNodeSize, int sX, int sY) {
	using namespace octree_utils::revelles;
	
	switch (data.nodeIndex) {
		case 0:
			data.nodeIndex = newNode(data.txm, 4, data.tym, 2, data.tzm, 1, epsilon);
			return traverseNewNode(foundSolid, intersectionData, stack, origRayOrigin, origRayDirection, data.tx0, data.ty0, data.tz0, data.txm, data.tym, data.tzm, childMortonRevelles(data.mortonCode,     a), minNodeSize, sX, sY);
		case 1:
			data.nodeIndex = newNode(data.txm, 5, data.tym, 3, data.tz1, 8, epsilon);
			return traverseNewNode(foundSolid, intersectionData, stack, origRayOrigin, origRayDirection, data.tx0, data.ty0, data.tzm, data.txm, data.tym, data.tz1, childMortonRevelles(data.mortonCode, 1 ^ a), minNodeSize, sX, sY);
		case 2:
			data.nodeIndex = newNode(data.txm, 6, data.ty1, 8, data.tzm, 3, epsilon);
			return traverseNewNode(foundSolid, intersectionData, stack, origRayOrigin, origRayDirection, data.tx0, data.tym, data.tz0, data.txm, data.ty1, data.tzm, childMortonRevelles(data.mortonCode, 2 ^ a), minNodeSize, sX, sY);
		case 3:
			data.nodeIndex = newNode(data.txm, 7, data.ty1, 8, data.tz1, 8, epsilon);
			return traverseNewNode(foundSolid, intersectionData, stack, origRayOrigin, origRayDirection, data.tx0, data.tym, data.tzm, data.txm, data.ty1, data.tz1, childMortonRevelles(data.mortonCode, 3 ^ a), minNodeSize, sX, sY);
		case 4:
			data.nodeIndex = newNode(data.tx1, 8, data.tym, 6, data.tzm, 5, epsilon);
			return traverseNewNode(foundSolid, intersectionData, stack, origRayOrigin, origRayDirection, data.txm, data.ty0, data.tz0, data.tx1, data.tym, data.tzm, childMortonRevelles(data.mortonCode, 4 ^ a), minNodeSize, sX, sY);
		case 5:
			data.nodeIndex = newNode(data.tx1, 8, data.tym, 7, data.tz1, 8, epsilon);
			return traverseNewNode(foundSolid, intersectionData, stack, origRayOrigin, origRayDirection, data.txm, data.ty0, data.tzm, data.tx1, data.tym, data.tz1, childMortonRevelles(data.mortonCode, 5 ^ a), minNodeSize, sX, sY);
		case 6:
			data.nodeIndex = newNode(data.tx1, 8, data.ty1, 8, data.tzm, 7, epsilon);
			return traverseNewNode(foundSolid, intersectionData, stack, origRayOrigin, origRayDirection, data.txm, data.tym, data.tz0, data.tx1, data.ty1, data.tzm, childMortonRevelles(data.mortonCode, 6 ^ a), minNodeSize, sX, sY);
		case 7:
			data.nodeIndex = 8;
			return traverseNewNode(foundSolid, intersectionData, stack, origRayOrigin, origRayDirection, data.txm, data.tym, data.tzm, data.tx1, data.ty1, data.tz1, childMortonRevelles(data.mortonCode, 7 ^ a), minNodeSize, sX, sY);
		case 8:
			stack.pop();
	}
}

__device__ Triple<scve::Vector3<int>, scve::Vector3<>, uint8_t> Octree::getRayIntersectionData(scve::Vector3<> rayOrigin, scve::Vector3<> rayDirection, int sX, int sY, int minNodeSize) {
	unsigned char a = 0;

	scve::Vector3<> origRayOrigin = rayOrigin;
	scve::Vector3<> origRayDirection = rayDirection;

	if (rayDirection.x < 0) {
		rayOrigin.x = -rayOrigin.x + (xMin * 2 + maxSize);
		rayDirection.x = -rayDirection.x;
		a |= 4;
	}
	if (rayDirection.y < 0) {
		rayOrigin.y = -rayOrigin.y + (yMin * 2 + maxSize);
		rayDirection.y = -rayDirection.y;
		a |= 2;
	}
	if (rayDirection.z < 0) {
		rayOrigin.z = -rayOrigin.z + (zMin * 2 + maxSize);
		rayDirection.z = -rayDirection.z;
		a |= 1;
	}

	float tx0 = (xMin - rayOrigin.x) / rayDirection.x;
	float tx1 = (xMin + maxSize - rayOrigin.x) / rayDirection.x;
	float ty0 = (yMin - rayOrigin.y) / rayDirection.y;
	float ty1 = (yMin + maxSize - rayOrigin.y) / rayDirection.y;
	float tz0 = (zMin - rayOrigin.z) / rayDirection.z;
	float tz1 = (zMin + maxSize - rayOrigin.z) / rayDirection.z;
	
	Triple<scve::Vector3<int>, scve::Vector3<>, uint8_t> intersectionData;

	if (maxv(maxv(tx0, ty0), tz0) < minv(minv(tx1, ty1), tz1)) {
		octree_utils::Stack stack;
		bool foundSolid = false;

		traverseNewNode(foundSolid, intersectionData, stack, origRayOrigin, origRayDirection, tx0, ty0, tz0, tx1, ty1, tz1, 1, minNodeSize, sX, sY);

		while (!stack.isEmpty() && !foundSolid) {
			traverseChildNodes(foundSolid, intersectionData, stack, stack.top(), origRayOrigin, origRayDirection, a, minNodeSize, sX, sY);
		}
	}

	return intersectionData;
}

__device__ __host__ void Octree::setMinPos(scve::Vector3<int> minPos) {
	xMin = minPos.x;
	yMin = minPos.y;
	zMin = minPos.z;
}

// __device__ __host__ scve::Vector3<int> Octree::getMinPos() const {
// 	return scve::Vector3<int>(xMin, yMin, zMin);
// };

// __device__ __host__ unsigned int Octree::getMaxLevel() const {
// 	return maxLevel;
// }

// __device__ __host__ unsigned int Octree::getMaxSize() const {
// 	return maxSize;
// }

__host__ unsigned int Octree::getMaxOctreeLevelByGPU() {
	size_t freeBytes, totalBytes;

	cudaMemGetInfo(&freeBytes, &totalBytes);

	unsigned int maxLevel = 1;

	while(getAllocatedMemoryInBytes(maxLevel) < freeBytes) {
		maxLevel++;
	}

	return minv(MAX_POSSIBLE_LEVEL, maxLevel - 1);
}

cudaError_t Octree::insertBlocks(uint8_t* blockIdArray, scve::Vector3<int> startOffset, bool isCalculatingInsertLODsEnabled, unsigned int chunkWidth, unsigned int gridSize, unsigned int blockSize) {
	insertBlocksKernel<<<gridSize, blockSize>>>(this, blockIdArray, chunkWidth, startOffset);

	// if(isCalculatingInsertLODsEnabled) {
	// 	for(int level = 0; level <= maxLevel; level++) {
	// 		// printf("grid size: %d, total: %d, side length: %d\n", gridSize, gridSize * blockSize, (int)cbrtf(float(gridSize) * float(blockSize)));

	// 		insertBlocksByXYZFrameFunctorFixLODKernel<<<gridSize, blockSize>>>(this, blockPosToIdFunction, frameNumber, level);

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

__global__ void insertBlocksKernel(Octree* octree, uint8_t* blockIdArray, unsigned int chunkWidth, scve::Vector3<int> startOffset) {            
	size_t index = threadIdx.x + blockIdx.x * blockDim.x;

    size_t x = index % chunkWidth 		  		 + startOffset.x;
    size_t y = (index / chunkWidth) % chunkWidth + startOffset.y;
    size_t z = index / (chunkWidth * chunkWidth) + startOffset.z;

	uint8_t blockId = blockIdArray[index];

    if(blockId == 0 || x >= chunkWidth || y >= chunkWidth || z >= chunkWidth) {
        return;
    }
	
    octree->insert(BlockInfo<>(scve::Vector3<int>(x, y, z), blockId));
}

cudaError_t Octree::getBlocks(uint8_t* outBlockIdArray, scve::Vector3<int> startOffset, unsigned int chunkWidth, unsigned int gridSize, unsigned int blockSize) {
	getBlocksKernel<<<gridSize, blockSize>>>(this, outBlockIdArray, chunkWidth, startOffset);

	return cudaDeviceSynchronize();
}

__global__ void getBlocksKernel(Octree* octree, uint8_t* outBlockIdArray, unsigned int chunkWidth, scve::Vector3<int> startOffset) {
	size_t index = threadIdx.x + blockIdx.x * blockDim.x;

    size_t x = index % chunkWidth 		  		 + startOffset.x;
    size_t y = (index / chunkWidth) % chunkWidth + startOffset.y;
    size_t z = index / (chunkWidth * chunkWidth) + startOffset.z;

	if(x >= chunkWidth || y >= chunkWidth || z >= chunkWidth) {
        return;
    }

	outBlockIdArray[index] = octree->nodes[octree->morton3Dencode(scve::Vector3<int>::add(scve::Vector3<int>(x, y, z), octree->getMinPos()))].blockId();
}

}