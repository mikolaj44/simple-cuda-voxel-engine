#include <bitset>
#include <cstddef>
#include <iostream>
#include <limits>
#include <cxxabi.h>
#include <string>
#include <cmath>

#include "octree/octree.cuh"
#include "octree/octree_utils.cuh"
#include "globals.cuh"
#include "blocks_data.cuh"
#include "cuda_math.cuh"
#include "renderer/cuda_renderer.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

constexpr float epsilon = 0.00001;

namespace {
	size_t getAllocatedMemoryInBytes(unsigned int maxLevel) {
		return size_t(2) << (3 * maxLevel + 1);
	}
}

__host__  cudaError_t Octree::allocateByMaxLevel(unsigned int newMaxLevel) {
	allocatedMemoryInBytes = getAllocatedMemoryInBytes(newMaxLevel);
	maxLevel = newMaxLevel;
	maxSize = 1 << maxLevel;

	// TODO: copy from the old node array

	cudaFree(nodes);
	cudaError_t error = cudaMalloc(&nodes, allocatedMemoryInBytes);

	if(error != cudaSuccess) {
		return error;
	}

	size_t freeBytes, totalBytes;
	cudaMemGetInfo(&freeBytes, &totalBytes);

	printf("\n%zu bytes free out of %zu\n", freeBytes, totalBytes);

	printf("allocating %zu bytes (%d levels)\n", allocatedMemoryInBytes, maxLevel);

	cudaMemGetInfo(&freeBytes, &totalBytes);

	printf("%zu bytes free out of %zu\n", freeBytes, totalBytes);

	return cudaSuccess;
}

__host__ cudaError_t Octree::create(int xMin_, int yMin_, int zMin_, unsigned int maxLevel_) {
	xMin = xMin_;
	yMin = yMin_;
	zMin = zMin_;
	return allocateByMaxLevel(maxLevel_);
}

__host__ cudaError_t Octree::create(unsigned int maxLevel) {
	return create(0, 0, 0, maxLevel);
}

__host__ cudaError_t Octree::cleanup() {
	return cudaFree(nodes);
}

__host__ cudaError_t Octree::clear() {
	return cudaMemset(nodes, 0, allocatedMemoryInBytes);
}

__device__ Vector3<int> Octree::morton3Ddecode(uint32_t mortonCode) {
	static const uint32_t mostSignificant1 = uint32_t(1) << 31;

	int index = 0;
	uint32_t code = mortonCode;

	// 0000101001

	while(code >>= 1) {
		index++;
	}

	mortonCode <<= (32 - index);

	int x = xMin;
	int y = yMin;
	int z = zMin;
	
	int size = maxSize;

	while(size >= 1) {
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

	return Vector3<int>(x, y, z);
}

// TODO: change this to a more efficient method, at least "magic bits": https://forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/
__device__ uint32_t Octree::morton3Dencode(Vector3<int> pos) {
	uint32_t mortonCode = 1;

	int x = pos.x;
	int y = pos.y;
	int z = pos.z;

	int xMinCopy = xMin;
	int yMinCopy = yMin;
	int zMinCopy = zMin;
	
	int size = maxSize;

	while(size >= 1) {
		mortonCode <<= 3;

		if(x < xMinCopy + size / 2) {
			mortonCode |= 0b001;
		}
		else {
			xMinCopy += size / 2;
		}

		if(y < yMinCopy + size / 2) {
			mortonCode |= 0b010;
		}
		else {
			yMinCopy += size / 2;
		}

		if(z < zMinCopy + size / 2) {
			mortonCode |= 0b100;
		}
		else {
			zMinCopy += size / 2;
		}

		size /= 2;
	}

	return mortonCode;
}

// First part of the insertion (top-down): inserting the correct leaf-voxel data
__device__ void Octree::insert(BlockInfo<>& block) {
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
	// uint32_t prevIndex = 1;

	// Iterate over all node levels up until the leaf node
	do {
		// Get the node at index (to insert the right block data)
		if (size == 1) {
			nodes[index].id = block.id;
			return;
		}

		nodes[index].id = 128;

		// prevIndex = index;

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

		// nodes[prevIndex].id = 128;

		size /= 2;

	} while (size >= 1);
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

__device__ void Octree::traverseNewNode(bool& foundSolid, Triple<Vector3<int>, Vector3<>, uint8_t>& intersectionData, octree_utils::Stack& stack, uchar4* pixels, Vector3<> origRayOrigin, Vector3<> origRayDirection, float tx0, float ty0, float tz0, float tx1, float ty1, float tz1, unsigned int nodeIdx, int minNodeSize, int sX, int sY) {
	using namespace octree_utils::revelles;
	using namespace octree_utils;
	
	// TODO: calculate CUDA_STACK_SIZE when maxSize is calculated
	if(stack.topIndex >= CUDA_STACK_SIZE - 1) {
		return;
	}

	if (!nodes[nodeIdx].isNotSolid() && nodes[nodeIdx].blockId() != 0) { // nodeLevel(nodeIdx, maxLevel) == 0 && nodes[nodeIdx].blockId() != 0
		// intersectionData.first = morton3Ddecode(nodeIdx);
		// intersectionData.second = getBlockHitPos(intersectionData.first, origRayOrigin, origRayDirection);
		
		intersectionData.third = nodes[nodeIdx].blockId();
		
		// TODO: return multiplecheck if the block is not translucent before setting this to true:
		foundSolid = true;
		return;
	}

	// (!nodes[nodeIdx].isNotSolid() && nodes[nodeIdx].blockId() == 0)
	if (tx1 < 0.0f || ty1 < 0.0f || tz1 < 0.0f) { // !nodes[nodeIdx].isNotSolid()
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

__device__ void Octree::traverseChildNodes(bool& foundSolid, Triple<Vector3<int>, Vector3<>, uint8_t>& intersectionData, octree_utils::Stack& stack, octree_utils::Stack::Frame& data, uchar4* pixels, Vector3<> origRayOrigin, Vector3<> origRayDirection, unsigned char a, int minNodeSize, int sX, int sY) {
	using namespace octree_utils::revelles;
	
	switch (data.nodeIndex) {
		case 0:
			data.nodeIndex = newNode(data.txm, 4, data.tym, 2, data.tzm, 1, epsilon);
			return traverseNewNode(foundSolid, intersectionData, stack, pixels, origRayOrigin, origRayDirection, data.tx0, data.ty0, data.tz0, data.txm, data.tym, data.tzm, childMortonRevelles(data.mortonCode,     a), minNodeSize, sX, sY);
		case 1:
			data.nodeIndex = newNode(data.txm, 5, data.tym, 3, data.tz1, 8, epsilon);
			return traverseNewNode(foundSolid, intersectionData, stack, pixels, origRayOrigin, origRayDirection, data.tx0, data.ty0, data.tzm, data.txm, data.tym, data.tz1, childMortonRevelles(data.mortonCode, 1 ^ a), minNodeSize, sX, sY);
		case 2:
			data.nodeIndex = newNode(data.txm, 6, data.ty1, 8, data.tzm, 3, epsilon);
			return traverseNewNode(foundSolid, intersectionData, stack, pixels, origRayOrigin, origRayDirection, data.tx0, data.tym, data.tz0, data.txm, data.ty1, data.tzm, childMortonRevelles(data.mortonCode, 2 ^ a), minNodeSize, sX, sY);
		case 3:
			data.nodeIndex = newNode(data.txm, 7, data.ty1, 8, data.tz1, 8, epsilon);
			return traverseNewNode(foundSolid, intersectionData, stack, pixels, origRayOrigin, origRayDirection, data.tx0, data.tym, data.tzm, data.txm, data.ty1, data.tz1, childMortonRevelles(data.mortonCode, 3 ^ a), minNodeSize, sX, sY);
		case 4:
			data.nodeIndex = newNode(data.tx1, 8, data.tym, 6, data.tzm, 5, epsilon);
			return traverseNewNode(foundSolid, intersectionData, stack, pixels, origRayOrigin, origRayDirection, data.txm, data.ty0, data.tz0, data.tx1, data.tym, data.tzm, childMortonRevelles(data.mortonCode, 4 ^ a), minNodeSize, sX, sY);
		case 5:
			data.nodeIndex = newNode(data.tx1, 8, data.tym, 7, data.tz1, 8, epsilon);
			return traverseNewNode(foundSolid, intersectionData, stack, pixels, origRayOrigin, origRayDirection, data.txm, data.ty0, data.tzm, data.tx1, data.tym, data.tz1, childMortonRevelles(data.mortonCode, 5 ^ a), minNodeSize, sX, sY);
		case 6:
			data.nodeIndex = newNode(data.tx1, 8, data.ty1, 8, data.tzm, 7, epsilon);
			return traverseNewNode(foundSolid, intersectionData, stack, pixels, origRayOrigin, origRayDirection, data.txm, data.tym, data.tz0, data.tx1, data.ty1, data.tzm, childMortonRevelles(data.mortonCode, 6 ^ a), minNodeSize, sX, sY);
		case 7:
			data.nodeIndex = 8;
			return traverseNewNode(foundSolid, intersectionData, stack, pixels, origRayOrigin, origRayDirection, data.txm, data.tym, data.tzm, data.tx1, data.ty1, data.tz1, childMortonRevelles(data.mortonCode, 7 ^ a), minNodeSize, sX, sY);
		case 8:
			stack.pop();
	}
}

__device__ Triple<Vector3<int>, Vector3<>, uint8_t> Octree::getRayIntersectionData(uchar4* pixels, Vector3<> rayOrigin, Vector3<> rayDirection, int sX, int sY, int minNodeSize) {
	unsigned char a = 0;

	Vector3<> origRayOrigin = rayOrigin;
	Vector3<> origRayDirection = rayDirection;

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
	
	Triple<Vector3<int>, Vector3<>, uint8_t> intersectionData;

	if (maxv(maxv(tx0, ty0), tz0) < minv(minv(tx1, ty1), tz1)) {
		octree_utils::Stack stack;
		bool foundSolid = false;

		traverseNewNode(foundSolid, intersectionData, stack, pixels, origRayOrigin, origRayDirection, tx0, ty0, tz0, tx1, ty1, tz1, 1, minNodeSize, sX, sY);

		while (!stack.isEmpty() && !foundSolid) {
			traverseChildNodes(foundSolid, intersectionData, stack, stack.top(), pixels, origRayOrigin, origRayDirection, a, minNodeSize, sX, sY);
		}
	}

	return intersectionData;
}

__device__ __host__ void Octree::setMinPos(Vector3<> minPos) {
	xMin = minPos.x;
	yMin = minPos.y;
	zMin = minPos.z;
}

__host__ cudaError_t Octree::setMaxLevel(unsigned int maxLevel_) {
	return allocateByMaxLevel(maxLevel_);
}

__device__ __host__ Vector3<> Octree::getMinPos() const {
	return Vector3<>(xMin, yMin, zMin);
};

__device__ __host__ unsigned int Octree::getMaxLevel() const {
	return maxLevel;
}

__device__ __host__ unsigned int Octree::getMaxSize() const {
	return maxSize;
}