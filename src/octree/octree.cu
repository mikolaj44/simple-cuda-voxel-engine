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
#include "cuda_renderer.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

constexpr float epsilon = 0.00001;

namespace {
	inline size_t getNumOfBytesToAllocate(unsigned int maxLevel) {
		return size_t(2) << (3 * maxLevel + 1);
	}
}

cudaError_t Octree::createOctree(int xMin_, int yMin_, int zMin_, unsigned int maxLevel_) {
	xMin = xMin_;
	yMin = yMin_;
	zMin = zMin_;
	maxLevel = maxLevel_;
	allocatedMemoryInBytes = getNumOfBytesToAllocate(maxLevel_);

	size_t freeBytes, totalBytes;
	cudaMemGetInfo(&freeBytes, &totalBytes);

	printf("%zu bytes free out of %zu\n", freeBytes, totalBytes);

	printf("allocating %zu bytes (%d levels)\n", allocatedMemoryInBytes, maxLevel);

	cudaError_t error = cudaMalloc(&nodes, allocatedMemoryInBytes);

	cudaMemGetInfo(&freeBytes, &totalBytes);

	printf("%zu bytes free out of %zu\n", freeBytes, totalBytes);

	return error;
}

cudaError_t Octree::createOctree(unsigned int maxLevel) {
	return createOctree(0, 0, 0, maxLevel);
}

cudaError_t Octree::clear() {
	return cudaMemset(nodes, 0, allocatedMemoryInBytes);
}

namespace {
	__device__ void getChildXYZindex(int& x, int& y, int& z, uint32_t& index, unsigned int level, unsigned int childIndex) {
		int size = 1 << level;
		index <<= 3;

		switch (childIndex) {
			case 0:
				break;
			case 1:
				z += size / 2;
				index |= (1 << 2);
				break;
			case 2:
				y += size / 2;
				index |= (1 << 1);
				break;
			case 3:
				y += size / 2;
				z += size / 2;
				index |= (1 << 1);
				index |= (1 << 2);
				break;
			case 4:
				x += size / 2;
				index |= 1;
				break;
			case 5:
				x += size / 2;
				z += size / 2;
				index |= 1;
				index |= (1 << 2);
				break;
			case 6:
				x += size / 2;
				y += size / 2;
				index |= 1;
				index |= (1 << 1);
				break;
			case 7:
				x += size / 2;
				y += size / 2;
				z += size / 2;
				index |= (1 << 1);
				index |= (1 << 2);
				index |= 1;
				break;
			default:
				break;
		}
	}
}

__device__ Vector3<int> Octree::morton3Ddecode(uint32_t mortonCode){
	static const uint32_t mostSignificant1 = uint32_t(1) << 31;

	int index = 0;
	uint32_t code = mortonCode;

	while(code >>= 1){
		index++;
	}

	mortonCode <<= (32 - index);

	int x = xMin;
	int y = yMin;
	int z = zMin;
	
	int level = Octree::maxLevel;
	int size;

	while(index > 0){
		size = 1 << level;
		
		if(mortonCode & mostSignificant1){
			z += size / 2;
		}
		if(mortonCode & (mostSignificant1 >> 1)){
			y += size / 2;
		}
		if(mortonCode & (mostSignificant1 >> 2)){
			x += size / 2;
		}

		mortonCode <<= 3;
		index -= 3;

		level--;
	}

	return Vector3<int>(x, y, z);
}

__device__ void Octree::insert(BlockInfo<>& block) {
	int x = block.pos.x;
	int y = block.pos.y;
	int z = block.pos.z;

	int level = Octree::maxLevel;
	int size = 1 << level;

	// Octree coordinate system is positive only, convert the coordinates to this system
	x -= Octree::xMin;
	y -= Octree::yMin;
	z -= Octree::zMin;

	int xMin = 0;
	int yMin = 0;
	int zMin = 0;

	int xM, yM, zM;

	// If the voxel is out of bounds (we don't grow the octree)
	if(x < 0 || y < 0 || z < 0 || x >= size || y >= size || z >= size){
		return;
	}

	uint32_t index = 1; // root node index
	uint32_t prevIndex = 1;

	// Iterate over all node levels up until the leaf node
	do {
		// Get the node at index (to insert the right block data)
		if (level == 0) {
			nodes[index].id = block.id;
			return;
		}

		prevIndex = index;

		// Get the midpoint
		int xM = (2 * xMin + size) / 2;
		int yM = (2 * yMin + size) / 2;
		int zM = (2 * zMin + size) / 2;

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

		nodes[prevIndex].id = block.id | 128;

		level--;
		size = 1 << level;

	} while (level >= 0);
}

__device__ void Octree::traverseNewNode(bool& foundSolid, Triple<Vector3<int>, Vector3<>, uint8_t>& intersectionData, octree_utils::Stack& stack, uchar4* pixels, Vector3<> origRayOrigin, Vector3<> origRayDirection, float tx0, float ty0, float tz0, float tx1, float ty1, float tz1, unsigned int nodeIdx, int minNodeSize, int sX, int sY) {
	using namespace octree_utils::revelles;
	using namespace octree_utils;
	
	if(stack.topIndex >= CUDA_STACK_SIZE - 1) {
		return;
	}

	if (nodeLevel(nodeIdx, maxLevel) == 0 && nodes[nodeIdx].blockId() != 0) {
		Vector3<int> blockPos = morton3Ddecode(nodeIdx);

		Vector3<> hitPos = getBlockHitPos(blockPos, origRayOrigin, origRayDirection);

		intersectionData.first = blockPos;
		intersectionData.second = hitPos;
		intersectionData.third = nodes[nodeIdx].blockId();
		
		// TODO: return multiplecheck if the block is not translucent before setting this to true:
		foundSolid = true;
		return;
	}

	if (!nodes[nodeIdx].hasChildren() || tx1 < 0.0f || ty1 < 0.0f || tz1 < 0.0f){
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

	int size = 1 << maxLevel;

	if (rayDirection.x < 0) {
		rayOrigin.x = -rayOrigin.x + (xMin * 2 + size);
		rayDirection.x = -rayDirection.x;
		a |= 4;
	}
	if (rayDirection.y < 0) {
		rayOrigin.y = -rayOrigin.y + (yMin * 2 + size);
		rayDirection.y = -rayDirection.y;
		a |= 2;
	}
	if (rayDirection.z < 0) {
		rayOrigin.z = -rayOrigin.z + (zMin * 2 + size);
		rayDirection.z = -rayDirection.z;
		a |= 1;
	}

	float tx0 = (xMin - rayOrigin.x) / rayDirection.x;
	float tx1 = (xMin + size - rayOrigin.x) / rayDirection.x;
	float ty0 = (yMin - rayOrigin.y) / rayDirection.y;
	float ty1 = (yMin + size - rayOrigin.y) / rayDirection.y;
	float tz0 = (zMin - rayOrigin.z) / rayDirection.z;
	float tz1 = (zMin + size - rayOrigin.z) / rayDirection.z;
	
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

void Octree::setMinPos(Vector3<> minPos) {
	xMin = minPos.x;
	yMin = minPos.y;
	zMin = minPos.z;
}

void Octree::setMaxLevel(unsigned int maxLevel_) {
	maxLevel = maxLevel_;
}

Vector3<> Octree::getMinPos() const {
	return Vector3<>(xMin, yMin, zMin);
};

unsigned int Octree::getMaxLevel() const {
	return maxLevel;
}