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
	cudaError_t create(int xMin, int yMin, int zMin, unsigned int maxLevel);

	cudaError_t create(unsigned int maxLevel);

	cudaError_t cleanup();

	cudaError_t clear();

	__device__ void insert(BlockInfo<>& block);

	__device__ Triple<Vector3<int>, Vector3<float>, uint8_t> getRayIntersectionData(uchar4* pixels, Vector3<> rayOrigin, Vector3<> rayDirection, int sX, int sY, int minNodeSize);

	void setMinPos(Vector3<> minPos);

	void setMaxLevel(unsigned int maxLevel);

	Vector3<> getMinPos() const;

	unsigned int getMaxLevel() const;
private:
	struct alignas(uint8_t) Node {
		uint8_t id; // most significant bit is 1 if the node has children, the rest of the bits are for block id

		__device__ Node() {};

		__device__ Node(uint8_t id_) : id(id_) {};

		__device__ inline bool hasChildren() const {
			return id & 128;
		}

		__device__ inline unsigned char blockId() const {
			return id & 127;
		}
	};

	Node* nodes;

	int xMin, yMin, zMin;

	unsigned int maxLevel; // level 0 is a terminal node

	size_t allocatedMemoryInBytes;

	__device__ Vector3<int> morton3Ddecode(uint32_t mortonCode);

	__device__ void traverseNewNode(bool& foundSolid, Triple<Vector3<int>, Vector3<float>, uint8_t>& intersectionData, octree_utils::Stack& stack, uchar4* pixels, Vector3<> origRayOrigin, Vector3<> origRayDirection, float tx0, float ty0, float tz0, float tx1, float ty1, float tz1, unsigned int nodeIdx, int minNodeSize, int sX, int sY);

	__device__ void traverseChildNodes(bool& foundSolid, Triple<Vector3<int>, Vector3<float>, uint8_t>& intersectionData, octree_utils::Stack& stack, octree_utils::Stack::Frame& data, uchar4* pixels, Vector3<> origRayOrigin, Vector3<> origRayDirection, unsigned char a, int minNodeSize, int sX, int sY);
};