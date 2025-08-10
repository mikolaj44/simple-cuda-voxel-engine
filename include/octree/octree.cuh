#pragma once

#include <stdint.h>

#include "vector3.cuh"
#include "octree/octree_utils.cuh"

#include "cuda_math.cuh"
#include "octree/octree_utils.cuh"

class Block {
public:
	int x, y, z;
	uint8_t blockId;

	__device__ __host__ Block() {};

	__device__ __host__ Block(int x_, int y_, int z_, uint8_t blockId_) : x(x_), y(y_), z(z_), blockId(blockId_) {};
};

class Octree {
public:
	void createOctree(int xMin, int yMin, int zMin, unsigned int maxLevel);

	void createOctree(unsigned int maxLevel);

	void clear();

	__device__ void insert(Block block);

	__device__ int8_t getRayIntersectionData(uchar4* pixels, Vector3 rayOrigin, Vector3 rayDirection, int sX, int sY, int minNodeSize);

	void setMinPos(Vector3 minPos);

	void setMaxLevel(unsigned int maxLevel);

	Vector3 getMinPos() const;

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

	__device__ void morton3Ddecode(uint32_t mortonCode, int& x, int& y, int& z);

	__device__ int8_t traverseNewNode(octree_utils::Stack& stack, uchar4* pixels, Vector3 origRayOrigin, Vector3 origRayDirection, float tx0, float ty0, float tz0, float tx1, float ty1, float tz1, unsigned int nodeIdx, int minNodeSize, int sX, int sY);

	__device__ int8_t traverseChildNodes(octree_utils::Stack& stack, octree_utils::Stack::Frame& data, uchar4* pixels, Vector3 origRayOrigin, Vector3 origRayDirection, unsigned char a, int minNodeSize, int sX, int sY);
};