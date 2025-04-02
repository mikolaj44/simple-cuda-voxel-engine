#pragma once
#include <string>
#include "globals.cuh"
#include "pixel_drawing.cuh"
#include "blocks_data.cuh"
#include "cuda_math.cuh"
#include <cmath>
#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

class Block {
public:
	int x, y, z;
	uint8_t blockId;

	__device__ __host__ Block() {};

	__device__ __host__ Block(int x_, int y_, int z_, uint8_t blockId_) : x(x_), y(y_), z(z_), blockId(blockId_) {};
};

class Octree {

public:

	enum class OctreeSpecialPosition { CENTERED };

	// nodeId : most significant bit is 1 if the node has children, the rest of the bits are for block id
	uint8_t* nodes;

	int xMin, yMin, zMin;
	unsigned int level; // level 0 is a terminal node

	// the root node is stored at the first position

	void createOctree(int xMin, int yMin, int zMin, unsigned int level);

	void createOctree(OctreeSpecialPosition position, unsigned int level);

	void createOctree();

	__device__ void insert(Block block);

	unsigned char get(int x, int y, int z);

	void clear();

	void display(uchar4* pixels, bool showBorder = true);

	__device__ void morton3Ddecode(uint32_t mortonCode, int&x, int& y, int& z);

	static void getChildXYZindex(int& x, int& y, int& z, uint32_t& index, unsigned int level, unsigned int childIndex);
	
private:

	void display(uchar4* pixels, uint32_t index = 1, bool showBorder = true, int x = 0, int y = 0, int z = 0, unsigned int level = 0);
};

// __device__ __host__ inline unsigned int nodeLevel(uint32_t mortonCode, unsigned int octreeLevel){
// 	#ifdef CUDA_ARCH
// 		return minv(octreeLevel - log2f(mortonCode) / 3, 1);
// 	#else
// 		return std::min(int(octreeLevel - log2f(mortonCode) / 3), 1);
// 	#endif
// }

__device__ __host__ inline unsigned int nodeLevel(uint32_t mortonCode, unsigned int octreeLevel){

	int depth = 0;

	for (uint32_t code = mortonCode; code != 1; code >>= 3, depth++);

    return octreeLevel - depth;
}

__device__ __host__ inline unsigned int nodeSize(uint32_t mortonCode, unsigned int octreeLevel){
	return 1 << nodeLevel(mortonCode, octreeLevel);
}

__device__ int firstNode(float tx0, float ty0, float tz0, float txm, float tym, float tzm);

__device__ int newNode(float tx, int i1, float ty, int i2, float tz, int i3);

__device__ uint32_t childMortonRevelles(uint32_t mortonCode, unsigned char revellesChildIndex);

__device__ void performRaycast(Octree* octree, float oX, float oY, float oZ, float dX, float dY, float dZ, int sX, int sY, int minNodeSize = 1, uchar4*  pixels = nullptr);

struct frame {
	float tx0, ty0, tz0, tx1, ty1, tz1, txm, tym, tzm; unsigned char nodeIndex; uint32_t mortonCode;
};

__device__ void drawTexturePixel(int blockX, int blockY, int blockZ, float oX, float oY, float oZ, float dX, float dY, float dZ, int sX, int sY, unsigned char blockId, uchar4* pixels, bool negativeDX, bool negativeDY, bool negativeDZ);

__device__ unsigned char raycastDrawPixel(Octree* octree, float oX, float oY, float oZ, float dX, float dY, float dZ, float tx0, float ty0, float tz0, float tx1, float ty1, float tz1, unsigned char a, int minNodeSize, int sX, int sY, uchar4* pixels, float origOX, float origOY, float origOZ, bool negativeDX, bool negativeDY, bool negativeDZ);