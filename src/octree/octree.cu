#include <bitset>
#include <cstddef>
#include <iostream>
#include <limits>
#include <cxxabi.h>
#include <string>
#include <cmath>

#include "octree.cuh"
#include "cuda_morton.cuh"
#include "globals.cuh"
#include "pixel_drawing.cuh"
#include "blocks_data.cuh"
#include "cuda_math.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

constexpr float epsilon = 0.00001;

namespace {
	inline size_t getNumOfBytesToAllocate(int maxLevel) {
		// sizeof(Node)
	}
}

void Octree::createOctree(int xMin_, int yMin_, int zMin_, int maxLevel_) {
	xMin = xMin_;
	yMin = yMin_;
	zMin = zMin_;
	maxLevel = maxLevel_;
	allocatedMemoryInBytes = getNumOfBytesToAllocate(maxLevel_);

	cudaMalloc(&nodes, allocatedMemoryInBytes);
}

void Octree::createOctree(size_t preallocatedMemoryInBytes) {
	createOctree(0, 0, 0, 1, preallocatedMemoryInBytes);
}

void Octree::clear() {
	cudaMemset(nodes, 0, PREALLOCATE_MB_AMOUNT * size_t(1024) * size_t(1024));
}

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

__device__ void morton3Ddecode(uint32_t mortonCode, int& x, int& y, int& z){
	const uint32_t mostSignificant1 = uint32_t(1) << 31;
	int index = 0;
	uint32_t code = mortonCode;

	while(code >>= 1){
		index++;
	}

	//printf("%d %llu %llu\n",index, mortonCode, code);

	mortonCode <<= (32 - index);

	x = xMin;
	y = yMin;
	z = zMin;
	
	int level = Octree::level;
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

}

__device__ unsigned char firstNode(float tx0, float ty0, float tz0, float txm, float tym, float tzm) {
	float maxV = maxv(maxv(tx0, ty0), tz0);

	unsigned char v = 0;

	if (equals(maxV,tx0, epsilon)) {
		if (tym < tx0)
			v |= 2;
		if (tzm < tx0)
			v |= 1;
		return v;
	}

	if (equals(maxV, ty0, epsilon)) {
		if (txm < ty0)
			v |= 4;
		if (tzm < ty0)
			v |= 1;
		return v;
	}

	if (txm < tz0)
		v |= 4;
	if (tym < tz0)
		v |= 2;
	return v;
}

__device__ unsigned char newNode(float tx, unsigned char i1, float ty, unsigned char i2, float tz, unsigned char i3) {
	float minV = minv(minv(tx, ty), tz);

	if (equals(minV, tx, epsilon)) {
		return i1;
	}
	if (equals(minV, ty, epsilon)) {
		return i2;
	}
	return i3;
}

__device__ uint32_t childMortonRevelles(uint32_t mortonCode, unsigned char revellesChildIndex){
	static unsigned char reversed[8] = {0, 4, 2, 6, 1, 5, 3, 7};

	return (mortonCode << 3) | reversed[revellesChildIndex];
}

__device__ void Octree::insert(Block block) {
	int x = block.x;
	int y = block.y;
	int z = block.z;

	int level = Octree::level;
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
			nodes[index].id = block.blockId;
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

		nodes[prevIndex].id = block.blockId | 128;

		level--;
		size = 1 << level;

	} while (level >= 0);
}

__device__ void performRaycast(Octree* octree, float oX, float oY, float oZ, float dX, float dY, float dZ, int sX, int sY, int minNodeSize, uchar4* pixels){
	unsigned char a = 0;

	float origOX = oX, origOY = oY, origOZ = oZ;
	float origDX = dX, origDY = dY, origDZ = dZ;

	int size = 1 << octree->level;

	if (dX < 0) {
		oX = -oX + (octree->xMin * 2 + size);// +dCenterX;
		dX = -dX;
		a |= 4;
	}
	if (dY < 0) {
		oY = -oY + (octree->yMin * 2 + size);// +dCenterY;
		dY = -dY;
		a |= 2;
	}
	if (dZ < 0) {
		oZ = -oZ + (octree->zMin * 2 + size);// +dCenterZ;
		dZ = -dZ;
		a |= 1;
	}

	float tx0 = (octree->xMin - oX) / dX;
	float tx1 = (octree->xMin + size - oX) / dX;
	float ty0 = (octree->yMin - oY) / dY;
	float ty1 = (octree->yMin + size - oY) / dY;
	float tz0 = (octree->zMin - oZ) / dZ;
	float tz1 = (octree->zMin + size - oZ) / dZ;

	if (maxv(maxv(tx0, ty0), tz0) < minv(minv(tx1, ty1), tz1)) {
        Stack stack;
        int foundNode = traverseNewNode(tx0, ty0, tz0, tx1, ty1, tz1, 1, minNodeSize, sX, sY, origOX, origOY, origOZ, origDX, origDY, origDZ, pixels, stack, octree);

		int index = 0;

        while (!stack.isEmpty() && foundNode == -1) {
            Stack::Frame* data = stack.top();

            foundNode = traverseChildNodes(data, a, minNodeSize, sX, sY, origOX, origOY, origOZ, origDX, origDY, origDZ, pixels, stack, octree);
        }

		if (foundNode <= -1) {
			setPixel(pixels, sX, sY, 30, 30, 30, 255); //30 30 255
		}
	}
}

__device__ int traverseChildNodes(Stack::Frame* data, unsigned char a, int minNodeSize, int sX, int sY, float origOX, float origOY, float origOZ, float origDX, float origDY, float origDZ, uchar4* pixels, Stack& stack, Octree* octree) {
	switch (data->nodeIndex) {
		case 0:
			data->nodeIndex = newNode(data->txm, 4, data->tym, 2, data->tzm, 1);
			return traverseNewNode(data->tx0, data->ty0, data->tz0, data->txm, data->tym, data->tzm, childMortonRevelles(data->mortonCode,     a), minNodeSize, sX, sY, origOX, origOY, origOZ, origDX, origDY, origDZ, pixels, stack, octree);
		case 1:
			data->nodeIndex = newNode(data->txm, 5, data->tym, 3, data->tz1, 8);
			return traverseNewNode(data->tx0, data->ty0, data->tzm, data->txm, data->tym, data->tz1, childMortonRevelles(data->mortonCode, 1 ^ a), minNodeSize, sX, sY, origOX, origOY, origOZ, origDX, origDY, origDZ, pixels, stack, octree);
		case 2:
			data->nodeIndex = newNode(data->txm, 6, data->ty1, 8, data->tzm, 3);
			return traverseNewNode(data->tx0, data->tym, data->tz0, data->txm, data->ty1, data->tzm, childMortonRevelles(data->mortonCode, 2 ^ a), minNodeSize, sX, sY, origOX, origOY, origOZ, origDX, origDY, origDZ, pixels, stack, octree);
		case 3:
			data->nodeIndex = newNode(data->txm, 7, data->ty1, 8, data->tz1, 8);
			return traverseNewNode(data->tx0, data->tym, data->tzm, data->txm, data->ty1, data->tz1, childMortonRevelles(data->mortonCode, 3 ^ a), minNodeSize, sX, sY, origOX, origOY, origOZ, origDX, origDY, origDZ, pixels, stack, octree);
		case 4:
			data->nodeIndex = newNode(data->tx1, 8, data->tym, 6, data->tzm, 5);
			return traverseNewNode(data->txm, data->ty0, data->tz0, data->tx1, data->tym, data->tzm, childMortonRevelles(data->mortonCode, 4 ^ a), minNodeSize, sX, sY, origOX, origOY, origOZ, origDX, origDY, origDZ, pixels, stack, octree);
		case 5:
			data->nodeIndex = newNode(data->tx1, 8, data->tym, 7, data->tz1, 8);
			return traverseNewNode(data->txm, data->ty0, data->tzm, data->tx1, data->tym, data->tz1, childMortonRevelles(data->mortonCode, 5 ^ a), minNodeSize, sX, sY, origOX, origOY, origOZ, origDX, origDY, origDZ, pixels, stack, octree);
		case 6:
			data->nodeIndex = newNode(data->tx1, 8, data->ty1, 8, data->tzm, 7);
			return traverseNewNode(data->txm, data->tym, data->tz0, data->tx1, data->ty1, data->tzm, childMortonRevelles(data->mortonCode, 6 ^ a), minNodeSize, sX, sY, origOX, origOY, origOZ, origDX, origDY, origDZ, pixels, stack, octree);
		case 7:
			data->nodeIndex = 8;
			return traverseNewNode(data->txm, data->tym, data->tzm, data->tx1, data->ty1, data->tz1, childMortonRevelles(data->mortonCode, 7 ^ a), minNodeSize, sX, sY, origOX, origOY, origOZ, origDX, origDY, origDZ, pixels, stack, octree);
		case 8:
			stack.pop();
			return -1;
	}
	
	return -1;
}

__device__ int traverseNewNode(float tx0, float ty0, float tz0, float&tx1, float ty1, float tz1, unsigned int nodeIdx, int minNodeSize, int sX, int sY, float origOX, float origOY, float origOZ, float origDX, float origDY, float origDZ, uchar4* pixels, Stack& stack, Octree* octree) {
        
	if(stack.topIndex >= CUDA_STACK_SIZE - 1) return -1;

	if (nodeLevel(nodeIdx, octree->level) == 0 && octree->nodes[nodeIdx].blockId() != 0) {
		int blockX, blockY, blockZ;

		octree->morton3Ddecode(nodeIdx, blockX, blockY, blockZ);

		drawTexturePixel(blockX, blockY, blockZ, origOX, origOY, origOZ, origDX, origDY, origDZ, sX, sY, octree->nodes[nodeIdx].blockId(), pixels, octree->textureRenderingEnabled);

		return octree->nodes[nodeIdx].blockId();
	}

	if (!octree->nodes[nodeIdx].hasChildren() || tx1 < 0.0f || ty1 < 0.0f || tz1 < 0.0f) return -1;

	const float txm = 0.5f * (tx0 + tx1);
	const float tym = 0.5f * (ty0 + ty1);
	const float tzm = 0.5f * (tz0 + tz1);

	stack.push({
		tx0, ty0, tz0,
		txm, tym, tzm,
		tx1, ty1, tz1,
		nodeIdx,
		firstNode(tx0, ty0, tz0, txm, tym, tzm),
	});
	
	return -1;
}