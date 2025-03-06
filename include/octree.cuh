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

#include <cuco/static_map.cuh>
#include <thrust/device_vector.h>

#define map_type cuco::static_map<uint64_t, Node, cuco::extent<unsigned long, NODE_MAP_CAPACITY>, (cuda::std::__4::thread_scope)1, cuda::std::__4::equal_to<uint64_t>, cuco::linear_probing<1, cuco::detail::XXHash_64<uint64_t> >, cuco::cuda_allocator<cuco::pair<uint64_t, Node> >, cuco::storage<1> >

using namespace std;

class Block {
public:
	int x, y, z;
	unsigned char blockId;

	__device__ __host__ Block() {};

	__device__ __host__ Block(int x_, int y_, int z_, unsigned char blockId_) : x(x_), y(y_), z(z_), blockId(blockId_) {};
};

class Node {

public:
	unsigned char blockId = 0; // air
	bool childMask = 0; // which children are present, indexed according to Figure 1: http://wscg.zcu.cz/wscg2000/Papers_2000/X31.pdf

	__host__ __device__ Node() {};

	__host__ __device__ Node(unsigned char blockId_, bool childMask_) : blockId(blockId_), childMask(childMask_) {};

	//__host__ __device__ Node(unsigned char blockId_, bool hasChildren_, uint64_t mortonCode_) : blockId(blockId_), hasChildren(hasChildren_), mortonCode(mortonCode_) {};
private:
	uint8_t padding[2]; // 2 bytes padding
};

class Octree {

public:

	enum class OctreeSpecialPosition { CENTERED };

	map_type nodeMap;

	int xMin, yMin, zMin;
	unsigned int level; // level 0 is a terminal node

	// the root node is stored at the first position

	void createOctree(int xMin, int yMin, int zMin, unsigned int level);

	void createOctree(OctreeSpecialPosition position, unsigned int level);

	void createOctree();

	unsigned char get(int x, int y, int z);

	void clear();

	void display(unsigned char* pixels, bool showBorder = true);

	__device__ void morton3Ddecode(uint64_t mortonCode, int&x, int& y, int& z);

	static void getChildXYZindex(int& x, int& y, int& z, uint64_t& index, unsigned int level, unsigned int childIndex);
	
private:

	void display(unsigned char* pixels, uint64_t index = 1, bool showBorder = true, int x = 0, int y = 0, int z = 0, unsigned int level = 0);

	void subdivide(Node* node); // Will be used later for the SVO

	void grow(int x, int y, int z);
};

// __device__ __host__ inline unsigned int nodeLevel(uint64_t mortonCode, unsigned int octreeLevel){
// 	#ifdef CUDA_ARCH
// 		return minv(octreeLevel - log2f(mortonCode) / 3, 1);
// 	#else
// 		return std::min(int(octreeLevel - log2f(mortonCode) / 3), 1);
// 	#endif
// }

__device__ __host__ inline unsigned int nodeLevel(uint64_t mortonCode, unsigned int octreeLevel){

	int depth = 0;

	for (uint64_t code = mortonCode; code != 1; code >>= 3, depth++);

    return octreeLevel - depth;
}

__device__ __host__ inline unsigned int nodeSize(uint64_t mortonCode, unsigned int octreeLevel){
	return 1 << nodeLevel(mortonCode, octreeLevel);
}

__device__ int firstNode(float tx0, float ty0, float tz0, float txm, float tym, float tzm);

__device__ int newNode(float tx, int i1, float ty, int i2, float tz, int i3);

__device__ uint64_t childMortonRevelles(uint64_t mortonCode, unsigned char revellesChildIndex);

__device__ void performRaycast(Octree* octree, float oX, float oY, float oZ, float dX, float dY, float dZ, int sX, int sY, int minNodeSize = 1, unsigned char* pixels = nullptr);

__device__ unsigned char raycastDrawPixel(float oX, float oY, float oZ, float dX, float dY, float dZ, float tx0, float ty0, float tz0, float tx1, float ty1, float tz1, unsigned char a, int minNodeSize, int sX, int sY, unsigned char* pixels, float origOX, float origOY, float origOZ, bool negativeDX, bool negativeDY, bool negativeDZ);

// the actual device insert function
template<typename MapInsertRef>
__device__ void insert(Octree* octree, MapInsertRef insertRef, Block block) {

	int x = block.x;
	int y = block.y;
	int z = block.z;

	//printf("%d %d %d\n", x, y, z);

	int level = octree->level;
	int size = 1 << level;

	// Octree coordinate system is positive only, convert the coordinates to this system
	x -= octree->xMin;
	y -= octree->yMin;
	z -= octree->zMin;

	int xMin = 0;
	int yMin = 0;
	int zMin = 0;

	int xM, yM, zM;
	unsigned char childMask;

	//printf("%d\n", x, y, z, xMin + size, yMin + size, zMin + size);

	// If the voxel is out of bounds (we don't grow the octree)
	if(x < 0 || y < 0 || z < 0 || x >= size || y >= size || z >= size){
		return;
	}

	uint64_t index = 1; // root node index
	int numShifts = 0;

	// Iterate over all node levels up until the leaf node
	do{
		// if(level == 17){
		// 	printf("%d %llu\n", level, index);
		// }

		//cout << level << " " << bitset<64>(index) << endl;

		if (level == 0) {

			// Get the node at index (to insert the right block data)
			// auto iterator = nodeMapRef.find(index);
			// iterator->second

			insertRef.insert(cuco::pair{index, Node{block.blockId, 0}});
			return;
		}

		// if(numShifts >= 21){ // Detect index overflow
		// 	return;
		// }

		childMask = 0;

		// Get the midpoint
		int xM = (2 * xMin + size) / 2;
		int yM = (2 * yMin + size) / 2;
		int zM = (2 * zMin + size) / 2;

		numShifts += 1;
		index <<= 3;

		// Compute the coordinates and morton code of the child node
		if (x >= xM) {
			xMin += size / 2;
			index |= 1;
			childMask |= 4;
		}
		
		if (y >= yM) {
			yMin += size / 2;
			index |= 2;
			childMask |= 2;
		}

		if (z >= zM) {	
			zMin += size / 2;
			index |= 4;
			childMask |= 1;
		}

		insertRef.insert(cuco::pair{index, Node{block.blockId, 1}});

		level--;
		size = 1 << level;

	} while (level >= 0);
}

template<typename MapInsertRef, typename MapFindRef>
__device__ void performRaycast(Octree* octree, MapInsertRef insertRef, MapFindRef findRef, float oX, float oY, float oZ, float dX, float dY, float dZ, int sX, int sY, int minNodeSize, unsigned char* pixels){

	unsigned char a = 0;

	bool negativeDX = false, negativeDY = false, negativeDZ = false;
	float origOX = oX, origOY = oY, origOZ = oZ;

	int size = 1 << octree->level;

	if (dX < 0) {
		//origin.x *= -1; //abs(octree->(root->xMin + root->size) - octree->root->xMin) / 2 - origin.x;
		//origin.x += abs(octree->(root->zMin + root->size) - octree->root->zMin) / 4; // abs(octree->(root->xMin + root->size) - octree->root->xMin);
		oX = -oX + (octree->xMin * 2 + size);// +dCenterX;
		dX = -dX;
		a |= 4;
		negativeDX = true;
	}
	if (dY < 0) {
		//origin.y *= -1;
		//origin.y += abs(octree->(root->zMin + root->size) - octree->root->zMin) / 4; //shift; //abs(octree->(root->yMin + root->size) - octree->root->yMin);
		oY = -oY + (octree->yMin * 2 + size);// +dCenterY;
		dY = -dY;
		a |= 2;
		negativeDY = true;
	}
	if (dZ < 0) {
		//origin.z *= -1; //abs(octree->(root->zMin + root->size) - octree->root->zMin) / 2- origin.z;
		//origin.z += abs(octree->(root->zMin + root->size) - octree->root->zMin) / 4; // abs(octree->(root->zMin + root->size) - octree->root->zMin);
		oZ = -oZ + (octree->zMin * 2 + size);// +dCenterZ;
		dZ = -dZ;
		a |= 1;
		negativeDZ = true;
	}

	float tx0 = (octree->xMin - oX) / dX;
	float tx1 = (octree->xMin + size - oX) / dX;
	float ty0 = (octree->yMin - oY) / dY;
	float ty1 = (octree->yMin + size - oY) / dY;
	float tz0 = (octree->zMin - oZ) / dZ;
	float tz1 = (octree->zMin + size - oZ) / dZ;

	//int color[3] = { 0,0,0 };

	//printf("\n-2");

	if (maxv(maxv(tx0, ty0), tz0) < minv(minv(tx1, ty1), tz1)) {

		//printf("%f %f %f\n",tx0, ty0, tz0);

		unsigned char result = raycastDrawPixel(octree, findRef, oX, oY, oZ, dX, dY, dZ, tx0, ty0, tz0, tx1, ty1, tz1, a, minNodeSize, sX, sY, pixels, origOX, origOY, origOZ, negativeDX, negativeDY, negativeDZ);

		if (result == 0) {
			setPixel(pixels, sX, sY, 30, 30, 30, 255); //30 30 255
		}

	}
}

struct frame {
	float tx0, ty0, tz0, tx1, ty1, tz1, txm, tym, tzm; unsigned char nodeIndex; uint64_t mortonCode;
};

__device__ void drawTexturePixel(int blockX, int blockY, int blockZ, float oX, float oY, float oZ, float dX, float dY, float dZ, int sX, int sY, unsigned char blockId, unsigned char* pixels, bool negativeDX, bool negativeDY, bool negativeDZ);

template<typename MapFindRef>
__device__ unsigned char raycastDrawPixel(Octree* octree, MapFindRef findRef, float oX, float oY, float oZ, float dX, float dY, float dZ, float tx0, float ty0, float tz0, float tx1, float ty1, float tz1, unsigned char a, int minNodeSize, int sX, int sY, unsigned char* pixels, float origOX, float origOY, float origOZ, bool negativeDX, bool negativeDY, bool negativeDZ) {

	unsigned int MAX_THREAD_STACK_SIZE = octree->level + 1;

	frame stack[21];

	for (int i = 0; i < MAX_THREAD_STACK_SIZE; i++) {
		stack[i].tx0 = tx0; stack[i].ty0 = ty0; stack[i].tz0 = tz0; stack[i].tx1 = tx1; stack[i].ty1 = ty1; stack[i].tz1 = tz1; stack[i].nodeIndex = 0; stack[i].mortonCode = uint64_t(1); stack[i].txm = -1; stack[i].tym = -1; stack[i].tzm = -1;
	}

	int currIndex = 0;

	while (currIndex >= 0 && currIndex < MAX_THREAD_STACK_SIZE) {

	start:

		auto found = findRef.find(stack[currIndex].mortonCode);
		Node node;

		if(found == findRef.end()){
			goto end;
		}
		// else if(nodeSize(stack[currIndex].mortonCode, octree->level) <= 2){
		// 	printf("%d %llu\n", nodeSize(stack[currIndex].mortonCode, octree->level), stack[currIndex].mortonCode);
		// }
		// else{
		// 	printf("%d %d %llu\n", nodeSize(stack[currIndex].mortonCode, octree->level), octree->level, stack[currIndex].mortonCode);
		// }

		node = found->second;

		//printf("%d - %f\n", currIndex, stack[currIndex].tx0);
		//if(nodeSize(stack[currIndex].mortonCode, octree->level) < 8)
		//	printf("%d\n", nodeSize(stack[currIndex].mortonCode, octree->level));

		// terminal (leaf) node (but not air)
		if (nodeLevel(stack[currIndex].mortonCode, octree->level) == 0 && node.blockId != 0) {

			int blockX, blockY, blockZ;
			octree->morton3Ddecode(stack[currIndex].mortonCode, blockX, blockY, blockZ);

			drawTexturePixel(blockX, blockY, blockZ, origOX, origOY, origOZ, dX, dY, dZ, sX, sY, node.blockId, pixels, negativeDX, negativeDY, negativeDZ);

			//unsigned char* color = BlockTypeToColor(stack[currIndex].node->blockId);
			//setPixel(pixels, sX, sY, 0, 255, 0);

			return node.blockId;
		}

		// out of the octree
		else if (!node.childMask || stack[currIndex].tx1 < 0 || stack[currIndex].ty1 < 0 || stack[currIndex].tz1 < 0) {

		end:
			currIndex--;

			if (currIndex < 0)
				return 0;

			unsigned char prevIndex;

			// set node index of the previous stack frame
			switch (stack[currIndex].nodeIndex) {

				prevIndex = stack[currIndex].nodeIndex;

				case 0:
					stack[currIndex].nodeIndex = newNode(stack[currIndex].txm, 4, stack[currIndex].tym, 2, stack[currIndex].tzm, 1);
					break;
				case 1:
					stack[currIndex].nodeIndex = newNode(stack[currIndex].txm, 5, stack[currIndex].tym, 3, stack[currIndex].tz1, 8);
					break;
				case 2:
					stack[currIndex].nodeIndex = newNode(stack[currIndex].txm, 6, stack[currIndex].ty1, 8, stack[currIndex].tzm, 3);
					break;
				case 3:
					stack[currIndex].nodeIndex = newNode(stack[currIndex].txm, 7, stack[currIndex].ty1, 8, stack[currIndex].tz1, 8);
					break;
				case 4:
					stack[currIndex].nodeIndex = newNode(stack[currIndex].tx1, 8, stack[currIndex].tym, 6, stack[currIndex].tzm, 5);
					break;
				case 5:
					stack[currIndex].nodeIndex = newNode(stack[currIndex].tx1, 8, stack[currIndex].tym, 7, stack[currIndex].tz1, 8);
					break;
				case 6:
					stack[currIndex].nodeIndex = newNode(stack[currIndex].tx1, 8, stack[currIndex].ty1, 8, stack[currIndex].tzm, 7);
					break;
				case 7:
					stack[currIndex].nodeIndex = 8;
					break;
			}

		// 	//printf("%d - %d\n", prevIndex, stack[currIndex].nodeIndex);
		 	goto loop;
		}

		stack[currIndex].txm = (stack[currIndex].tx0 + stack[currIndex].tx1) / 2.0;
		stack[currIndex].tym = (stack[currIndex].ty0 + stack[currIndex].ty1) / 2.0;
		stack[currIndex].tzm = (stack[currIndex].tz0 + stack[currIndex].tz1) / 2.0;
		//unsigned char x = stack[currIndex].nodeIndex;
		stack[currIndex].nodeIndex = firstNode(stack[currIndex].tx0, stack[currIndex].ty0, stack[currIndex].tz0, stack[currIndex].txm, stack[currIndex].tym, stack[currIndex].tzm);
	loop:
			
		//printf("%d\n", stack[currIndex].nodeIndex);
		//printf("%d\n", stack[currIndex].mortonCode);

		// TODO: CHECK IN ADVANCE IF THE CHILD EXISTS
		switch (stack[currIndex].nodeIndex) {

			case 0: {
				if (currIndex >= MAX_THREAD_STACK_SIZE - 1)
					return 0;
				currIndex++;
				stack[currIndex].tx0 = stack[currIndex - 1].tx0; stack[currIndex].ty0 = stack[currIndex - 1].ty0; stack[currIndex].tz0 = stack[currIndex - 1].tz0; stack[currIndex].tx1 = stack[currIndex - 1].txm; stack[currIndex].ty1 = stack[currIndex - 1].tym; stack[currIndex].tz1 = stack[currIndex - 1].tzm; stack[currIndex].nodeIndex = 0; stack[currIndex].mortonCode = childMortonRevelles(stack[currIndex - 1].mortonCode, a);
				goto start;
			}
			case 1: {
				if (currIndex >= MAX_THREAD_STACK_SIZE - 1)
					return 0;
				currIndex++;
				stack[currIndex].tx0 = stack[currIndex - 1].tx0; stack[currIndex].ty0 = stack[currIndex - 1].ty0; stack[currIndex].tz0 = stack[currIndex - 1].tzm; stack[currIndex].tx1 = stack[currIndex - 1].txm; stack[currIndex].ty1 = stack[currIndex - 1].tym; stack[currIndex].tz1 = stack[currIndex - 1].tz1; stack[currIndex].nodeIndex = 1; stack[currIndex].mortonCode = childMortonRevelles(stack[currIndex - 1].mortonCode, 1 ^ a);
				goto start;
			}
			case 2: {
				if (currIndex >= MAX_THREAD_STACK_SIZE - 1)
					return 0;
				currIndex++;
				stack[currIndex].tx0 = stack[currIndex - 1].tx0; stack[currIndex].ty0 = stack[currIndex - 1].tym; stack[currIndex].tz0 = stack[currIndex - 1].tz0; stack[currIndex].tx1 = stack[currIndex - 1].txm; stack[currIndex].ty1 = stack[currIndex - 1].ty1; stack[currIndex].tz1 = stack[currIndex - 1].tzm; stack[currIndex].nodeIndex = 2; stack[currIndex].mortonCode = childMortonRevelles(stack[currIndex - 1].mortonCode, 2 ^ a);
				goto start;
			}
			case 3: {
				if (currIndex >= MAX_THREAD_STACK_SIZE - 1)
					return 0;
				currIndex++;
				stack[currIndex].tx0 = stack[currIndex - 1].tx0; stack[currIndex].ty0 = stack[currIndex - 1].tym; stack[currIndex].tz0 = stack[currIndex - 1].tzm; stack[currIndex].tx1 = stack[currIndex - 1].txm; stack[currIndex].ty1 = stack[currIndex - 1].ty1; stack[currIndex].tz1 = stack[currIndex - 1].tz1; stack[currIndex].nodeIndex = 3; stack[currIndex].mortonCode = childMortonRevelles(stack[currIndex - 1].mortonCode, 3 ^ a);
				goto start;
			}
			case 4: {
				if (currIndex >= MAX_THREAD_STACK_SIZE - 1)
					return 0;
				currIndex++;
				stack[currIndex].tx0 = stack[currIndex - 1].txm; stack[currIndex].ty0 = stack[currIndex - 1].ty0; stack[currIndex].tz0 = stack[currIndex - 1].tz0; stack[currIndex].tx1 = stack[currIndex - 1].tx1; stack[currIndex].ty1 = stack[currIndex - 1].tym; stack[currIndex].tz1 = stack[currIndex - 1].tzm; stack[currIndex].nodeIndex = 4; stack[currIndex].mortonCode = childMortonRevelles(stack[currIndex - 1].mortonCode, 4 ^ a);
				goto start;
			}
			case 5: {
				if (currIndex >= MAX_THREAD_STACK_SIZE - 1)
					return 0;
				currIndex++;
				stack[currIndex].tx0 = stack[currIndex - 1].txm; stack[currIndex].ty0 = stack[currIndex - 1].ty0; stack[currIndex].tz0 = stack[currIndex - 1].tzm; stack[currIndex].tx1 = stack[currIndex - 1].tx1; stack[currIndex].ty1 = stack[currIndex - 1].tym; stack[currIndex].tz1 = stack[currIndex - 1].tz1; stack[currIndex].nodeIndex = 5; stack[currIndex].mortonCode = childMortonRevelles(stack[currIndex - 1].mortonCode, 5 ^ a);
				goto start;
			}
			case 6: {
				if (currIndex >= MAX_THREAD_STACK_SIZE - 1)
					return 0;
				currIndex++;
				stack[currIndex].tx0 = stack[currIndex - 1].txm; stack[currIndex].ty0 = stack[currIndex - 1].tym; stack[currIndex].tz0 = stack[currIndex - 1].tz0; stack[currIndex].tx1 = stack[currIndex - 1].tx1; stack[currIndex].ty1 = stack[currIndex - 1].ty1; stack[currIndex].tz1 = stack[currIndex - 1].tzm; stack[currIndex].nodeIndex = 6; stack[currIndex].mortonCode = childMortonRevelles(stack[currIndex - 1].mortonCode, 6 ^ a);
				goto start;
			}
			case 7: {
				if (currIndex >= MAX_THREAD_STACK_SIZE - 1)
					return 0;
				currIndex++;
				stack[currIndex].tx0 = stack[currIndex - 1].txm; stack[currIndex].ty0 = stack[currIndex - 1].tym; stack[currIndex].tz0 = stack[currIndex - 1].tzm; stack[currIndex].tx1 = stack[currIndex - 1].tx1; stack[currIndex].ty1 = stack[currIndex - 1].ty1; stack[currIndex].tz1 = stack[currIndex - 1].tz1; stack[currIndex].nodeIndex = 7; stack[currIndex].mortonCode = childMortonRevelles(stack[currIndex - 1].mortonCode, 7 ^ a);
				goto start;
			}
		}

		if (stack[currIndex].nodeIndex >= 8)
			goto end;
	}
	return 0;
}