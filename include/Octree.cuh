#pragma once
#include <string>
#include "globals.cuh"
#include "pixeldrawing.cuh"
#include "blocksdata.cuh"
#include "cudamath.cuh"
#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cuco/static_map.cuh>

#define map_type 		cuco::static_map<uint64_t, Node, cuco::extent<unsigned long, NODE_MAP_CAPACITY>, (cuda::std::__4::thread_scope)1, cuda::std::__4::equal_to<uint64_t>, cuco::linear_probing<1, cuco::detail::XXHash_64<uint64_t> >, cuco::cuda_allocator<cuco::pair<uint64_t, Node> >, cuco::storage<1> >
#define insert_ref_type cuco::static_map_ref<uint64_t, Node, (cuda::std::__4::thread_scope)1, cuda::std::__4::equal_to<uint64_t>, cuco::linear_probing<1, cuco::detail::XXHash_64<uint64_t> >, cuco::bucket_storage_ref<cuco::pair<uint64_t, Node>, 1, cuco::bucket_extent<uint64_t, NODE_MAP_CAPACITY + 9> >, cuco::op::insert_tag>
#define find_ref_type   cuco::static_map_ref<uint64_t, Node, (cuda::std::__4::thread_scope)1, cuda::std::__4::equal_to<uint64_t>, cuco::linear_probing<1, cuco::detail::XXHash_64<uint64_t> >, cuco::bucket_storage_ref<cuco::pair<uint64_t, Node>, 1, cuco::bucket_extent<uint64_t, NODE_MAP_CAPACITY + 9> >, cuco::op::find_tag>

using namespace std;

class Node {

public:
	unsigned char blockId = 0; // air
	bool hasChildren = false; // for the regular octree, not an SVO - either has 0 or 8 children
	uint8_t padding[2]; // 2 bytes padding

	//unsigned char childMask = 0; // which children are present, indexed according to Figure 1: http://wscg.zcu.cz/wscg2000/Papers_2000/X31.pdf

	__host__ __device__ Node() {};

	__host__ __device__ Node(unsigned char blockId_, bool hasChildren_) : blockId(blockId_), hasChildren(hasChildren_) {};

	//__host__ __device__ Node(unsigned char blockId_, bool hasChildren_, uint64_t mortonCode_) : blockId(blockId_), hasChildren(hasChildren_), mortonCode(mortonCode_) {};
};

class Octree {

public:

	static size_t memoryTakenBytes;
	static size_t memoryAvailableBytes;

	map_type nodeMap;

	insert_ref_type nodeMapInsertRef;
	find_ref_type nodeMapFindRef;

	int xMin, yMin, zMin;
	unsigned int level; // level 0 is a terminal node

	// the root node is stored at the first position

	void createOctree(int xMin_, int yMin_, int zMin_, unsigned int level_);

	// the actual device insert function
	template<typename MapInsertRef>
	__device__ void insert(MapInsertRef insertRef, int x, int y, int z, unsigned char blockId) {

		// Octree coordinate system is positive only, convert the coordinates to this system
		x -= xMin;
		y -= yMin;
		z -= zMin;

		// If the voxel is out of bounds (we don't grow the octree)
		if(x < 0 || y < 0 || z < 0 || x >= xMin || y >= yMin || z >= zMin){
			return;
		}

		int level = Octree::level;
		uint64_t index = 1; // root node index

		int nodeX = xMin;
		int nodeY = yMin;
		int nodeZ = zMin;

		int size = 1 << level;

		int numShifts = 0;

		// Iterate over all node levels up until the leaf node
		do{

			if(numShifts >= sizeof(uint64_t) * 8){ // Detect index overflow
				return;
			}

			//cout << level << " " << bitset<64>(index) << endl;

			if (level == 1) {

				// Get the node at index (to insert the right block data)
				// auto iterator = nodeMapRef.find(index);
				// iterator->second

				//nodeMapInsertRef

				insertRef.insert(cuco::pair{index, Node{false, blockId}});
				return;
			}

			// We are still assuming that the octree is not sparse
			
			//insertRef.insert(cuco::pair{index, Node{true, blockId}});

			// Get the midpoint
			int xM = (2 * nodeX + size) / 2;
			int yM = (2 * nodeY + size) / 2;
			int zM = (2 * nodeZ + size) / 2;

			// Compute the coordinates and morton code of the child node

			numShifts += 3;
			index <<= 3;

			if (x >= xM) {
				nodeX += size / 2;
				index |= (1 << 2);
			}
			if (y >= yM) {
				nodeY += size / 2;
				index |= (1 << 1);
			}
			if (z >= zM) {	
				nodeZ += size / 2;
				index |= 1;
			}

			level--;

			size = 1 << level;

		} while (level >= 1);
		
	}

	unsigned char get(int x, int y, int z);

	void display(bool showBorder = true, unsigned int level = INT_MAX);

	void display(int xMin, int yMin, int zMin, unsigned int level = INT_MAX, bool showBorder = true);
	
private:

	void subdivide(Node* node); // Will be used later for the SVO

	void grow(int x, int y, int z);

	void getChildXYZ(int xMin, int yMin, int zMin, unsigned int level, int childIndex, int& x, int& y, int& z);
};

__device__ int firstNode(float tx0, float ty0, float tz0, float txm, float tym, float tzm);

__device__ int newNode(float tx, int i1, float ty, int i2, float tz, int i3);

__device__ void rayParameter(Octree* octree, float oX, float oY, float oZ, float dX, float dY, float dZ, int sX, int sY, int minNodeSize = 1, unsigned char* pixels = nullptr);

__device__ unsigned char procSubtree(float oX, float oY, float oZ, float dX, float dY, float dZ, float tx0, float ty0, float tz0, float tx1, float ty1, float tz1, Node* node, unsigned char a, int minNodeSize, int sX, int sY, unsigned char* pixels, float origOX, float origOY, float origOZ, bool negativeDX, bool negativeDY, bool negativeDZ);

// the kernel for inserting nodes into the octree
__global__ void insert(Octree* octree, int x, int y, int z, unsigned char blockId);