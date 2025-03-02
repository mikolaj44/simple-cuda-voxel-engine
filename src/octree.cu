//cudaMemcpy(memory, memory, size_t(PREALLOCATE_MB_AMOUNT * 1024 * 1024), cudaMemcpyDeviceToDevice);
//cudaSetDevice(0);
//int id = 0;
//cudaGetDevice(&id);
//cudaMemAdvise(memory, size_t(PREALLOCATE_MB_AMOUNT * 1024 * 1024), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
//cudaMemAdvise(memory, size_t(PREALLOCATE_MB_AMOUNT * 1024 * 1024), cudaMemAdviseSetReadMostly, 0);
//cudaMemPrefetchAsync(memory, PREALLOCATE_BYTES_AMOUNT, id);
//cudaMemAdvise(memory, PREALLOCATE_BYTES_AMOUNT, cudaMemAdviseSetAccessedBy, 0);

#include <cmath>
#include <bitset>

#include "octree.cuh"
#include "cuda_morton.cuh"

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/logical.h>
#include <thrust/sequence.h>
#include <thrust/tuple.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <cxxabi.h>

#define epsilon 0.00001

using namespace std;

size_t Octree::memoryAvailableBytes = 0;
size_t Octree::memoryTakenBytes = 0;


// template<typename MapFindRef, typename MapInsertRef>
// __global__ void insertNode(MapFindRef mapFindRef, MapInsertRef mapInsertRef, uint64_t key, Node node){

// 	nodeMapInsertRef.insert(cuco::pair{key, node});

// 	auto found = nodeMapFindRef.find(0);

// 	printf("%d\n", (int)found->second.blockId );
// }


void Octree::createOctree(int xMin_, int yMin_, int zMin_, unsigned int level_) {

	xMin = xMin_;
	yMin = yMin_;
	zMin = zMin_;
	level = level_;
	
	//memoryAvailableBytes = size_t(PREALLOCATE_MB_AMOUNT * 1024 * 1024) / sizeof(Node);

	nodeMap = cuco::static_map{cuco::extent<std::size_t, NODE_MAP_CAPACITY>{},
								cuco::empty_key{uint64_t{0}},
								cuco::empty_value{Node{}},
								thrust::equal_to<uint64_t>{},
								cuco::linear_probing<1,cuco::xxhash_64<uint64_t>>{}};

	// int status;
    // char* demangled = abi::__cxa_demangle(typeid(nodeMapFindRef).name(), nullptr, nullptr, &status);
	// cout << demangled << endl;

	//insertNode<<<1,1>>>(nodeMapFindRef, nodeMapInsertRef, 1, Node(44, false));
}

void Octree::createOctree(OctreeSpecialPosition position, unsigned int level){

	if(position == CENTERED){
		createOctree(-(1 << (level - 1)), -(1 << (level - 1)), -(1 << (level - 1)), level);
	}
}

//void Octree::subdivide(Node* node) {
//
//	int xM = (2 * node->xMin + node->size) / 2;
//	int yM = (2 * node->yMin + node->size) / 2;
//	int zM = (2 * node->zMin + node->size) / 2;
//
//	//libmorton::morton3D_64_encode
//
//	// labeled according to Figure 1: http://wscg.zcu.cz/wscg2000/Papers_2000/X31.pdf
//
//	node->children[0] = Node::createNode(node->xMin, xM, node->yMin, yM, node->zMin, zM);
//	node->children[1] = Node::createNode(node->xMin, xM, node->yMin, yM, zM, node->zMin + node->size);
//	node->children[2] = Node::createNode(node->xMin, xM, yM, node->yMin + node->size, node->zMin, zM);
//	node->children[3] = Node::createNode(node->xMin, xM, yM, node->yMin + node->size, zM, node->zMin + node->size);
//
//	node->children[4] = Node::createNode(xM, node->xMin + node->size, node->yMin, yM, node->zMin, zM);
//	node->children[5] = Node::createNode(xM, node->xMin + node->size, node->yMin, yM, zM, node->zMin + node->size);
//	node->children[6] = Node::createNode(xM, node->xMin + node->size, yM, node->yMin + node->size, node->zMin, zM);
//	node->children[7] = Node::createNode(xM, node->xMin + node->size, yM, node->yMin + node->size, zM, node->zMin + node->size);
//}

// void Octree::grow(int x, int y, int z) {

// 	int size = 1 << level;

// 	// Grow the octree (resize the root) until the point fits
// 	while (x < xMin || x >= xMin + size || y < yMin || y >= yMin + size || z < zMin || z >= zMin + size) {

// 		// Mark the node that has the previous root position as having children
// 		uint64_t index = octree_morton3D_64_encode(xMin, yMin, zMin, level);
// 		nodes[index].hasChildren = true;

// 		cout << xMin << " " << yMin << " " << zMin << " " << bitset<64>(index) << endl;

// 		// Handle the 8 possible growth cases
// 		if (x < xMin) {
// 			xMin -= size;
// 		}
// 		if (y < yMin) {
// 			yMin -= size;
// 		}
// 		if (z < yMin) {
// 			zMin -= size;
// 		}

// 		// Update the octree level
// 		level++;
// 		size = 1 << level;
// 	}
// }

//unsigned char Octree::get(int x, int y, int z, Node* node){
//
//	unsigned char octantIndex = ((x < 0) * (1 << 2)) | ((y < 0) * (1 << 1)) | (z < 0);
//	uint64_t nodeIndex = libmorton::morton3D_64_encode(x, y, z);
//
//	if (node == nullptr || x < root->xMin || x >= (root->xMin + root->size) || y < root->yMin || y >= (root->yMin + root->size) || z < root->zMin || z >= (root->zMin + root->size))
//		return -1;
//
//	if ((node->xMin + node->size) - node->xMin == 1)
//		return node->blockId;
//
//	int xM = (node->xMin + (node->xMin + node->size)) / 2;
//	int yM = (node->yMin + (node->yMin + node->size)) / 2;
//	int zM = (node->zMin + (node->zMin + node->size)) / 2;
//
//	if (x < xM) {
//
//		if (y < yM) {
//
//			if (z < zM)
//				get(x, y, z, node->children[0]);
//			else
//				get(x, y, z, node->children[1]);
//		}
//		else {
//			if (z < zM)
//				get(x, y, z, node->children[2]);
//			else
//				get(x, y, z, node->children[3]);
//		}
//	}
//	else {
//		if (y < yM) {
//
//			if (z < zM)
//				get(x, y, z, node->children[4]);
//			else
//				get(x, y, z, node->children[5]);
//		}
//		else {
//			if (z < zM)
//				get(x, y, z, node->children[6]);
//			else
//				get(x, y, z, node->children[7]);
//		}
//	}
//
//	return -1;
//}

//unsigned char Octree::get(int x, int y, int z) {
//	return get(x, y, z, root);
//}

void Octree::getChildXYZindex(int& x, int& y, int& z, uint64_t& index, unsigned int level, unsigned int childIndex) {

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

void Octree::display(unsigned char* pixels, uint64_t index, bool showBorder, int x, int y, int z, unsigned int level){

	if(index == 1){
		x = xMin;
		y = yMin;
		z = zMin;
		level = Octree::level;
	}

	//cout << bitset<64>(index) << endl;

	thrust::device_vector<uint64_t> key(1);
	thrust::device_vector<Node> value(1);
	key[0] = index;

	nodeMap.find(key.begin(), key.end(), value.begin());

	Node node = value[0];

	//cout << (int)node.blockId << " " << node.hasChildren << endl;

	//if(level == 17)
	//	cout << (int)node.blockId << " " << node.hasChildren << " " << index << " " << level << endl;

	int size = 1 << level;

	if (showBorder || node.blockId != 0) {

		float coordinates[8][2];
		float* coordinate;

		coordinate = _3d2dProjection(x, y, z);
		coordinates[0][0] = coordinate[0];
		coordinates[0][1] = coordinate[1];

		coordinate = _3d2dProjection(x + size, y, z);
		coordinates[1][0] = coordinate[0];
		coordinates[1][1] = coordinate[1];

		coordinate = _3d2dProjection(x + size, y + size, z);
		coordinates[2][0] = coordinate[0];
		coordinates[2][1] = coordinate[1];

		coordinate = _3d2dProjection(x, y + size, z);
		coordinates[3][0] = coordinate[0];
		coordinates[3][1] = coordinate[1];

		coordinate = _3d2dProjection(x, y, z + size);
		coordinates[4][0] = coordinate[0];
		coordinates[4][1] = coordinate[1];

		coordinate = _3d2dProjection(x + size, y, z + size);
		coordinates[5][0] = coordinate[0];
		coordinates[5][1] = coordinate[1];

		coordinate = _3d2dProjection(x + size, y + size, z + size);
		coordinates[6][0] = coordinate[0];
		coordinates[6][1] = coordinate[1];

		coordinate = _3d2dProjection(x, y + size, z + size);
		coordinates[7][0] = coordinate[0];
		coordinates[7][1] = coordinate[1];

		//unsigned char type = 0;

		//if (node->blockId != 0)
		//	type = node->blockId;

		//unsigned char* color = BlockTypeToColor(type);
		//int color[3] = { rand() % 255, rand() % 255, rand() % 255 };
		int color[3] = { 0, 255, 0 };

		drawLine(pixels, (int)coordinates[0][0], (int)coordinates[0][1], (int)coordinates[1][0], (int)coordinates[1][1], color[0], color[1], color[2]);
		drawLine(pixels, (int)coordinates[1][0], (int)coordinates[1][1], (int)coordinates[2][0], (int)coordinates[2][1], color[0], color[1], color[2]);
		drawLine(pixels, (int)coordinates[2][0], (int)coordinates[2][1], (int)coordinates[3][0], (int)coordinates[3][1], color[0], color[1], color[2]);
		drawLine(pixels, (int)coordinates[3][0], (int)coordinates[3][1], (int)coordinates[0][0], (int)coordinates[0][1], color[0], color[1], color[2]);
		drawLine(pixels, (int)coordinates[4][0], (int)coordinates[4][1], (int)coordinates[5][0], (int)coordinates[5][1], color[0], color[1], color[2]);
		drawLine(pixels, (int)coordinates[5][0], (int)coordinates[5][1], (int)coordinates[6][0], (int)coordinates[6][1], color[0], color[1], color[2]);
		drawLine(pixels, (int)coordinates[6][0], (int)coordinates[6][1], (int)coordinates[7][0], (int)coordinates[7][1], color[0], color[1], color[2]);
		drawLine(pixels, (int)coordinates[7][0], (int)coordinates[7][1], (int)coordinates[4][0], (int)coordinates[4][1], color[0], color[1], color[2]);
		drawLine(pixels, (int)coordinates[0][0], (int)coordinates[0][1], (int)coordinates[4][0], (int)coordinates[4][1], color[0], color[1], color[2]);
		drawLine(pixels, (int)coordinates[1][0], (int)coordinates[1][1], (int)coordinates[5][0], (int)coordinates[5][1], color[0], color[1], color[2]);
		drawLine(pixels, (int)coordinates[2][0], (int)coordinates[2][1], (int)coordinates[6][0], (int)coordinates[6][1], color[0], color[1], color[2]);
		drawLine(pixels, (int)coordinates[3][0], (int)coordinates[3][1], (int)coordinates[7][0], (int)coordinates[7][1], color[0], color[1], color[2]);
	}

	if (level <= 1 || !node.hasChildren) {
		return;
	}

	for (int i = 0; i < 8; i++) {
		int x_ = x;
		int y_ = y;
		int z_ = z;
		uint64_t index_ = index;
		getChildXYZindex(x_, y_, z_, index_, level, i);
		display(pixels, index_, showBorder, x_, y_, z_, level - 1);
	}

	// // For now the octree isn't sparse

	// /*if (node->children[0] != nullptr)
	// 	display(node->children[0], showBorder, ++depth);
	// if (node->children[1] != nullptr)
	// 	display(node->children[1], showBorder, ++depth);
	// if (node->children[2] != nullptr)
	// 	display(node->children[2], showBorder, ++depth);
	// if (node->children[3] != nullptr)
	// 	display(node->children[3], showBorder, ++depth);
	// if (node->children[4] != nullptr)
	// 	display(node->children[4], showBorder, ++depth);
	// if (node->children[5] != nullptr)
	// 	display(node->children[5], showBorder, ++depth);
	// if (node->children[6] != nullptr)
	// 	display(node->children[6], showBorder, ++depth);
	// if (node->children[7] != nullptr)
	// 	display(node->children[7], showBorder, ++depth);*/
}

void Octree::display(unsigned char* pixels, bool showBorder){
	display(pixels, 1, showBorder);
}

void insert(Octree* octree, thrust::device_vector<Block> blocks, size_t numBlocks, unsigned int gridSize, unsigned int blockSize){
	insertKernel<<<gridSize, blockSize>>>(octree, octree->nodeMap.ref(cuco::insert), thrust::raw_pointer_cast(blocks.data()), numBlocks);
	cudaDeviceSynchronize(); // maybe remove this later
}

__device__
int firstNode(float tx0, float ty0, float tz0, float txm, float tym, float tzm) {

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

__device__
int newNode(float tx, int i1, float ty, int i2, float tz, int i3) {

	//printf("%f - %d - %f - %d - %f - %d\n",tx,i1,ty,i2,tz,i3);

	float minV = minv(minv(tx, ty), tz);

	if (equals(minV, tx, epsilon)) {
		return i1;
	}
	if (equals(minV, ty, epsilon)) {
		return i2;
	}
	return i3;
}

__device__ uint64_t childMortonRevelles(uint64_t mortonCode, unsigned char revellesChildIndex){

	uint64_t code = mortonCode << 3;

	switch (revellesChildIndex){
		case 0:
			return code;
		case 1:
			code |= (1 << 2);
			return code;
		case 2:
			code |= (1 << 1);
			return code;
		case 3:
			code |= (1 << 1);
			code |= (1 << 2);
			return code;
		case 4:
			code |= 1;
			return code;
		case 5:
			code |= 1;
			code |= (1 << 2);
			return code;
		case 6:
			code |= 1;
			code |= (1 << 1);
			return code;
		case 7:
			code |= 1;
			code |= (1 << 1);
			code |= (1 << 2);
			return code;
		default:
			return code;
	}
}

__device__ void drawTexturePixel(int blockX, int blockY, int blockZ, float oX, float oY, float oZ, float dX, float dY, float dZ, int sX, int sY, unsigned char blockId, unsigned char* pixels, bool negativeDX, bool negativeDY, bool negativeDZ) {

	if (dX == 0 || dY == 0 || dZ == 0) { // for now
		return;
	}

	if (negativeDX) {
		dX *= -1.0;
	}
	if (negativeDY) {
		dY *= -1.0;
	}
	if (negativeDZ) {
		dZ *= -1.0;
	}
	
	float tmin =  minv(((float)blockX - oX) / dX, ((float)blockX + 1.0 - oX) / dX);
	float tymin = minv(((float)blockY - oY) / dY, ((float)blockY + 1.0 - oY) / dY);
	float tzmin = minv(((float)blockZ - oZ) / dZ, ((float)blockZ + 1.0 - oZ) / dZ);

	tmin = maxv(maxv(tmin, tymin), tzmin);

	float x = oX + tmin * dX;
	float y = oY + tmin * dY;
	float z = oZ + tmin * dZ;

	//printf("%d\n", negativeDX);

	setPixelById(sX, sY, blockX, blockY, blockZ, x, y, z, blockId, pixels);
}
