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

#include "Octree.cuh"
#include "cudamorton.cuh"

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

	//auto map = cuco::static_multimap<int, int>{N * 2, cuco::empty_key{-1}, cuco::empty_value{-1}};

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

	nodeMapFindRef = nodeMap.ref(cuco::find);
	nodeMapInsertRef = nodeMap.ref(cuco::insert);

	// int status;
    // char* demangled = abi::__cxa_demangle(typeid(nodeMapFindRef).name(), nullptr, nullptr, &status);
	// cout << demangled << endl;

	//insertNode<<<1,1>>>(nodeMapFindRef, nodeMapInsertRef, 1, Node(44, false));
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

__device__ void Octree::insert(int x, int y, int z, unsigned char blockId){
	insert(nodeMapInsertRef, x, y, z, blockId);
}

__global__ void insert(Octree* octree, int x, int y, int z, unsigned char blockId){

	// thread index etc.

	octree->insert(octree->nodeMapInsertRef, x, y, z, blockId);
}

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

void Octree::getChildXYZ(int xMin, int yMin, int zMin, unsigned int level, int childIndex, int& x, int& y, int& z) {

	int size = 1 << level;

	switch (childIndex) {
		case 0:
			x = xMin;
			y = yMin;
			z = zMin;
			break;
		case 1:
			x = xMin;
			y = yMin;
			z = zMin + size;
			break;
		case 2:
			x = xMin;
			y = yMin + size;
			z = zMin;
			break;
		case 3:
			x = xMin;
			y = yMin + size;
			z = zMin + size;
			break;
		case 4:
			x = xMin + size;
			y = yMin;
			z = zMin;
			break;
		case 5:
			x = xMin + size;
			y = yMin;
			z = zMin + size;
			break;
		case 6:
			x = xMin + size;
			y = yMin + size;
			z = zMin;
			break;
		case 7:
			x = xMin + size;
			y = yMin + size;
			z = zMin + size;
			break;
		default:
			break;
	}
}

void Octree::display(int xMin, int yMin, int zMin, unsigned int level, bool showBorder) {

	uint64_t index;

	if (level >= Octree::level) {
		level = Octree::level;
		index = 0;
	}
	else {
		index = octree_morton3D_64_encode(xMin, yMin, zMin, level);
	}

	//cout << bitset<64>(index) << endl;
	
	int size = 1 << level;

	//if (showBorder || nodes[index].blockId != 0) {

	//	float coordinates[8][2];
	//	float* coordinate;

	//	coordinate = _3d2dProjection(xMin, yMin, zMin);
	//	coordinates[0][0] = coordinate[0];
	//	coordinates[0][1] = coordinate[1];

	//	coordinate = _3d2dProjection((xMin + size), yMin, zMin);
	//	coordinates[1][0] = coordinate[0];
	//	coordinates[1][1] = coordinate[1];

	//	coordinate = _3d2dProjection((xMin + size), (yMin + size), zMin);
	//	coordinates[2][0] = coordinate[0];
	//	coordinates[2][1] = coordinate[1];

	//	coordinate = _3d2dProjection(xMin, (yMin + size), zMin);
	//	coordinates[3][0] = coordinate[0];
	//	coordinates[3][1] = coordinate[1];

	//	coordinate = _3d2dProjection(xMin, yMin, (zMin + size));
	//	coordinates[4][0] = coordinate[0];
	//	coordinates[4][1] = coordinate[1];

	//	coordinate = _3d2dProjection((xMin + size), yMin, (zMin + size));
	//	coordinates[5][0] = coordinate[0];
	//	coordinates[5][1] = coordinate[1];

	//	coordinate = _3d2dProjection((xMin + size), (yMin + size), (zMin + size));
	//	coordinates[6][0] = coordinate[0];
	//	coordinates[6][1] = coordinate[1];

	//	coordinate = _3d2dProjection(xMin, (yMin + size), (zMin + size));
	//	coordinates[7][0] = coordinate[0];
	//	coordinates[7][1] = coordinate[1];

	//	//unsigned char type = 0;

	//	//if (node->blockId != 0)
	//	//	type = node->blockId;

	//	//unsigned char* color = BlockTypeToColor(type);
	//	int color[3] = { 255,0,0 };

	//	DrawLine((int)coordinates[0][0], (int)coordinates[0][1], (int)coordinates[1][0], (int)coordinates[1][1], color[0], color[1], color[2]);
	//	DrawLine((int)coordinates[1][0], (int)coordinates[1][1], (int)coordinates[2][0], (int)coordinates[2][1], color[0], color[1], color[2]);
	//	DrawLine((int)coordinates[2][0], (int)coordinates[2][1], (int)coordinates[3][0], (int)coordinates[3][1], color[0], color[1], color[2]);
	//	DrawLine((int)coordinates[3][0], (int)coordinates[3][1], (int)coordinates[0][0], (int)coordinates[0][1], color[0], color[1], color[2]);
	//	DrawLine((int)coordinates[4][0], (int)coordinates[4][1], (int)coordinates[5][0], (int)coordinates[5][1], color[0], color[1], color[2]);
	//	DrawLine((int)coordinates[5][0], (int)coordinates[5][1], (int)coordinates[6][0], (int)coordinates[6][1], color[0], color[1], color[2]);
	//	DrawLine((int)coordinates[6][0], (int)coordinates[6][1], (int)coordinates[7][0], (int)coordinates[7][1], color[0], color[1], color[2]);
	//	DrawLine((int)coordinates[7][0], (int)coordinates[7][1], (int)coordinates[4][0], (int)coordinates[4][1], color[0], color[1], color[2]);
	//	DrawLine((int)coordinates[0][0], (int)coordinates[0][1], (int)coordinates[4][0], (int)coordinates[4][1], color[0], color[1], color[2]);
	//	DrawLine((int)coordinates[1][0], (int)coordinates[1][1], (int)coordinates[5][0], (int)coordinates[5][1], color[0], color[1], color[2]);
	//	DrawLine((int)coordinates[2][0], (int)coordinates[2][1], (int)coordinates[6][0], (int)coordinates[6][1], color[0], color[1], color[2]);
	//	DrawLine((int)coordinates[3][0], (int)coordinates[3][1], (int)coordinates[7][0], (int)coordinates[7][1], color[0], color[1], color[2]);
	//}

	if (level == 1) { // || !nodes[index].hasChildren
		return;
	}

	level--;

	int x, y, z;

	for (int i = 0; i < 8; i++) {
		getChildXYZ(xMin, yMin, zMin, level, i, x, y, z);
		display(x, y, z, level, showBorder);
	}

	// For now the octree isn't sparse

	/*if (node->children[0] != nullptr)
		display(node->children[0], showBorder, ++depth);
	if (node->children[1] != nullptr)
		display(node->children[1], showBorder, ++depth);
	if (node->children[2] != nullptr)
		display(node->children[2], showBorder, ++depth);
	if (node->children[3] != nullptr)
		display(node->children[3], showBorder, ++depth);
	if (node->children[4] != nullptr)
		display(node->children[4], showBorder, ++depth);
	if (node->children[5] != nullptr)
		display(node->children[5], showBorder, ++depth);
	if (node->children[6] != nullptr)
		display(node->children[6], showBorder, ++depth);
	if (node->children[7] != nullptr)
		display(node->children[7], showBorder, ++depth);*/
}

void Octree::display(bool showBorder, unsigned int level) {

	display(xMin, yMin, zMin, level, showBorder);
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

//__device__
//void rayParameter(Octree* octree, float oX, float oY, float oZ, float dX, float dY, float dZ, int sX, int sY, int minNodeSize, unsigned char* pixels){
//
//	unsigned char a = 0;
//
//	bool negativeDX = false, negativeDY = false, negativeDZ = false;
//	float origOX = oX, origOY = oY, origOZ = oZ;
//
//	if (dX < 0) {
//		//origin.x *= -1; //abs(octree->(root->xMin + root->size) - octree->root->xMin) / 2 - origin.x;
//		//origin.x += abs(octree->(root->zMin + root->size) - octree->root->zMin) / 4; // abs(octree->(root->xMin + root->size) - octree->root->xMin);
//		oX = -oX + (octree->root->xMin + octree->root->size + octree->root->xMin);// +dCenterX;
//		dX = -dX;
//		a |= 4;
//		negativeDX = true;
//	}
//	if (dY < 0) {
//		//origin.y *= -1;
//		//origin.y += abs(octree->(root->zMin + root->size) - octree->root->zMin) / 4; //shift; //abs(octree->(root->yMin + root->size) - octree->root->yMin);
//		oY = -oY + (octree->root->yMin + octree->root->size + octree->root->yMin);// +dCenterY;
//		dY = -dY;
//		a |= 2;
//		negativeDY = true;
//	}
//	if (dZ < 0) {
//		//origin.z *= -1; //abs(octree->(root->zMin + root->size) - octree->root->zMin) / 2- origin.z;
//		//origin.z += abs(octree->(root->zMin + root->size) - octree->root->zMin) / 4; // abs(octree->(root->zMin + root->size) - octree->root->zMin);
//		oZ = -oZ + (octree->root->zMin + octree->root->size + octree->root->zMin);// +dCenterZ;
//		dZ = -dZ;
//		a |= 1;
//		negativeDZ = true;
//	}
//
//	float tx0 = (octree->root->xMin - oX) / dX;
//	float tx1 = (octree->root->xMin + octree->root->size - oX) / dX;
//	float ty0 = (octree->root->yMin - oY) / dY;
//	float ty1 = (octree->root->yMin + octree->root->size - oY) / dY;
//	float tz0 = (octree->root->zMin - oZ) / dZ;
//	float tz1 = (octree->root->zMin + octree->root->size - oZ) / dZ;
//
//	//int color[3] = { 0,0,0 };
//
//	//printf("\n-2");
//
//	if (maxv(maxv(tx0, ty0), tz0) < minv(minv(tx1, ty1), tz1)) {
//
//		//printf("%f %f %f\n",tx0, ty0, tz0);
//
//		unsigned char result = procSubtree(oX, oY, oZ, dX, dY, dZ, tx0, ty0, tz0, tx1, ty1, tz1, octree->root, a, minNodeSize, sX, sY, pixels, origOX, origOY, origOZ, negativeDX, negativeDY, negativeDZ);
//
//		if (result == 0) {
//
//			SetPixel(sX, sY, 0, 0, 0, 255, pixels); //30 30 255
//		}
//
//	}
//}
//
//struct frame {
//	float tx0, ty0, tz0, tx1, ty1, tz1, txm, tym, tzm; unsigned char nodeIndex; Node* node;
//};
//
//__device__
//void drawTexturePixel(int blockX, int blockY, int blockZ, float oX, float oY, float oZ, float dX, float dY, float dZ, int sX, int sY, unsigned char blockId, unsigned char* pixels, bool negativeDX, bool negativeDY, bool negativeDZ) {
//
//	if (dX == 0 || dY == 0 || dZ == 0) { // for now
//		return;
//	}
//
//	if (negativeDX) {
//		dX *= -1.0;
//	}
//	if (negativeDY) {
//		dY *= -1.0;
//	}
//	if (negativeDZ) {
//		dZ *= -1.0;
//	}
//	
//	float tmin =  minv(((float)blockX - oX) / dX, ((float)blockX + 1.0 - oX) / dX);
//	float tymin = minv(((float)blockY - oY) / dY, ((float)blockY + 1.0 - oY) / dY);
//	float tzmin = minv(((float)blockZ - oZ) / dZ, ((float)blockZ + 1.0 - oZ) / dZ);
//
//	tmin = maxv(maxv(tmin, tymin), tzmin);
//
//	float x = oX + tmin * dX;
//	float y = oY + tmin * dY;
//	float z = oZ + tmin * dZ;
//
//	//printf("%d\n", negativeDX);
//
//	setPixelById(sX, sY, blockX, blockY, blockZ, x, y, z, blockId, pixels);
//}
//
//__device__
//unsigned char procSubtree(float oX, float oY, float oZ, float dX, float dY, float dZ, float tx0, float ty0, float tz0, float tx1, float ty1, float tz1, Node* node, unsigned char a, int minNodeSize, int sX, int sY, unsigned char* pixels, float origOX, float origOY, float origOZ, bool negativeDX, bool negativeDY, bool negativeDZ) {
//
//	frame stack[MAX_THREAD_STACK_SIZE];
//
//	for (int i = 0; i < MAX_THREAD_STACK_SIZE; i++) {
//		stack[i].tx0 = tx0; stack[i].ty0 = ty0; stack[i].tz0 = tz0; stack[i].tx1 = tx1; stack[i].ty1 = ty1; stack[i].tz1 = tz1; stack[i].nodeIndex = 0; stack[i].node = node; stack[i].txm = -1; stack[i].tym = -1; stack[i].tzm = -1;
//	}
//
//	int currIndex = 0;
//
//	while (currIndex >= 0 && currIndex < MAX_THREAD_STACK_SIZE) {
//
//	start:
//
//		if (stack[currIndex].node == nullptr) {
//			goto end;
//		}
//
//		//printf("%d - %f\n", currIndex, stack[currIndex].tx0);
//
//		// terminal (leaf) node (but not air)
//		if (stack[currIndex].node->xMin + stack[currIndex].node->size - stack[currIndex].node->xMin <= minNodeSize && stack[currIndex].node->blockId != 0) {
//
//			//printf("%f %f %f %f\n", absv(stack[currIndex].tx1 - (-256 - oX) / dX), absv(stack[currIndex].ty1 * dY + oY), absv(stack[currIndex].tz1 * dZ + oZ), oX);
//
//			drawTexturePixel(stack[currIndex].node->xMin, stack[currIndex].node->yMin, stack[currIndex].node->zMin, origOX, origOY, origOZ, dX, dY, dZ, sX, sY, stack[currIndex].node->blockId, pixels, negativeDX, negativeDY, negativeDZ);
//
//			/*unsigned char* color = BlockTypeToColor(stack[currIndex].node->blockId);
//			SetPixel(sX, sY, color[0], color[1], color[2], 255, pixels);*/
//			return stack[currIndex].node->blockId;
//		}
//
//		// out of the octree
//		else if (stack[currIndex].node->children[0] == nullptr || stack[currIndex].tx1 < 0 || stack[currIndex].ty1 < 0 || stack[currIndex].tz1 < 0) {
//
//		end:
//			currIndex--;
//
//			if (currIndex < 0)
//				return 0;
//
//			unsigned char prevIndex;
//
//			// set node index of the previous stack frame
//			switch (stack[currIndex].nodeIndex) {
//
//				prevIndex = stack[currIndex].nodeIndex;
//
//				case 0:
//					stack[currIndex].nodeIndex = newNode(stack[currIndex].txm, 4, stack[currIndex].tym, 2, stack[currIndex].tzm, 1);
//					break;
//				case 1:
//					stack[currIndex].nodeIndex = newNode(stack[currIndex].txm, 5, stack[currIndex].tym, 3, stack[currIndex].tz1, 8);
//					break;
//				case 2:
//					stack[currIndex].nodeIndex = newNode(stack[currIndex].txm, 6, stack[currIndex].ty1, 8, stack[currIndex].tzm, 3);
//					break;
//				case 3:
//					stack[currIndex].nodeIndex = newNode(stack[currIndex].txm, 7, stack[currIndex].ty1, 8, stack[currIndex].tz1, 8);
//					break;
//				case 4:
//					stack[currIndex].nodeIndex = newNode(stack[currIndex].tx1, 8, stack[currIndex].tym, 6, stack[currIndex].tzm, 5);
//					break;
//				case 5:
//					stack[currIndex].nodeIndex = newNode(stack[currIndex].tx1, 8, stack[currIndex].tym, 7, stack[currIndex].tz1, 8);
//					break;
//				case 6:
//					stack[currIndex].nodeIndex = newNode(stack[currIndex].tx1, 8, stack[currIndex].ty1, 8, stack[currIndex].tzm, 7);
//					break;
//				case 7:
//					stack[currIndex].nodeIndex = 8;
//					break;
//			}
//
//			//printf("%d - %d\n", prevIndex, stack[currIndex].nodeIndex);
//			goto loop;
//		}
//
//		stack[currIndex].txm = (stack[currIndex].tx0 + stack[currIndex].tx1) / 2.0;
//		stack[currIndex].tym = (stack[currIndex].ty0 + stack[currIndex].ty1) / 2.0;
//		stack[currIndex].tzm = (stack[currIndex].tz0 + stack[currIndex].tz1) / 2.0;
//		//unsigned char x = stack[currIndex].nodeIndex;
//		stack[currIndex].nodeIndex = firstNode(stack[currIndex].tx0, stack[currIndex].ty0, stack[currIndex].tz0, stack[currIndex].txm, stack[currIndex].tym, stack[currIndex].tzm);
//	loop:
//			
//		//printf("%d\n", stack[currIndex].nodeIndex);
//
//		switch (stack[currIndex].nodeIndex) {
//
//			case 0: {
//				if (currIndex >= MAX_THREAD_STACK_SIZE - 1)
//					return 0;
//				currIndex++;
//				stack[currIndex].tx0 = stack[currIndex - 1].tx0; stack[currIndex].ty0 = stack[currIndex - 1].ty0; stack[currIndex].tz0 = stack[currIndex - 1].tz0; stack[currIndex].tx1 = stack[currIndex - 1].txm; stack[currIndex].ty1 = stack[currIndex - 1].tym; stack[currIndex].tz1 = stack[currIndex - 1].tzm; stack[currIndex].nodeIndex = 0; stack[currIndex].node = stack[currIndex - 1].node->children[a];
//				goto start;
//			}
//			case 1: {
//				if (currIndex >= MAX_THREAD_STACK_SIZE - 1)
//					return 0;
//				currIndex++;
//				stack[currIndex].tx0 = stack[currIndex - 1].tx0; stack[currIndex].ty0 = stack[currIndex - 1].ty0; stack[currIndex].tz0 = stack[currIndex - 1].tzm; stack[currIndex].tx1 = stack[currIndex - 1].txm; stack[currIndex].ty1 = stack[currIndex - 1].tym; stack[currIndex].tz1 = stack[currIndex - 1].tz1; stack[currIndex].nodeIndex = 1; stack[currIndex].node = stack[currIndex - 1].node->children[1 ^ a];
//				goto start;
//			}
//			case 2: {
//				if (currIndex >= MAX_THREAD_STACK_SIZE - 1)
//					return 0;
//				currIndex++;
//				stack[currIndex].tx0 = stack[currIndex - 1].tx0; stack[currIndex].ty0 = stack[currIndex - 1].tym; stack[currIndex].tz0 = stack[currIndex - 1].tz0; stack[currIndex].tx1 = stack[currIndex - 1].txm; stack[currIndex].ty1 = stack[currIndex - 1].ty1; stack[currIndex].tz1 = stack[currIndex - 1].tzm; stack[currIndex].nodeIndex = 2; stack[currIndex].node = stack[currIndex - 1].node->children[2 ^ a];
//				goto start;
//			}
//			case 3: {
//				if (currIndex >= MAX_THREAD_STACK_SIZE - 1)
//					return 0;
//				currIndex++;
//				stack[currIndex].tx0 = stack[currIndex - 1].tx0; stack[currIndex].ty0 = stack[currIndex - 1].tym; stack[currIndex].tz0 = stack[currIndex - 1].tzm; stack[currIndex].tx1 = stack[currIndex - 1].txm; stack[currIndex].ty1 = stack[currIndex - 1].ty1; stack[currIndex].tz1 = stack[currIndex - 1].tz1; stack[currIndex].nodeIndex = 3; stack[currIndex].node = stack[currIndex - 1].node->children[3 ^ a];
//				goto start;
//			}
//			case 4: {
//				if (currIndex >= MAX_THREAD_STACK_SIZE - 1)
//					return 0;
//				currIndex++;
//				stack[currIndex].tx0 = stack[currIndex - 1].txm; stack[currIndex].ty0 = stack[currIndex - 1].ty0; stack[currIndex].tz0 = stack[currIndex - 1].tz0; stack[currIndex].tx1 = stack[currIndex - 1].tx1; stack[currIndex].ty1 = stack[currIndex - 1].tym; stack[currIndex].tz1 = stack[currIndex - 1].tzm; stack[currIndex].nodeIndex = 4; stack[currIndex].node = stack[currIndex - 1].node->children[4 ^ a];
//				goto start;
//			}
//			case 5: {
//				if (currIndex >= MAX_THREAD_STACK_SIZE - 1)
//					return 0;
//				currIndex++;
//				stack[currIndex].tx0 = stack[currIndex - 1].txm; stack[currIndex].ty0 = stack[currIndex - 1].ty0; stack[currIndex].tz0 = stack[currIndex - 1].tzm; stack[currIndex].tx1 = stack[currIndex - 1].tx1; stack[currIndex].ty1 = stack[currIndex - 1].tym; stack[currIndex].tz1 = stack[currIndex - 1].tz1; stack[currIndex].nodeIndex = 5; stack[currIndex].node = stack[currIndex - 1].node->children[5 ^ a];
//				goto start;
//			}
//			case 6: {
//				if (currIndex >= MAX_THREAD_STACK_SIZE - 1)
//					return 0;
//				currIndex++;
//				stack[currIndex].tx0 = stack[currIndex - 1].txm; stack[currIndex].ty0 = stack[currIndex - 1].tym; stack[currIndex].tz0 = stack[currIndex - 1].tz0; stack[currIndex].tx1 = stack[currIndex - 1].tx1; stack[currIndex].ty1 = stack[currIndex - 1].ty1; stack[currIndex].tz1 = stack[currIndex - 1].tzm; stack[currIndex].nodeIndex = 6; stack[currIndex].node = stack[currIndex - 1].node->children[6 ^ a];
//				goto start;
//			}
//			case 7: {
//				if (currIndex >= MAX_THREAD_STACK_SIZE - 1)
//					return 0;
//				currIndex++;
//				stack[currIndex].tx0 = stack[currIndex - 1].txm; stack[currIndex].ty0 = stack[currIndex - 1].tym; stack[currIndex].tz0 = stack[currIndex - 1].tzm; stack[currIndex].tx1 = stack[currIndex - 1].tx1; stack[currIndex].ty1 = stack[currIndex - 1].ty1; stack[currIndex].tz1 = stack[currIndex - 1].tz1; stack[currIndex].nodeIndex = 7; stack[currIndex].node = stack[currIndex - 1].node->children[7 ^ a];
//				goto start;
//			}
//		}
//
//		if (stack[currIndex].nodeIndex >= 8)
//			goto end;
//	}
//	return 0;
//}