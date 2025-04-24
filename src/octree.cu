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

#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <cxxabi.h>

#define epsilon 0.00001

using namespace std;


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

	cudaMalloc(&nodes, PREALLOCATE_MB_AMOUNT * sizeof(Node) * size_t(1024) * size_t(1024));
	cudaDeviceSynchronize();
	
	//memoryAvailableBytes = size_t(PREALLOCATE_MB_AMOUNT * 1024 * 1024) / sizeof(Node);

	// int status;
    // char* demangled = abi::__cxa_demangle(typeid(nodeMapFindRef).name(), nullptr, nullptr, &status);
	// cout << demangled << endl;

	//insertNode<<<1,1>>>(nodeMapFindRef, nodeMapInsertRef, 1, Node(44, false));
}

void Octree::createOctree(OctreeSpecialPosition position, unsigned int level){

	switch(position){

		case OctreeSpecialPosition::CENTERED:
			createOctree(-(1 << (level - 1)), -(1 << (level - 1)), -(1 << (level - 1)), level);
			break;
	}
}

void Octree::createOctree(){
	createOctree(0, 0, 0, 1);
}

void Octree::clear(){
	cudaMemset(nodes, 0, PREALLOCATE_MB_AMOUNT * size_t(1024) * size_t(1024));
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

void Octree::getChildXYZindex(int& x, int& y, int& z, uint32_t& index, unsigned int level, unsigned int childIndex) {

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

__device__ void Octree::morton3Ddecode(uint32_t mortonCode, int& x, int& y, int& z){

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

void Octree::display(uchar4* pixels, uint32_t index, bool showBorder, int x, int y, int z, unsigned int level){

	return;

	if(index == 1){
		x = xMin;
		y = yMin;
		z = zMin;
		level = Octree::level;
	}

	//cout << bitset<32>(index) << endl;

	uint8_t nodeId = nodes[index].id;

	//cout << (int)node.blockId << " " << node.hasChildren << endl;

	//if(level == 17)
	//	cout << (int)node.blockId << " " << node.hasChildren << " " << index << " " << level << endl;

	int size = 1 << level;

	if ((showBorder /*&& level < 8*/) || (size == 1 && nodeId & 127 != 0)) {

		//printf("%d\n", nodeId);

		// if(size == 1){
		// 	cout << nodeLevel(index, Octree::level) << " " << index << " " << Octree::level << endl;
		// }

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

		if(level == 0){
			color[0] = 255;
		}
		// else if(level == 1){
		// 	color[2] = 255;
		// }

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

	if (level <= 0 || nodeId & 128 == 0) {
		return;
	}

	for (int i = 0; i < 8; i++) {
		int x_ = x;
		int y_ = y;
		int z_ = z;
		uint32_t index_ = index;
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

void Octree::display(uchar4* pixels, bool showBorder){
	display(pixels, 1, showBorder);
}

__device__
unsigned char firstNode(float tx0, float ty0, float tz0, float txm, float tym, float tzm) {

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
unsigned char newNode(float tx, unsigned char i1, float ty, unsigned char i2, float tz, unsigned char i3) {

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

// TODO: OPTIMIZE THIS - IT SHOULD NEED ONLY 3 IF'S, NOT 8
__device__ uint32_t childMortonRevelles(uint32_t mortonCode, unsigned char revellesChildIndex){

	uint32_t code = mortonCode << 3;

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

__device__ void drawTexturePixel(int blockX, int blockY, int blockZ, float oX, float oY, float oZ, float dX, float dY, float dZ, int sX, int sY, unsigned char blockId, uchar4* pixels, bool textureRenderingEnabled) {
	if (dX == 0 || dY == 0 || dZ == 0) { // for now
		return;
	}
	
	float tmin =  minv((float)((float)blockX - oX) / dX, (float)((float)blockX + 1.0 - oX) / dX);
	float tymin = minv((float)((float)blockY - oY) / dY, (float)((float)blockY + 1.0 - oY) / dY);
	float tzmin = minv((float)((float)blockZ - oZ) / dZ, (float)((float)blockZ + 1.0 - oZ) / dZ);

	tmin = maxv(maxv(tmin, tymin), tzmin);

	float x = oX + tmin * dX;
	float y = oY + tmin * dY;
	float z = oZ + tmin * dZ;

	//printf("%d %d %d\n", x, y, z);

	setPixelById(sX, sY, blockX, blockY, blockZ, x, y, z, blockId, pixels, Vector3(oX, oY, oZ), Material(Vector3(255,255,255), 0, 1, 15), PointLight(Vector3(0,0,-500), Vector3(255, 0, 255)), true);
}

// the actual device insert function
__device__ void Octree::insert(Block block) {

	int x = block.x;
	int y = block.y;
	int z = block.z;

	//printf("%d %d %d\n", x, y, z);

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

	//printf("%d\n", x, y, z, xMin + size, yMin + size, zMin + size);

	// If the voxel is out of bounds (we don't grow the octree)
	if(x < 0 || y < 0 || z < 0 || x >= size || y >= size || z >= size){
		return;
	}

	uint32_t index = 1; // root node index
	uint32_t prevIndex = 1;
	//int numShifts = 0;

	// Iterate over all node levels up until the leaf node
	do{
		// if(level == 17){
		// 	printf("%d %llu\n", level, index);
		// }

		//cout << level << " " << bitset<32>(index) << endl;

		if (level == 0) {

			// Get the node at index (to insert the right block data)
			// auto iterator = nodeMapRef.find(index);
			// iterator->second

			nodes[index].id = block.blockId;
			return;
		}

		// if(numShifts >= 21){ // Detect index overflow
		// 	return;
		// }

		prevIndex = index;

		// Get the midpoint
		int xM = (2 * xMin + size) / 2;
		int yM = (2 * yMin + size) / 2;
		int zM = (2 * zMin + size) / 2;

		//numShifts += 1;
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
		//origin.x *= -1; //abs(octree->(root->xMin + root->size) - octree->root->xMin) / 2 - origin.x;
		//origin.x += abs(octree->(root->zMin + root->size) - octree->root->zMin) / 4; // abs(octree->(root->xMin + root->size) - octree->root->xMin);
		oX = -oX + (octree->xMin * 2 + size);// +dCenterX;
		dX = -dX;
		a |= 4;
	}
	if (dY < 0) {
		//origin.y *= -1;
		//origin.y += abs(octree->(root->zMin + root->size) - octree->root->zMin) / 4; //shift; //abs(octree->(root->yMin + root->size) - octree->root->yMin);
		oY = -oY + (octree->yMin * 2 + size);// +dCenterY;
		dY = -dY;
		a |= 2;
	}
	if (dZ < 0) {
		//origin.z *= -1; //abs(octree->(root->zMin + root->size) - octree->root->zMin) / 2- origin.z;
		//origin.z += abs(octree->(root->zMin + root->size) - octree->root->zMin) / 4; // abs(octree->(root->zMin + root->size) - octree->root->zMin);
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

	//int color[3] = { 0,0,0 };

	//printf("\n-2");

	if (maxv(maxv(tx0, ty0), tz0) < minv(minv(tx1, ty1), tz1)) {

        Stack stack;
        int foundNode = traverseNewNode(tx0, ty0, tz0, tx1, ty1, tz1, 1, minNodeSize, sX, sY, origOX, origOY, origOZ, origDX, origDY, origDZ, pixels, stack, octree);

		int index = 0;

        while (!stack.isEmpty() && foundNode == -1) {
			//printf("1");
            Stack::Frame* data = stack.top();

            foundNode = traverseChildNodes(data, a, minNodeSize, sX, sY, origOX, origOY, origOZ, origDX, origDY, origDZ, pixels, stack, octree);
			//printf("%d %d\n", foundNode, index);
			//index++;
        }

		if (foundNode <= -1) {
			setPixel(pixels, sX, sY, 30, 30, 30, 255); //30 30 255
		}

		//unsigned char result = raycastDrawPixel(octree, oX, oY, oZ, dX, dY, dZ, tx0, ty0, tz0, tx1, ty1, tz1, a, minNodeSize, sX, sY, pixels, origOX, origOY, origOZ, negativeDX, negativeDY, negativeDZ);
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

		//setPixel(pixels, sX, sY, 0, 255, 0);

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

// __device__ unsigned char raycastDrawPixel(Octree* octree, float oX, float oY, float oZ, float dX, float dY, float dZ, float tx0, float ty0, float tz0, float tx1, float ty1, float tz1, unsigned char a, int minNodeSize, int sX, int sY, uchar4* pixels, float origOX, float origOY, float origOZ, bool negativeDX, bool negativeDY, bool negativeDZ) {

// 	unsigned int MAX_THREAD_STACK_SIZE = octree->level + 1;

// 	//printf("%d\n", MAX_THREAD_STACK_SIZE);

// 	frame stack[21];

// 	for (int i = 0; i < MAX_THREAD_STACK_SIZE; i++) {
// 		stack[i].tx0 = tx0; stack[i].ty0 = ty0; stack[i].tz0 = tz0; stack[i].tx1 = tx1; stack[i].ty1 = ty1; stack[i].tz1 = tz1; stack[i].nodeIndex = 0; stack[i].mortonCode = uint32_t(1); stack[i].txm = -1; stack[i].tym = -1; stack[i].tzm = -1;
// 	}

// 	int currIndex = 0;

// 	while (currIndex >= 0 && currIndex < MAX_THREAD_STACK_SIZE) {

// 	start:

// 		uint8_t nodeId = octree->nodes[stack[currIndex].mortonCode];

// 		if(nodeId == 0){
// 			goto end;
// 		}

// 		// else if(nodeSize(stack[currIndex].mortonCode, octree->level) <= 2){
// 		// 	printf("%d %llu\n", nodeSize(stack[currIndex].mortonCode, octree->level), stack[currIndex].mortonCode);
// 		// }
// 		// else{
// 		// 	printf("%d %d %llu\n", nodeSize(stack[currIndex].mortonCode, octree->level), octree->level, stack[currIndex].mortonCode);
// 		// }

// 		//printf("%d - %f\n", currIndex, stack[currIndex].tx0);
// 		//if(nodeSize(stack[currIndex].mortonCode, octree->level) < 8)
// 		//	printf("%d\n", nodeSize(stack[currIndex].mortonCode, octree->level));

// 		// terminal (leaf) node (but not air)
// 		if (nodeLevel(stack[currIndex].mortonCode, octree->level) == 0 && nodeId != 0) {

// 			int blockX, blockY, blockZ;
// 			octree->morton3Ddecode(stack[currIndex].mortonCode, blockX, blockY, blockZ);

// 			//printf("%d\n", nodeId);

// 			drawTexturePixel(blockX, blockY, blockZ, origOX, origOY, origOZ, dX, dY, dZ, sX, sY, nodeId & 127, pixels, negativeDX, negativeDY, negativeDZ);

// 			//unsigned char* color = BlockTypeToColor(stack[currIndex].node->blockId);
// 			//setPixel(pixels, sX, sY, 0, 255, 0);

// 			return nodeId & 127;
// 		}

// 		// out of the octree
// 		else if (nodeId & 128 == 0 || stack[currIndex].tx1 < 0 || stack[currIndex].ty1 < 0 || stack[currIndex].tz1 < 0) {

// 		end:
// 			currIndex--;

// 			if (currIndex < 0)
// 				return 0;

// 			unsigned char prevIndex;

// 			// set node index of the previous stack frame
// 			switch (stack[currIndex].nodeIndex) {

// 				prevIndex = stack[currIndex].nodeIndex;

// 				case 0:
// 					stack[currIndex].nodeIndex = newNode(stack[currIndex].txm, 4, stack[currIndex].tym, 2, stack[currIndex].tzm, 1);
// 					break;
// 				case 1:
// 					stack[currIndex].nodeIndex = newNode(stack[currIndex].txm, 5, stack[currIndex].tym, 3, stack[currIndex].tz1, 8);
// 					break;
// 				case 2:
// 					stack[currIndex].nodeIndex = newNode(stack[currIndex].txm, 6, stack[currIndex].ty1, 8, stack[currIndex].tzm, 3);
// 					break;
// 				case 3:
// 					stack[currIndex].nodeIndex = newNode(stack[currIndex].txm, 7, stack[currIndex].ty1, 8, stack[currIndex].tz1, 8);
// 					break;
// 				case 4:
// 					stack[currIndex].nodeIndex = newNode(stack[currIndex].tx1, 8, stack[currIndex].tym, 6, stack[currIndex].tzm, 5);
// 					break;
// 				case 5:
// 					stack[currIndex].nodeIndex = newNode(stack[currIndex].tx1, 8, stack[currIndex].tym, 7, stack[currIndex].tz1, 8);
// 					break;
// 				case 6:
// 					stack[currIndex].nodeIndex = newNode(stack[currIndex].tx1, 8, stack[currIndex].ty1, 8, stack[currIndex].tzm, 7);
// 					break;
// 				case 7:
// 					stack[currIndex].nodeIndex = 8;
// 					break;
// 			}

// 		// 	//printf("%d - %d\n", prevIndex, stack[currIndex].nodeIndex);
// 		 	goto loop;
// 		}

// 		stack[currIndex].txm = (stack[currIndex].tx0 + stack[currIndex].tx1) / 2.0;
// 		stack[currIndex].tym = (stack[currIndex].ty0 + stack[currIndex].ty1) / 2.0;
// 		stack[currIndex].tzm = (stack[currIndex].tz0 + stack[currIndex].tz1) / 2.0;
// 		//unsigned char x = stack[currIndex].nodeIndex;
// 		stack[currIndex].nodeIndex = firstNode(stack[currIndex].tx0, stack[currIndex].ty0, stack[currIndex].tz0, stack[currIndex].txm, stack[currIndex].tym, stack[currIndex].tzm);
// 	loop:
			
// 		//printf("%d\n", stack[currIndex].nodeIndex);
// 		//printf("%d\n", stack[currIndex].mortonCode);

// 		// TODO: CHECK IN ADVANCE IF THE CHILD EXISTS
// 		switch (stack[currIndex].nodeIndex) {

// 			case 0: {
// 				if (currIndex >= MAX_THREAD_STACK_SIZE - 1)
// 					return 0;
// 				currIndex++;
// 				stack[currIndex].tx0 = stack[currIndex - 1].tx0; stack[currIndex].ty0 = stack[currIndex - 1].ty0; stack[currIndex].tz0 = stack[currIndex - 1].tz0; stack[currIndex].tx1 = stack[currIndex - 1].txm; stack[currIndex].ty1 = stack[currIndex - 1].tym; stack[currIndex].tz1 = stack[currIndex - 1].tzm; stack[currIndex].nodeIndex = 0; stack[currIndex].mortonCode = childMortonRevelles(stack[currIndex - 1].mortonCode, a);
// 				goto start;
// 			}
// 			case 1: {
// 				if (currIndex >= MAX_THREAD_STACK_SIZE - 1)
// 					return 0;
// 				currIndex++;
// 				stack[currIndex].tx0 = stack[currIndex - 1].tx0; stack[currIndex].ty0 = stack[currIndex - 1].ty0; stack[currIndex].tz0 = stack[currIndex - 1].tzm; stack[currIndex].tx1 = stack[currIndex - 1].txm; stack[currIndex].ty1 = stack[currIndex - 1].tym; stack[currIndex].tz1 = stack[currIndex - 1].tz1; stack[currIndex].nodeIndex = 1; stack[currIndex].mortonCode = childMortonRevelles(stack[currIndex - 1].mortonCode, 1 ^ a);
// 				goto start;
// 			}
// 			case 2: {
// 				if (currIndex >= MAX_THREAD_STACK_SIZE - 1)
// 					return 0;
// 				currIndex++;
// 				stack[currIndex].tx0 = stack[currIndex - 1].tx0; stack[currIndex].ty0 = stack[currIndex - 1].tym; stack[currIndex].tz0 = stack[currIndex - 1].tz0; stack[currIndex].tx1 = stack[currIndex - 1].txm; stack[currIndex].ty1 = stack[currIndex - 1].ty1; stack[currIndex].tz1 = stack[currIndex - 1].tzm; stack[currIndex].nodeIndex = 2; stack[currIndex].mortonCode = childMortonRevelles(stack[currIndex - 1].mortonCode, 2 ^ a);
// 				goto start;
// 			}
// 			case 3: {
// 				if (currIndex >= MAX_THREAD_STACK_SIZE - 1)
// 					return 0;
// 				currIndex++;
// 				stack[currIndex].tx0 = stack[currIndex - 1].tx0; stack[currIndex].ty0 = stack[currIndex - 1].tym; stack[currIndex].tz0 = stack[currIndex - 1].tzm; stack[currIndex].tx1 = stack[currIndex - 1].txm; stack[currIndex].ty1 = stack[currIndex - 1].ty1; stack[currIndex].tz1 = stack[currIndex - 1].tz1; stack[currIndex].nodeIndex = 3; stack[currIndex].mortonCode = childMortonRevelles(stack[currIndex - 1].mortonCode, 3 ^ a);
// 				goto start;
// 			}
// 			case 4: {
// 				if (currIndex >= MAX_THREAD_STACK_SIZE - 1)
// 					return 0;
// 				currIndex++;
// 				stack[currIndex].tx0 = stack[currIndex - 1].txm; stack[currIndex].ty0 = stack[currIndex - 1].ty0; stack[currIndex].tz0 = stack[currIndex - 1].tz0; stack[currIndex].tx1 = stack[currIndex - 1].tx1; stack[currIndex].ty1 = stack[currIndex - 1].tym; stack[currIndex].tz1 = stack[currIndex - 1].tzm; stack[currIndex].nodeIndex = 4; stack[currIndex].mortonCode = childMortonRevelles(stack[currIndex - 1].mortonCode, 4 ^ a);
// 				goto start;
// 			}
// 			case 5: {
// 				if (currIndex >= MAX_THREAD_STACK_SIZE - 1)
// 					return 0;
// 				currIndex++;
// 				stack[currIndex].tx0 = stack[currIndex - 1].txm; stack[currIndex].ty0 = stack[currIndex - 1].ty0; stack[currIndex].tz0 = stack[currIndex - 1].tzm; stack[currIndex].tx1 = stack[currIndex - 1].tx1; stack[currIndex].ty1 = stack[currIndex - 1].tym; stack[currIndex].tz1 = stack[currIndex - 1].tz1; stack[currIndex].nodeIndex = 5; stack[currIndex].mortonCode = childMortonRevelles(stack[currIndex - 1].mortonCode, 5 ^ a);
// 				goto start;
// 			}
// 			case 6: {
// 				if (currIndex >= MAX_THREAD_STACK_SIZE - 1)
// 					return 0;
// 				currIndex++;
// 				stack[currIndex].tx0 = stack[currIndex - 1].txm; stack[currIndex].ty0 = stack[currIndex - 1].tym; stack[currIndex].tz0 = stack[currIndex - 1].tz0; stack[currIndex].tx1 = stack[currIndex - 1].tx1; stack[currIndex].ty1 = stack[currIndex - 1].ty1; stack[currIndex].tz1 = stack[currIndex - 1].tzm; stack[currIndex].nodeIndex = 6; stack[currIndex].mortonCode = childMortonRevelles(stack[currIndex - 1].mortonCode, 6 ^ a);
// 				goto start;
// 			}
// 			case 7: {
// 				if (currIndex >= MAX_THREAD_STACK_SIZE - 1)
// 					return 0;
// 				currIndex++;
// 				stack[currIndex].tx0 = stack[currIndex - 1].txm; stack[currIndex].ty0 = stack[currIndex - 1].tym; stack[currIndex].tz0 = stack[currIndex - 1].tzm; stack[currIndex].tx1 = stack[currIndex - 1].tx1; stack[currIndex].ty1 = stack[currIndex - 1].ty1; stack[currIndex].tz1 = stack[currIndex - 1].tz1; stack[currIndex].nodeIndex = 7; stack[currIndex].mortonCode = childMortonRevelles(stack[currIndex - 1].mortonCode, 7 ^ a);
// 				goto start;
// 			}
// 		}

// 		if (stack[currIndex].nodeIndex >= 8)
// 			goto end;
// 	}
// 	return 0;
// }