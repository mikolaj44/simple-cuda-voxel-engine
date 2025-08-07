#pragma once

class Block {
public:
	int x, y, z;
	uint8_t blockId;

	__device__ __host__ Block() {};

	__device__ __host__ Block(int x_, int y_, int z_, uint8_t blockId_) : x(x_), y(y_), z(z_), blockId(blockId_) {};
};

class Octree {
public:
	void createOctree(int xMin, int yMin, int zMin, int maxLevel);

	void createOctree(int maxLevel);

	void clear();

	__device__ void insert(Block block);

	Vector3 getMinPos() const;

	unsigned int getLevel() const;

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
};

__device__ unsigned char firstNode(float tx0, float ty0, float tz0, float txm, float tym, float tzm);

__device__ unsigned char newNode(float tx, unsigned char i1, float ty, unsigned char i2, float tz, unsigned char i3);

__device__ uint32_t childMortonRevelles(uint32_t mortonCode, unsigned char revellesChildIndex);

__device__ void performRaycast(Octree* octree, float oX, float oY, float oZ, float dX, float dY, float dZ, int sX, int sY, int minNodeSize = 1, uchar4*  pixels = nullptr);

__device__ void drawTexturePixel(int blockX, int blockY, int blockZ, float oX, float oY, float oZ, float dX, float dY, float dZ, int sX, int sY, unsigned char blockId, uchar4* pixels, bool textureRenderingEnabled);

__device__ unsigned char raycastDrawPixel(Octree* octree, float oX, float oY, float oZ, float dX, float dY, float dZ, float tx0, float ty0, float tz0, float tx1, float ty1, float tz1, unsigned char a, int minNodeSize, int sX, int sY, uchar4* pixels, float origOX, float origOY, float origOZ, bool negativeDX, bool negativeDY, bool negativeDZ);


__device__ int proc_subtree(Octree* octree, float oX, float oY, float oZ, float dX, float dY, float dZ, float tx0, float ty0, float tz0, float tx1, float ty1, float tz1, unsigned char a, int minNodeSize, int sX, int sY, uchar4* pixels, int morton = 1);

__device__ int traverseChildNodes(Stack::Frame* data, unsigned char a, int minNodeSize, int sX, int sY, float origOX, float origOY, float origOZ, float origDX, float origDY, float origDZ, uchar4* pixels, Stack& stack, Octree* octree);

__device__ int traverseNewNode(float tx0, float ty0, float tz0, float&tx1, float ty1, float tz1, unsigned int nodeIdx, int minNodeSize, int sX, int sY, float origOX, float origOY, float origOZ, float origDX, float origDY, float origDZ, uchar4* pixels, Stack& stack, Octree* octree);