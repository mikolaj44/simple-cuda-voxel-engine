#include "block.cuh"

__device__ Block::Block (BlockType type_, BlockTexture* texture_) {

	type = type;
	texture = texture_;
}