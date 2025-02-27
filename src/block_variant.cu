#include "block_variant.cuh"

__device__ BlockVariant::BlockVariant (BlockType type_, BlockTexture* texture_) {

	type = type;
	texture = texture_;
}