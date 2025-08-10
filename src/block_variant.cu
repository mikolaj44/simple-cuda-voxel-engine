#include "block_variant.cuh"

__device__ BlockVariant::BlockVariant (Material material_, BlockTexture* texture_) {
	material = material_;
	texture = texture_;
}