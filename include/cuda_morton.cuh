#pragma once

#include <stdint.h>
#include <cstdio>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda_math.cuh"

using namespace std;

//https://github.com/Forceflow/libmorton
//https://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/

constexpr unsigned char OCTREE_DEPTH_BYTES = 5;
constexpr uint64_t DEPTH_MASK = (uint64_t(~0) << (sizeof(uint64_t) * 8 - OCTREE_DEPTH_BYTES)) >> 3;
constexpr uint64_t OCTANT_MASK = (uint64_t(~0) << (sizeof(uint64_t) * 8 - 3));

extern const uint_fast32_t morton3D_encode_x_256_host[256];
extern const uint_fast32_t morton3D_encode_y_256_host[256];
extern const uint_fast32_t morton3D_encode_z_256_host[256];

extern const uint_fast8_t morton3D_decode_x_512_host[512];
extern const uint_fast8_t morton3D_decode_y_512_host[512];
extern const uint_fast8_t morton3D_decode_z_512_host[512];

extern const uint64_t binaryNumbersHost[256];

extern __constant__ uint_fast32_t morton3D_encode_x_256[256];
extern __constant__ uint_fast32_t morton3D_encode_y_256[256];
extern __constant__ uint_fast32_t morton3D_encode_z_256[256];

extern __constant__ uint_fast8_t morton3D_decode_x_512[512];
extern __constant__ uint_fast8_t morton3D_decode_y_512[512];
extern __constant__ uint_fast8_t morton3D_decode_z_512[512];

extern __constant__ uint64_t binaryNumbers[256];

void initDeviceLUTs();

__device__ __host__ unsigned int binaryToDecimal(unsigned int val);

__device__ __host__ inline void morton3D_64_encode(uint64_t& m, const unsigned int x, const unsigned int y, const unsigned int z) {

	#ifdef __CUDA_ARCH__
		m = morton3D_encode_z_256[(z >> 16) & 0xFF] | // we start by shifting the third byte, since we only look at the first 21 bits
			morton3D_encode_y_256[(y >> 16) & 0xFF] |
			morton3D_encode_x_256[(x >> 16) & 0xFF];
		m = m << 48 | morton3D_encode_z_256[(z >> 8) & 0xFF] | // shifting second byte
			morton3D_encode_y_256[(y >> 8) & 0xFF] |
			morton3D_encode_x_256[(x >> 8) & 0xFF];
		m = m << 24 |
			morton3D_encode_z_256[(z) & 0xFF] | // first byte
			morton3D_encode_y_256[(y) & 0xFF] |
			morton3D_encode_x_256[(x) & 0xFF];
	#else
		m = morton3D_encode_z_256_host[(z >> 16) & 0xFF] | // we start by shifting the third byte, since we only look at the first 21 bits
			morton3D_encode_y_256_host[(y >> 16) & 0xFF] |
			morton3D_encode_x_256_host[(x >> 16) & 0xFF];
		m = m << 48 | morton3D_encode_z_256_host[(z >> 8) & 0xFF] | // shifting second byte
			morton3D_encode_y_256_host[(y >> 8) & 0xFF] |
			morton3D_encode_x_256_host[(x >> 8) & 0xFF];
		m = m << 24 |
			morton3D_encode_z_256_host[(z) & 0xFF] | // first byte
			morton3D_encode_y_256_host[(y) & 0xFF] |
			morton3D_encode_x_256_host[(x) & 0xFF];
	#endif
}

__device__ __host__ inline unsigned int morton3D_64_decodeCoord(const uint64_t m, const uint_fast8_t* LUT, const unsigned int startshift) {
	uint64_t a = 0;
	for (unsigned int i = 0; i < 7; ++i) {
		a |= (uint64_t)(LUT[(m >> ((i * 9) + startshift)) & 0x000001FF] << uint64_t(3 * i));
	}
	return static_cast<unsigned int>(a);
}

__device__ __host__ inline void morton3D_64_decode(const uint64_t m, unsigned int& x, unsigned int& y, unsigned int& z) {
	#ifdef __CUDA_ARCH__
		x = morton3D_64_decodeCoord(m, morton3D_decode_x_512, 0);
		y = morton3D_64_decodeCoord(m, morton3D_decode_y_512, 0);
		z = morton3D_64_decodeCoord(m, morton3D_decode_z_512, 0);
	#else
		x = morton3D_64_decodeCoord(m, morton3D_decode_x_512_host, 0);
		y = morton3D_64_decodeCoord(m, morton3D_decode_y_512_host, 0);
		z = morton3D_64_decodeCoord(m, morton3D_decode_z_512_host, 0);
	#endif
}

__device__ __host__ inline uint64_t mortonEncode_for(unsigned int x, unsigned int y, unsigned int z){
    uint64_t answer = 0;
    for (uint64_t i = 0; i < (sizeof(uint64_t)* CHAR_BIT)/3; ++i) {
    	answer |= ((x & ((uint64_t)1 << i)) << 2*i) | ((y & ((uint64_t)1 << i)) << (2*i + 1)) | ((z & ((uint64_t)1 << i)) << (2*i + 2));
    }
    return answer;
}

//__device__ __host__ 
inline uint64_t octree_morton3D_64_encode(unsigned int x, unsigned int y, unsigned int z, unsigned int level, unsigned int octreeX, unsigned int octreeY, unsigned int octreeZ, unsigned int octreeLevel) {

	uint64_t code = mortonEncode_for(x - octreeX, y - octreeY, z - octreeZ);
	return code;

	// code >>= level * 3;

	// uint64_t shifted_1 = 1;

	// for(int i = 0; i < octreeLevel - level; i++){
	// 	shifted_1 <<= 3;
	// }

	// return code |= shifted_1;
}

__device__ __host__ inline void octree_morton3D_64_decode(uint64_t code, int& x, int& y, int& z, unsigned int& level) {

	code -= 1;
	level = binaryToDecimal((code & DEPTH_MASK) >> 56);

	unsigned int xAbs, yAbs, zAbs;
	morton3D_64_decode(code & ~(OCTANT_MASK + DEPTH_MASK), xAbs, yAbs, zAbs);

	if (code & (uint64_t(1) << 63)) {
		x *= -1;
	}
	if (code & (uint64_t(1) << 62)) {
		y *= -1;
	}
	if (code & (uint64_t(1) << 61)) {
		z *= -1;
	}
}