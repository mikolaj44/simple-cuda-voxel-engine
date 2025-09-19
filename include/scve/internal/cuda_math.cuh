#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace scve {

template<typename Number>
__device__ __host__ inline Number maxv(Number a, Number b) {
	if (a > b)
		return a;
	return b;
}

template<typename Number>
__device__ __host__ inline Number minv(Number a, Number b) {
	if (a < b)
		return a;
	return b;
}

template<typename Number>
__device__ __host__ inline Number absv(Number a) {
	if (a < 0)
		return -a;
	return a;
}

template<typename Number>
__device__ __host__ inline bool equals(Number a, Number b, Number epsilon) {
	if (absv(a - b) <= epsilon)
		return true;
	return false;
}

}