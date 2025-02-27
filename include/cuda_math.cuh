#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>


__device__ __host__ inline float maxv(float a, float b) {
	if (a > b)
		return a;
	return b;
}

__device__ __host__ inline float minv(float a, float b) {
	if (a < b)
		return a;
	return b;
}

__device__ __host__ inline float absv(float a) {
	if (a < 0)
		return -a;
	return a;
}

__device__ __host__ inline bool equals(float a, float b, float epsilon) {
	if (absv(a - b) <= epsilon)
		return true;
	return false;
}