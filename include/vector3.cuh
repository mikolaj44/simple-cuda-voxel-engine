#pragma once

#include <cuda_runtime.h>

class Vector3 {
public:
    float x = 0.0, y = 0.0, z = 0.0;

    __device__ __host__ Vector3() {};

    __device__ __host__ Vector3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {};
};

__device__ __host__ float len(Vector3 v);

__device__ __host__ float dot(Vector3 v1, Vector3 v2);

__device__ __host__ Vector3 mul(Vector3 v, float val);

__device__ __host__ Vector3 div(Vector3 v, float val);

__device__ __host__ Vector3 norm(Vector3 v);

__device__ __host__ Vector3 vmul(Vector3 v1, Vector3 v2);

__device__ __host__ Vector3 add(Vector3 v1, Vector3 v2);

__device__ __host__ Vector3 sub(Vector3 v1, Vector3 v2);

__device__ __host__ Vector3 pow(Vector3 v, float val);