#pragma once

#include "vector3.cuh"

#include <cuda_runtime.h>

class PointLight {
public:
    Vector3 pos;
    Vector3 color;

    __device__ __host__ PointLight() {};

    __device__ __host__ PointLight(Vector3 pos_, Vector3 color_) : pos(pos_), color(color_) {};
};