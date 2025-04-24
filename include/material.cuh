#pragma once

#include "vector3.cuh"
#include <cuda_runtime.h>

class Material {
public:
    Vector3 color;

    float diffuse;
    float specular;
    float specularExponent;

    __device__ __host__ Material() {};

    __device__ __host__ Material(Vector3 color_, float diffuse_, float specular_, float specularExponent_) : color(color_), diffuse(diffuse_), specular(specular_), specularExponent(specularExponent_) {};
};