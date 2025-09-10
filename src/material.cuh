#pragma once

#include "vector3.cuh"

class Material {
public:
    scve::Vector3<> color;

    float diffuse = 0.0;
    float specular = 0.0;
    float specularExponent = 0.0;

    __device__ __host__ Material() {};

    __device__ __host__ Material(scve::Vector3<> color_) : color(color_) {};

    __device__ __host__ Material(scve::Vector3<> color_, float diffuse_, float specular_, float specularExponent_) : color(color_), diffuse(diffuse_), specular(specular_), specularExponent(specularExponent_) {};
};