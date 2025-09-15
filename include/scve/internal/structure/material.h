#pragma once

#include "scve/internal/structure/vector3.h"

namespace scve {

class Material {
public:
    Vector3<> color;

    float diffuse = 0.0;
    float specular = 0.0;
    float specularExponent = 0.0;

    __host__ __device__ Material() {};

    __host__ __device__ Material(Vector3<> color_) : color(color_) {};

    __host__ __device__ Material(Vector3<> color_, float diffuse_, float specular_, float specularExponent_) : color(color_), diffuse(diffuse_), specular(specular_), specularExponent(specularExponent_) {};
};

}