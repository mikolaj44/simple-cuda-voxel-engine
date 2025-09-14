#pragma once

#include "scve/internal/structure/vector3.h"

namespace scve {

/**
* @brief The Material used by the Phong illumination model. You can set the **color**, **diffuse** which means the diffuse reflection that makes objects
look more opaque, **specular** that makes them appear shiny and **specularExponent** that controls the size of the light spots.
* */
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