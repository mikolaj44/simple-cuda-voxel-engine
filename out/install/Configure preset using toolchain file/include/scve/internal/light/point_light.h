#pragma once

#include "scve/internal/structure/vector3.h"

#include <cuda_runtime.h>

namespace scve {

class PointLight {
public:
    Vector3<> pos = Vector3<>();
    Vector3<> color = Vector3<>(255, 255, 255);

    float intensity = 1;

    PointLight() {};

    PointLight(Vector3<> pos_, Vector3<> color_ = Vector3<>(255, 255, 255), float intensity_ = 1) : pos(pos_), color(color_), intensity(intensity_) {};
};

}