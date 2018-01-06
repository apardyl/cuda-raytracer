#ifndef RAY_TRACER_MATERIAL_H
#define RAY_TRACER_MATERIAL_H

#include "Color.h"

struct Material {
    Color ambient, diffuse, specular;
    float specularExponent, dissolve;

    Material() = default;
    Material(Color ambient, Color diffuse, Color specular, float specularExponent, float dissolve);
};

#endif // RAY_TRACER_MATERIAL_H
