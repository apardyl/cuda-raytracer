#ifndef RAY_TRACER_MATERIAL_H
#define RAY_TRACER_MATERIAL_H

#include "Color.h"

struct Material {
    Color ambient, diffuse, specular, transparent;
    float specularExponent, dissolve, refractiveIndex;

    Material();
    Material(Color ambient, Color diffuse, Color specular, float specularExponent, float dissolve);
    Material(Color ambient, Color diffuse, Color specular, Color transparent,
             float specularExponent, float dissolve, float refractiveIndex);
};

#endif // RAY_TRACER_MATERIAL_H
