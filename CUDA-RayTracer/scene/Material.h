#ifndef RAY_TRACER_MATERIAL_H
#define RAY_TRACER_MATERIAL_H

#include "Color.h"

struct Material {
	float Ks, Kd, Ka, alfa;

	Material();

	Material(float Ks, float Kd, float Ka, float alfa);
};

#endif // RAY_TRACER_MATERIAL_H
