#ifndef RAY_TRACER_LIGHT_H
#define RAY_TRACER_LIGHT_H

#include "scene/Color.h"
#include "scene/Point.h"

struct Light {
	Point point;
	Color specular, diffuse;

	Light();

	Light(Point point, Color Is, Color Id);
};
#endif //RAY_TRACER_LIGHT_H