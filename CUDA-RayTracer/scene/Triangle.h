#ifndef RAY_TRACER_TRIANGLE_H
#define RAY_TRACER_TRIANGLE_H

#include "Point.h"
#include "Material.h"
#include "Intersection.h"

struct Triangle {
	Point x, y, z;

	Triangle() = default;

	Triangle(Point a, Point b, Point c);

	Point getMidpoint() const;

	Intersection intersect(Vector vector) const;

	float getDist(Vector vector) const;

	Vector getReflectedVector(Vector vector) const;

	Vector getNormal() const;

	float getMinX() const;

	float getMinY() const;

	float getMinZ() const;

	float getMaxX() const;

	float getMaxY() const;

	float getMaxZ() const;
};

#endif // RAY_TRACER_TRIANGLE_H
