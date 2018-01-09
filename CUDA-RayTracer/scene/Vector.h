#ifndef RAY_TRACER_VECTOR_H
#define RAY_TRACER_VECTOR_H

#include "Point.h"

struct Vector {
    Point startPoint;
    float x, y, z;

    Vector() = default;
    Vector(const Point &a, const Point &b);
    Vector(const Point &startPoint, float x, float y, float z);

    Vector add(const Vector &a) const;

    Vector mul(float scl) const;

    Vector crossProduct(Vector a) const;

    float dot(const Vector &a) const;

    float len() const;

    Vector& normalize();

	void translateStartedPoint(float eps);

	float getAngle(Vector vector);

	bool isObtuse(Vector vector);
};

#endif // RAY_TRACER_VECTOR_H
