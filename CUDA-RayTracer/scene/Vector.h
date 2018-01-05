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

    Vector crossProduct(const Vector a) const;

    float dot(const Vector &a) const;

    float len() const;

    void normalize();
};

#endif // RAY_TRACER_VECTOR_H
