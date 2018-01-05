#ifndef RAY_TRACER_TRIANGLE_H
#define RAY_TRACER_TRIANGLE_H

#include "Point.h"

struct Triangle {
    Point x, y, z;

    Triangle() = default;

    Triangle(const Point &x, const Point &y, const Point &z);

    Point getMidpoint() const;
};

#endif // RAY_TRACER_TRIANGLE_H
