#ifndef RAY_TRACER_POINT_H
#define RAY_TRACER_POINT_H

#include "Models.h"

struct Point {
    float x, y, z;

    Point() = default;
    Point(float x, float y, float z);

    float getDist(const Point &a) const;
    Point translate(const Vector &a) const;
};

#endif // RAY_TRACER_POINT_H
