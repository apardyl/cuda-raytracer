#ifndef RAY_TRACER_INTERSECTION_H
#define RAY_TRACER_INTERSECTION_H

#include "Point.h"

struct Intersection {
    Point point;
    float distance;

    static const Intersection NO_INTERSECTION;

    Intersection(const Point &point, float distance);
};

#endif //RAY_TRACER_INTERSECTION_H
