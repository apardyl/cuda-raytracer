#ifndef RAY_TRACER_BOX_H
#define RAY_TRACER_BOX_H

#include <algorithm>
#include "scene/Point.h"
#include "scene/Vector.h"

struct Box {
    Point point;

	// width, length, height
    float x, y, z;

    Box();

    Box(Point point, float x, float y, float z);

    float getDist(Vector vector); // not implemented

    Point getMin() const;

    Point getMax() const;

    bool isIntersecting(Vector vector);

    Box &operator=(const Box &other);
};
#endif //RAY_TRACER_BOX_H