#ifndef RAY_TRACER_BOX_H
#define RAY_TRACER_BOX_H

#include <algorithm>

#include "scene/Point.h"
#include "scene/Vector.h"

class Box {
private:
    Point point;
    float width, length, height;

public:
    Box();

    Box(Point point, float x, float y, float z);

    Point getMin() const;

    Point getMax() const;

    bool isIntersecting(Vector vector);

    Box &operator=(const Box &other);
};

#endif //RAY_TRACER_BOX_H
