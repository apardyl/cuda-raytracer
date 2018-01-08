#include "Intersection.h"

Intersection::Intersection(const Point &point, float distance) :
        point(point), distance(distance) {}

const Intersection Intersection::NO_INTERSECTION(Point(0, 0, 0), -1);
