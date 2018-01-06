#include "Point.h"
#include "Vector.h"
#include <cmath>

Point::Point(): x(0), y(0), z(0) {
}

Point::Point(float x, float y, float z) {
    this->x = x;
    this->y = y;
    this->z = z;
}

float Point::getDist(const Point &a) const {
    const float u = this->x - a.x;
    const float v = this->y - a.y;
    const float m = this->z - a.z;
    return sqrt(u * u + v * v + m * m);
}

Point Point::translate(const Vector &a) const {
    return {x + a.x, y + a.y, z + a.z};
}
