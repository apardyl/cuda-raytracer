#include "Triangle.h"

Triangle::Triangle(const Point &x, const Point &y, const Point &z)
    : x(x), y(y), z(z) {
}

Point Triangle::getMidpoint() const {
    const float mid_x = (x.x + y.x + z.x) / 3;
    const float mid_y = (x.y + y.y + z.y) / 3;
    const float mid_z = (x.z + y.z + z.z) / 3;
    return {mid_x, mid_y, mid_z};
}
