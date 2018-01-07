#include "Vector.h"

#ifdef __NVCC__
#include <math.h>
#else
#include <cmath>
#endif

Vector::Vector(const Point &a, const Point &b) : startPoint(a) {
    x = b.x - a.x;
    y = b.y - a.x;
    z = b.z - a.z;
}

Vector::Vector(const Point &startPoint, float x, float y, float z)
    : startPoint(startPoint), x(x), y(y), z(z) {
}

Vector Vector::add(const Vector &a) const {
    return {startPoint, x + a.x, y + a.y, z + a.z};
}

Vector Vector::cross_product(const Vector a) const {
    return {
        startPoint,
        y * a.z - a.y * z,
        z * a.x - a.z * x,
        x * a.y - a.x * y
    };
}

float Vector::dot(const Vector &a) const {
    return x * a.x + y * a.y + z * a.z;
}

float Vector::len() const {
    return sqrtf(x * x + y * y + z * z);
}

void Vector::normalize() {
    const float div = len();
    if (div != 0) {
        x /= div;
        y /= div;
        z /= div;
    }
}

Vector Vector::mul(float scl) const {
	return Vector(startPoint, this->x*scl, this->y*scl, this->z*scl);
}