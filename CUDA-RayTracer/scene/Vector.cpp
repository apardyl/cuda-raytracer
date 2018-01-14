#include "Vector.h"

#ifdef __NVCC__
#include <math.h>
#else
#include <cmath>
#endif

const Vector Vector::ZERO(Point(0, 0, 0), 0, 0, 0);

Vector::Vector() : x(0), y(0), z(0) {
}

Vector::Vector(const Point &a, const Point &b) : startPoint(a) {
    x = b.x - a.x;
    y = b.y - a.y;
    z = b.z - a.z;
}

Vector::Vector(const Point &startPoint, float x, float y, float z)
    : startPoint(startPoint), x(x), y(y), z(z) {
}

Vector Vector::add(const Vector &a) const {
    return {startPoint, x + a.x, y + a.y, z + a.z};
}

Vector Vector::crossProduct(const Vector a) const {
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

Vector& Vector::normalize() {
    const float div = len();
    if (div != 0) {
        x /= div;
        y /= div;
        z /= div;
    }
    return *this;
}

Vector Vector::mul(float scl) const {
    return {startPoint, this->x * scl, this->y * scl, this->z * scl};
}

void Vector::translateStartedPoint(float eps) {
	startPoint = startPoint.translate(this->mul(eps / this->len()));
}

float Vector::getAngle(Vector vector) {
	Vector temp = *this;
	Vector other = vector;
	temp.normalize();
	other.normalize();
	return acos(temp.dot(other));
}

bool Vector::isObtuse(Vector vector) {
    Vector temp = *this;
    Vector other = vector;
    temp.normalize();
    other.normalize();
    return temp.dot(other) < 0;
}

Vector Vector::rotateX(float angle) const {
    Vector vector(startPoint, x, y, z);
    performRotation(angle, vector.y, vector.z, y, -z, y, z);
    return vector;
}

Vector Vector::rotateY(float angle) const {
    Vector vector(startPoint, x, y, z);
    performRotation(angle, vector.x, vector.z, x, z, -x, z);
    return vector;
}

Vector Vector::rotateZ(float angle) const {
    Vector vector(startPoint, x, y, z);
    performRotation(angle, vector.x, vector.y, x, -y, x, y);
    return vector;
}

void Vector::performRotation(float angle,
                             float &axis1, float &axis2,
                             float axis1Mult1, float axis1Mult2,
                             float axis2Mult1, float axis2Mult2) const {
    float angleSin = std::sin(angle);
    float angleCos = std::cos(angle);
    axis1 = axis1Mult1 * angleCos + axis1Mult2 * angleSin;
    axis2 = axis2Mult1 * angleSin + axis2Mult2 * angleCos;
}
