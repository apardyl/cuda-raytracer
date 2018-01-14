#ifndef RAY_TRACER_VECTOR_H
#define RAY_TRACER_VECTOR_H

#include "Point.h"

struct Vector {
    Point startPoint;
    float x, y, z;

    Vector();
    Vector(const Point &a, const Point &b);
    Vector(const Point &startPoint, float x, float y, float z);

    Vector& operator=(const Vector& v) = default;

    Vector add(const Vector &a) const;

    Vector mul(float scl) const;

    Vector crossProduct(Vector a) const;

    float dot(const Vector &a) const;

    Vector rotateX(float angle) const;

    Vector rotateY(float angle) const;

    Vector rotateZ(float angle) const;

    float len() const;

    Vector& normalize();

	void translateStartedPoint(float eps);

	float getAngle(Vector vector);

	bool isObtuse(Vector vector);

private:
    void performRotation(float angle,
                         float &axis1, float &axis2,
                         float axis1Mult1, float axis1Mult2,
                         float axis2Mult1, float axis2Mult2) const;
};

#endif // RAY_TRACER_VECTOR_H
