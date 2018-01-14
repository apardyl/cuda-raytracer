#include "Triangle.h"
#include <algorithm>
#include <cmath>
#include <cfloat>

Triangle::Triangle(const Point a, const Point b, const Point c)
        : x(a), y(b), z(c) {
}

Triangle::Triangle(Point a, Point b, Point c, int materialCode)
        : Triangle(a, b, c) {
    this->materialCode = materialCode;
}

Point Triangle::getMidpoint() const {
    const float midX = (x.x + y.x + z.x) / 3;
    const float midY = (x.y + y.y + z.y) / 3;
    const float midZ = (x.z + y.z + z.z) / 3;
    return {midX, midY, midZ};
}

Intersection Triangle::intersect(Vector vector) const {
    vector.normalize();
    Vector aB(x, y);
    const Vector aC(x, z);
    Vector normal = aB.crossProduct(aC);
    if (fabsf(normal.dot(vector)) < FLT_EPSILON) {
        // Triangle is parallel to vector
        return Intersection::NO_INTERSECTION;
    }

    const Vector origin(
            Point(0, 0, 0),
            vector.startPoint.x, vector.startPoint.y, vector.startPoint.z);
    const Vector a(Point(0, 0, 0), x.x, x.y, x.z);
    const float d = -normal.dot(a);
    const float distToPlane = -(normal.dot(origin) + d) / (normal.dot(vector));
    if (distToPlane < 0) {
        // Vector is directed in opposite direction
        return Intersection::NO_INTERSECTION;
    }

    // Check if intersection point is inside the triangle
    const Point p(vector.startPoint.translate(vector.mul(distToPlane)));
    Vector edgeAB(x, y);
    Vector edgeBC(y, z);
    Vector edgeCA(z, x);
    const Vector aP(x, p);
    const Vector bP(y, p);
    const Vector cP(z, p);
    const bool onLeftAB = normal.dot(edgeAB.crossProduct(aP)) > 0;
    const bool onLeftBC = normal.dot(edgeBC.crossProduct(bP)) > 0;
    const bool onLeftCA = normal.dot(edgeCA.crossProduct(cP)) > 0;
    if (onLeftAB && onLeftBC && onLeftCA) {
        return Intersection(p, distToPlane);
    }

    return Intersection::NO_INTERSECTION;
}

float Triangle::getDist(Vector vector) const {
    return intersect(vector).distance;
}

Vector Triangle::getReflectedVector(Vector vector) const {
    vector.normalize();
    Vector normal = getNormal();
    Vector res = vector.add(normal.mul((-2) * vector.dot(normal)));
    res.startPoint = intersect(vector).point;
    return res.normalize();
}

Vector Triangle::getNormal() const {
    Vector aB(x, y);
    const Vector aC(x, z);
    return aB.crossProduct(aC).normalize();
}

float Triangle::getMinX() const {
    return std::min(x.x, std::min(y.x, z.x));
}

float Triangle::getMinY() const {
    return std::min(x.y, std::min(y.y, z.y));
}

float Triangle::getMinZ() const {
    return std::min(x.z, std::min(y.z, z.z));
}

float Triangle::getMaxX() const {
    return std::max(x.x, std::max(y.x, z.x));
}

float Triangle::getMaxY() const {
    return std::max(x.y, std::max(y.y, z.y));
}

float Triangle::getMaxZ() const {
    return std::max(x.z, std::max(y.z, z.z));
}
