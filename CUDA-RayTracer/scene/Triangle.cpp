#include "Triangle.h"
#include "Material.h"
#include "Vector.h"
#include <algorithm>
#include <cfloat>

Triangle::Triangle(const Point a, const Point b, const Point c, Material material) {
    x = a;
    y = b;
    z = c;
    this->material = material;
}

Point Triangle::getMidpoint() const {
    const float midX = (x.x + y.x + z.x) / 3;
    const float midY = (x.y + y.y + z.y) / 3;
    const float midZ = (x.z + y.z + z.z) / 3;
    return {midX, midY, midZ};
}

// if the is no intersection return -1
float Triangle::getDist(Vector vector) const {
    vector.normalize();
    Vector aB(x, y);
    const Vector aC(x, z);
    Vector normal = aB.crossProduct(aC);
    const Vector origin(Point(0, 0, 0), vector.startPoint.x, vector.startPoint.y,
                        vector.startPoint.z);
    const Vector a(Point(0, 0, 0), x.x, x.y, x.z);
    if (fabs(normal.dot(vector)) < FLT_EPSILON) // if triangle is parallel to vector return -1
        return -1;
    const float d = -normal.dot(a);
    const float distToPlane = -(normal.dot(origin) + d) / (normal.dot(vector));
    if (distToPlane < 0) // vector is directed in opposite direction
        return -1;
    // check if intersection point is inside the triangle
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
        return distToPlane;
    }
    return -1;
}

Vector Triangle::getReflectedVector(Vector vector) const {
    vector.normalize();
    Vector aB(x, y);
    const Vector aC(x, z);
    Vector normal = aB.crossProduct(aC).normalize();
    return vector.add(normal.mul((-2) * vector.dot(normal))).normalize();
}

Vector Triangle::getNormal() const {
    Vector aB(x, y);
    const Vector aC(x, z);
    return aB.crossProduct(aC);
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
