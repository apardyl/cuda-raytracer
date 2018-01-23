#include "Box.h"

Box::Box() = default;

Box::Box(Point point, float x, float y, float z) {
    this->point = point;
    this->width = x;
    this->length = y;
    this->height = z;
}

Point Box::getMin() const {
    return point;
}

Point Box::getMax() const {
    Point p;
    p.x = point.x + width;
    p.y = point.y + length;
    p.z = point.z + height;
    return p;
}

bool Box::isIntersecting(Vector vector) {
    Point pMin = getMin();
    Point pMax = getMax();

    float txMin = (pMin.x - vector.startPoint.x) / vector.x;
    float txMax = (pMax.x - vector.startPoint.x) / vector.x;

    if (txMin > txMax) {
        std::swap(txMin, txMax);
    }

    float tyMin = (pMin.y - vector.startPoint.y) / vector.y;
    float tyMax = (pMax.y - vector.startPoint.y) / vector.y;

    if (tyMin > tyMax) {
        std::swap(tyMin, tyMax);
    }

    if ((txMin > tyMax) || (tyMin > txMax)) {
        return false;
    }

    if (tyMin > txMin) {
        txMin = tyMin;
    }

    if (tyMax < txMax) {
        txMax = tyMax;
    }

    float tzMin = (pMin.z - vector.startPoint.z) / vector.z;
    float tzMax = (pMax.z - vector.startPoint.z) / vector.z;

    if (tzMin > tzMax) {
        std::swap(tzMin, tzMax);
    }

    if ((txMin > tzMax) || (tzMin > txMax)) {
        return false;
    }

    if (tzMin > txMin) {
        txMin = tzMin;
    }

    if (tzMax < txMax) {
        txMax = tzMax;
    }

    return true;
}

Box &Box::operator=(const Box &other) {
    if (this != &other) {
        this->width = other.width;
        this->length = other.length;
        this->height = other.height;
        this->point = other.point;
    }
    return *this;
}
