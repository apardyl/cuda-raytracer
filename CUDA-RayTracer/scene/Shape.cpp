#include "Shape.h"
#include <utility>

Shape::Shape() {
    materialCode = -1;
    triangleCount = 0;
}

Shape::Shape(int triangleCount, int materialCode): materialCode(materialCode),
                                                   triangleCount(triangleCount) {
    this->triangles = new Triangle[triangleCount];
}

Shape::Shape(const Shape &shape): materialCode(shape.materialCode),
                                  triangleCount(shape.triangleCount) {
    this->triangles = new Triangle[triangleCount];
    for (int i = 0; i < triangleCount; i++) {
        this->triangles[i] = shape.triangles[i];
    }
}

Shape::Shape(Shape &&shape) noexcept: materialCode(shape.materialCode),
                                      triangleCount(shape.triangleCount) {
    std::swap(triangles, shape.triangles);
}

Shape::~Shape() {
    delete[] triangles;
}

Shape& Shape::operator=(const Shape &shape) {
    *this = Shape(shape);
    return *this;
}

Shape& Shape::operator=(Shape &&shape) noexcept {
    std::swap(materialCode, shape.materialCode);
    std::swap(triangleCount, shape.triangleCount);
    std::swap(triangles, shape.triangles);
    return *this;
}

Triangle* Shape::getTriangles() const {
    return triangles;
}

int Shape::getTriangleCount() const {
    return triangleCount;
}

int Shape::getMaterialCode() const {
    return materialCode;
}
