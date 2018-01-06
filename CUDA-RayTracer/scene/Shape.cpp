#include "Shape.h"

Shape::Shape(int triangleCount): material(), triangleCount(triangleCount) {
    this->triangles = new Triangle[triangleCount];
}

Shape::Shape(int triangleCount, const Material &material): Shape(triangleCount) {
    this->material = material;
}

Shape::Shape(const Shape &shape): material(shape.material), triangleCount(shape.triangleCount) {
    this->triangles = new Triangle[triangleCount];
    for (int i = 0; i < triangleCount; i++) {
        this->triangles[i] = shape.triangles[i];
    }
}

Shape::Shape(Shape &&shape) noexcept: material(shape.material), triangleCount(shape.triangleCount) {
    triangles = shape.triangles;
}

Shape::~Shape() {
    delete[] triangles;
}

Shape& Shape::operator=(const Shape &shape) {
    material = shape.material;
    triangleCount = shape.triangleCount;
    triangles = new Triangle[triangleCount];
    for (int i = 0; i < triangleCount; i++) {
        triangles[i] = shape.triangles[i];
    }
    return *this;
}
