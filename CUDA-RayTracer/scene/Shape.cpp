#include "Shape.h"
#include <utility>

Shape::Shape(int triangleCount) : triangleCount(triangleCount) {
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
    std::swap(material, shape.material);
    std::swap(triangleCount, shape.triangleCount);
    std::swap(triangles, shape.triangles);
    return *this;
}
