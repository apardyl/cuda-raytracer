#ifndef RAY_TRACER_SHAPE_H
#define RAY_TRACER_SHAPE_H

#include "Material.h"
#include "Triangle.h"

struct Shape {
private:
    Triangle *triangles = nullptr;
    int materialCode;
    int triangleCount;

public:
    Shape();
    Shape(int triangleCount, int materialCode);

    Shape(const Shape &shape);
    Shape(Shape &&shape) noexcept;

    ~Shape();

    Shape& operator=(const Shape &shape);
    Shape& operator=(Shape &&shape) noexcept;

    Triangle* getTriangles() const;
    int getTriangleCount() const;
    int getMaterialCode() const;
};

#endif // RAY_TRACER_SHAPE_H
