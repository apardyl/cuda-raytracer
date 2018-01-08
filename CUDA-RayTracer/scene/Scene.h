#ifndef RAY_TRACER_SCENE_H
#define RAY_TRACER_SCENE_H

#include "Material.h"
#include "Triangle.h"

class Scene {
private:
    Material *materials = nullptr;
    Triangle *triangles = nullptr;
public:
    const int materialsCount;
    const int trianglesCount;

    Scene(int trianglesCount, int shapesCount);

    Scene(const Scene &scene);

    Scene(Scene &&scene) noexcept;

    ~Scene();

    Scene& operator=(const Scene &scene) = delete;
    Scene& operator=(Scene &&scene) = delete;

    Material* getMaterials() const;

    const Material& getMaterial(int id) const;

    Triangle* getTriangles() const;
};

#endif // RAY_TRACER_SCENE_H
