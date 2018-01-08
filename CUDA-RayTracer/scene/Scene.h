#ifndef RAY_TRACER_SCENE_H
#define RAY_TRACER_SCENE_H

#include "Material.h"
#include "Shape.h"

class Scene {
private:
    Material *materials = nullptr;
    Shape *shapes = nullptr;
public:
    const int materialsCount;
    const int shapesCount;

    Scene(int materialsCount, int shapesCount);

    Scene(const Scene &scene);

    Scene(Scene &&scene) noexcept;

    Scene& operator=(const Scene &scene) = delete;
    Scene& operator=(Scene &&scene) = delete;

    Material* getMaterials() const;

    Material getMaterial(int id) const {
        if (id < 0 || id >= materialsCount) {
            return {};
        } else {
            return materials[id];
        }
    }

    Shape* getShapes() const;
};

#endif // RAY_TRACER_SCENE_H
