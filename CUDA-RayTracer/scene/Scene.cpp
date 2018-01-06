#include "Scene.h"

Scene::Scene(int materialsCount, int shapesCount): materialsCount(materialsCount),
                                                   shapesCount(shapesCount) {
    materials = new Material[materialsCount];
    shapes = new Shape[shapesCount];
}

Scene::Scene(const Scene &scene): Scene(scene.materialsCount, scene.shapesCount) {
    for (int i = 0; i < materialsCount; i++) {
        materials[i] = scene.materials[i];
    }
    for (int i = 0; i < shapesCount; i++) {
        shapes[i] = scene.shapes[i];
    }
}

Scene::Scene(Scene &&scene) noexcept: materialsCount(scene.materialsCount),
                                      shapesCount(scene.shapesCount) {
    std::swap(materials, scene.materials);
    std::swap(shapes, scene.shapes);
}

Material* Scene::getMaterials() const {
    return materials;
}

Shape* Scene::getShapes() const {
    return shapes;
}
