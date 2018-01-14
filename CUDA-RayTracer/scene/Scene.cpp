#include "Scene.h"
#include "utility"

Scene::Scene(int materialsCount, Material *materials, int trianglesCount, Triangle *triangles) :
    materials(materials), triangles(triangles), materialsCount(materialsCount),
    trianglesCount(trianglesCount) {
}

Scene::Scene(int materialsCount, int trianglesCount) : materialsCount(materialsCount),
                                                       trianglesCount(trianglesCount) {
    materials = new Material[materialsCount];
    triangles = new Triangle[trianglesCount];
}

Scene::Scene(const Scene &scene) : Scene(scene.materialsCount, scene.trianglesCount) {
    for (int i = 0; i < materialsCount; i++) {
        materials[i] = scene.materials[i];
    }
    for (int i = 0; i < trianglesCount; i++) {
        triangles[i] = scene.triangles[i];
    }
}

Scene::Scene(Scene &&scene) noexcept: materialsCount(scene.materialsCount),
                                      trianglesCount(scene.trianglesCount) {
    std::swap(materials, scene.materials);
    std::swap(triangles, scene.triangles);
}

Material* Scene::getMaterials() const {
    return materials;
}

const Material& Scene::getMaterial(int id) const {
    if (id < 0 || id >= materialsCount) {
        return materials[0];
    } else {
        return materials[id];
    }
}

Triangle* Scene::getTriangles() const {
    return triangles;
}

Scene::~Scene() {
    delete[] triangles;
    delete[] materials;
}
