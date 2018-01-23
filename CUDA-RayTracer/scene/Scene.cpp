#include "Scene.h"
#include "utility"

Scene::Scene(int materialsCount, Material *materials, int trianglesCount, Triangle *triangles,
    int lightsCount, Light *lights) : materialsCount(materialsCount), materials(materials), triangles(triangles), trianglesCount(trianglesCount), lights(lights), lightsCount(lightsCount) {
}

Scene::Scene(Scene &&scene) noexcept: materialsCount(scene.materialsCount),
                                      trianglesCount(scene.trianglesCount), lightsCount(scene.lightsCount) {
    std::swap(materials, scene.materials);
    std::swap(triangles, scene.triangles);
    std::swap(lights, scene.lights);
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

Light * Scene::getLights() const {
    return lights;
}

Scene::~Scene() {
    delete[] triangles;
    delete[] materials;
    delete[] lights;
}
