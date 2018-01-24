#ifndef RAY_TRACER_SCENE_H
#define RAY_TRACER_SCENE_H

#include "Material.h"
#include "Triangle.h"
#include "Light.h"
#include "CameraPosition.h"

class Scene {
private:
    Material *materials = nullptr;
    Triangle *triangles = nullptr;
    Light *lights = nullptr;
public:
    const int materialsCount;
    const int trianglesCount;
    const int lightsCount;
    const CameraPosition camera;

    Scene(int materialsCount, Material * materials, int trianglesCount, 
          Triangle * triangles, int lightsCount, Light * lights, const CameraPosition& camera);

    Scene(const Scene &scene);

    Scene(Scene &&scene) noexcept;

    ~Scene();

    Scene& operator=(const Scene &scene) = delete;
    Scene& operator=(Scene &&scene) = delete;

    Material* getMaterials() const;

    const Material& getMaterial(int id) const;

    Triangle* getTriangles() const;

    Light* getLights() const;
};

#endif // RAY_TRACER_SCENE_H
