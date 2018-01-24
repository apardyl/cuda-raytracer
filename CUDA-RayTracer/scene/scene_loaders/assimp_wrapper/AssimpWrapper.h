#pragma once

#include "scene/scene_loaders/SceneLoader.h"
#include "scene/Scene.h"
#include <assimp/scene.h>
#include <unordered_map>

class AssimpWrapper : SceneLoader {
private:
    const aiScene *loadedScene = nullptr;
    Material *materials = nullptr;
    unsigned materialsCount = 0;
    Triangle *triangles = nullptr;
    unsigned triangleCount = 0;
    Light *lights = nullptr;
    unsigned lightsCount = 0;
    CameraPosition camera;

    std::unordered_map<std::string, aiNode*> nodes;

    static Color getColor(const aiMaterial *material, const char *pKey,
                          unsigned int type, unsigned int idx, Color defaulColor = Color());
    static float getFloat(const aiMaterial *material, const char *pKey,
                          unsigned int type, unsigned int idx, float defaultValue = 1.f);
    static Point getPoint(const aiMesh *mesh, unsigned idx);
    static Vector getNormal(const aiMesh *mesh, unsigned idx);

    void readGraph(aiNode *root);
    void loadMaterials();
    void loadTriangles();
    void loadLights();
    void loadCamera();
public:
    std::unique_ptr<Scene> load(const std::string &filename) override;

};
