#ifndef RAY_TRACER_SCENELOADER_H
#define RAY_TRACER_SCENELOADER_H

#include <string>
#include "scene/Scene.h"
#include <memory>

class SceneLoader {
public:
    virtual ~SceneLoader() = default;

    virtual std::unique_ptr<Scene> load(const std::string &filename) = 0;

    static std::unique_ptr<Scene>  loadFromFile(const std::string &fileName);
};

#endif // RAY_TRACER_SCENELOADER_H
