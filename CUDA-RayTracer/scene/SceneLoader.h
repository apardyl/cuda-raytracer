#ifndef RAY_TRACER_SCENELOADER_H
#define RAY_TRACER_SCENELOADER_H

#include "Scene.h"
#include <string>

class SceneLoader {
public:
    virtual ~SceneLoader() = default;

    virtual Scene load(const std::string &filename) = 0;

    static Scene loadFromFile(const std::string &fileName);
};

#endif // RAY_TRACER_SCENELOADER_H
