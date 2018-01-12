#ifndef RAY_TRACER_OBJLOADER_H
#define RAY_TRACER_OBJLOADER_H

#include "scene/scene_loaders/SceneLoader.h"
#include "scene/Scene.h"

class ObjLoader : public SceneLoader {
public:
    std::unique_ptr<Scene> load(const std::string &filename) override;
};

#endif // RAY_TRACER_OBJLOADER_H
