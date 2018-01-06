#ifndef RAY_TRACER_OBJLOADER_H
#define RAY_TRACER_OBJLOADER_H

#include "scene/SceneLoader.h"
#include "MtlParser.h"

class ObjLoader : public SceneLoader {
public:
    Scene load(const std::string &filename) override;
};

#endif // RAY_TRACER_OBJLOADER_H
