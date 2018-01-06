#pragma once
#include "Scene.h"

class SceneLoader {
public:
    virtual ~SceneLoader() = default;

    virtual Scene load() = 0;

    static Scene loadFile(const std::string& fileName) {
        
    }
};
