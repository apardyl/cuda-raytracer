#include "SceneLoader.h"
#include "scene/Scene.h"
#include "UnknownFormatException.h"
#include "assimp_wrapper/AssimpWrapper.h"

std::unique_ptr<Scene> SceneLoader::loadFromFile(const std::string &fileName) {
    return AssimpWrapper().load(fileName);
}
