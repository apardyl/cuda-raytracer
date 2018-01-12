#include "SceneLoader.h"
#include <boost/algorithm/string/predicate.hpp>
#include "scene/Scene.h"
#include "obj_loader/ObjLoader.h"
#include "UnknownFormatException.h"

std::unique_ptr<Scene>  SceneLoader::loadFromFile(const std::string &fileName) {
    if (boost::ends_with(fileName, ".obj")) {
        return ObjLoader().load(fileName);
    }

    throw UnknowFormatException("Unknown file format: " + fileName);
}
