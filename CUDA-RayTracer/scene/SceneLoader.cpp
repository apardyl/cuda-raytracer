#include "SceneLoader.h"
#include <boost/algorithm/string/predicate.hpp>
#include "scene_loaders/ObjLoader.h"
#include "scene_loaders/UnknownFormatException.h"

Scene SceneLoader::loadFromFile(const std::string &fileName) {
    if (boost::ends_with(fileName, ".obj")) {
        return ObjLoader().load(fileName);
    }

    throw UnknowFormatException("Unknown file format: " + fileName);
}
