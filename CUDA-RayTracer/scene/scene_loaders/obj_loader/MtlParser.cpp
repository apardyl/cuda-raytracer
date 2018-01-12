#include "MtlParser.h"
#include <boost/algorithm/string/predicate.hpp>
#include <fstream>
#include <string>
#include "scene/scene_loaders/UnknownFormatException.h"
#include "scene/scene_loaders/IOException.h"

using std::string;

MtlParser::MtlParser(MaterialMap &materialMap): materialMap(materialMap) {
}

void MtlParser::parse(const std::string &fileName) const {
    if (!boost::ends_with(fileName, ".mtl")) {
        throw UnknowFormatException("Unknown file extension " + fileName);
    }

    std::ifstream mtl(fileName);
    if (!mtl.is_open()) {
        throw IOException("Unable to open " + fileName);
    }

    string name;
    Material material;

    while (mtl.good()) {
        string keyword;
        mtl >> keyword;

        if (keyword == "newmtl") {
            if (!name.empty()) {
                materialMap[name] = material;
            }

            material = Material();
            mtl >> name;
        } else if (keyword == "Ns") {
            mtl >> material.specularExponent;
        } else if (keyword == "Ka") {
            mtl >> material.ambient.red >> material.ambient.green >> material.ambient.blue;
        } else if (keyword == "Kd") {
            mtl >> material.diffuse.red >> material.diffuse.green >> material.diffuse.blue;
        } else if (keyword == "Ks") {
            mtl >> material.specular.red >> material.specular.green >> material.specular.blue;
        } else if (keyword == "d") {
            mtl >> material.dissolve;
        } else if (keyword == "Tr") {
            mtl >> material.dissolve;
            material.dissolve = 1 - material.dissolve;
        }

        mtl.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }

    if (!name.empty()) {
        materialMap[name] = material;
    }
}
