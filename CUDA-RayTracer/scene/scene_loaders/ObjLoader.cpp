#include "ObjLoader.h"
#include <boost/algorithm/string/predicate.hpp>
#include "IOException.h"
#include "UnknownFormatException.h"
#include <fstream>
#include <vector>
#include "ParseError.h"
#include <boost/filesystem.hpp>
#include <sstream>

using std::string;
using std::vector;

Scene ObjLoader::load(const string &fileName) {
    if (!boost::ends_with(fileName, ".obj")) {
        throw UnknowFormatException("Unknown file extension " + fileName);
    }

    std::ifstream obj(fileName);
    if (!obj.is_open()) {
        throw IOException("Unable to open " + fileName);
    }

    const boost::filesystem::path parent = boost::filesystem::absolute(fileName).parent_path();
    MaterialMap knownMaterials;
    vector<std::pair<vector<Triangle>, string>> shapes;
    vector<Point> vertices(1);
    vector<Triangle> currentShapeTriangles;
    string currentMaterialName;

    MtlParser mtlParser(knownMaterials);

    while (obj.good()) {
        string keyword;
        obj >> keyword;

        if (keyword == "o") {
            if (!currentShapeTriangles.empty()) {
                shapes.emplace_back(vector<Triangle>(), currentMaterialName);
                shapes[shapes.size() - 1].first.swap(currentShapeTriangles);
            }
        } else if (keyword == "v") {
            Point p;
            obj >> p.x >> p.y >> p.z;
            vertices.push_back(p);
        } else if (keyword == "usemtl") {
            string materialName;
            obj >> materialName;
            currentMaterialName = materialName;
        } else if (keyword == "f") {
            string face;
            std::getline(obj, face);
            std::istringstream stream(face);
            int x[3];
            for (int &i : x) {
                stream >> i;
                if (i < 1 || i >= vertices.size()) {
                    throw ParseError("Unknow vertex " + std::to_string(i));
                }
                stream.ignore(std::numeric_limits<std::streamsize>::max(), ' ');
            }
            if (stream.good()) {
                throw ParseError("Too many vertices, only triangles supported: " + face);
            }
            currentShapeTriangles.emplace_back(vertices[x[0]], vertices[x[1]], vertices[x[2]]);
        } else if (keyword == "mtllib") {
            string pathstr;
            obj >> pathstr;
            boost::filesystem::path path = boost::filesystem::absolute(pathstr, parent);
            mtlParser.parse(path.string());
        }

        obj.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    if (!currentShapeTriangles.empty()) {
        shapes.emplace_back(vector<Triangle>(), currentMaterialName);
        shapes[shapes.size() - 1].first.swap(currentShapeTriangles);
    }

    Scene scene(knownMaterials.size(), shapes.size());
    vector<string> materialNames(knownMaterials.size());
    int i = 0;
    for (const auto matPair : knownMaterials) {
        scene.getMaterials()[i] = matPair.second;
        materialNames[i] = matPair.first;
        i++;
    }
    i = 0;
    for (auto p : shapes) {
        const int materialNum = std::lower_bound(materialNames.begin(),
                                                 materialNames.end(), p.second) -
            materialNames.begin();
        if (materialNum >= scene.materialsCount || materialNum < 0) {
            throw ParseError("Unknown material " + p.second);
        }
        Shape shape(p.first.size(), materialNum);
        int j = 0;
        for (const auto t : p.first) {
            shape.getTriangles()[j++] = t;
        }
        scene.getShapes()[i++] = std::move(shape);
    }

    return scene;
}
