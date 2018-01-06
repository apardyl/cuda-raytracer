#ifndef RAY_TRACER_MTLPARSER_H
#define RAY_TRACER_MTLPARSER_H

#include <map>
#include "scene/Material.h"

using MaterialMap = std::map<std::string, Material>;

class MtlParser {
    MaterialMap& materialMap;
public:
    MtlParser(MaterialMap &materialMap);

    void parse(const std::string& fileName) const;
};

#endif // RAY_TRACER_MTLPARSER_H
